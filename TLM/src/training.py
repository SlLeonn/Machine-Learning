"""Reproducible training and experiment orchestration for TLM."""

from __future__ import annotations

import gc
import json
import logging
import os
import random
import time
import warnings
from contextlib import redirect_stdout
from dataclasses import asdict, dataclass, field
from hashlib import sha256
from io import StringIO
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import torch
from numpy.typing import NDArray
from sklearn.metrics import balanced_accuracy_score, log_loss
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

from src.data import AlignedMultisensorData, PreparedFlightFold
from src.models import (
    LogisticRegressionWrapper,
    TabNetWrapper,
    TorchClassifierWrapper,
    create_model,
    metadata_from_state,
)


LOGGER = logging.getLogger(__name__)
FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
SUPPORTED_MODELS = (
    "logistic_regression",
    "tabnet",
    "tab_transformer",
    "ft_transformer",
    "saint_supervised",
)
SUPPORTED_VIEWS = ("sensor_core", "full_diagnostic")


@dataclass(frozen=True)
class BenchmarkConfig:
    """Central, serializable configuration for one benchmark profile."""

    project_root: Path
    profile: str
    seeds: tuple[int, ...]
    implementation_version: str = "tlm_multiclass_v1"
    model_names: tuple[str, ...] = SUPPORTED_MODELS
    feature_views: tuple[str, ...] = SUPPORTED_VIEWS
    device: str = "auto"
    deterministic: bool = True
    batch_size: int = 256
    inference_batch_size: int = 1_024
    max_epochs: int = 60
    patience: int = 10
    num_workers: int = 0
    selection_metric: str = "balanced_accuracy"
    min_delta: float = 1e-5
    gradient_clip_norm: float = 1.0
    model_configs: Mapping[str, Mapping[str, Any]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        root = Path(self.project_root).resolve()
        object.__setattr__(self, "project_root", root)
        if self.profile not in {"smoke", "study"}:
            raise ValueError("Profile must be either 'smoke' or 'study'")
        if not self.implementation_version.strip():
            raise ValueError("Implementation version cannot be empty")
        if not self.seeds or len(self.seeds) != len(set(self.seeds)):
            raise ValueError("Seeds must be nonempty and unique")
        unknown_models = sorted(set(self.model_names).difference(SUPPORTED_MODELS))
        if unknown_models:
            raise ValueError(f"Unsupported models: {unknown_models}")
        unknown_views = sorted(set(self.feature_views).difference(SUPPORTED_VIEWS))
        if unknown_views:
            raise ValueError(f"Unsupported feature views: {unknown_views}")
        positive = {
            "batch_size": self.batch_size,
            "inference_batch_size": self.inference_batch_size,
            "max_epochs": self.max_epochs,
            "patience": self.patience,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"Configuration values must be positive: {invalid}")
        if self.num_workers < 0:
            raise ValueError("num_workers cannot be negative")
        if self.selection_metric != "balanced_accuracy":
            raise ValueError("Only balanced_accuracy is supported for selection")
        missing_configs = sorted(set(self.model_names).difference(self.model_configs))
        if missing_configs:
            raise ValueError(f"Missing model configurations: {missing_configs}")

    @property
    def output_dir(self) -> Path:
        """Return the profile-specific output directory."""

        return self.project_root / "results" / "benchmark" / self.profile


@dataclass
class ExperimentResult:
    """Transparent record of one model, view, fold, and seed."""

    run_id: str
    profile: str
    model_name: str
    view_name: str
    fold_name: str
    development_flight: int
    test_flight: int
    seed: int
    config_fingerprint: str
    prepared_fingerprint: str
    best_epoch: int
    selection_metric: str
    best_validation_score: float | None
    tuning_seconds: float
    refit_seconds: float
    inference_seconds: float
    parameter_count: int
    reload_max_abs_difference: float
    checkpoint_path: str
    prediction_path: str
    history_path: str
    metric_path: str
    metrics: dict[str, dict[str, float]]
    per_class: dict[str, list[dict[str, Any]]]
    confusion_matrices: dict[str, list[list[int]]]
    status: str = "completed"

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible representation."""

        return asdict(self)

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> ExperimentResult:
        """Restore a result from its JSON representation."""

        return cls(**dict(values))


def make_benchmark_config(
    project_root: Path | str,
    profile: str = "smoke",
    **overrides: Any,
) -> BenchmarkConfig:
    """Build a conservative smoke profile or the full three-seed study."""

    normalized = profile.lower().strip()
    if normalized == "smoke":
        profile_values: dict[str, Any] = {
            "seeds": (42,),
            "max_epochs": 2,
            "patience": 1,
        }
    elif normalized == "study":
        profile_values = {
            "seeds": (17, 42, 73),
            "max_epochs": 60,
            "patience": 10,
        }
    else:
        raise ValueError("Profile must be either 'smoke' or 'study'")
    values: dict[str, Any] = {
        "project_root": Path(project_root),
        "profile": normalized,
        "implementation_version": "tlm_multiclass_v1",
        "device": "auto",
        "deterministic": True,
        "batch_size": 256,
        "inference_batch_size": 1_024,
        "num_workers": 0,
        "selection_metric": "balanced_accuracy",
        "min_delta": 1e-5,
        "gradient_clip_norm": 1.0,
        "model_configs": default_model_configs(),
        **profile_values,
    }
    values.update(overrides)
    return BenchmarkConfig(**values)


def default_model_configs() -> dict[str, dict[str, Any]]:
    """Return documented base configurations without an exhaustive search."""

    return {
        "logistic_regression": {
            "max_iter": 2_000,
            "solver": "lbfgs",
            "C": 1.0,
        },
        "tabnet": {
            "n_d": 16,
            "n_a": 16,
            "n_steps": 4,
            "gamma": 1.3,
            "lambda_sparse": 1e-4,
            "cat_emb_dim": 4,
            "mask_type": "entmax",
            "learning_rate": 2e-2,
            "weight_decay": 1e-5,
            "batch_size": 256,
            "virtual_batch_size": 64,
        },
        "tab_transformer": {
            "d_token": 32,
            "n_heads": 4,
            "n_layers": 2,
            "dropout": 0.10,
            "mlp_hidden": (64, 32),
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
        },
        "ft_transformer": {
            "d_token": 32,
            "n_heads": 4,
            "n_layers": 2,
            "dropout": 0.10,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
        },
        "saint_supervised": {
            "d_token": 32,
            "n_heads": 4,
            "n_layers": 2,
            "dropout": 0.10,
            "numerical_embedding_hidden": 16,
            "use_row_attention": False,
            "learning_rate": 1e-3,
            "weight_decay": 1e-5,
            "batch_size": 128,
        },
    }


def resolve_device(requested: str = "auto") -> str:
    """Resolve CUDA availability with an explicit CPU fallback."""

    normalized = requested.lower().strip()
    if normalized not in {"auto", "cpu", "cuda"}:
        raise ValueError("Device must be 'auto', 'cpu', or 'cuda'")
    if normalized == "cpu":
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if normalized == "cuda":
        LOGGER.warning("CUDA was requested but is unavailable; using CPU")
    return "cpu"


def set_reproducible_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch, and all visible CUDA devices."""

    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.enable_flash_sdp(not deterministic)
        torch.backends.cuda.enable_mem_efficient_sdp(not deterministic)
        torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = deterministic
    torch.use_deterministic_algorithms(deterministic, warn_only=False)


def fit_torch_model(
    wrapper: TorchClassifierWrapper,
    X_num: FloatArray,
    X_cat: IntArray,
    y: IntArray,
    *,
    X_num_valid: FloatArray | None,
    X_cat_valid: IntArray | None,
    y_valid: IntArray | None,
    class_weights: FloatArray,
    fit_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Optimize a torch classifier with optional validation early stopping."""

    _validate_fit_arrays(X_num, X_cat, y, wrapper.metadata.n_classes)
    has_validation = _validation_is_complete(
        X_num_valid,
        X_cat_valid,
        y_valid,
    )
    max_epochs = int(fit_config["max_epochs"])
    patience = int(fit_config.get("patience", max_epochs))
    batch_size = int(
        wrapper.model_config.get("batch_size", fit_config["batch_size"])
    )
    inference_batch_size = int(fit_config["inference_batch_size"])
    loader = _make_training_loader(
        X_num,
        X_cat,
        y,
        batch_size=batch_size,
        seed=wrapper.seed,
        num_workers=int(fit_config.get("num_workers", 0)),
    )
    weights = torch.as_tensor(
        class_weights,
        dtype=torch.float32,
        device=wrapper.device,
    )
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        wrapper.module.parameters(),
        lr=float(wrapper.model_config.get("learning_rate", 1e-3)),
        weight_decay=float(wrapper.model_config.get("weight_decay", 1e-5)),
    )
    best_score = -np.inf
    best_epoch = 0
    best_state: dict[str, Tensor] | None = None
    stale_epochs = 0
    history: list[dict[str, float]] = []
    started = _start_timer(wrapper.device)
    for epoch in range(1, max_epochs + 1):
        train_loss = _train_torch_epoch(
            wrapper,
            loader,
            criterion,
            optimizer,
            gradient_clip_norm=float(
                fit_config.get("gradient_clip_norm", 1.0)
            ),
        )
        row: dict[str, float] = {
            "epoch": float(epoch),
            "train_loss": float(train_loss),
        }
        if has_validation:
            assert X_num_valid is not None
            assert X_cat_valid is not None
            assert y_valid is not None
            probabilities = wrapper.predict_proba(
                X_num_valid,
                X_cat_valid,
                inference_batch_size,
            )
            score = float(
                balanced_accuracy_score(
                    y_valid,
                    probabilities.argmax(axis=1),
                )
            )
            row["valid_balanced_accuracy"] = score
            row["valid_log_loss"] = float(
                log_loss(
                    y_valid,
                    probabilities,
                    labels=np.arange(wrapper.metadata.n_classes),
                )
            )
            if score > best_score + float(fit_config.get("min_delta", 1e-5)):
                best_score = score
                best_epoch = epoch
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in wrapper.module.state_dict().items()
                }
                stale_epochs = 0
            else:
                stale_epochs += 1
        else:
            best_epoch = epoch
        history.append(row)
        if has_validation and stale_epochs >= patience:
            break
    elapsed = _stop_timer(started, wrapper.device)
    if has_validation:
        if best_state is None:
            raise RuntimeError("Early stopping did not capture a valid state")
        wrapper.module.load_state_dict(best_state, strict=True)
    wrapper.module.eval()
    wrapper.history = history
    wrapper.best_epoch = best_epoch
    return {
        "elapsed_seconds": elapsed,
        "best_epoch": best_epoch,
        "best_validation_score": (
            float(best_score) if has_validation else None
        ),
        "epochs_ran": len(history),
    }


def fit_tabnet_model(
    wrapper: TabNetWrapper,
    X_num: FloatArray,
    X_cat: IntArray,
    y: IntArray,
    *,
    X_num_valid: FloatArray | None,
    X_cat_valid: IntArray | None,
    y_valid: IntArray | None,
    class_weights: FloatArray,
    fit_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit native TabNet under the same validation/refit contract."""

    _validate_fit_arrays(X_num, X_cat, y, wrapper.metadata.n_classes)
    has_validation = _validation_is_complete(
        X_num_valid,
        X_cat_valid,
        y_valid,
    )
    train_matrix = _tabnet_matrix(X_num, X_cat)
    max_epochs = int(fit_config["max_epochs"])
    batch_size = int(
        wrapper.model_config.get("batch_size", fit_config["batch_size"])
    )
    virtual_batch_size = min(
        batch_size,
        int(wrapper.model_config.get("virtual_batch_size", 64)),
    )
    class_weight_mapping = {
        index: float(value) for index, value in enumerate(class_weights)
    }
    kwargs: dict[str, Any] = {
        "X_train": train_matrix,
        "y_train": y,
        "eval_metric": ["balanced_accuracy"],
        "weights": class_weight_mapping,
        "max_epochs": max_epochs,
        "patience": int(fit_config.get("patience", max_epochs)),
        "batch_size": batch_size,
        "virtual_batch_size": virtual_batch_size,
        "num_workers": int(fit_config.get("num_workers", 0)),
        "drop_last": False,
        "pin_memory": wrapper.device == "cuda",
        "compute_importance": True,
    }
    if has_validation:
        assert X_num_valid is not None
        assert X_cat_valid is not None
        assert y_valid is not None
        kwargs["eval_set"] = [
            (train_matrix, y),
            (_tabnet_matrix(X_num_valid, X_cat_valid), y_valid),
        ]
        kwargs["eval_name"] = ["train", "valid"]
    else:
        kwargs["eval_set"] = [(train_matrix, y)]
        kwargs["eval_name"] = ["train"]
        kwargs["patience"] = 0
    started = _start_timer(torch.device(wrapper.device))
    with warnings.catch_warnings(), redirect_stdout(StringIO()):
        warnings.filterwarnings(
            "ignore",
            message="Best weights from best epoch are automatically used!",
        )
        warnings.filterwarnings(
            "ignore",
            message="No early stopping will be performed.*",
        )
        wrapper.model.fit(**kwargs)
    elapsed = _stop_timer(started, torch.device(wrapper.device))
    raw_history = dict(wrapper.model.history.history)
    wrapper.history = _tabnet_history_rows(raw_history)
    if has_validation:
        wrapper.best_epoch = int(wrapper.model.best_epoch) + 1
        best_score: float | None = float(wrapper.model.best_cost)
    else:
        wrapper.best_epoch = max_epochs
        best_score = None
    return {
        "elapsed_seconds": elapsed,
        "best_epoch": wrapper.best_epoch,
        "best_validation_score": best_score,
        "epochs_ran": len(wrapper.history),
    }


def fit_logistic_model(
    wrapper: LogisticRegressionWrapper,
    X_num: FloatArray,
    X_cat: IntArray,
    y: IntArray,
    *,
    X_num_valid: FloatArray | None,
    X_cat_valid: IntArray | None,
    y_valid: IntArray | None,
    class_weights: FloatArray,
    fit_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Fit the fixed linear baseline without data-driven hyperparameters."""

    del fit_config
    _validate_fit_arrays(X_num, X_cat, y, wrapper.metadata.n_classes)
    started = time.perf_counter()
    design = wrapper._design_matrix(X_num, X_cat)
    wrapper.model.fit(design, y, sample_weight=class_weights[y])
    elapsed = time.perf_counter() - started
    row: dict[str, float] = {
        "epoch": 0.0,
        "iterations": float(np.max(wrapper.model.n_iter_)),
    }
    best_score: float | None = None
    if _validation_is_complete(X_num_valid, X_cat_valid, y_valid):
        assert X_num_valid is not None
        assert X_cat_valid is not None
        assert y_valid is not None
        prediction = wrapper.predict(
            X_num_valid,
            X_cat_valid,
            batch_size=len(X_num_valid),
        )
        best_score = float(balanced_accuracy_score(y_valid, prediction))
        row["valid_balanced_accuracy"] = best_score
    wrapper.history = [row]
    wrapper.best_epoch = 0
    return {
        "elapsed_seconds": elapsed,
        "best_epoch": 0,
        "best_validation_score": best_score,
        "epochs_ran": 1,
    }


def run_multiclass_experiment(
    model_name: str,
    prepared: PreparedFlightFold,
    aligned: AlignedMultisensorData,
    config: BenchmarkConfig,
    seed: int,
    *,
    force: bool = False,
) -> ExperimentResult:
    """Tune, refit, restore, evaluate, and persist one external-flight run."""

    if model_name not in config.model_names:
        raise ValueError(f"Model {model_name!r} is disabled in this configuration")
    if seed not in config.seeds:
        raise ValueError(f"Seed {seed} is disabled in this configuration")
    if prepared.feature_view.name not in config.feature_views:
        raise ValueError("Prepared feature view is disabled in this configuration")
    if not bool(prepared.checks.all()):
        raise AssertionError("Prepared fold contains failed integrity checks")
    device = resolve_device(config.device)
    fingerprint = _experiment_fingerprint(model_name, prepared, config, seed)
    run_id = _run_identifier(model_name, prepared, seed, fingerprint)
    paths = _run_paths(config, run_id, model_name)
    if not force:
        cached = _load_cached_result(paths, fingerprint, config.project_root)
        if cached is not None:
            LOGGER.info("Reusing completed run %s", run_id)
            return cached

    LOGGER.info(
        "Running %s | %s | %s | seed=%d | device=%s",
        model_name,
        prepared.feature_view.name,
        prepared.fold.name,
        seed,
        device,
    )
    model_config = dict(config.model_configs[model_name])
    tuning_metadata = metadata_from_state(
        prepared.tuning_state.numerical_columns,
        prepared.tuning_state.categorical_columns,
        prepared.tuning_state.categorical_cardinalities,
        prepared.class_names,
    )
    final_metadata = metadata_from_state(
        prepared.final_state.numerical_columns,
        prepared.final_state.categorical_columns,
        prepared.final_state.categorical_cardinalities,
        prepared.class_names,
    )
    _validate_metadata_compatibility(tuning_metadata, final_metadata)
    fit_values = _fit_config_values(config)

    set_reproducible_seed(seed, config.deterministic)
    tuning_model = create_model(
        model_name,
        data_metadata=tuning_metadata,
        model_config=model_config,
        device=device,
        seed=seed,
    )
    tuning_outcome = tuning_model.fit(
        prepared.train.X_num,
        prepared.train.X_cat,
        prepared.y_train,
        X_num_valid=prepared.valid.X_num,
        X_cat_valid=prepared.valid.X_cat,
        y_valid=prepared.y_valid,
        class_weights=prepared.tuning_class_weights,
        fit_config=fit_values,
    )
    tuning_history = tuning_model.get_training_history()
    selected_epoch = int(tuning_outcome["best_epoch"])
    refit_epochs = max(1, selected_epoch)
    del tuning_model
    release_resources()

    set_reproducible_seed(seed, config.deterministic)
    final_model = create_model(
        model_name,
        data_metadata=final_metadata,
        model_config=model_config,
        device=device,
        seed=seed,
    )
    refit_values = {**fit_values, "max_epochs": refit_epochs}
    refit_outcome = final_model.fit(
        prepared.development.X_num,
        prepared.development.X_cat,
        prepared.y_development,
        X_num_valid=None,
        X_cat_valid=None,
        y_valid=None,
        class_weights=prepared.final_class_weights,
        fit_config=refit_values,
    )
    refit_history = final_model.get_training_history()
    checkpoint_path = final_model.save(paths["checkpoint"])

    restored = create_model(
        model_name,
        data_metadata=final_metadata,
        model_config=model_config,
        device=device,
        seed=seed,
    )
    restored.load(checkpoint_path)
    sample_size = min(256, len(prepared.development.X_num))
    original_sample = final_model.predict_proba(
        prepared.development.X_num[:sample_size],
        prepared.development.X_cat[:sample_size],
        config.inference_batch_size,
    )
    restored_sample = restored.predict_proba(
        prepared.development.X_num[:sample_size],
        prepared.development.X_cat[:sample_size],
        config.inference_batch_size,
    )
    reload_difference = float(
        np.max(np.abs(original_sample - restored_sample), initial=0.0)
    )
    if reload_difference > 1e-6:
        raise AssertionError(
            f"Checkpoint reload changed probabilities by {reload_difference}"
        )
    del final_model
    release_resources()

    inference_started = _start_timer(torch.device(device))
    probabilities = restored.predict_proba(
        prepared.test.X_num,
        prepared.test.X_cat,
        config.inference_batch_size,
    )
    inference_seconds = _stop_timer(inference_started, torch.device(device))
    parameter_count = restored.parameter_count()
    predictions = probabilities.argmax(axis=1).astype(np.int64)

    from src.evaluation import evaluate_multiclass_regions

    stable_rows = set(prepared.fold.stable_test_row_ids.tolist())
    stable_mask = np.asarray(
        [int(row_id) in stable_rows for row_id in prepared.test.row_ids],
        dtype=bool,
    )
    evaluation = evaluate_multiclass_regions(
        prepared.y_test,
        probabilities,
        stable_mask,
        prepared.class_names,
    )
    _persist_predictions(
        paths["prediction"],
        aligned,
        prepared,
        predictions,
        probabilities,
        stable_mask,
    )
    result = ExperimentResult(
        run_id=run_id,
        profile=config.profile,
        model_name=model_name,
        view_name=prepared.feature_view.name,
        fold_name=prepared.fold.name,
        development_flight=prepared.fold.development_flight,
        test_flight=prepared.fold.test_flight,
        seed=int(seed),
        config_fingerprint=fingerprint,
        prepared_fingerprint=prepared.fingerprint,
        best_epoch=selected_epoch,
        selection_metric=config.selection_metric,
        best_validation_score=tuning_outcome["best_validation_score"],
        tuning_seconds=float(tuning_outcome["elapsed_seconds"]),
        refit_seconds=float(refit_outcome["elapsed_seconds"]),
        inference_seconds=inference_seconds,
        parameter_count=parameter_count,
        reload_max_abs_difference=reload_difference,
        checkpoint_path=_relative_path(checkpoint_path, config.project_root),
        prediction_path=_relative_path(paths["prediction"], config.project_root),
        history_path=_relative_path(paths["history"], config.project_root),
        metric_path=_relative_path(paths["metric"], config.project_root),
        metrics=evaluation["metrics"],
        per_class=evaluation["per_class"],
        confusion_matrices=evaluation["confusion_matrices"],
    )
    _persist_metric_rows(paths["metric"], result)
    _persist_run_history(
        paths["history"],
        result,
        config,
        model_config,
        tuning_history,
        refit_history,
        device,
    )
    del restored
    release_resources()
    return result


def run_multiclass_benchmark(
    aligned: AlignedMultisensorData,
    prepared_folds: Sequence[PreparedFlightFold],
    config: BenchmarkConfig,
    *,
    force: bool = False,
) -> list[ExperimentResult]:
    """Run every enabled model on identical prepared folds and seeds."""

    selected = [
        item
        for item in prepared_folds
        if item.feature_view.name in config.feature_views
    ]
    expected_pairs = {
        (fold_name, view_name)
        for fold_name in {item.fold.name for item in selected}
        for view_name in config.feature_views
    }
    actual_pairs = {
        (item.fold.name, item.feature_view.name) for item in selected
    }
    if actual_pairs != expected_pairs:
        raise ValueError("Prepared folds do not cover every enabled fold/view pair")
    results: list[ExperimentResult] = []
    for model_name in config.model_names:
        for prepared in selected:
            for seed in config.seeds:
                try:
                    result = run_multiclass_experiment(
                        model_name,
                        prepared,
                        aligned,
                        config,
                        seed,
                        force=force,
                    )
                except Exception:
                    LOGGER.exception(
                        "Experiment failed: %s/%s/%s/seed=%d",
                        model_name,
                        prepared.feature_view.name,
                        prepared.fold.name,
                        seed,
                    )
                    release_resources()
                    raise
                results.append(result)
    _persist_benchmark_outputs(aligned, prepared_folds, config, results)
    return results


def release_resources() -> None:
    """Collect Python objects and release unused CUDA cache."""

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _make_training_loader(
    X_num: FloatArray,
    X_cat: IntArray,
    y: IntArray,
    *,
    batch_size: int,
    seed: int,
    num_workers: int,
) -> DataLoader:
    dataset = TensorDataset(
        torch.as_tensor(np.array(X_num, copy=True), dtype=torch.float32),
        torch.as_tensor(np.array(X_cat, copy=True), dtype=torch.long),
        torch.as_tensor(np.array(y, copy=True), dtype=torch.long),
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        generator=generator,
        worker_init_fn=_seed_worker if num_workers else None,
    )


def _seed_worker(worker_id: int) -> None:
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed + worker_id)
    random.seed(worker_seed + worker_id)


def _train_torch_epoch(
    wrapper: TorchClassifierWrapper,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    gradient_clip_norm: float,
) -> float:
    wrapper.module.train()
    total_loss = 0.0
    total_rows = 0
    for X_num, X_cat, target in loader:
        X_num = X_num.to(wrapper.device, non_blocking=True)
        X_cat = X_cat.to(wrapper.device, non_blocking=True)
        target = target.to(wrapper.device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        logits = wrapper.module(X_num, X_cat)
        loss = criterion(logits, target)
        if not torch.isfinite(loss):
            raise FloatingPointError("Training loss became nonfinite")
        loss.backward()
        if gradient_clip_norm > 0:
            nn.utils.clip_grad_norm_(
                wrapper.module.parameters(),
                gradient_clip_norm,
            )
        optimizer.step()
        rows = len(target)
        total_loss += float(loss.detach().cpu()) * rows
        total_rows += rows
    return total_loss / total_rows


def _validate_fit_arrays(
    X_num: FloatArray,
    X_cat: IntArray,
    y: IntArray,
    n_classes: int,
) -> None:
    rows = len(y)
    if X_num.ndim != 2 or X_cat.ndim != 2 or y.ndim != 1:
        raise ValueError("Training arrays have invalid dimensions")
    if len(X_num) != rows or len(X_cat) != rows or rows == 0:
        raise ValueError("Training arrays have incompatible row counts")
    if not np.isfinite(X_num).all():
        raise ValueError("Numerical training values must be finite")
    if X_cat.min(initial=0) < 0:
        raise ValueError("Categorical indices cannot be negative")
    if set(np.unique(y)) != set(range(n_classes)):
        raise ValueError("Every target class must appear in training")


def _validation_is_complete(
    X_num: FloatArray | None,
    X_cat: IntArray | None,
    y: IntArray | None,
) -> bool:
    supplied = (X_num is not None, X_cat is not None, y is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("Validation arrays must be supplied together")
    if not all(supplied):
        return False
    assert X_num is not None and X_cat is not None and y is not None
    if len(X_num) != len(X_cat) or len(X_num) != len(y) or len(y) == 0:
        raise ValueError("Validation arrays have incompatible row counts")
    return True


def _tabnet_matrix(X_num: FloatArray, X_cat: IntArray) -> FloatArray:
    if X_cat.shape[1] == 0:
        return np.asarray(X_num, dtype=np.float32)
    return np.concatenate(
        [
            np.asarray(X_num, dtype=np.float32),
            np.asarray(X_cat, dtype=np.float32),
        ],
        axis=1,
    )


def _tabnet_history_rows(
    history: Mapping[str, Sequence[Any]],
) -> list[dict[str, float]]:
    if not history:
        return []
    epochs = max(len(values) for values in history.values())
    rows: list[dict[str, float]] = []
    for epoch in range(epochs):
        row = {"epoch": float(epoch + 1)}
        for key, values in history.items():
            if epoch < len(values):
                row[str(key)] = float(values[epoch])
        rows.append(row)
    return rows


def _fit_config_values(config: BenchmarkConfig) -> dict[str, Any]:
    return {
        "batch_size": config.batch_size,
        "inference_batch_size": config.inference_batch_size,
        "max_epochs": config.max_epochs,
        "patience": config.patience,
        "num_workers": config.num_workers,
        "selection_metric": config.selection_metric,
        "min_delta": config.min_delta,
        "gradient_clip_norm": config.gradient_clip_norm,
    }


def _validate_metadata_compatibility(tuning: Any, final: Any) -> None:
    if tuning.n_classes != final.n_classes:
        raise ValueError("Tuning and refit class counts differ")
    if tuning.n_num != final.n_num or tuning.n_cat != final.n_cat:
        raise ValueError(
            "Train-only constant removal changed model dimensions between phases"
        )
    if tuning.numerical_columns != final.numerical_columns:
        raise ValueError("Tuning and refit numerical columns differ")
    if tuning.categorical_columns != final.categorical_columns:
        raise ValueError("Tuning and refit categorical columns differ")
    if (
        tuning.categorical_cardinalities
        != final.categorical_cardinalities
    ):
        raise ValueError("Tuning and refit categorical cardinalities differ")


def _experiment_fingerprint(
    model_name: str,
    prepared: PreparedFlightFold,
    config: BenchmarkConfig,
    seed: int,
) -> str:
    payload = {
        "model_name": model_name,
        "prepared_fingerprint": prepared.fingerprint,
        "seed": int(seed),
        "profile": config.profile,
        "implementation_version": config.implementation_version,
        "training": _fit_config_values(config),
        "model_config": dict(config.model_configs[model_name]),
        "deterministic": config.deterministic,
    }
    return sha256(
        json.dumps(payload, sort_keys=True, default=list).encode("utf-8")
    ).hexdigest()


def _run_identifier(
    model_name: str,
    prepared: PreparedFlightFold,
    seed: int,
    fingerprint: str,
) -> str:
    return "__".join(
        [
            model_name,
            prepared.feature_view.name,
            prepared.fold.name,
            f"seed_{seed}",
            fingerprint[:10],
        ]
    )


def _run_paths(
    config: BenchmarkConfig,
    run_id: str,
    model_name: str,
) -> dict[str, Path]:
    extension = {
        "logistic_regression": ".joblib",
        "tabnet": ".zip",
    }.get(model_name, ".pt")
    paths = {
        "checkpoint": config.output_dir / "checkpoints" / f"{run_id}{extension}",
        "prediction": config.output_dir / "predictions" / f"{run_id}.csv",
        "history": config.output_dir / "histories" / f"{run_id}.json",
        "metric": config.output_dir / "metrics" / f"{run_id}.csv",
    }
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    return paths


def _load_cached_result(
    paths: Mapping[str, Path],
    fingerprint: str,
    project_root: Path,
) -> ExperimentResult | None:
    history_path = paths["history"]
    if not history_path.is_file():
        return None
    payload = json.loads(history_path.read_text(encoding="utf-8"))
    result = ExperimentResult.from_dict(payload["result"])
    required = (
        project_root / result.checkpoint_path,
        project_root / result.prediction_path,
        project_root / result.metric_path,
    )
    if result.config_fingerprint != fingerprint or not all(
        path.is_file() for path in required
    ):
        return None
    return result


def _persist_predictions(
    path: Path,
    aligned: AlignedMultisensorData,
    prepared: PreparedFlightFold,
    predictions: IntArray,
    probabilities: FloatArray,
    stable_mask: NDArray[np.bool_],
) -> None:
    indexed = aligned.frame.set_index("row_id")
    source = indexed.loc[prepared.test.row_ids]
    frame = pd.DataFrame(
        {
            "row_id": prepared.test.row_ids,
            "flight_id": source["flight_id"].to_numpy(dtype=np.int64),
            "time_us": source["time_us"].to_numpy(dtype=np.int64),
            "episode_id": source["episode_id"].to_numpy(dtype=np.int64),
            "y_true": prepared.y_test,
            "y_pred": predictions,
            "evaluation_region": np.where(
                stable_mask,
                "stable",
                "transition_guard",
            ),
        }
    )
    for index, class_name in enumerate(prepared.class_names):
        frame[f"probability_{index}_{class_name}"] = probabilities[:, index]
    frame.to_csv(path, index=False)


def _persist_metric_rows(path: Path, result: ExperimentResult) -> None:
    rows = []
    for region, metrics in result.metrics.items():
        rows.append(
            {
                "run_id": result.run_id,
                "profile": result.profile,
                "model_name": result.model_name,
                "view_name": result.view_name,
                "fold_name": result.fold_name,
                "development_flight": result.development_flight,
                "test_flight": result.test_flight,
                "seed": result.seed,
                "region": region,
                "best_epoch": result.best_epoch,
                "tuning_seconds": result.tuning_seconds,
                "refit_seconds": result.refit_seconds,
                "inference_seconds": result.inference_seconds,
                "parameter_count": result.parameter_count,
                **metrics,
            }
        )
    pd.DataFrame(rows).to_csv(path, index=False)


def _persist_run_history(
    path: Path,
    result: ExperimentResult,
    config: BenchmarkConfig,
    model_config: Mapping[str, Any],
    tuning_history: list[dict[str, float]],
    refit_history: list[dict[str, float]],
    device: str,
) -> None:
    payload = {
        "result": result.to_dict(),
        "configuration": _public_config(config),
        "model_config": dict(model_config),
        "resolved_device": device,
        "tuning_history": tuning_history,
        "refit_history": refit_history,
        "methodological_contract": {
            "tuning_fit_rows": "inner_train_only",
            "selection_rows": "validation_only",
            "refit_rows": "development_flight_only",
            "external_test_use": "inference_and_metrics_only_after_refit",
            "primary_region": "all_external_rows",
            "transition_guard_region": "secondary_sensitivity_only",
        },
    }
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, default=list),
        encoding="utf-8",
    )


def _persist_benchmark_outputs(
    aligned: AlignedMultisensorData,
    prepared_folds: Sequence[PreparedFlightFold],
    config: BenchmarkConfig,
    results: Sequence[ExperimentResult],
) -> None:
    from src.evaluation import persist_benchmark_tables

    config.output_dir.mkdir(parents=True, exist_ok=True)
    summary_paths = persist_benchmark_tables(results, config.output_dir)
    manifest = {
        "profile": config.profile,
        "configuration": _public_config(config),
        "alignment_fingerprint": aligned.alignment_fingerprint,
        "prepared_fingerprints": sorted(
            item.fingerprint for item in prepared_folds
        ),
        "expected_runs": (
            len(config.model_names)
            * len(config.feature_views)
            * len({item.fold.name for item in prepared_folds})
            * len(config.seeds)
        ),
        "completed_runs": len(results),
        "run_ids": [result.run_id for result in results],
        "summary_artifacts": {
            name: _relative_path(path, config.project_root)
            for name, path in summary_paths.items()
        },
        "test_used_for_selection": False,
    }
    (config.output_dir / "benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def _public_config(config: BenchmarkConfig) -> dict[str, Any]:
    values = asdict(config)
    values["project_root"] = "."
    return values


def _relative_path(path: Path | str, project_root: Path) -> str:
    return Path(path).resolve().relative_to(project_root).as_posix()


def _start_timer(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()


def _stop_timer(started: float, device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter() - started
