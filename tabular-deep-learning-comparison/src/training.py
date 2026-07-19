"""Training orchestration, reproducibility, early stopping, and checkpoints."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from .evaluation import (
    compute_classification_metrics,
    compute_regression_metrics,
    labels_from_probabilities,
    optimize_binary_threshold,
    persist_classification_results,
    persist_regression_results,
    validate_probabilities,
    validate_regression_predictions,
)
from .models import TORCH_AVAILABLE, create_model

if TORCH_AVAILABLE:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
else:  # pragma: no cover - exercised only without torch.
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


LOGGER = logging.getLogger(__name__)


def set_reproducible_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, PyTorch, and CUDA when available."""

    random.seed(seed)
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        if deterministic:
            os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            if deterministic:
                cuda_backend = torch.backends.cuda
                if hasattr(cuda_backend, "enable_flash_sdp"):
                    cuda_backend.enable_flash_sdp(False)
                if hasattr(cuda_backend, "enable_mem_efficient_sdp"):
                    cuda_backend.enable_mem_efficient_sdp(False)
                if hasattr(cuda_backend, "enable_math_sdp"):
                    cuda_backend.enable_math_sdp(True)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = deterministic
        if deterministic:
            try:
                torch.use_deterministic_algorithms(True, warn_only=False)
            except TypeError:
                torch.use_deterministic_algorithms(True)
        else:
            torch.use_deterministic_algorithms(False)


def resolve_device(requested_device: str = "auto") -> str:
    """Resolve an experiment device string without scattering device logic."""

    requested = requested_device.lower()
    if requested == "auto":
        if TORCH_AVAILABLE and torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if requested.startswith("cuda") and (
        not TORCH_AVAILABLE or not torch.cuda.is_available()
    ):
        raise ValueError("CUDA was requested but is not available.")
    return requested_device


def run_classification_experiment(
    model_name: str,
    data: Any,
    config: dict[str, Any],
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train and validate a classifier, optionally deferring final test access."""

    seed = int(config["seed"])
    set_reproducible_seed(seed, deterministic=bool(config.get("deterministic", True)))
    device = resolve_device(str(config.get("device", "auto")))
    output_dirs = _classification_output_dirs(Path(config["results_dir"]))

    run_id = _make_run_id(model_name, seed)
    merged_model_config = _merge_model_config(
        model_name=model_name,
        base_config=config,
        model_config=model_config,
        device=device,
    )
    training_config = {
        "seed": seed,
        "device": device,
        "batch_size": int(merged_model_config.get("batch_size", config["batch_size"])),
        "learning_rate": float(
            merged_model_config.get("learning_rate", config["learning_rate"])
        ),
        "weight_decay": float(
            merged_model_config.get("weight_decay", config.get("weight_decay", 0.0))
        ),
        "max_epochs": int(
            merged_model_config.get("max_epochs", config["max_epochs"])
        ),
        "patience": int(merged_model_config.get("patience", config["patience"])),
        "num_workers": int(
            merged_model_config.get("num_workers", config.get("num_workers", 0))
        ),
        "selection_metric": str(
            merged_model_config.get(
                "selection_metric", config.get("selection_metric", "roc_auc")
            )
        ),
        "threshold": float(
            merged_model_config.get("threshold", config.get("threshold", 0.5))
        ),
        "inference_batch_size": int(
            merged_model_config.get(
                "inference_batch_size", config.get("inference_batch_size", 4096)
            )
        ),
        "max_grad_norm": merged_model_config.get(
            "max_grad_norm", config.get("max_grad_norm", None)
        ),
    }
    canonical_name = _canonical_model_name(model_name)
    uses_row_attention = canonical_name == "saint_supervised" and bool(
        merged_model_config.get("use_row_attention", False)
    )
    if uses_row_attention and bool(
        config.get("require_row_independent_inference", False)
    ):
        raise ValueError(
            "This experiment requires row-independent inference; configure SAINT "
            "with use_row_attention=False."
        )
    if (
        uses_row_attention
        and training_config["inference_batch_size"] != training_config["batch_size"]
    ):
        raise ValueError(
            "SAINT row attention requires the same batch_size for training, "
            "validation, and inference because predictions are batch-dependent."
        )
    if data.split_report.get("test_split_used_for_model_selection") is not False:
        raise ValueError("The prepared data does not certify test isolation.")
    checkpoint_path = _checkpoint_path(output_dirs["checkpoints"], run_id, model_name)

    model = create_model(
        model_name=model_name,
        task="classification",
        data_metadata=data.metadata(),
        model_config=merged_model_config,
    )

    LOGGER.info("Training %s with run_id=%s", model_name, run_id)
    train_started = time.perf_counter()
    try:
        model.fit(data, training_config, checkpoint_path)
        train_time = time.perf_counter() - train_started

        valid_probabilities = model.predict_proba(
            data,
            split="valid",
            batch_size=training_config["inference_batch_size"],
        )
        validate_probabilities(valid_probabilities)
        threshold = _select_threshold(data, valid_probabilities, config)
        valid_metrics = compute_classification_metrics(
            y_true=data.y_valid,
            probabilities=valid_probabilities,
            class_names=data.class_names,
            threshold=threshold,
        )

        history = model.get_training_history()
        epochs_trained = _epochs_trained(history)
        result = {
            "run_id": run_id,
            "experiment_id": _experiment_id(config),
            "protocol_fingerprint": _protocol_fingerprint(config),
            "model_name": canonical_name,
            "implementation_version": _implementation_version(
                canonical_name, merged_model_config
            ),
            "seed": seed,
            "threshold": threshold,
            "class_names": data.class_names,
            "positive_class": data.positive_class,
            "history": history,
            "best_epoch": _best_epoch(history),
            "epochs_trained": epochs_trained,
            "reached_epoch_budget": (
                epochs_trained >= training_config["max_epochs"]
                if canonical_name != "baseline_logistic"
                else None
            ),
            "valid_metrics": valid_metrics,
            "train_time_seconds": float(train_time),
            "n_parameters": int(model.count_parameters()),
            "split_fingerprint": data.split_fingerprint(),
            "checkpoint_path": checkpoint_path,
            "prediction_scope": (
                "batch_contextual" if uses_row_attention else "row_independent"
            ),
            "test_evaluated": False,
            "config": {
                "experiment": config,
                "model": merged_model_config,
                "training": training_config,
                "data_metadata": data.metadata(),
                "test_split_used_for_model_selection": False,
            },
        }
        result["reload_check"] = verify_checkpoint_reload(
            model_name=model_name,
            data=data,
            model_config=merged_model_config,
            checkpoint_path=checkpoint_path,
            reference_probabilities=valid_probabilities,
            batch_size=training_config["inference_batch_size"],
        )
        if bool(config.get("defer_test_until_final", False)):
            return result
        return _complete_classification_result(
            result=result,
            model=model,
            data=data,
            output_dirs=output_dirs,
            batch_size=training_config["inference_batch_size"],
        )
    except Exception:
        LOGGER.exception("Experiment failed for model %s", model_name)
        raise
    finally:
        if not bool(config.get("keep_model", False)):
            del model
            release_model_resources()


def finalize_classification_experiment(
    pending_result: dict[str, Any],
    data: Any,
) -> dict[str, Any]:
    """Evaluate one validation-selected classifier on the isolated test split."""

    if pending_result.get("test_evaluated") is not False:
        raise ValueError("Only a pending classification result can be finalized.")
    if str(pending_result["split_fingerprint"]) != data.split_fingerprint():
        raise ValueError("The pending result and prepared data use different splits.")
    if pending_result.get("reload_check", {}).get("checkpoint_reloaded") is not True:
        raise ValueError("Checkpoint integrity must be verified before final test use.")

    result = dict(pending_result)
    stored_config = result["config"]
    model_config = dict(stored_config["model"])
    training_config = dict(stored_config["training"])
    experiment_config = stored_config["experiment"]
    output_dirs = _classification_output_dirs(Path(experiment_config["results_dir"]))
    model = create_model(
        model_name=str(result["model_name"]),
        task="classification",
        data_metadata=data.metadata(),
        model_config=model_config,
    )
    try:
        model.load(Path(result["checkpoint_path"]))
        return _complete_classification_result(
            result=result,
            model=model,
            data=data,
            output_dirs=output_dirs,
            batch_size=int(training_config["inference_batch_size"]),
        )
    except Exception:
        LOGGER.exception(
            "Final test evaluation failed for model %s",
            result["model_name"],
        )
        raise
    finally:
        del model
        release_model_resources()


def _complete_classification_result(
    result: dict[str, Any],
    model: Any,
    data: Any,
    output_dirs: dict[str, Path],
    batch_size: int,
) -> dict[str, Any]:
    inference_started = time.perf_counter()
    test_probabilities = model.predict_proba(
        data,
        split="test",
        batch_size=batch_size,
    )
    inference_time = time.perf_counter() - inference_started
    validate_probabilities(test_probabilities)
    threshold = result.get("threshold")
    result.update(
        {
            "test_metrics": compute_classification_metrics(
                y_true=data.y_test,
                probabilities=test_probabilities,
                class_names=data.class_names,
                threshold=threshold,
            ),
            "inference_time_seconds": float(inference_time),
            "probabilities": test_probabilities,
            "y_pred": labels_from_probabilities(test_probabilities, threshold),
            "y_true": data.y_test,
            "test_indices": data.test_indices,
            "test_evaluated": True,
        }
    )
    result["paths"] = persist_classification_results(result, output_dirs)
    return result


def run_regression_experiment(
    model_name: str,
    data: Any,
    config: dict[str, Any],
    model_config: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Train and validate a regressor, optionally deferring final test access."""

    seed = int(config["seed"])
    set_reproducible_seed(seed, deterministic=bool(config.get("deterministic", True)))
    device = resolve_device(str(config.get("device", "auto")))
    output_dirs = _regression_output_dirs(Path(config["results_dir"]))
    canonical_name = _canonical_regression_model_name(model_name)
    run_id = _make_regression_run_id(model_name, seed)
    merged_model_config = _merge_model_config(
        model_name=canonical_name,
        base_config=config,
        model_config=model_config,
        device=device,
    )
    training_config = {
        "seed": seed,
        "device": device,
        "batch_size": int(merged_model_config.get("batch_size", config["batch_size"])),
        "learning_rate": float(
            merged_model_config.get("learning_rate", config["learning_rate"])
        ),
        "weight_decay": float(
            merged_model_config.get("weight_decay", config.get("weight_decay", 0.0))
        ),
        "max_epochs": int(
            merged_model_config.get("max_epochs", config["max_epochs"])
        ),
        "patience": int(merged_model_config.get("patience", config["patience"])),
        "num_workers": int(
            merged_model_config.get("num_workers", config.get("num_workers", 0))
        ),
        "selection_metric": str(
            merged_model_config.get(
                "selection_metric", config.get("selection_metric", "rmse")
            )
        ),
        "inference_batch_size": int(
            merged_model_config.get(
                "inference_batch_size", config.get("inference_batch_size", 4096)
            )
        ),
        "max_grad_norm": merged_model_config.get(
            "max_grad_norm", config.get("max_grad_norm", None)
        ),
    }
    uses_row_attention = canonical_name == "saint_supervised" and bool(
        merged_model_config.get("use_row_attention", False)
    )
    if uses_row_attention and bool(
        config.get("require_row_independent_inference", False)
    ):
        raise ValueError(
            "This experiment requires row-independent inference; configure SAINT "
            "with use_row_attention=False."
        )
    if (
        uses_row_attention
        and training_config["inference_batch_size"] != training_config["batch_size"]
    ):
        raise ValueError(
            "SAINT row attention requires the same batch_size for training, "
            "validation, and inference because predictions are batch-dependent."
        )
    if data.split_report.get("test_split_used_for_model_selection") is not False:
        raise ValueError("The prepared data does not certify test isolation.")
    checkpoint_path = _checkpoint_path(output_dirs["checkpoints"], run_id, model_name)
    model = create_model(
        model_name=model_name,
        task="regression",
        data_metadata=data.metadata(),
        model_config=merged_model_config,
    )

    LOGGER.info("Training %s with run_id=%s", model_name, run_id)
    train_started = time.perf_counter()
    try:
        model.fit(data, training_config, checkpoint_path)
        train_time = time.perf_counter() - train_started
        valid_predictions = model.predict(
            data,
            split="valid",
            batch_size=training_config["inference_batch_size"],
        )
        validate_regression_predictions(data.y_valid, valid_predictions)
        valid_metrics = compute_regression_metrics(data.y_valid, valid_predictions)
        history = model.get_training_history()
        epochs_trained = _epochs_trained(history)
        result = {
            "run_id": run_id,
            "experiment_id": _experiment_id(config),
            "protocol_fingerprint": _protocol_fingerprint(config),
            "model_name": canonical_name,
            "implementation_version": _implementation_version(
                canonical_name, merged_model_config
            ),
            "seed": seed,
            "target_name": data.target_name,
            "target_unit": data.target_unit,
            "history": history,
            "best_epoch": _best_epoch(history),
            "epochs_trained": epochs_trained,
            "reached_epoch_budget": (
                epochs_trained >= training_config["max_epochs"]
                if canonical_name != "baseline_ridge"
                else None
            ),
            "valid_metrics": valid_metrics,
            "train_time_seconds": float(train_time),
            "n_parameters": int(model.count_parameters()),
            "split_fingerprint": data.split_fingerprint(),
            "checkpoint_path": checkpoint_path,
            "prediction_scope": (
                "batch_contextual" if uses_row_attention else "row_independent"
            ),
            "test_evaluated": False,
            "config": {
                "experiment": config,
                "model": merged_model_config,
                "training": training_config,
                "data_metadata": data.metadata(),
                "test_split_used_for_model_selection": False,
            },
        }
        result["reload_check"] = verify_regression_checkpoint_reload(
            model_name=model_name,
            data=data,
            model_config=merged_model_config,
            checkpoint_path=checkpoint_path,
            reference_predictions=valid_predictions,
            batch_size=training_config["inference_batch_size"],
        )
        if bool(config.get("defer_test_until_final", False)):
            return result
        return _complete_regression_result(
            result=result,
            model=model,
            data=data,
            output_dirs=output_dirs,
            batch_size=training_config["inference_batch_size"],
        )
    except Exception:
        LOGGER.exception("Experiment failed for model %s", model_name)
        raise
    finally:
        if not bool(config.get("keep_model", False)):
            del model
            release_model_resources()


def finalize_regression_experiment(
    pending_result: dict[str, Any],
    data: Any,
) -> dict[str, Any]:
    """Evaluate one validation-selected checkpoint on the isolated test split."""

    if pending_result.get("test_evaluated") is not False:
        raise ValueError("Only a pending regression result can be finalized.")
    if str(pending_result["split_fingerprint"]) != data.split_fingerprint():
        raise ValueError("The pending result and prepared data use different splits.")
    if pending_result.get("reload_check", {}).get("checkpoint_reloaded") is not True:
        raise ValueError("Checkpoint integrity must be verified before final test use.")

    result = dict(pending_result)
    stored_config = result["config"]
    model_config = dict(stored_config["model"])
    training_config = dict(stored_config["training"])
    experiment_config = stored_config["experiment"]
    output_dirs = _regression_output_dirs(Path(experiment_config["results_dir"]))
    model = create_model(
        model_name=str(result["model_name"]),
        task="regression",
        data_metadata=data.metadata(),
        model_config=model_config,
    )
    try:
        model.load(Path(result["checkpoint_path"]))
        return _complete_regression_result(
            result=result,
            model=model,
            data=data,
            output_dirs=output_dirs,
            batch_size=int(training_config["inference_batch_size"]),
        )
    except Exception:
        LOGGER.exception(
            "Final test evaluation failed for model %s",
            result["model_name"],
        )
        raise
    finally:
        del model
        release_model_resources()


def _complete_regression_result(
    result: dict[str, Any],
    model: Any,
    data: Any,
    output_dirs: dict[str, Path],
    batch_size: int,
) -> dict[str, Any]:
    inference_started = time.perf_counter()
    test_predictions = model.predict(
        data,
        split="test",
        batch_size=batch_size,
    )
    inference_time = time.perf_counter() - inference_started
    validate_regression_predictions(data.y_test, test_predictions)
    result.update(
        {
            "test_metrics": compute_regression_metrics(
                data.y_test,
                test_predictions,
            ),
            "inference_time_seconds": float(inference_time),
            "predictions": test_predictions,
            "y_true": data.y_test,
            "test_indices": data.test_indices,
            "test_evaluated": True,
        }
    )
    result["paths"] = persist_regression_results(result, output_dirs)
    return result


def train_torch_classifier(
    model: Any,
    data: Any,
    training_config: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Train a PyTorch wrapper with validation early stopping."""

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for this model.")

    device = torch.device(str(training_config["device"]))
    train_loader = _torch_loader(
        data.X_cat_train,
        data.X_num_train,
        data.y_train,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        seed=int(training_config["seed"]),
        num_workers=int(training_config.get("num_workers", 0)),
    )
    valid_loader = _torch_loader(
        data.X_cat_valid,
        data.X_num_valid,
        data.y_valid,
        batch_size=int(training_config.get("inference_batch_size", 4096)),
        shuffle=False,
        seed=int(training_config["seed"]),
        num_workers=int(training_config.get("num_workers", 0)),
    )

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.network.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )

    selection_metric = str(training_config.get("selection_metric", "roc_auc"))
    patience = int(training_config["patience"])
    max_epochs = int(training_config["max_epochs"])
    threshold = float(training_config.get("threshold", 0.5))
    max_grad_norm = training_config.get("max_grad_norm", None)

    history: dict[str, Any] = {
        "train_loss": [],
        "valid_loss": [],
        "valid_accuracy": [],
        "valid_balanced_accuracy": [],
        "valid_f1": [],
        "valid_roc_auc": [],
        "selection_metric": selection_metric,
    }
    best_score = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = _train_one_epoch(
            network=model.network,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_grad_norm=max_grad_norm,
        )
        valid_loss, valid_probabilities = _evaluate_torch(
            network=model.network,
            loader=valid_loader,
            criterion=criterion,
            device=device,
        )
        valid_metrics = compute_classification_metrics(
            y_true=data.y_valid,
            probabilities=valid_probabilities,
            class_names=data.class_names,
            threshold=threshold,
        )

        history["train_loss"].append(float(train_loss))
        history["valid_loss"].append(float(valid_loss))
        history["valid_accuracy"].append(float(valid_metrics["accuracy"]))
        history["valid_balanced_accuracy"].append(
            float(valid_metrics["balanced_accuracy"])
        )
        history["valid_f1"].append(float(valid_metrics["f1"]))
        history["valid_roc_auc"].append(float(valid_metrics["roc_auc"]))

        score = _selection_score(selection_metric, valid_loss, valid_metrics)
        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            model.history = dict(history)
            model.history.update(
                {
                    "best_epoch": best_epoch,
                    "best_score": float(best_score),
                }
            )
            model.save(checkpoint_path)
        else:
            epochs_without_improvement += 1

        LOGGER.info(
            "epoch=%03d train_loss=%.4f valid_loss=%.4f %s=%.4f",
            epoch,
            train_loss,
            valid_loss,
            selection_metric,
            score,
        )
        if epochs_without_improvement >= patience:
            break

    model.load(checkpoint_path)
    history["best_epoch"] = best_epoch
    history["best_score"] = float(best_score)
    history["epochs_trained"] = len(history["train_loss"])
    history["reached_epoch_budget"] = history["epochs_trained"] >= max_epochs
    history["stopped_early"] = history["epochs_trained"] < max_epochs
    model.history = history
    model.save(checkpoint_path)
    return history


def train_torch_regressor(
    model: Any,
    data: Any,
    training_config: dict[str, Any],
    checkpoint_path: Path,
) -> dict[str, Any]:
    """Train a scalar PyTorch regressor with validation early stopping."""

    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch is required for this model.")
    device = torch.device(str(training_config["device"]))
    train_loader = _torch_regression_loader(
        data.X_cat_train,
        data.X_num_train,
        data.y_train_scaled,
        batch_size=int(training_config["batch_size"]),
        shuffle=True,
        seed=int(training_config["seed"]),
        num_workers=int(training_config.get("num_workers", 0)),
    )
    valid_loader = _torch_regression_loader(
        data.X_cat_valid,
        data.X_num_valid,
        data.y_valid_scaled,
        batch_size=int(training_config.get("inference_batch_size", 4096)),
        shuffle=False,
        seed=int(training_config["seed"]),
        num_workers=int(training_config.get("num_workers", 0)),
    )
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(
        model.network.parameters(),
        lr=float(training_config["learning_rate"]),
        weight_decay=float(training_config.get("weight_decay", 0.0)),
    )
    selection_metric = str(training_config.get("selection_metric", "rmse"))
    patience = int(training_config["patience"])
    max_epochs = int(training_config["max_epochs"])
    max_grad_norm = training_config.get("max_grad_norm", None)
    history: dict[str, Any] = {
        "train_loss": [],
        "valid_loss": [],
        "valid_rmse": [],
        "valid_mae": [],
        "valid_r2": [],
        "selection_metric": selection_metric,
        "loss_scale": "standardized_target",
        "metrics_scale": "original_target_units",
    }
    best_score = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_loss = _train_one_epoch(
            network=model.network,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            max_grad_norm=max_grad_norm,
        )
        valid_loss, valid_scaled_predictions = _evaluate_torch_regressor(
            network=model.network,
            loader=valid_loader,
            criterion=criterion,
            device=device,
        )
        valid_predictions = data.inverse_transform_target(valid_scaled_predictions)
        valid_metrics = compute_regression_metrics(data.y_valid, valid_predictions)
        history["train_loss"].append(float(train_loss))
        history["valid_loss"].append(float(valid_loss))
        history["valid_rmse"].append(float(valid_metrics["rmse"]))
        history["valid_mae"].append(float(valid_metrics["mae"]))
        history["valid_r2"].append(float(valid_metrics["r2"]))

        score = _regression_selection_score(
            selection_metric,
            valid_loss,
            valid_metrics,
        )
        if score > best_score:
            best_score = score
            best_epoch = epoch
            epochs_without_improvement = 0
            model.history = dict(history)
            model.history.update(
                {
                    "best_epoch": best_epoch,
                    "best_score": float(best_score),
                }
            )
            model.save(checkpoint_path)
        else:
            epochs_without_improvement += 1

        LOGGER.info(
            "epoch=%03d train_loss=%.4f valid_loss=%.4f %s=%.4f",
            epoch,
            train_loss,
            valid_loss,
            selection_metric,
            -score if selection_metric.lower() != "r2" else score,
        )
        if epochs_without_improvement >= patience:
            break

    model.load(checkpoint_path)
    history["best_epoch"] = best_epoch
    history["best_score"] = float(best_score)
    history["epochs_trained"] = len(history["train_loss"])
    history["reached_epoch_budget"] = history["epochs_trained"] >= max_epochs
    history["stopped_early"] = history["epochs_trained"] < max_epochs
    model.history = history
    model.save(checkpoint_path)
    return history


def verify_checkpoint_reload(
    model_name: str,
    data: Any,
    model_config: dict[str, Any],
    checkpoint_path: Path,
    reference_probabilities: np.ndarray,
    batch_size: int,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Reload a checkpoint and compare validation probabilities."""

    reloaded = create_model(
        model_name=model_name,
        task="classification",
        data_metadata=data.metadata(),
        model_config=model_config,
    )
    reloaded.load(checkpoint_path)
    reloaded_probabilities = reloaded.predict_proba(
        data,
        split="valid",
        batch_size=batch_size,
    )
    max_abs_difference = float(
        np.max(np.abs(reference_probabilities - reloaded_probabilities))
    )
    if max_abs_difference > tolerance:
        raise ValueError(
            "Reloaded checkpoint predictions differ from in-memory predictions by "
            f"{max_abs_difference:.6f}."
        )
    del reloaded
    release_model_resources()
    return {
        "checkpoint_reloaded": True,
        "max_abs_probability_difference": max_abs_difference,
        "tolerance": tolerance,
    }


def verify_regression_checkpoint_reload(
    model_name: str,
    data: Any,
    model_config: dict[str, Any],
    checkpoint_path: Path,
    reference_predictions: np.ndarray,
    batch_size: int,
    tolerance: float = 1e-4,
) -> dict[str, Any]:
    """Reload a regression checkpoint and compare validation predictions."""

    reloaded = create_model(
        model_name=model_name,
        task="regression",
        data_metadata=data.metadata(),
        model_config=model_config,
    )
    reloaded.load(checkpoint_path)
    restored_predictions = reloaded.predict(
        data,
        split="valid",
        batch_size=batch_size,
    )
    max_abs_difference = float(
        np.max(np.abs(reference_predictions - restored_predictions))
    )
    if max_abs_difference > tolerance:
        raise ValueError(
            "Reloaded checkpoint predictions differ from in-memory predictions by "
            f"{max_abs_difference:.6f}."
        )
    del reloaded
    release_model_resources()
    return {
        "checkpoint_reloaded": True,
        "max_abs_prediction_difference": max_abs_difference,
        "tolerance": tolerance,
    }


def release_model_resources() -> None:
    """Release Python and CUDA memory between model runs."""

    gc.collect()
    if TORCH_AVAILABLE and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _torch_loader(
    X_cat: np.ndarray,
    X_num: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> Any:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(
        torch.as_tensor(X_cat, dtype=torch.long),
        torch.as_tensor(X_num, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.long),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _torch_regression_loader(
    X_cat: np.ndarray,
    X_num: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> Any:
    generator = torch.Generator()
    generator.manual_seed(seed)
    dataset = TensorDataset(
        torch.as_tensor(X_cat, dtype=torch.long),
        torch.as_tensor(X_num, dtype=torch.float32),
        torch.as_tensor(y, dtype=torch.float32).reshape(-1, 1),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        generator=generator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _train_one_epoch(
    network: Any,
    loader: Any,
    criterion: Any,
    optimizer: Any,
    device: Any,
    max_grad_norm: float | None,
) -> float:
    network.train()
    total_loss = 0.0
    total_examples = 0
    for x_cat, x_num, y in loader:
        x_cat = x_cat.to(device)
        x_num = x_num.to(device)
        y = y.to(device)

        optimizer.zero_grad(set_to_none=True)
        logits = network(x_cat, x_num)
        loss = criterion(logits, y)
        loss.backward()
        if max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(network.parameters(), float(max_grad_norm))
        optimizer.step()

        batch_size = y.shape[0]
        total_loss += float(loss.detach().cpu()) * batch_size
        total_examples += batch_size
    return total_loss / max(total_examples, 1)


def _evaluate_torch(
    network: Any,
    loader: Any,
    criterion: Any,
    device: Any,
) -> tuple[float, np.ndarray]:
    network.eval()
    total_loss = 0.0
    total_examples = 0
    probabilities: list[np.ndarray] = []
    with torch.no_grad():
        for x_cat, x_num, y in loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            y = y.to(device)
            logits = network(x_cat, x_num)
            loss = criterion(logits, y)
            batch_size = y.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size
            probabilities.append(
                torch.softmax(logits.detach().cpu(), dim=1).numpy()
            )
    return total_loss / max(total_examples, 1), np.concatenate(probabilities, axis=0)


def _evaluate_torch_regressor(
    network: Any,
    loader: Any,
    criterion: Any,
    device: Any,
) -> tuple[float, np.ndarray]:
    network.eval()
    total_loss = 0.0
    total_examples = 0
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for x_cat, x_num, y in loader:
            x_cat = x_cat.to(device)
            x_num = x_num.to(device)
            y = y.to(device)
            output = network(x_cat, x_num)
            loss = criterion(output, y)
            batch_size = y.shape[0]
            total_loss += float(loss.detach().cpu()) * batch_size
            total_examples += batch_size
            predictions.append(output.detach().cpu().numpy().reshape(-1))
    return total_loss / max(total_examples, 1), np.concatenate(predictions)


def _selection_score(
    selection_metric: str,
    valid_loss: float,
    valid_metrics: dict[str, Any],
) -> float:
    normalized = selection_metric.lower()
    if normalized in {"loss", "valid_loss"}:
        return -float(valid_loss)
    metric_name = normalized.replace("valid_", "")
    if metric_name not in valid_metrics:
        raise ValueError(f"Unknown selection metric: {selection_metric!r}")
    return float(valid_metrics[metric_name])


def _regression_selection_score(
    selection_metric: str,
    valid_loss: float,
    valid_metrics: dict[str, float],
) -> float:
    normalized = selection_metric.lower().replace("valid_", "")
    if normalized == "loss":
        return -float(valid_loss)
    if normalized not in valid_metrics:
        raise ValueError(f"Unknown regression selection metric: {selection_metric!r}")
    value = float(valid_metrics[normalized])
    if normalized in {"rmse", "mae", "median_absolute_error", "mape_percent"}:
        return -value
    if normalized == "r2":
        return value
    raise ValueError(
        f"Regression metric {selection_metric!r} has no selection direction."
    )


def _select_threshold(
    data: Any,
    valid_probabilities: np.ndarray,
    config: dict[str, Any],
) -> float | None:
    if len(data.class_names) != 2:
        return None
    if bool(config.get("optimize_threshold", False)):
        threshold, _ = optimize_binary_threshold(
            y_valid=data.y_valid,
            valid_probabilities=valid_probabilities,
            metric=str(config.get("threshold_metric", "f1")),
            grid_size=int(config.get("threshold_grid_size", 181)),
        )
        return threshold
    return float(config.get("threshold", 0.5))


def _classification_output_dirs(results_dir: Path) -> dict[str, Path]:
    return {
        "metrics": results_dir / "metrics",
        "histories": results_dir / "histories",
        "predictions": results_dir / "predictions",
        "checkpoints": results_dir / "checkpoints",
    }


def _regression_output_dirs(results_dir: Path) -> dict[str, Path]:
    return {
        "metrics": results_dir / "metrics",
        "histories": results_dir / "histories",
        "predictions": results_dir / "predictions",
        "checkpoints": results_dir / "checkpoints",
    }


def _merge_model_config(
    model_name: str,
    base_config: dict[str, Any],
    model_config: dict[str, Any] | None,
    device: str,
) -> dict[str, Any]:
    configured_models = base_config.get("model_configs", {})
    canonical_name = _canonical_model_name(model_name)
    merged = dict(configured_models.get(canonical_name, {}))
    if model_config:
        merged.update(model_config)
    merged.setdefault("seed", int(base_config["seed"]))
    merged.setdefault("device", device)
    merged.setdefault("learning_rate", float(base_config["learning_rate"]))
    merged.setdefault("weight_decay", float(base_config.get("weight_decay", 0.0)))
    return merged


def _checkpoint_path(checkpoint_dir: Path, run_id: str, model_name: str) -> Path:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    normalized = model_name.lower().replace("-", "_")
    if normalized in {
        "baseline",
        "baseline_logistic",
        "logistic_regression",
        "baseline_ridge",
        "ridge",
    }:
        suffix = ".joblib"
    elif normalized in {
        "tabnet",
        "tabnetclassifier",
        "tabnet_classifier",
        "tabnetregressor",
        "tabnet_regressor",
    }:
        suffix = ".zip"
    else:
        suffix = ".pt"
    return checkpoint_dir / f"{run_id}{suffix}"


def _make_run_id(model_name: str, seed: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{_canonical_model_name(model_name)}_seed{seed}_{timestamp}"


def _experiment_id(config: dict[str, Any]) -> str:
    value = str(config.get("experiment_id", "unlabeled_experiment")).strip()
    if not value:
        raise ValueError("experiment_id must be a non-empty string.")
    return value


def _protocol_fingerprint(config: dict[str, Any]) -> str:
    """Hash settings shared by all model/seed runs in one campaign."""

    excluded = {"keep_model", "results_dir", "seed"}
    payload = {
        str(key): _config_json_value(value)
        for key, value in config.items()
        if key not in excluded
    }
    serialized = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _config_json_value(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            str(key): _config_json_value(item)
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        }
    if isinstance(value, (list, tuple)):
        return [_config_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _make_regression_run_id(model_name: str, seed: int) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    return f"{_canonical_regression_model_name(model_name)}_seed{seed}_{timestamp}"


def _canonical_model_name(model_name: str) -> str:
    normalized = model_name.lower().replace("-", "_")
    aliases = {
        "baseline": "baseline_logistic",
        "logistic_regression": "baseline_logistic",
        "tabnetclassifier": "tabnet",
        "tabnet_classifier": "tabnet",
        "tabtransformer": "tab_transformer",
        "fttransformer": "ft_transformer",
        "saint": "saint_supervised",
    }
    return aliases.get(normalized, normalized)


def _canonical_regression_model_name(model_name: str) -> str:
    normalized = model_name.lower().replace("-", "_")
    aliases = {
        "baseline": "baseline_ridge",
        "ridge": "baseline_ridge",
        "tabnetregressor": "tabnet",
        "tabnet_regressor": "tabnet",
        "tabtransformer": "tab_transformer",
        "fttransformer": "ft_transformer",
        "saint": "saint_supervised",
    }
    return aliases.get(normalized, normalized)


def _best_epoch(history: dict[str, Any]) -> int | None:
    value = history.get("best_epoch")
    if value is None:
        return None
    return int(value)


def _epochs_trained(history: dict[str, Any]) -> int:
    value = history.get("epochs_trained")
    if value is not None:
        return int(value)
    for key in ("train_loss", "loss", "train_log_loss"):
        values = history.get(key)
        if isinstance(values, list):
            return len(values)
    return 0


def _implementation_version(
    canonical_name: str,
    model_config: dict[str, Any],
) -> str:
    configured = model_config.get("implementation_version")
    if configured:
        return str(configured)
    defaults = {
        "baseline_logistic": "sklearn_logistic_regression",
        "baseline_ridge": "sklearn_ridge",
        "tabnet": "pytorch_tabnet_4_1",
        "tab_transformer": "tab_transformer_v1",
        "ft_transformer": "ft_transformer_v1",
        "saint_supervised": "saint_column_v1",
    }
    return defaults.get(canonical_name, "unspecified")
