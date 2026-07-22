"""Multiclass metrics, result validation, summaries, and plots for TLM."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from numpy.typing import NDArray
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    log_loss,
    matthews_corrcoef,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.preprocessing import label_binarize

from src.data import PreparedFlightFold


FloatArray = NDArray[np.float32]
IntArray = NDArray[np.int64]
PRIMARY_REGION = "all_external_rows"
STABLE_REGION = "outside_transition_guard"


def validate_probabilities(
    probabilities: NDArray[np.floating[Any]],
    n_classes: int,
    *,
    atol: float = 1e-5,
) -> FloatArray:
    """Validate and return a float32 multiclass probability matrix."""

    values = np.asarray(probabilities, dtype=np.float32)
    if values.ndim != 2 or values.shape[1] != n_classes:
        raise ValueError(
            f"Expected probability shape (n, {n_classes}), found {values.shape}"
        )
    if len(values) == 0:
        raise ValueError("Probability matrix cannot be empty")
    if not np.isfinite(values).all():
        raise ValueError("Probabilities contain nonfinite values")
    if values.min(initial=0.0) < -atol or values.max(initial=0.0) > 1.0 + atol:
        raise ValueError("Probabilities lie outside [0, 1]")
    if not np.allclose(values.sum(axis=1), 1.0, atol=atol):
        raise ValueError("Probability rows do not sum to one")
    return values


def multiclass_evaluation(
    y_true: IntArray,
    probabilities: NDArray[np.floating[Any]],
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    """Compute probability-aware multiclass metrics and per-class diagnostics."""

    target = np.asarray(y_true, dtype=np.int64)
    values = validate_probabilities(probabilities, len(class_names))
    if target.ndim != 1 or len(target) != len(values):
        raise ValueError("Targets and probabilities have incompatible shapes")
    expected = set(range(len(class_names)))
    if set(np.unique(target)) != expected:
        raise ValueError("Every class must be represented in the evaluated region")
    prediction = values.argmax(axis=1)
    labels = np.arange(len(class_names))
    macro = precision_recall_fscore_support(
        target,
        prediction,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    weighted = precision_recall_fscore_support(
        target,
        prediction,
        labels=labels,
        average="weighted",
        zero_division=0,
    )
    per_class_values = precision_recall_fscore_support(
        target,
        prediction,
        labels=labels,
        average=None,
        zero_division=0,
    )
    binary_target = label_binarize(target, classes=labels)
    metrics = {
        "n_rows": float(len(target)),
        "accuracy": float(accuracy_score(target, prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(target, prediction)
        ),
        "precision_macro": float(macro[0]),
        "recall_macro": float(macro[1]),
        "f1_macro": float(macro[2]),
        "f1_weighted": float(weighted[2]),
        "mcc": float(matthews_corrcoef(target, prediction)),
        "log_loss": float(log_loss(target, values, labels=labels)),
        "roc_auc_ovr_macro": float(
            roc_auc_score(
                target,
                values,
                labels=labels,
                multi_class="ovr",
                average="macro",
            )
        ),
        "pr_auc_macro": float(
            average_precision_score(binary_target, values, average="macro")
        ),
    }
    per_class: list[dict[str, Any]] = []
    for index, class_name in enumerate(class_names):
        per_class.append(
            {
                "class_index": index,
                "class_name": class_name,
                "precision": float(per_class_values[0][index]),
                "recall": float(per_class_values[1][index]),
                "f1": float(per_class_values[2][index]),
                "support": int(per_class_values[3][index]),
                "roc_auc_ovr": float(
                    roc_auc_score(binary_target[:, index], values[:, index])
                ),
                "pr_auc_ovr": float(
                    average_precision_score(
                        binary_target[:, index],
                        values[:, index],
                    )
                ),
            }
        )
    matrix = confusion_matrix(target, prediction, labels=labels)
    return {
        "metrics": metrics,
        "per_class": per_class,
        "confusion_matrix": matrix.astype(int).tolist(),
    }


def evaluate_multiclass_regions(
    y_true: IntArray,
    probabilities: NDArray[np.floating[Any]],
    stable_mask: NDArray[np.bool_],
    class_names: tuple[str, ...],
) -> dict[str, Any]:
    """Evaluate primary full-flight and secondary transition-guard regions."""

    target = np.asarray(y_true, dtype=np.int64)
    mask = np.asarray(stable_mask, dtype=bool)
    if mask.ndim != 1 or len(mask) != len(target):
        raise ValueError("Stable-region mask has an invalid shape")
    if mask.all() or not mask.any():
        raise ValueError("Transition sensitivity requires both mask regions")
    values = validate_probabilities(probabilities, len(class_names))
    primary = multiclass_evaluation(target, values, class_names)
    stable = multiclass_evaluation(target[mask], values[mask], class_names)
    return {
        "metrics": {
            PRIMARY_REGION: primary["metrics"],
            STABLE_REGION: stable["metrics"],
        },
        "per_class": {
            PRIMARY_REGION: primary["per_class"],
            STABLE_REGION: stable["per_class"],
        },
        "confusion_matrices": {
            PRIMARY_REGION: primary["confusion_matrix"],
            STABLE_REGION: stable["confusion_matrix"],
        },
    }


def results_frame(results: Sequence[Any]) -> pd.DataFrame:
    """Flatten experiment results into one row per evaluation region."""

    rows: list[dict[str, Any]] = []
    for result in results:
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
                    "best_validation_score": result.best_validation_score,
                    "tuning_seconds": result.tuning_seconds,
                    "refit_seconds": result.refit_seconds,
                    "inference_seconds": result.inference_seconds,
                    "parameter_count": result.parameter_count,
                    "reload_max_abs_difference": (
                        result.reload_max_abs_difference
                    ),
                    **metrics,
                }
            )
    return pd.DataFrame(rows)


def per_class_frame(results: Sequence[Any]) -> pd.DataFrame:
    """Flatten per-class diagnostics for analysis and plotting."""

    rows: list[dict[str, Any]] = []
    for result in results:
        for region, class_rows in result.per_class.items():
            for values in class_rows:
                rows.append(
                    {
                        "run_id": result.run_id,
                        "model_name": result.model_name,
                        "view_name": result.view_name,
                        "fold_name": result.fold_name,
                        "test_flight": result.test_flight,
                        "seed": result.seed,
                        "region": region,
                        **values,
                    }
                )
    return pd.DataFrame(rows)


def flightwise_comparison_table(
    results: Sequence[Any],
    *,
    region: str = PRIMARY_REGION,
) -> pd.DataFrame:
    """Summarize seeds within each external-flight direction."""

    frame = results_frame(results)
    selected = frame.loc[frame["region"].eq(region)]
    metrics = [
        "balanced_accuracy",
        "f1_macro",
        "mcc",
        "roc_auc_ovr_macro",
        "pr_auc_macro",
    ]
    grouped = selected.groupby(
        ["view_name", "model_name", "test_flight"],
        sort=False,
    )[metrics]
    mean = grouped.mean().add_suffix("_mean")
    standard_deviation = grouped.std(ddof=0).add_suffix("_seed_std")
    table = mean.join(standard_deviation).reset_index()
    return table.sort_values(
        ["view_name", "test_flight", "f1_macro_mean"],
        ascending=[True, True, False],
        kind="stable",
    ).reset_index(drop=True)


def descriptive_comparison_table(
    results: Sequence[Any],
    *,
    region: str = PRIMARY_REGION,
) -> pd.DataFrame:
    """Rank descriptively while retaining the worst external-flight result."""

    flightwise = flightwise_comparison_table(results, region=region)
    grouped = flightwise.groupby(
        ["view_name", "model_name"],
        sort=False,
    )
    table = grouped.agg(
        balanced_accuracy_mean=("balanced_accuracy_mean", "mean"),
        f1_macro_mean=("f1_macro_mean", "mean"),
        f1_macro_worst_flight=("f1_macro_mean", "min"),
        mcc_mean=("mcc_mean", "mean"),
        roc_auc_ovr_macro_mean=("roc_auc_ovr_macro_mean", "mean"),
        pr_auc_macro_mean=("pr_auc_macro_mean", "mean"),
        maximum_seed_std=("f1_macro_seed_std", "max"),
    ).reset_index()
    return table.sort_values(
        ["view_name", "f1_macro_mean"],
        ascending=[True, False],
        kind="stable",
    ).reset_index(drop=True)


def computational_cost_table(results: Sequence[Any]) -> pd.DataFrame:
    """Summarize train, refit, inference, epoch, and parameter costs."""

    frame = results_frame(results)
    primary = frame.loc[frame["region"].eq(PRIMARY_REGION)]
    return (
        primary.groupby(["view_name", "model_name"], sort=False)
        .agg(
            tuning_seconds_mean=("tuning_seconds", "mean"),
            refit_seconds_mean=("refit_seconds", "mean"),
            inference_seconds_mean=("inference_seconds", "mean"),
            best_epoch_mean=("best_epoch", "mean"),
            parameter_count_mean=("parameter_count", "mean"),
        )
        .reset_index()
        .sort_values(
            ["view_name", "refit_seconds_mean"],
            kind="stable",
        )
        .reset_index(drop=True)
    )


def persist_benchmark_tables(
    results: Sequence[Any],
    output_dir: Path | str,
) -> dict[str, Path]:
    """Persist run-level and aggregate tables used by the main notebook."""

    destination = Path(output_dir) / "metrics"
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "summary": destination / "summary_metrics.csv",
        "descriptive": destination / "descriptive_comparison.csv",
        "flightwise": destination / "flightwise_comparison.csv",
        "per_class": destination / "per_class_metrics.csv",
        "costs": destination / "computational_costs.csv",
    }
    results_frame(results).to_csv(paths["summary"], index=False)
    descriptive_comparison_table(results).to_csv(
        paths["descriptive"],
        index=False,
    )
    flightwise_comparison_table(results).to_csv(
        paths["flightwise"],
        index=False,
    )
    per_class_frame(results).to_csv(paths["per_class"], index=False)
    computational_cost_table(results).to_csv(paths["costs"], index=False)
    return paths


def validate_benchmark_results(
    results: Sequence[Any],
    prepared_folds: Sequence[PreparedFlightFold],
    project_root: Path | str,
    *,
    expected_models: Sequence[str],
    expected_views: Sequence[str],
    expected_seeds: Sequence[int],
    raise_on_failure: bool = True,
) -> pd.Series:
    """Recompute persisted predictions and verify the complete run grid."""

    root = Path(project_root).resolve()
    prepared_lookup = {
        (item.fold.name, item.feature_view.name): item
        for item in prepared_folds
    }
    expected_keys = {
        (model, view, fold, int(seed))
        for model in expected_models
        for view in expected_views
        for fold in {item.fold.name for item in prepared_folds}
        for seed in expected_seeds
    }
    result_keys = {
        (
            result.model_name,
            result.view_name,
            result.fold_name,
            result.seed,
        )
        for result in results
    }
    probability_checks: list[bool] = []
    row_checks: list[bool] = []
    metric_checks: list[bool] = []
    artifact_checks: list[bool] = []
    for result in results:
        prepared = prepared_lookup[(result.fold_name, result.view_name)]
        prediction_path = root / result.prediction_path
        history_path = root / result.history_path
        metric_path = root / result.metric_path
        checkpoint_path = root / result.checkpoint_path
        artifacts_exist = all(
            path.is_file()
            for path in (
                prediction_path,
                history_path,
                metric_path,
                checkpoint_path,
            )
        )
        artifact_checks.append(artifacts_exist)
        if not artifacts_exist:
            probability_checks.append(False)
            row_checks.append(False)
            metric_checks.append(False)
            continue
        prediction = pd.read_csv(prediction_path)
        probability_columns = sorted(
            (
                column
                for column in prediction
                if column.startswith("probability_")
            ),
            key=lambda value: int(value.split("_", 2)[1]),
        )
        try:
            probabilities = validate_probabilities(
                prediction[probability_columns].to_numpy(),
                len(prepared.class_names),
            )
            probability_checks.append(True)
        except ValueError:
            probability_checks.append(False)
            row_checks.append(False)
            metric_checks.append(False)
            continue
        row_checks.append(
            np.array_equal(
                prediction["row_id"].to_numpy(dtype=np.int64),
                prepared.test.row_ids,
            )
            and np.array_equal(
                prediction["y_true"].to_numpy(dtype=np.int64),
                prepared.y_test,
            )
            and prediction["flight_id"].eq(
                prepared.fold.test_flight
            ).all()
        )
        stable_mask = prediction["evaluation_region"].eq("stable").to_numpy()
        recomputed = evaluate_multiclass_regions(
            prepared.y_test,
            probabilities,
            stable_mask,
            prepared.class_names,
        )
        metric_checks.append(
            _metric_dicts_close(result.metrics, recomputed["metrics"])
        )
    checks = pd.Series(
        {
            "run_grid_is_complete": result_keys == expected_keys,
            "run_keys_are_unique": len(result_keys) == len(results),
            "all_artifacts_exist": all(artifact_checks),
            "all_probabilities_are_valid": all(probability_checks),
            "persisted_rows_match_external_folds": all(row_checks),
            "persisted_metrics_recompute_exactly": all(metric_checks),
            "all_checkpoints_reload_within_tolerance": all(
                result.reload_max_abs_difference <= 1e-6
                for result in results
            ),
            "only_primary_and_secondary_regions_exist": all(
                set(result.metrics) == {PRIMARY_REGION, STABLE_REGION}
                for result in results
            ),
        },
        dtype=bool,
        name="passed",
    )
    if raise_on_failure and not bool(checks.all()):
        failed = checks.index[~checks].tolist()
        raise AssertionError(f"Benchmark result checks failed: {failed}")
    return checks


def load_benchmark_results(output_dir: Path | str) -> list[Any]:
    """Load result records without restoring model checkpoints."""

    from src.training import ExperimentResult

    destination = Path(output_dir)
    histories = destination / "histories"
    if not histories.is_dir():
        raise FileNotFoundError(f"History directory not found: {histories}")
    manifest_path = destination / "benchmark_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        history_paths = [
            histories / f"{run_id}.json" for run_id in manifest["run_ids"]
        ]
    else:
        history_paths = sorted(histories.glob("*.json"))
    results = []
    for path in history_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Manifest history is missing: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        results.append(ExperimentResult.from_dict(payload["result"]))
    if not results:
        raise ValueError(f"No completed result records found in {histories}")
    return results


def plot_flightwise_metric(
    results: Sequence[Any],
    *,
    metric: str = "f1_macro",
    region: str = PRIMARY_REGION,
    output_path: Path | str | None = None,
) -> Figure:
    """Plot each external-flight score without hiding direction asymmetry."""

    frame = results_frame(results)
    selected = frame.loc[frame["region"].eq(region)]
    grouped = selected.groupby(
        ["view_name", "model_name", "test_flight"],
        sort=False,
    )[metric].agg(
        mean="mean",
        std=lambda values: values.std(ddof=0),
    ).reset_index()
    views = grouped["view_name"].drop_duplicates().tolist()
    figure, axes = plt.subplots(
        1,
        len(views),
        figsize=(6.5 * len(views), 4.2),
        sharey=True,
        squeeze=False,
    )
    for axis, view in zip(axes[0], views, strict=True):
        subset = grouped.loc[grouped["view_name"].eq(view)]
        models = subset["model_name"].drop_duplicates().tolist()
        x = np.arange(2, dtype=float)
        offsets = np.linspace(-0.28, 0.28, len(models))
        for offset, model in zip(offsets, models, strict=True):
            rows = subset.loc[subset["model_name"].eq(model)].set_index(
                "test_flight"
            )
            means = np.asarray([rows.loc[index, "mean"] for index in (0, 1)])
            errors = np.asarray([rows.loc[index, "std"] for index in (0, 1)])
            axis.errorbar(
                x + offset,
                means,
                yerr=errors,
                marker="o",
                capsize=3,
                linestyle="none",
                label=model,
            )
        axis.set_title(view)
        axis.set_xticks(x, ["test flight 0", "test flight 1"])
        axis.set_xlabel("External direction")
        axis.grid(axis="y", alpha=0.25)
    axes[0, 0].set_ylabel(metric)
    axes[0, -1].legend(fontsize=8, loc="best")
    figure.tight_layout()
    _save_figure(figure, output_path)
    return figure


def plot_mean_confusion_matrices(
    results: Sequence[Any],
    class_names: tuple[str, ...],
    *,
    view_name: str = "sensor_core",
    region: str = PRIMARY_REGION,
    output_path: Path | str | None = None,
) -> Figure:
    """Plot mean row-normalized confusion matrices across seeds and flights."""

    selected = [result for result in results if result.view_name == view_name]
    models = list(dict.fromkeys(result.model_name for result in selected))
    if not models:
        raise ValueError(f"No results found for view {view_name!r}")
    columns = 3
    rows = int(np.ceil(len(models) / columns))
    figure, axes = plt.subplots(
        rows,
        columns,
        figsize=(4.4 * columns, 4.0 * rows),
        squeeze=False,
        layout="constrained",
    )
    image = None
    for axis, model in zip(axes.flat, models, strict=False):
        matrices = [
            np.asarray(result.confusion_matrices[region], dtype=float)
            for result in selected
            if result.model_name == model
        ]
        normalized = []
        for matrix in matrices:
            denominators = matrix.sum(axis=1, keepdims=True)
            normalized.append(
                np.divide(
                    matrix,
                    denominators,
                    out=np.zeros_like(matrix),
                    where=denominators != 0,
                )
            )
        mean_matrix = np.mean(normalized, axis=0)
        image = axis.imshow(mean_matrix, vmin=0.0, vmax=1.0, cmap="Blues")
        _annotate_confusion(axis, mean_matrix)
        axis.set_title(model)
        axis.set_xticks(range(len(class_names)), class_names, rotation=45)
        axis.set_yticks(range(len(class_names)), class_names)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
    for axis in axes.flat[len(models) :]:
        axis.set_visible(False)
    if image is not None:
        figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.75)
    figure.suptitle(f"Mean normalized confusion: {view_name}", y=1.02)
    _save_figure(figure, output_path)
    return figure


def _metric_dicts_close(
    stored: dict[str, dict[str, float]],
    recomputed: dict[str, dict[str, float]],
) -> bool:
    if set(stored) != set(recomputed):
        return False
    for region in stored:
        if set(stored[region]) != set(recomputed[region]):
            return False
        for metric, value in stored[region].items():
            if not np.isclose(
                float(value),
                float(recomputed[region][metric]),
                rtol=1e-6,
                atol=1e-7,
            ):
                return False
    return True


def _annotate_confusion(axis: Axes, matrix: NDArray[np.float64]) -> None:
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            value = matrix[row, column]
            axis.text(
                column,
                row,
                f"{value:.2f}",
                ha="center",
                va="center",
                color="white" if value > 0.55 else "black",
                fontsize=8,
            )


def _save_figure(
    figure: Figure,
    output_path: Path | str | None,
) -> None:
    if output_path is None:
        return
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=160, bbox_inches="tight")
