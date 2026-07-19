"""Evaluation metrics, thresholding, plotting, and result persistence."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
    median_absolute_error,
    precision_recall_curve,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.preprocessing import label_binarize


@dataclass(frozen=True)
class PersistedTaskResults:
    """Validated persisted artifacts for one experiment campaign."""

    task: str
    experiment_id: str
    metrics: pd.DataFrame
    histories: dict[str, dict[str, Any]]
    configs: dict[str, dict[str, Any]]
    predictions: dict[str, pd.DataFrame]
    checkpoints: dict[str, Path]
    report: dict[str, Any]

    @property
    def ready(self) -> bool:
        """Return whether the campaign passed every integrity check."""

        return bool(self.report.get("ready", False))


def validate_probabilities(
    probabilities: np.ndarray,
    tolerance: float = 1e-4,
) -> None:
    """Check probability ranges and row sums."""

    if probabilities.ndim != 2:
        raise ValueError("Probabilities must be a 2D array.")
    if probabilities.shape[0] == 0 or probabilities.shape[1] < 2:
        raise ValueError("Probabilities must contain rows for at least two classes.")
    if not np.isfinite(probabilities).all():
        raise ValueError("Probabilities contain NaN or infinite values.")
    if probabilities.min() < -tolerance or probabilities.max() > 1.0 + tolerance:
        raise ValueError("Probabilities must lie in [0, 1].")
    row_sums = probabilities.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=tolerance):
        raise ValueError("Each probability row must sum to 1.")


def labels_from_probabilities(
    probabilities: np.ndarray,
    threshold: float | None = None,
) -> np.ndarray:
    """Convert probabilities to labels without using test data for tuning."""

    validate_probabilities(probabilities)
    if probabilities.shape[1] == 2 and threshold is not None:
        return (probabilities[:, 1] >= threshold).astype(np.int64)
    return np.argmax(probabilities, axis=1).astype(np.int64)


def compute_classification_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: tuple[str, ...],
    threshold: float | None = 0.5,
) -> dict[str, Any]:
    """Compute classification metrics for binary or multiclass tasks."""

    validate_probabilities(probabilities)
    y_true = np.asarray(y_true)
    if y_true.ndim != 1 or len(y_true) != len(probabilities):
        raise ValueError("y_true must be one-dimensional and match probabilities.")
    y_pred = labels_from_probabilities(probabilities, threshold)
    n_classes = len(class_names)
    if probabilities.shape[1] != n_classes:
        raise ValueError("Probability width must match the number of class names.")
    if not set(np.unique(y_true)).issubset(set(range(n_classes))):
        raise ValueError("y_true contains labels outside class_names.")

    metrics: dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }

    if n_classes == 2:
        metrics.update(
            {
                "precision": float(
                    precision_score(y_true, y_pred, zero_division=0)
                ),
                "recall": float(recall_score(y_true, y_pred, zero_division=0)),
                "f1": float(f1_score(y_true, y_pred, zero_division=0)),
                "roc_auc": float(roc_auc_score(y_true, probabilities[:, 1])),
                "pr_auc": float(average_precision_score(y_true, probabilities[:, 1])),
                "threshold": float(threshold if threshold is not None else 0.5),
                "average": "binary",
            }
        )
    else:
        y_binary = label_binarize(y_true, classes=np.arange(n_classes))
        metrics.update(
            {
                "precision": float(
                    precision_score(
                        y_true,
                        y_pred,
                        average="weighted",
                        zero_division=0,
                    )
                ),
                "recall": float(
                    recall_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
                "f1": float(
                    f1_score(y_true, y_pred, average="weighted", zero_division=0)
                ),
                "roc_auc": float(
                    roc_auc_score(
                        y_binary,
                        probabilities,
                        average="macro",
                        multi_class="ovr",
                    )
                ),
                "pr_auc": float(
                    average_precision_score(y_binary, probabilities, average="macro")
                ),
                "threshold": float("nan"),
                "average": "weighted; auc/pr_auc macro one-vs-rest",
            }
        )
    return metrics


def optimize_binary_threshold(
    y_valid: np.ndarray,
    valid_probabilities: np.ndarray,
    metric: str = "f1",
    grid_size: int = 181,
) -> tuple[float, float]:
    """Select a binary threshold using validation probabilities only."""

    validate_probabilities(valid_probabilities)
    if valid_probabilities.shape[1] != 2:
        raise ValueError("Threshold optimization is defined only for binary tasks.")
    if grid_size < 3:
        raise ValueError("grid_size must be at least 3.")

    thresholds = np.linspace(0.05, 0.95, grid_size)
    best_threshold = 0.5
    best_score = -np.inf
    scores = {
        "f1": lambda yt, yp: f1_score(yt, yp, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score,
        "precision": lambda yt, yp: precision_score(yt, yp, zero_division=0),
        "recall": lambda yt, yp: recall_score(yt, yp, zero_division=0),
    }
    if metric not in scores:
        raise ValueError(f"Unsupported threshold metric: {metric!r}")

    for threshold in thresholds:
        y_pred = (valid_probabilities[:, 1] >= threshold).astype(np.int64)
        score = float(scores[metric](y_valid, y_pred))
        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    return best_threshold, best_score


def confusion_matrix_frame(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    class_names: tuple[str, ...],
    threshold: float | None = 0.5,
) -> pd.DataFrame:
    """Return a labeled confusion matrix."""

    y_pred = labels_from_probabilities(probabilities, threshold)
    matrix = confusion_matrix(y_true, y_pred, labels=np.arange(len(class_names)))
    return pd.DataFrame(matrix, index=class_names, columns=class_names)


def validate_experiment_results(
    results: list[dict[str, Any]],
    metric_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Verify shared test data, metrics, probabilities, and checkpoints."""

    if not results:
        raise ValueError("At least one experiment result is required.")

    reference = results[0]
    reference_y = np.asarray(reference["y_true"])
    reference_indices = np.asarray(reference["test_indices"])
    reference_fingerprint = str(reference["split_fingerprint"])
    reference_classes = tuple(reference["class_names"])
    reference_experiment = str(reference.get("experiment_id", "unlabeled_experiment"))
    reference_protocol = str(reference.get("protocol_fingerprint", "unlabeled"))
    seen_runs: set[tuple[str, int]] = set()

    for result in results:
        run_key = (str(result["model_name"]), int(result["seed"]))
        if run_key in seen_runs:
            raise ValueError(f"Duplicate model/seed result in comparison: {run_key}")
        seen_runs.add(run_key)
        if result.get("test_evaluated") is not True:
            raise ValueError(f"Test evaluation is incomplete for {run_key}.")
        if str(result.get("experiment_id", "unlabeled_experiment")) != (
            reference_experiment
        ):
            raise ValueError("All results must belong to the same experiment campaign.")
        if str(result.get("protocol_fingerprint", "unlabeled")) != (
            reference_protocol
        ):
            raise ValueError("All results must use the same experiment protocol.")
        probabilities = np.asarray(result["probabilities"])
        validate_probabilities(probabilities)
        if tuple(result["class_names"]) != reference_classes:
            raise ValueError("All results must use the same ordered class names.")
        if not np.array_equal(np.asarray(result["y_true"]), reference_y):
            raise ValueError("All results must use identical ordered test labels.")
        if not np.array_equal(np.asarray(result["test_indices"]), reference_indices):
            raise ValueError("All results must use identical ordered test indices.")
        if str(result["split_fingerprint"]) != reference_fingerprint:
            raise ValueError("All results must use the same split fingerprint.")

        expected_predictions = labels_from_probabilities(
            probabilities,
            result.get("threshold"),
        )
        if not np.array_equal(expected_predictions, np.asarray(result["y_pred"])):
            raise ValueError(f"Stored predictions are inconsistent for {run_key}.")

        recomputed = compute_classification_metrics(
            y_true=reference_y,
            probabilities=probabilities,
            class_names=reference_classes,
            threshold=result.get("threshold"),
        )
        for metric_name, expected_value in result["test_metrics"].items():
            if metric_name not in recomputed or not isinstance(
                expected_value, (int, float, np.number)
            ):
                continue
            if not np.isclose(
                float(recomputed[metric_name]),
                float(expected_value),
                atol=metric_tolerance,
                rtol=0.0,
                equal_nan=True,
            ):
                raise ValueError(
                    f"Metric {metric_name!r} is inconsistent for {run_key}."
                )

        checkpoint_path = Path(result["checkpoint_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for {run_key}: {checkpoint_path}"
            )
        if result.get("reload_check", {}).get("checkpoint_reloaded") is not True:
            raise ValueError(f"Checkpoint reload was not verified for {run_key}.")

    return {
        "n_results": len(results),
        "experiment_id": reference_experiment,
        "protocol_fingerprint": reference_protocol,
        "shared_split_fingerprint": reference_fingerprint,
        "shared_test_rows": int(len(reference_y)),
        "identical_test_labels": True,
        "identical_test_indices": True,
        "probabilities_valid": True,
        "predictions_recomputed": True,
        "metrics_recomputed": True,
        "checkpoints_present_and_reloaded": True,
    }


def persist_classification_results(
    result: dict[str, Any],
    output_dirs: dict[str, Path],
) -> dict[str, Path]:
    """Persist metrics, history, predictions, and configuration."""

    run_id = result["run_id"]
    metrics_dir = output_dirs["metrics"]
    histories_dir = output_dirs["histories"]
    predictions_dir = output_dirs["predictions"]
    for directory in (metrics_dir, histories_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / f"{run_id}_metrics.csv"
    history_path = histories_dir / f"{run_id}_history.json"
    config_path = histories_dir / f"{run_id}_config.json"
    predictions_path = predictions_dir / f"{run_id}_predictions.csv"
    summary_path = metrics_dir / "summary_metrics.csv"

    metrics_row = {
        "run_id": run_id,
        "experiment_id": result.get("experiment_id", "unlabeled_experiment"),
        "protocol_fingerprint": result.get("protocol_fingerprint", "unlabeled"),
        "model_name": result["model_name"],
        "implementation_version": result.get("implementation_version"),
        "prediction_scope": result.get("prediction_scope"),
        "seed": result["seed"],
        "best_epoch": result.get("best_epoch"),
        "epochs_trained": result.get("epochs_trained"),
        "reached_epoch_budget": result.get("reached_epoch_budget"),
        "split_fingerprint": result.get("split_fingerprint"),
        "train_time_seconds": result["train_time_seconds"],
        "inference_time_seconds": result["inference_time_seconds"],
        "n_parameters": result["n_parameters"],
        **result["test_metrics"],
    }
    pd.DataFrame([metrics_row]).to_csv(metrics_path, index=False)
    _upsert_campaign_summary(summary_path, metrics_row)

    history_payload = {
        "history": _json_safe(result.get("history", {})),
        "validation_metrics": _json_safe(result.get("valid_metrics", {})),
        "reload_check": _json_safe(result.get("reload_check", {})),
    }
    history_path.write_text(
        json.dumps(history_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(_json_safe(result.get("config", {})), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    prediction_frame = pd.DataFrame(
        {
            "source_index": result["test_indices"],
            "y_true": result["y_true"],
            "y_pred": result["y_pred"],
        }
    )
    for class_idx, class_name in enumerate(result["class_names"]):
        prediction_frame[f"prob_{class_name}"] = result["probabilities"][:, class_idx]
    prediction_frame.to_csv(predictions_path, index=False)

    return {
        "metrics": metrics_path,
        "summary_metrics": summary_path,
        "history": history_path,
        "config": config_path,
        "predictions": predictions_path,
    }


def comparison_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a compact comparison table from experiment results."""

    rows: list[dict[str, Any]] = []
    for result in results:
        rows.append(
            {
                "experiment_id": result.get(
                    "experiment_id", "unlabeled_experiment"
                ),
                "model_name": result["model_name"],
                "implementation_version": result.get("implementation_version"),
                "prediction_scope": result.get("prediction_scope"),
                "seed": result["seed"],
                "best_epoch": result.get("best_epoch"),
                "epochs_trained": result.get("epochs_trained"),
                "reached_epoch_budget": result.get("reached_epoch_budget"),
                "n_parameters": result["n_parameters"],
                "train_time_seconds": result["train_time_seconds"],
                "inference_time_seconds": result["inference_time_seconds"],
                **result["test_metrics"],
            }
        )
    return pd.DataFrame(rows)


def plot_roc_pr_curves(
    results: list[dict[str, Any]],
    figures_dir: Path,
    prefix: str = "classification",
) -> dict[str, Path]:
    """Plot ROC and precision-recall curves for binary experiments."""

    import matplotlib.pyplot as plt

    if not results or len(results[0]["class_names"]) != 2:
        raise ValueError("ROC/PR plotting currently expects binary results.")

    figures_dir.mkdir(parents=True, exist_ok=True)
    roc_path = figures_dir / f"{prefix}_roc_curves.png"
    pr_path = figures_dir / f"{prefix}_precision_recall_curves.png"

    fig, ax = plt.subplots(figsize=(7, 5))
    for result in results:
        fpr, tpr, _ = roc_curve(result["y_true"], result["probabilities"][:, 1])
        roc_auc = result.get("test_metrics", {}).get("roc_auc")
        label = str(result["model_name"])
        implementation = result.get("implementation_version")
        if implementation and result["model_name"] == "saint_supervised":
            label = f"{label} [{implementation}]"
        if roc_auc is not None:
            label = f"{label} (AUC={float(roc_auc):.4f})"
        ax.plot(fpr, tpr, label=label)
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.4", label="chance")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title("ROC curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(roc_path, dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 5))
    for result in results:
        precision, recall, _ = precision_recall_curve(
            result["y_true"],
            result["probabilities"][:, 1],
        )
        pr_auc = result.get("test_metrics", {}).get("pr_auc")
        label = str(result["model_name"])
        implementation = result.get("implementation_version")
        if implementation and result["model_name"] == "saint_supervised":
            label = f"{label} [{implementation}]"
        if pr_auc is not None:
            label = f"{label} (AP={float(pr_auc):.4f})"
        ax.plot(recall, precision, label=label)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-recall curves")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(pr_path, dpi=160)
    plt.close(fig)

    return {"roc": roc_path, "precision_recall": pr_path}


def plot_confusion_matrices(
    results: list[dict[str, Any]],
    figures_dir: Path,
    prefix: str = "classification",
) -> Path:
    """Plot labeled confusion matrices for the test split."""

    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("No results were provided.")

    n_results = len(results)
    n_cols = min(3, n_results)
    n_rows = int(np.ceil(n_results / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.5 * n_cols, 4 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()
    matrices = [
        confusion_matrix_frame(
            result["y_true"],
            result["probabilities"],
            result["class_names"],
            result["threshold"],
        )
        for result in results
    ]
    color_max = max(int(matrix.values.max()) for matrix in matrices)

    for axis, result, matrix in zip(axes_array, results, matrices):
        image = axis.imshow(matrix.values, cmap="Blues", vmin=0, vmax=color_max)
        title = str(result["model_name"])
        implementation = result.get("implementation_version")
        if implementation and result["model_name"] == "saint_supervised":
            title = f"{title}\n{implementation}"
        axis.set_title(title)
        axis.set_xticks(np.arange(len(result["class_names"])))
        axis.set_yticks(np.arange(len(result["class_names"])))
        axis.set_xticklabels(result["class_names"], rotation=35, ha="right")
        axis.set_yticklabels(result["class_names"])
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        for row_idx in range(matrix.shape[0]):
            for col_idx in range(matrix.shape[1]):
                axis.text(
                    col_idx,
                    row_idx,
                    int(matrix.iloc[row_idx, col_idx]),
                    ha="center",
                    va="center",
                    color=(
                        "white"
                        if matrix.iloc[row_idx, col_idx] > color_max / 2
                        else "black"
                    ),
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    for axis in axes_array[n_results:]:
        axis.axis("off")

    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{prefix}_confusion_matrices.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def validate_regression_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Validate and return one-dimensional finite regression arrays."""

    targets = np.asarray(y_true, dtype=np.float64)
    predictions = np.asarray(y_pred, dtype=np.float64)
    if targets.ndim != 1 or predictions.ndim != 1:
        raise ValueError("Regression targets and predictions must be one-dimensional.")
    if targets.shape != predictions.shape or targets.size == 0:
        raise ValueError("Regression targets and predictions must have equal size.")
    if not np.isfinite(targets).all():
        raise ValueError("Regression targets contain NaN or infinite values.")
    if not np.isfinite(predictions).all():
        raise ValueError("Regression predictions contain NaN or infinite values.")
    return targets, predictions


def compute_regression_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Compute complementary error and goodness-of-fit regression metrics."""

    targets, predictions = validate_regression_predictions(y_true, y_pred)
    residuals = predictions - targets
    return {
        "rmse": float(np.sqrt(mean_squared_error(targets, predictions))),
        "mae": float(mean_absolute_error(targets, predictions)),
        "median_absolute_error": float(
            median_absolute_error(targets, predictions)
        ),
        "r2": float(r2_score(targets, predictions)),
        "mape_percent": float(
            100.0 * mean_absolute_percentage_error(targets, predictions)
        ),
        "mean_error": float(residuals.mean()),
    }


def validate_regression_experiment_results(
    results: list[dict[str, Any]],
    metric_tolerance: float = 1e-10,
) -> dict[str, Any]:
    """Verify shared test rows, predictions, metrics, and regression checkpoints."""

    if not results:
        raise ValueError("At least one experiment result is required.")
    reference = results[0]
    reference_y = np.asarray(reference["y_true"], dtype=np.float64)
    reference_indices = np.asarray(reference["test_indices"])
    reference_fingerprint = str(reference["split_fingerprint"])
    reference_target = str(reference["target_name"])
    reference_experiment = str(reference.get("experiment_id", "unlabeled_experiment"))
    reference_protocol = str(reference.get("protocol_fingerprint", "unlabeled"))
    seen_runs: set[tuple[str, int]] = set()

    for result in results:
        run_key = (str(result["model_name"]), int(result["seed"]))
        if run_key in seen_runs:
            raise ValueError(f"Duplicate model/seed result in comparison: {run_key}")
        seen_runs.add(run_key)
        if result.get("test_evaluated") is not True:
            raise ValueError(f"Test evaluation is incomplete for {run_key}.")
        if str(result.get("experiment_id", "unlabeled_experiment")) != (
            reference_experiment
        ):
            raise ValueError("All results must belong to the same experiment campaign.")
        if str(result.get("protocol_fingerprint", "unlabeled")) != (
            reference_protocol
        ):
            raise ValueError("All results must use the same experiment protocol.")
        if str(result["target_name"]) != reference_target:
            raise ValueError("All results must use the same regression target.")
        if not np.array_equal(np.asarray(result["y_true"]), reference_y):
            raise ValueError("All results must use identical ordered test targets.")
        if not np.array_equal(np.asarray(result["test_indices"]), reference_indices):
            raise ValueError("All results must use identical ordered test indices.")
        if str(result["split_fingerprint"]) != reference_fingerprint:
            raise ValueError("All results must use the same split fingerprint.")

        predictions = np.asarray(result["predictions"], dtype=np.float64)
        validate_regression_predictions(reference_y, predictions)
        recomputed = compute_regression_metrics(reference_y, predictions)
        for metric_name, expected_value in result["test_metrics"].items():
            if metric_name not in recomputed:
                continue
            if not np.isclose(
                recomputed[metric_name],
                float(expected_value),
                atol=metric_tolerance,
                rtol=0.0,
            ):
                raise ValueError(
                    f"Metric {metric_name!r} is inconsistent for {run_key}."
                )

        checkpoint_path = Path(result["checkpoint_path"])
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Missing checkpoint for {run_key}: {checkpoint_path}"
            )
        if result.get("reload_check", {}).get("checkpoint_reloaded") is not True:
            raise ValueError(f"Checkpoint reload was not verified for {run_key}.")

    return {
        "n_results": len(results),
        "experiment_id": reference_experiment,
        "protocol_fingerprint": reference_protocol,
        "shared_split_fingerprint": reference_fingerprint,
        "shared_test_rows": int(len(reference_y)),
        "identical_test_targets": True,
        "identical_test_indices": True,
        "predictions_are_finite": True,
        "metrics_recomputed": True,
        "checkpoints_present_and_reloaded": True,
    }


def persist_regression_results(
    result: dict[str, Any],
    output_dirs: dict[str, Path],
) -> dict[str, Path]:
    """Persist regression metrics, history, predictions, and configuration."""

    run_id = result["run_id"]
    metrics_dir = output_dirs["metrics"]
    histories_dir = output_dirs["histories"]
    predictions_dir = output_dirs["predictions"]
    for directory in (metrics_dir, histories_dir, predictions_dir):
        directory.mkdir(parents=True, exist_ok=True)

    metrics_path = metrics_dir / f"{run_id}_metrics.csv"
    history_path = histories_dir / f"{run_id}_history.json"
    config_path = histories_dir / f"{run_id}_config.json"
    predictions_path = predictions_dir / f"{run_id}_predictions.csv"
    summary_path = metrics_dir / "summary_metrics.csv"
    metrics_row = {
        "run_id": run_id,
        "experiment_id": result.get("experiment_id", "unlabeled_experiment"),
        "protocol_fingerprint": result.get("protocol_fingerprint", "unlabeled"),
        "model_name": result["model_name"],
        "implementation_version": result.get("implementation_version"),
        "prediction_scope": result.get("prediction_scope"),
        "seed": result["seed"],
        "best_epoch": result.get("best_epoch"),
        "epochs_trained": result.get("epochs_trained"),
        "reached_epoch_budget": result.get("reached_epoch_budget"),
        "split_fingerprint": result.get("split_fingerprint"),
        "train_time_seconds": result["train_time_seconds"],
        "inference_time_seconds": result["inference_time_seconds"],
        "n_parameters": result["n_parameters"],
        **result["test_metrics"],
    }
    pd.DataFrame([metrics_row]).to_csv(metrics_path, index=False)
    _upsert_campaign_summary(summary_path, metrics_row)

    history_payload = {
        "history": _json_safe(result.get("history", {})),
        "validation_metrics": _json_safe(result.get("valid_metrics", {})),
        "reload_check": _json_safe(result.get("reload_check", {})),
    }
    history_path.write_text(
        json.dumps(history_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    config_path.write_text(
        json.dumps(_json_safe(result.get("config", {})), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    predictions = np.asarray(result["predictions"], dtype=np.float64)
    prediction_frame = pd.DataFrame(
        {
            "source_index": result["test_indices"],
            "y_true": result["y_true"],
            "y_pred": predictions,
            "residual": np.asarray(result["y_true"]) - predictions,
        }
    )
    prediction_frame.to_csv(predictions_path, index=False)
    return {
        "metrics": metrics_path,
        "summary_metrics": summary_path,
        "history": history_path,
        "config": config_path,
        "predictions": predictions_path,
    }


def regression_comparison_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Build a compact table from regression experiment results."""

    rows = [
        {
            "experiment_id": result.get("experiment_id", "unlabeled_experiment"),
            "model_name": result["model_name"],
            "implementation_version": result.get("implementation_version"),
            "prediction_scope": result.get("prediction_scope"),
            "seed": result["seed"],
            "best_epoch": result.get("best_epoch"),
            "epochs_trained": result.get("epochs_trained"),
            "reached_epoch_budget": result.get("reached_epoch_budget"),
            "n_parameters": result["n_parameters"],
            "train_time_seconds": result["train_time_seconds"],
            "inference_time_seconds": result["inference_time_seconds"],
            **result["test_metrics"],
        }
        for result in results
    ]
    return pd.DataFrame(rows)


def plot_regression_predictions(
    results: list[dict[str, Any]],
    figures_dir: Path,
    prefix: str = "regression",
) -> Path:
    """Plot observed versus predicted targets with a common scale."""

    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("No results were provided.")
    n_results = len(results)
    n_cols = min(3, n_results)
    n_rows = int(np.ceil(n_results / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
    axes_array = np.atleast_1d(axes).ravel()
    lower = min(
        min(float(np.min(result["y_true"])), float(np.min(result["predictions"])))
        for result in results
    )
    upper = max(
        max(float(np.max(result["y_true"])), float(np.max(result["predictions"])))
        for result in results
    )
    for axis, result in zip(axes_array, results):
        axis.scatter(
            result["y_true"],
            result["predictions"],
            s=8,
            alpha=0.25,
            edgecolors="none",
        )
        axis.plot([lower, upper], [lower, upper], "--", color="black", linewidth=1)
        axis.set_xlim(lower, upper)
        axis.set_ylim(lower, upper)
        axis.set_title(
            f"{result['model_name']} | R2={result['test_metrics']['r2']:.3f}"
        )
        axis.set_xlabel("Observed")
        axis.set_ylabel("Predicted")
        axis.grid(alpha=0.2)
    for axis in axes_array[n_results:]:
        axis.axis("off")
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{prefix}_observed_vs_predicted.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_regression_residuals(
    results: list[dict[str, Any]],
    figures_dir: Path,
    prefix: str = "regression",
) -> Path:
    """Plot residual magnitude and distribution for each model."""

    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("No results were provided.")
    fig, axes = plt.subplots(
        len(results),
        2,
        figsize=(11, max(3.2 * len(results), 4)),
        squeeze=False,
    )
    for row, result in enumerate(results):
        residuals = np.asarray(result["y_true"]) - np.asarray(result["predictions"])
        axes[row, 0].scatter(
            result["predictions"], residuals, s=8, alpha=0.25, edgecolors="none"
        )
        axes[row, 0].axhline(0.0, linestyle="--", color="black", linewidth=1)
        axes[row, 0].set_title(f"{result['model_name']}: residual vs prediction")
        axes[row, 0].set_xlabel("Predicted")
        axes[row, 0].set_ylabel("Observed - predicted")
        axes[row, 0].grid(alpha=0.2)
        axes[row, 1].hist(residuals, bins=45, alpha=0.8)
        axes[row, 1].axvline(0.0, linestyle="--", color="black", linewidth=1)
        axes[row, 1].set_title(f"{result['model_name']}: residual distribution")
        axes[row, 1].set_xlabel("Residual")
        axes[row, 1].set_ylabel("Rows")
        axes[row, 1].grid(alpha=0.2)
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{prefix}_residual_diagnostics.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_regression_histories(
    results: list[dict[str, Any]],
    figures_dir: Path,
    prefix: str = "regression",
) -> Path:
    """Plot validation RMSE histories when iterative histories are available."""

    import matplotlib.pyplot as plt

    iterative = [
        result
        for result in results
        if result["model_name"] != "baseline_ridge"
        and isinstance(result.get("history", {}).get("valid_rmse"), list)
        and result["history"]["valid_rmse"]
    ]
    if not iterative:
        raise ValueError("No validation RMSE histories are available.")
    for result in iterative:
        if (
            result["model_name"] == "tabnet"
            and result["history"].get("rmse_history_units")
            != "original_target_units"
        ):
            raise ValueError(
                "TabNet RMSE history is not certified in original target units."
            )
    fig, axis = plt.subplots(figsize=(8, 5))
    for result in iterative:
        values = result["history"]["valid_rmse"]
        epochs = np.arange(1, len(values) + 1)
        axis.plot(epochs, values, marker="o", markersize=3, label=result["model_name"])
        best_epoch = result.get("best_epoch")
        if best_epoch is not None and 1 <= int(best_epoch) <= len(values):
            axis.scatter(
                [int(best_epoch)],
                [values[int(best_epoch) - 1]],
                s=45,
                zorder=3,
            )
    axis.set_xlabel("Epoch")
    axis.set_ylabel("Validation RMSE (original target units)")
    axis.set_title("Validation trajectories")
    axis.legend()
    axis.grid(alpha=0.2)
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{prefix}_validation_histories.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_regression_costs(
    results: list[dict[str, Any]],
    figures_dir: Path,
    prefix: str = "regression",
) -> Path:
    """Compare training time and parameter count on separate axes."""

    import matplotlib.pyplot as plt

    if not results:
        raise ValueError("No results were provided.")
    names = [str(result["model_name"]) for result in results]
    train_times = [float(result["train_time_seconds"]) for result in results]
    parameters = [max(1, int(result["n_parameters"])) for result in results]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].bar(names, train_times)
    axes[0].set_ylabel("Seconds")
    axes[0].set_title("Training time")
    axes[1].bar(names, parameters)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("Trainable parameters (log scale)")
    axes[1].set_title("Model size")
    for axis in axes:
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.2)
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / f"{prefix}_computational_cost.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def load_persisted_task_results(
    results_dir: Path,
    task: str,
    experiment_id: str,
    expected_models: tuple[str, ...],
    expected_seeds: tuple[int, ...],
    expected_implementations: dict[str, str],
    require_row_independent: bool = True,
) -> PersistedTaskResults:
    """Load and independently validate one persisted experiment campaign."""

    if task not in {"classification", "regression"}:
        raise ValueError("task must be 'classification' or 'regression'.")
    results_dir = Path(results_dir)
    summary_path = results_dir / "metrics" / "summary_metrics.csv"
    issues: list[str] = []
    histories: dict[str, dict[str, Any]] = {}
    configs: dict[str, dict[str, Any]] = {}
    predictions: dict[str, pd.DataFrame] = {}
    checkpoints: dict[str, Path] = {}
    available_experiments: tuple[str, ...] = ()

    if not summary_path.exists():
        issues.append(f"Missing summary file: {summary_path}")
        return _persisted_results(
            task,
            experiment_id,
            pd.DataFrame(),
            histories,
            configs,
            predictions,
            checkpoints,
            issues,
            available_experiments,
            expected_models,
            expected_seeds,
        )

    summary = pd.read_csv(summary_path)
    required_columns = _comparison_required_columns(task)
    missing_columns = sorted(required_columns.difference(summary.columns))
    if missing_columns:
        issues.append(
            "Summary belongs to an incompatible protocol; missing columns: "
            + ", ".join(missing_columns)
        )
        campaign = pd.DataFrame(columns=summary.columns)
    else:
        available_experiments = tuple(
            sorted(summary["experiment_id"].astype(str).unique())
        )
        campaign = summary[
            summary["experiment_id"].astype(str) == str(experiment_id)
        ].copy()

    if campaign.empty:
        issues.append(
            f"No rows found for experiment_id={experiment_id!r}. "
            f"Available campaigns: {available_experiments or ('none',)}"
        )
        return _persisted_results(
            task,
            experiment_id,
            campaign,
            histories,
            configs,
            predictions,
            checkpoints,
            issues,
            available_experiments,
            expected_models,
            expected_seeds,
        )

    campaign["seed"] = campaign["seed"].astype(int)
    duplicate_slots = campaign.duplicated(["model_name", "seed"], keep=False)
    if duplicate_slots.any():
        slots = campaign.loc[duplicate_slots, ["model_name", "seed"]]
        issues.append(f"Duplicate model/seed rows: {slots.to_dict('records')}")

    expected_slots = {
        (model_name, int(seed))
        for model_name in expected_models
        for seed in expected_seeds
    }
    actual_slots = set(
        campaign[["model_name", "seed"]].itertuples(index=False, name=None)
    )
    missing_slots = sorted(expected_slots.difference(actual_slots))
    extra_slots = sorted(actual_slots.difference(expected_slots))
    if missing_slots:
        issues.append(f"Missing model/seed runs: {missing_slots}")
    if extra_slots:
        issues.append(f"Unexpected model/seed runs: {extra_slots}")
    if campaign["protocol_fingerprint"].astype(str).nunique() != 1:
        issues.append("Campaign contains more than one protocol fingerprint.")
    if campaign["split_fingerprint"].astype(str).nunique() != 1:
        issues.append("Campaign contains more than one split fingerprint.")
    if require_row_independent and set(campaign["prediction_scope"].astype(str)) != {
        "row_independent"
    }:
        issues.append("Campaign contains batch-contextual predictions.")

    for model_name, implementation in expected_implementations.items():
        observed = set(
            campaign.loc[
                campaign["model_name"] == model_name,
                "implementation_version",
            ].astype(str)
        )
        if observed != {implementation}:
            issues.append(
                f"{model_name} implementation mismatch: expected "
                f"{implementation!r}, found {sorted(observed)}"
            )

    reference_indices: np.ndarray | None = None
    reference_targets: np.ndarray | None = None
    for row in campaign.to_dict("records"):
        run_id = str(row["run_id"])
        metrics_path = results_dir / "metrics" / f"{run_id}_metrics.csv"
        history_path = results_dir / "histories" / f"{run_id}_history.json"
        config_path = results_dir / "histories" / f"{run_id}_config.json"
        prediction_path = results_dir / "predictions" / f"{run_id}_predictions.csv"
        checkpoint_matches = sorted(
            (results_dir / "checkpoints").glob(f"{run_id}.*")
        )
        missing_paths = [
            path
            for path in (metrics_path, history_path, config_path, prediction_path)
            if not path.exists()
        ]
        if missing_paths:
            issues.append(f"{run_id} is missing artifacts: {missing_paths}")
            continue
        if len(checkpoint_matches) != 1:
            issues.append(
                f"{run_id} must have exactly one checkpoint; found "
                f"{checkpoint_matches}"
            )
            continue

        individual_metrics = pd.read_csv(metrics_path)
        if len(individual_metrics) != 1:
            issues.append(f"{metrics_path} must contain exactly one row.")
            continue
        for column in campaign.columns.intersection(individual_metrics.columns):
            if not _persisted_values_match(
                row[column], individual_metrics.iloc[0][column]
            ):
                issues.append(
                    f"{run_id} summary differs from its metrics file at {column}."
                )

        history_payload = json.loads(history_path.read_text(encoding="utf-8"))
        config_payload = json.loads(config_path.read_text(encoding="utf-8"))
        prediction_frame = pd.read_csv(prediction_path)
        histories[run_id] = history_payload
        configs[run_id] = config_payload
        predictions[run_id] = prediction_frame
        checkpoints[run_id] = checkpoint_matches[0]

        experiment_config = config_payload.get("experiment", {})
        if experiment_config.get("experiment_id") != experiment_id:
            issues.append(f"{run_id} config has a different experiment_id.")
        if int(experiment_config.get("seed", -1)) != int(row["seed"]):
            issues.append(f"{run_id} config seed differs from its metrics row.")
        configured_seeds = tuple(
            int(seed) for seed in experiment_config.get("seeds", [])
        )
        if set(configured_seeds) != set(expected_seeds):
            issues.append(f"{run_id} config has unexpected seeds: {configured_seeds}")
        if experiment_config.get("defer_test_until_final") is not True:
            issues.append(f"{run_id} did not defer test evaluation until finalization.")
        if (
            require_row_independent
            and experiment_config.get("require_row_independent_inference")
            is not True
        ):
            issues.append(f"{run_id} did not require row-independent inference.")
        if config_payload.get("test_split_used_for_model_selection") is not False:
            issues.append(f"{run_id} does not certify test isolation.")
        expected_metric = str(_task_metric_specification(task)["primary"])
        training_config = config_payload.get("training", {})
        if training_config.get("selection_metric") != expected_metric:
            issues.append(
                f"{run_id} selected checkpoints with an unexpected metric."
            )
        reload_check = history_payload.get("reload_check", {})
        if reload_check.get("checkpoint_reloaded") is not True:
            issues.append(f"{run_id} does not certify checkpoint reload.")
        difference_keys = (
            "max_abs_probability_difference",
            "max_abs_prediction_difference",
        )
        difference_values = [
            float(reload_check[key])
            for key in difference_keys
            if key in reload_check
        ]
        tolerance = float(reload_check.get("tolerance", -1.0))
        if (
            len(difference_values) != 1
            or tolerance < 0.0
            or difference_values[0] > tolerance
        ):
            issues.append(f"{run_id} fails the checkpoint prediction reload check.")
        metadata = config_payload.get("data_metadata", {})
        if str(metadata.get("split_fingerprint")) != str(row["split_fingerprint"]):
            issues.append(f"{run_id} config and metrics fingerprints differ.")

        try:
            recomputed = _recompute_persisted_metrics(
                task=task,
                row=row,
                prediction_frame=prediction_frame,
                metadata=metadata,
            )
            for metric_name, value in recomputed.items():
                if metric_name in row and not np.isclose(
                    float(row[metric_name]),
                    float(value),
                    atol=1e-9,
                    rtol=0.0,
                    equal_nan=True,
                ):
                    issues.append(
                        f"{run_id} has inconsistent persisted {metric_name}."
                    )
        except (KeyError, TypeError, ValueError) as error:
            issues.append(f"{run_id} prediction validation failed: {error}")

        indices = prediction_frame.get("source_index")
        targets = prediction_frame.get("y_true")
        if indices is None or targets is None:
            issues.append(f"{run_id} predictions lack source_index or y_true.")
        elif reference_indices is None:
            reference_indices = indices.to_numpy()
            reference_targets = targets.to_numpy()
        elif not np.array_equal(
            indices.to_numpy(), reference_indices
        ) or not np.allclose(
            targets.to_numpy(),
            reference_targets,
            atol=0.0,
            rtol=0.0,
        ):
            issues.append(f"{run_id} does not use the shared ordered test rows.")

    campaign = campaign.sort_values(["model_name", "seed"]).reset_index(drop=True)
    return _persisted_results(
        task,
        experiment_id,
        campaign,
        histories,
        configs,
        predictions,
        checkpoints,
        issues,
        available_experiments,
        expected_models,
        expected_seeds,
    )


def summarize_task_metrics(metrics: pd.DataFrame, task: str) -> pd.DataFrame:
    """Aggregate test metrics across seeds without mixing task scales."""

    specification = _task_metric_specification(task)
    if metrics.empty:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    grouping = ["model_name", "implementation_version", "prediction_scope"]
    for keys, group in metrics.groupby(grouping, sort=True, dropna=False):
        row: dict[str, Any] = dict(zip(grouping, keys))
        row["n_seeds"] = int(group["seed"].nunique())
        for metric_name in specification["metrics"]:
            values = pd.to_numeric(group[metric_name], errors="raise")
            row[f"{metric_name}_mean"] = float(values.mean())
            row[f"{metric_name}_std"] = float(values.std(ddof=1))
        row["n_parameters"] = int(round(group["n_parameters"].mean()))
        row["best_epoch_mean"] = float(group["best_epoch"].mean())
        row["epochs_trained_mean"] = float(group["epochs_trained"].mean())
        budget = pd.to_numeric(group["reached_epoch_budget"], errors="coerce")
        row["budget_exhaustion_rate"] = float(budget.mean())
        rows.append(row)

    summary = pd.DataFrame(rows)
    primary = str(specification["primary"])
    ascending = bool(specification["ascending"])
    summary["primary_metric"] = primary
    summary["primary_mean"] = summary[f"{primary}_mean"]
    summary["primary_std"] = summary[f"{primary}_std"]
    summary["within_task_rank"] = summary["primary_mean"].rank(
        method="min",
        ascending=ascending,
    ).astype(int)

    baseline_name = str(specification["baseline"])
    baseline_rows = summary[summary["model_name"] == baseline_name]
    if len(baseline_rows) != 1:
        raise ValueError(f"Expected exactly one {baseline_name} aggregate row.")
    baseline_value = float(baseline_rows.iloc[0]["primary_mean"])
    if task == "classification":
        summary["primary_effect_vs_baseline"] = (
            summary["primary_mean"] - baseline_value
        )
        summary["effect_unit"] = "absolute ROC-AUC difference"
    else:
        summary["primary_effect_vs_baseline"] = 100.0 * (
            baseline_value - summary["primary_mean"]
        ) / baseline_value
        summary["effect_unit"] = "RMSE reduction (%)"
    return summary.sort_values("within_task_rank").reset_index(drop=True)


def paired_seed_differences(metrics: pd.DataFrame, task: str) -> pd.DataFrame:
    """Compare the best mean model with alternatives using matched seeds."""

    specification = _task_metric_specification(task)
    primary = str(specification["primary"])
    ascending = bool(specification["ascending"])
    model_means = metrics.groupby("model_name")[primary].mean()
    reference_model = (
        model_means.idxmin() if ascending else model_means.idxmax()
    )
    reference = metrics.loc[
        metrics["model_name"] == reference_model,
        ["seed", primary],
    ].rename(columns={primary: "reference_value"})
    rows: list[dict[str, Any]] = []
    for competitor in sorted(set(metrics["model_name"]) - {reference_model}):
        alternative = metrics.loc[
            metrics["model_name"] == competitor,
            ["seed", primary],
        ].rename(columns={primary: "competitor_value"})
        paired = reference.merge(alternative, on="seed", validate="one_to_one")
        if ascending:
            advantage = paired["competitor_value"] - paired["reference_value"]
        else:
            advantage = paired["reference_value"] - paired["competitor_value"]
        rows.append(
            {
                "reference_model": reference_model,
                "competitor_model": competitor,
                "metric": primary,
                "n_paired_seeds": int(len(paired)),
                "mean_reference_advantage": float(advantage.mean()),
                "std_reference_advantage": float(advantage.std(ddof=1)),
                "reference_wins": int((advantage > 0.0).sum()),
                "ties": int(np.isclose(advantage, 0.0, atol=1e-12).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        "mean_reference_advantage",
        ascending=False,
    ).reset_index(drop=True)


def build_cross_task_architecture_table(
    classification_summary: pd.DataFrame,
    regression_summary: pd.DataFrame,
) -> pd.DataFrame:
    """Align deep architectures while preserving task-specific metrics."""

    deep_models = {
        "tabnet",
        "tab_transformer",
        "ft_transformer",
        "saint_supervised",
    }
    class_columns = {
        "model_name": "model_name",
        "within_task_rank": "classification_overall_rank",
        "roc_auc_mean": "classification_roc_auc_mean",
        "roc_auc_std": "classification_roc_auc_std",
        "balanced_accuracy_mean": "classification_balanced_accuracy_mean",
        "primary_effect_vs_baseline": "classification_auc_gain_vs_baseline",
        "train_time_seconds_mean": "classification_train_time_mean",
        "n_parameters": "classification_parameters",
    }
    regression_columns = {
        "model_name": "model_name",
        "within_task_rank": "regression_overall_rank",
        "rmse_mean": "regression_rmse_mean",
        "rmse_std": "regression_rmse_std",
        "mae_mean": "regression_mae_mean",
        "r2_mean": "regression_r2_mean",
        "primary_effect_vs_baseline": "regression_rmse_reduction_percent",
        "train_time_seconds_mean": "regression_train_time_mean",
        "n_parameters": "regression_parameters",
    }
    classification = classification_summary[
        classification_summary["model_name"].isin(deep_models)
    ][list(class_columns)].rename(columns=class_columns)
    classification["classification_deep_rank"] = classification[
        "classification_roc_auc_mean"
    ].rank(method="min", ascending=False).astype(int)
    regression = regression_summary[
        regression_summary["model_name"].isin(deep_models)
    ][list(regression_columns)].rename(columns=regression_columns)
    regression["regression_deep_rank"] = regression[
        "regression_rmse_mean"
    ].rank(method="min", ascending=True).astype(int)
    aligned = classification.merge(
        regression,
        on="model_name",
        how="inner",
        validate="one_to_one",
    )
    aligned["descriptive_mean_rank"] = aligned[
        ["classification_deep_rank", "regression_deep_rank"]
    ].mean(axis=1)
    aligned["rank_gap"] = (
        aligned["classification_deep_rank"] - aligned["regression_deep_rank"]
    ).abs()
    return aligned.sort_values(
        ["descriptive_mean_rank", "rank_gap", "model_name"]
    ).reset_index(drop=True)


def build_seed_rank_table(metrics: pd.DataFrame, task: str) -> pd.DataFrame:
    """Return within-seed ranks for the task's primary metric."""

    specification = _task_metric_specification(task)
    primary = str(specification["primary"])
    ascending = bool(specification["ascending"])
    table = metrics[["model_name", "seed", primary]].copy()
    table["rank"] = table.groupby("seed")[primary].rank(
        method="min",
        ascending=ascending,
    ).astype(int)
    return table.sort_values(["seed", "rank", "model_name"]).reset_index(drop=True)


def plot_cross_task_performance(
    classification_summary: pd.DataFrame,
    regression_summary: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    """Plot task-specific primary metrics with between-seed error bars."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    panels = (
        (classification_summary, "ROC-AUC", "Classification: ROC-AUC", False),
        (regression_summary, "RMSE", "Regression: RMSE", True),
    )
    for axis, (summary, label, title, lower_is_better) in zip(axes, panels):
        ordered = summary.sort_values("primary_mean", ascending=lower_is_better)
        positions = np.arange(len(ordered))
        errors = ordered["primary_std"].fillna(0.0).to_numpy()
        axis.errorbar(
            ordered["primary_mean"],
            positions,
            xerr=errors,
            fmt="o",
            capsize=4,
        )
        axis.set_yticks(positions)
        axis.set_yticklabels(
            [_display_model_name(name) for name in ordered["model_name"]]
        )
        axis.invert_yaxis()
        axis.set_xlabel(f"Mean {label} +/- 1 SD across seeds")
        axis.set_title(title)
        axis.grid(axis="x", alpha=0.25)
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "comparison_task_performance.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_cross_task_resources(
    classification_summary: pd.DataFrame,
    regression_summary: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    """Compare time and model size separately inside each task."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    for column, (summary, task_label) in enumerate(
        (
            (classification_summary, "Classification"),
            (regression_summary, "Regression"),
        )
    ):
        ordered = summary.sort_values("train_time_seconds_mean")
        names = [_display_model_name(name) for name in ordered["model_name"]]
        axes[0, column].bar(names, ordered["train_time_seconds_mean"])
        axes[0, column].set_yscale("log")
        axes[0, column].set_ylabel("Training seconds (log scale)")
        axes[0, column].set_title(f"{task_label}: training time")
        axes[1, column].bar(names, ordered["n_parameters"])
        axes[1, column].set_yscale("log")
        axes[1, column].set_ylabel("Trainable parameters (log scale)")
        axes[1, column].set_title(f"{task_label}: model size")
        for row in range(2):
            axes[row, column].tick_params(axis="x", rotation=30)
            axes[row, column].grid(axis="y", alpha=0.2)
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "comparison_resources.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_seed_rank_consistency(
    classification_ranks: pd.DataFrame,
    regression_ranks: pd.DataFrame,
    figures_dir: Path,
) -> Path:
    """Visualize whether model ordering is stable across matched seeds."""

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, ranks, title in (
        (axes[0], classification_ranks, "Classification ranks"),
        (axes[1], regression_ranks, "Regression ranks"),
    ):
        matrix = ranks.pivot(index="model_name", columns="seed", values="rank")
        matrix = matrix.sort_values(list(matrix.columns))
        image = axis.imshow(matrix.to_numpy(), cmap="viridis_r", vmin=1)
        axis.set_xticks(np.arange(len(matrix.columns)))
        axis.set_xticklabels(matrix.columns)
        axis.set_yticks(np.arange(len(matrix.index)))
        axis.set_yticklabels([_display_model_name(name) for name in matrix.index])
        axis.set_xlabel("Seed")
        axis.set_title(title)
        for row_index in range(matrix.shape[0]):
            for column_index in range(matrix.shape[1]):
                axis.text(
                    column_index,
                    row_index,
                    int(matrix.iloc[row_index, column_index]),
                    ha="center",
                    va="center",
                )
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04, label="Rank")
    figures_dir.mkdir(parents=True, exist_ok=True)
    path = figures_dir / "comparison_seed_ranks.png"
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _upsert_campaign_summary(
    summary_path: Path,
    metrics_row: dict[str, Any],
) -> None:
    """Keep one current row per model/seed within one declared campaign."""

    experiment_id = str(metrics_row["experiment_id"])
    protocol_fingerprint = str(metrics_row["protocol_fingerprint"])
    if summary_path.exists():
        summary = pd.read_csv(summary_path)
        if not {"experiment_id", "protocol_fingerprint"}.issubset(
            summary.columns
        ):
            summary = pd.DataFrame()
        else:
            same_campaign = summary["experiment_id"].astype(str) == experiment_id
            same_protocol = (
                summary["protocol_fingerprint"].astype(str)
                == protocol_fingerprint
            )
            summary = summary[same_campaign & same_protocol]
    else:
        summary = pd.DataFrame()

    if not summary.empty:
        same_run_slot = (
            (summary["model_name"].astype(str) == str(metrics_row["model_name"]))
            & (summary["seed"].astype(int) == int(metrics_row["seed"]))
        )
        summary = summary.loc[~same_run_slot]
    summary = pd.concat([summary, pd.DataFrame([metrics_row])], ignore_index=True)
    summary.sort_values(["model_name", "seed"]).to_csv(summary_path, index=False)


def _comparison_required_columns(task: str) -> set[str]:
    common = {
        "run_id",
        "experiment_id",
        "protocol_fingerprint",
        "model_name",
        "implementation_version",
        "prediction_scope",
        "seed",
        "best_epoch",
        "epochs_trained",
        "reached_epoch_budget",
        "split_fingerprint",
        "train_time_seconds",
        "inference_time_seconds",
        "n_parameters",
    }
    task_metrics = set(_task_metric_specification(task)["metrics"])
    return common.union(task_metrics)


def _task_metric_specification(task: str) -> dict[str, Any]:
    if task == "classification":
        return {
            "primary": "roc_auc",
            "ascending": False,
            "baseline": "baseline_logistic",
            "metrics": (
                "accuracy",
                "balanced_accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "pr_auc",
                "train_time_seconds",
                "inference_time_seconds",
            ),
        }
    if task == "regression":
        return {
            "primary": "rmse",
            "ascending": True,
            "baseline": "baseline_ridge",
            "metrics": (
                "rmse",
                "mae",
                "median_absolute_error",
                "r2",
                "mape_percent",
                "mean_error",
                "train_time_seconds",
                "inference_time_seconds",
            ),
        }
    raise ValueError("task must be 'classification' or 'regression'.")


def _persisted_results(
    task: str,
    experiment_id: str,
    metrics: pd.DataFrame,
    histories: dict[str, dict[str, Any]],
    configs: dict[str, dict[str, Any]],
    predictions: dict[str, pd.DataFrame],
    checkpoints: dict[str, Path],
    issues: list[str],
    available_experiments: tuple[str, ...],
    expected_models: tuple[str, ...],
    expected_seeds: tuple[int, ...],
) -> PersistedTaskResults:
    report = {
        "ready": not issues,
        "task": task,
        "experiment_id": experiment_id,
        "available_experiments": available_experiments,
        "expected_models": expected_models,
        "expected_seeds": expected_seeds,
        "expected_runs": len(expected_models) * len(expected_seeds),
        "loaded_runs": int(len(metrics)),
        "loaded_histories": len(histories),
        "loaded_predictions": len(predictions),
        "loaded_checkpoints": len(checkpoints),
        "issues": tuple(issues),
    }
    return PersistedTaskResults(
        task=task,
        experiment_id=experiment_id,
        metrics=metrics,
        histories=histories,
        configs=configs,
        predictions=predictions,
        checkpoints=checkpoints,
        report=report,
    )


def _recompute_persisted_metrics(
    task: str,
    row: dict[str, Any],
    prediction_frame: pd.DataFrame,
    metadata: dict[str, Any],
) -> dict[str, float]:
    if task == "classification":
        class_names = tuple(str(name) for name in metadata["class_names"])
        probability_columns = [f"prob_{name}" for name in class_names]
        missing = set(probability_columns).difference(prediction_frame.columns)
        if missing:
            raise ValueError(f"Missing probability columns: {sorted(missing)}")
        probabilities = prediction_frame[probability_columns].to_numpy(dtype=float)
        validate_probabilities(probabilities)
        y_true = prediction_frame["y_true"].to_numpy(dtype=np.int64)
        threshold = float(row.get("threshold", 0.5))
        expected_labels = labels_from_probabilities(probabilities, threshold)
        stored_labels = prediction_frame["y_pred"].to_numpy(dtype=np.int64)
        if not np.array_equal(expected_labels, stored_labels):
            raise ValueError("Stored labels do not match persisted probabilities.")
        computed = compute_classification_metrics(
            y_true=y_true,
            probabilities=probabilities,
            class_names=class_names,
            threshold=threshold,
        )
    else:
        y_true = prediction_frame["y_true"].to_numpy(dtype=float)
        y_pred = prediction_frame["y_pred"].to_numpy(dtype=float)
        computed = compute_regression_metrics(y_true, y_pred)
        if "residual" in prediction_frame:
            expected_residuals = y_true - y_pred
            stored_residuals = prediction_frame["residual"].to_numpy(dtype=float)
            if not np.allclose(
                expected_residuals,
                stored_residuals,
                atol=1e-10,
                rtol=0.0,
            ):
                raise ValueError("Stored residuals are inconsistent.")
    return {
        key: float(value)
        for key, value in computed.items()
        if isinstance(value, (int, float, np.number))
    }


def _persisted_values_match(first: Any, second: Any) -> bool:
    if pd.isna(first) and pd.isna(second):
        return True
    numeric_types = (bool, int, float, np.bool_, np.number)
    if isinstance(first, numeric_types) and isinstance(
        second, numeric_types
    ):
        return bool(
            np.isclose(
                float(first),
                float(second),
                atol=1e-10,
                rtol=0.0,
                equal_nan=True,
            )
        )
    return str(first) == str(second)


def _display_model_name(model_name: str) -> str:
    names = {
        "baseline_logistic": "Logistic",
        "baseline_ridge": "Ridge",
        "tabnet": "TabNet",
        "tab_transformer": "TabTransformer",
        "ft_transformer": "FT-Transformer",
        "saint_supervised": "SAINT",
    }
    return names.get(str(model_name), str(model_name))
