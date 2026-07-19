"""Data loading, splitting, preprocessing, and leakage checks."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype, is_string_dtype
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split


UNKNOWN_CATEGORY = "__UNKNOWN__"
MISSING_CATEGORY = "__MISSING__"


@dataclass(frozen=True)
class DatasetConfig:
    """Editable dataset definition for a classification experiment."""

    name: str
    data_dir: Path
    source_files: tuple[str, ...]
    target_name: str
    positive_class: str | None
    negative_class: str | None
    numerical_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    identifier_columns: tuple[str, ...]
    leakage_columns: tuple[str, ...]
    positive_class_meaning: str


@dataclass(frozen=True)
class SplitConfig:
    """Split policy with future extension points for grouped time series data."""

    train_size: float = 0.70
    valid_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    strategy: Literal["stratified_random", "group", "temporal_block"] = (
        "stratified_random"
    )
    group_column: str | None = None
    time_column: str | None = None

    def validate(self) -> None:
        total = self.train_size + self.valid_size + self.test_size
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split proportions must sum to 1.0; got {total:.4f}.")
        if min(self.train_size, self.valid_size, self.test_size) <= 0:
            raise ValueError("All split proportions must be positive.")
        if self.strategy == "group" and not self.group_column:
            raise ValueError("A group split requires group_column.")
        if self.strategy == "temporal_block" and not self.time_column:
            raise ValueError("A temporal split requires time_column.")


@dataclass(frozen=True)
class RegressionDatasetConfig:
    """Editable definition of a tabular regression dataset."""

    name: str
    data_dir: Path
    target_name: str
    numerical_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    excluded_columns: tuple[str, ...]
    identifier_columns: tuple[str, ...]
    leakage_columns: tuple[str, ...]
    target_unit: str
    spatial_columns: tuple[str, ...] = ()


@dataclass(frozen=True)
class RegressionSplitConfig:
    """Split policy for regression with explicit non-random extension points."""

    train_size: float = 0.70
    valid_size: float = 0.15
    test_size: float = 0.15
    random_state: int = 42
    strategy: Literal["random", "group", "spatial_block", "temporal_block"] = (
        "random"
    )
    group_column: str | None = None
    time_column: str | None = None

    def validate(self) -> None:
        total = self.train_size + self.valid_size + self.test_size
        if not np.isclose(total, 1.0):
            raise ValueError(f"Split proportions must sum to 1.0; got {total:.4f}.")
        if min(self.train_size, self.valid_size, self.test_size) <= 0:
            raise ValueError("All split proportions must be positive.")
        if self.strategy == "group" and not self.group_column:
            raise ValueError("A group split requires group_column.")
        if self.strategy == "temporal_block" and not self.time_column:
            raise ValueError("A temporal split requires time_column.")


@dataclass(frozen=True)
class PreprocessingState:
    """Parameters fitted on the training split only."""

    numerical_medians: dict[str, float]
    numerical_means: dict[str, float]
    numerical_stds: dict[str, float]
    categorical_mappings: dict[str, dict[str, int]]
    fitted_on: Literal["train"]
    fit_row_count: int


@dataclass(frozen=True)
class RegressionPreprocessingState:
    """Feature and target transforms fitted exclusively on training rows."""

    feature_state: PreprocessingState
    target_mean: float
    target_std: float
    fitted_on: Literal["train"]
    fit_row_count: int


@dataclass
class PreparedClassificationData:
    """Prepared arrays and metadata consumed by all classification models."""

    X_train: np.ndarray
    X_valid: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    numerical_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    categorical_indices: tuple[int, ...]
    categorical_cardinalities: tuple[int, ...]
    class_names: tuple[str, ...]
    target_name: str
    dataset_name: str
    X_num_train: np.ndarray
    X_num_valid: np.ndarray
    X_num_test: np.ndarray
    X_cat_train: np.ndarray
    X_cat_valid: np.ndarray
    X_cat_test: np.ndarray
    numerical_indices: tuple[int, ...]
    feature_columns: tuple[str, ...]
    train_indices: np.ndarray
    valid_indices: np.ndarray
    test_indices: np.ndarray
    positive_class: str | None
    preprocessing_state: PreprocessingState
    split_report: dict[str, Any]

    def split_fingerprint(self) -> str:
        """Hash split membership, labels, and prepared feature values."""

        digest = hashlib.sha256()
        for split_name, indices, labels, features in (
            ("train", self.train_indices, self.y_train, self.X_train),
            ("valid", self.valid_indices, self.y_valid, self.X_valid),
            ("test", self.test_indices, self.y_test, self.X_test),
        ):
            digest.update(split_name.encode("ascii"))
            digest.update(np.asarray(indices, dtype="<i8").tobytes())
            digest.update(np.asarray(labels, dtype="<i8").tobytes())
            digest.update(
                np.ascontiguousarray(features, dtype="<f4").tobytes()
            )
        digest.update("\x1f".join(self.feature_columns).encode("utf-8"))
        digest.update("\x1f".join(self.class_names).encode("utf-8"))
        return digest.hexdigest()

    def metadata(self) -> dict[str, Any]:
        """Return model-facing metadata without exposing preprocessing internals."""

        return {
            "dataset_name": self.dataset_name,
            "n_features": int(self.X_train.shape[1]),
            "n_numerical_features": int(self.X_num_train.shape[1]),
            "n_categorical_features": int(self.X_cat_train.shape[1]),
            "categorical_cardinalities": self.categorical_cardinalities,
            "categorical_indices": self.categorical_indices,
            "numerical_indices": self.numerical_indices,
            "class_names": self.class_names,
            "n_classes": len(self.class_names),
            "target_name": self.target_name,
            "positive_class": self.positive_class,
            "feature_columns": self.feature_columns,
            "split_fingerprint": self.split_fingerprint(),
        }


@dataclass
class PreparedRegressionData:
    """Prepared arrays and metadata consumed by all regression models."""

    X_train: np.ndarray
    X_valid: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_valid: np.ndarray
    y_test: np.ndarray
    y_train_scaled: np.ndarray
    y_valid_scaled: np.ndarray
    y_test_scaled: np.ndarray
    numerical_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    categorical_indices: tuple[int, ...]
    categorical_cardinalities: tuple[int, ...]
    target_name: str
    target_unit: str
    dataset_name: str
    X_num_train: np.ndarray
    X_num_valid: np.ndarray
    X_num_test: np.ndarray
    X_cat_train: np.ndarray
    X_cat_valid: np.ndarray
    X_cat_test: np.ndarray
    numerical_indices: tuple[int, ...]
    feature_columns: tuple[str, ...]
    train_indices: np.ndarray
    valid_indices: np.ndarray
    test_indices: np.ndarray
    preprocessing_state: RegressionPreprocessingState
    split_report: dict[str, Any]

    def transform_target(self, values: np.ndarray) -> np.ndarray:
        """Apply the train-fitted target standardization."""

        array = np.asarray(values, dtype=np.float64)
        state = self.preprocessing_state
        return ((array - state.target_mean) / state.target_std).astype(np.float32)

    def inverse_transform_target(self, values: np.ndarray) -> np.ndarray:
        """Return standardized predictions to the original target units."""

        array = np.asarray(values, dtype=np.float64)
        state = self.preprocessing_state
        return array * state.target_std + state.target_mean

    def split_fingerprint(self) -> str:
        """Hash split membership, targets, and prepared feature values."""

        digest = hashlib.sha256()
        for split_name, indices, targets, features in (
            ("train", self.train_indices, self.y_train, self.X_train),
            ("valid", self.valid_indices, self.y_valid, self.X_valid),
            ("test", self.test_indices, self.y_test, self.X_test),
        ):
            digest.update(split_name.encode("ascii"))
            digest.update(np.asarray(indices, dtype="<i8").tobytes())
            digest.update(np.asarray(targets, dtype="<f8").tobytes())
            digest.update(
                np.ascontiguousarray(features, dtype="<f4").tobytes()
            )
        digest.update("\x1f".join(self.feature_columns).encode("utf-8"))
        return digest.hexdigest()

    def metadata(self) -> dict[str, Any]:
        """Return model-facing regression metadata."""

        return {
            "task": "regression",
            "dataset_name": self.dataset_name,
            "n_features": int(self.X_train.shape[1]),
            "n_numerical_features": int(self.X_num_train.shape[1]),
            "n_categorical_features": int(self.X_cat_train.shape[1]),
            "categorical_cardinalities": self.categorical_cardinalities,
            "categorical_indices": self.categorical_indices,
            "numerical_indices": self.numerical_indices,
            "n_outputs": 1,
            "target_name": self.target_name,
            "target_unit": self.target_unit,
            "target_mean": self.preprocessing_state.target_mean,
            "target_std": self.preprocessing_state.target_std,
            "feature_columns": self.feature_columns,
            "split_fingerprint": self.split_fingerprint(),
        }


def make_airlines_config(data_dir: Path | str = Path("archive")) -> DatasetConfig:
    """Return the editable Airlines dataset configuration used in this study."""

    return DatasetConfig(
        name="airlines_passenger_satisfaction",
        data_dir=Path(data_dir),
        source_files=("train.csv", "test.csv"),
        target_name="satisfaction",
        positive_class="satisfied",
        negative_class="neutral or dissatisfied",
        categorical_columns=(
            "Gender",
            "Customer Type",
            "Type of Travel",
            "Class",
        ),
        numerical_columns=(
            "Age",
            "Flight Distance",
            "Inflight wifi service",
            "Departure/Arrival time convenient",
            "Ease of Online booking",
            "Gate location",
            "Food and drink",
            "Online boarding",
            "Seat comfort",
            "Inflight entertainment",
            "On-board service",
            "Leg room service",
            "Baggage handling",
            "Checkin service",
            "Inflight service",
            "Cleanliness",
            "Departure Delay in Minutes",
            "Arrival Delay in Minutes",
        ),
        excluded_columns=("Unnamed: 0", "id"),
        identifier_columns=("id",),
        leakage_columns=(
            "Inflight wifi service",
            "Departure/Arrival time convenient",
            "Ease of Online booking",
            "Gate location",
            "Food and drink",
            "Online boarding",
            "Seat comfort",
            "Inflight entertainment",
            "On-board service",
            "Leg room service",
            "Baggage handling",
            "Checkin service",
            "Inflight service",
            "Cleanliness",
            "Departure Delay in Minutes",
            "Arrival Delay in Minutes",
        ),
        positive_class_meaning=(
            "The passenger reports being satisfied with the flight experience."
        ),
    )


def make_california_housing_config(
    data_dir: Path | str = Path("archive"),
) -> RegressionDatasetConfig:
    """Return the editable California Housing regression configuration."""

    return RegressionDatasetConfig(
        name="california_housing",
        data_dir=Path(data_dir),
        target_name="MedHouseVal",
        numerical_columns=(
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ),
        categorical_columns=(),
        excluded_columns=(),
        identifier_columns=(),
        leakage_columns=(),
        target_unit="hundreds of thousands of US dollars",
        spatial_columns=("Latitude", "Longitude"),
    )


def load_regression_dataframe(
    config: RegressionDatasetConfig,
    download_if_missing: bool = True,
) -> pd.DataFrame:
    """Load a configured regression dataset as a self-contained DataFrame."""

    if config.name != "california_housing":
        raise ValueError(
            "No loader is registered for regression dataset "
            f"{config.name!r}. Add its loading rule in src.data."
        )
    dataset = fetch_california_housing(
        data_home=config.data_dir,
        as_frame=True,
        download_if_missing=download_if_missing,
    )
    frame = dataset.frame.copy()
    _validate_regression_dataset_columns(frame, config)
    return frame


def audit_regression_dataframe(
    df: pd.DataFrame,
    config: RegressionDatasetConfig,
) -> dict[str, Any]:
    """Return structural, quality, and target diagnostics for regression."""

    _validate_regression_dataset_columns(df, config)
    target = pd.to_numeric(df[config.target_name], errors="coerce")
    finite_target = target[np.isfinite(target)]
    if finite_target.empty:
        raise ValueError("The regression target has no finite observations.")
    target_max = float(finite_target.max())
    target_min = float(finite_target.min())
    return {
        "shape": tuple(df.shape),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": df.isna().sum().loc[lambda values: values > 0].to_dict(),
        "infinite_values": {
            column: int(np.isinf(pd.to_numeric(df[column], errors="coerce")).sum())
            for column in config.numerical_columns + (config.target_name,)
        },
        "duplicate_rows": int(df.duplicated().sum()),
        "target_summary": {
            "count": int(finite_target.size),
            "mean": float(finite_target.mean()),
            "std": float(finite_target.std(ddof=0)),
            "min": target_min,
            "q25": float(finite_target.quantile(0.25)),
            "median": float(finite_target.median()),
            "q75": float(finite_target.quantile(0.75)),
            "max": target_max,
        },
        "rows_at_observed_target_maximum": int(np.isclose(target, target_max).sum()),
        "observed_target_maximum_fraction": float(
            np.isclose(target, target_max).mean()
        ),
        "identifier_columns": list(config.identifier_columns),
        "configured_leakage_columns": list(config.leakage_columns),
        "spatial_columns": list(config.spatial_columns),
    }


def sample_regression_dataframe(
    df: pd.DataFrame,
    sample_size: int | None,
    random_state: int,
) -> pd.DataFrame:
    """Optionally draw a deterministic development sample before splitting."""

    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive or None.")
    return df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)


def prepare_regression_data(
    df: pd.DataFrame,
    dataset_config: RegressionDatasetConfig,
    split_config: RegressionSplitConfig,
) -> PreparedRegressionData:
    """Split and transform regression data using training rows only."""

    split_config.validate()
    _validate_regression_dataset_columns(df, dataset_config)
    df = df.reset_index(drop=True)
    y = pd.to_numeric(df[dataset_config.target_name], errors="coerce").to_numpy(
        dtype=np.float64
    )
    if not np.isfinite(y).all():
        raise ValueError("The regression target contains NaN or infinite values.")

    train_indices, valid_indices, test_indices = _make_regression_split_indices(
        n_rows=len(df),
        config=split_config,
    )
    feature_state = _fit_preprocessing(
        df=df,
        train_indices=train_indices,
        numerical_columns=dataset_config.numerical_columns,
        categorical_columns=dataset_config.categorical_columns,
    )
    target_mean = float(y[train_indices].mean())
    target_std = float(y[train_indices].std(ddof=0))
    if not np.isfinite(target_std) or target_std <= 0:
        raise ValueError("The training target must have positive finite variance.")
    state = RegressionPreprocessingState(
        feature_state=feature_state,
        target_mean=target_mean,
        target_std=target_std,
        fitted_on="train",
        fit_row_count=int(len(train_indices)),
    )

    X_cat_train = _transform_categorical(
        df.iloc[train_indices], dataset_config.categorical_columns, feature_state
    )
    X_cat_valid = _transform_categorical(
        df.iloc[valid_indices], dataset_config.categorical_columns, feature_state
    )
    X_cat_test = _transform_categorical(
        df.iloc[test_indices], dataset_config.categorical_columns, feature_state
    )
    X_num_train = _transform_numerical(
        df.iloc[train_indices], dataset_config.numerical_columns, feature_state
    )
    X_num_valid = _transform_numerical(
        df.iloc[valid_indices], dataset_config.numerical_columns, feature_state
    )
    X_num_test = _transform_numerical(
        df.iloc[test_indices], dataset_config.numerical_columns, feature_state
    )

    n_cat = len(dataset_config.categorical_columns)
    cardinalities = tuple(
        len(feature_state.categorical_mappings[column]) + 1
        for column in dataset_config.categorical_columns
    )
    feature_columns = (
        dataset_config.categorical_columns + dataset_config.numerical_columns
    )
    def scale_target(values: np.ndarray) -> np.ndarray:
        return ((values - target_mean) / target_std).astype(np.float32)

    prepared = PreparedRegressionData(
        X_train=_combine_features(X_cat_train, X_num_train),
        X_valid=_combine_features(X_cat_valid, X_num_valid),
        X_test=_combine_features(X_cat_test, X_num_test),
        y_train=y[train_indices],
        y_valid=y[valid_indices],
        y_test=y[test_indices],
        y_train_scaled=scale_target(y[train_indices]),
        y_valid_scaled=scale_target(y[valid_indices]),
        y_test_scaled=scale_target(y[test_indices]),
        numerical_columns=dataset_config.numerical_columns,
        categorical_columns=dataset_config.categorical_columns,
        categorical_indices=tuple(range(n_cat)),
        categorical_cardinalities=cardinalities,
        target_name=dataset_config.target_name,
        target_unit=dataset_config.target_unit,
        dataset_name=dataset_config.name,
        X_num_train=X_num_train,
        X_num_valid=X_num_valid,
        X_num_test=X_num_test,
        X_cat_train=X_cat_train,
        X_cat_valid=X_cat_valid,
        X_cat_test=X_cat_test,
        numerical_indices=tuple(
            range(n_cat, n_cat + len(dataset_config.numerical_columns))
        ),
        feature_columns=feature_columns,
        train_indices=train_indices,
        valid_indices=valid_indices,
        test_indices=test_indices,
        preprocessing_state=state,
        split_report={},
    )
    prepared.split_report = validate_prepared_regression_data(prepared, y)
    return prepared


def validate_prepared_regression_data(
    data: PreparedRegressionData,
    all_targets: np.ndarray | None = None,
) -> dict[str, Any]:
    """Assert split, transformation, shape, and target integrity for regression."""

    _assert_disjoint_indices(data.train_indices, data.valid_indices, "train", "valid")
    _assert_disjoint_indices(data.train_indices, data.test_indices, "train", "test")
    _assert_disjoint_indices(data.valid_indices, data.test_indices, "valid", "test")
    all_indices = np.concatenate(
        [data.train_indices, data.valid_indices, data.test_indices]
    )
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("Each source row must belong to exactly one split.")
    if all_targets is not None:
        expected_indices = np.arange(len(all_targets), dtype=np.int64)
        if not np.array_equal(np.sort(all_indices), expected_indices):
            raise ValueError("The splits do not cover every source row exactly once.")

    for split_name in ("train", "valid", "test"):
        features = np.asarray(getattr(data, f"X_{split_name}"))
        numerical = np.asarray(getattr(data, f"X_num_{split_name}"))
        categorical = np.asarray(getattr(data, f"X_cat_{split_name}"))
        targets = np.asarray(getattr(data, f"y_{split_name}"))
        scaled_targets = np.asarray(getattr(data, f"y_{split_name}_scaled"))
        if features.shape != (len(targets), len(data.feature_columns)):
            raise ValueError(f"X_{split_name} has an inconsistent shape.")
        if numerical.shape[1] != len(data.numerical_columns):
            raise ValueError(f"X_num_{split_name} has an inconsistent width.")
        if categorical.shape[1] != len(data.categorical_columns):
            raise ValueError(f"X_cat_{split_name} has an inconsistent width.")
        for name, array in (
            (f"X_{split_name}", features),
            (f"X_num_{split_name}", numerical),
            (f"y_{split_name}", targets),
            (f"y_{split_name}_scaled", scaled_targets),
        ):
            if not np.isfinite(array).all():
                raise ValueError(f"{name} contains NaN or infinite values.")
        if targets.ndim != 1 or scaled_targets.shape != targets.shape:
            raise ValueError(f"Targets for {split_name} must be one-dimensional.")
        expected_scaled = data.transform_target(targets)
        if not np.allclose(scaled_targets, expected_scaled, atol=1e-6, rtol=1e-6):
            raise ValueError(f"Target scaling is inconsistent for {split_name}.")
        for col_idx, cardinality in enumerate(data.categorical_cardinalities):
            values = categorical[:, col_idx]
            if values.size and (values.min() < 0 or values.max() >= cardinality):
                raise ValueError(
                    f"{split_name} categorical column {col_idx} exceeds cardinality."
                )

    state = data.preprocessing_state
    if state.fitted_on != "train" or state.feature_state.fitted_on != "train":
        raise ValueError("All preprocessing state must be fitted on train.")
    if state.fit_row_count != len(data.y_train):
        raise ValueError("Target preprocessing fit row count does not match train.")
    if state.feature_state.fit_row_count != len(data.y_train):
        raise ValueError("Feature preprocessing fit row count does not match train.")
    if not np.isclose(data.y_train_scaled.mean(), 0.0, atol=1e-6):
        raise ValueError("The scaled training target is not centered at zero.")
    if not np.isclose(data.y_train_scaled.std(ddof=0), 1.0, atol=1e-6):
        raise ValueError("The scaled training target does not have unit variance.")

    target_distribution = {
        split: _regression_target_summary(getattr(data, f"y_{split}"))
        for split in ("train", "valid", "test")
    }
    if all_targets is not None:
        target_distribution["full"] = _regression_target_summary(all_targets)
    reference_std = max(target_distribution["train"]["std"], np.finfo(float).eps)
    mean_drift_std = {
        split: abs(
            target_distribution[split]["mean"]
            - target_distribution["train"]["mean"]
        )
        / reference_std
        for split in ("valid", "test")
    }
    std_ratio = {
        split: target_distribution[split]["std"] / reference_std
        for split in ("valid", "test")
    }
    distribution_reasonable = all(value <= 0.20 for value in mean_drift_std.values())
    distribution_reasonable = distribution_reasonable and all(
        0.75 <= value <= 1.25 for value in std_ratio.values()
    )

    return {
        "split_sizes": {
            "train": int(len(data.y_train)),
            "valid": int(len(data.y_valid)),
            "test": int(len(data.y_test)),
        },
        "no_index_overlap": True,
        "all_rows_assigned_once": True,
        "no_nan_after_preprocessing": True,
        "targets_are_finite": True,
        "target_scaling_is_consistent": True,
        "cardinalities_are_valid": True,
        "preprocessing_fitted_only_on_train": True,
        "test_split_used_for_model_selection": False,
        "split_fingerprint": data.split_fingerprint(),
        "target_distribution": target_distribution,
        "target_mean_drift_in_train_std": mean_drift_std,
        "target_std_ratio_to_train": std_ratio,
        "target_distribution_reasonable": distribution_reasonable,
    }


def load_classification_dataframe(config: DatasetConfig) -> pd.DataFrame:
    """Load and concatenate the configured source CSV files."""

    frames: list[pd.DataFrame] = []
    for file_name in config.source_files:
        path = config.data_dir / file_name
        if not path.exists():
            raise FileNotFoundError(f"Missing dataset file: {path}")
        frame = pd.read_csv(path)
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    return df


def audit_dataframe(df: pd.DataFrame, config: DatasetConfig) -> dict[str, Any]:
    """Return a compact structural and quality audit for notebook display."""

    missing = df.isna().sum().sort_values(ascending=False)
    class_counts = df[config.target_name].value_counts(dropna=False)
    return {
        "shape": tuple(df.shape),
        "columns": list(df.columns),
        "dtypes": {column: str(dtype) for column, dtype in df.dtypes.items()},
        "missing_values": missing[missing > 0].to_dict(),
        "target_distribution": class_counts.to_dict(),
        "target_distribution_pct": (
            df[config.target_name].value_counts(normalize=True, dropna=False)
            .mul(100)
            .round(3)
            .to_dict()
        ),
        "duplicate_rows": int(df.duplicated().sum()),
        "identifier_uniqueness": {
            column: bool(df[column].is_unique)
            for column in config.identifier_columns
            if column in df.columns
        },
        "configured_leakage_columns": [
            column for column in config.leakage_columns if column in df.columns
        ],
    }


def sample_classification_dataframe(
    df: pd.DataFrame,
    target_name: str,
    sample_size: int | None,
    random_state: int,
) -> pd.DataFrame:
    """Optionally draw a stratified development sample before splitting."""

    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True)
    if sample_size <= 0:
        raise ValueError("sample_size must be positive or None.")

    _, sampled = train_test_split(
        df,
        test_size=sample_size,
        random_state=random_state,
        stratify=df[target_name],
    )
    return sampled.reset_index(drop=True)


def prepare_classification_data(
    df: pd.DataFrame,
    dataset_config: DatasetConfig,
    split_config: SplitConfig,
) -> PreparedClassificationData:
    """Split and preprocess data while fitting all transformers on train only."""

    split_config.validate()
    _validate_dataset_columns(df, dataset_config)
    df = df.reset_index(drop=True)

    y, class_names = _encode_target(
        df[dataset_config.target_name],
        dataset_config.positive_class,
        dataset_config.negative_class,
    )
    train_indices, valid_indices, test_indices = _make_split_indices(
        df=df,
        y=y,
        config=split_config,
    )
    state = _fit_preprocessing(
        df=df,
        train_indices=train_indices,
        numerical_columns=dataset_config.numerical_columns,
        categorical_columns=dataset_config.categorical_columns,
    )

    X_cat_train = _transform_categorical(
        df.iloc[train_indices], dataset_config.categorical_columns, state
    )
    X_cat_valid = _transform_categorical(
        df.iloc[valid_indices], dataset_config.categorical_columns, state
    )
    X_cat_test = _transform_categorical(
        df.iloc[test_indices], dataset_config.categorical_columns, state
    )
    X_num_train = _transform_numerical(
        df.iloc[train_indices], dataset_config.numerical_columns, state
    )
    X_num_valid = _transform_numerical(
        df.iloc[valid_indices], dataset_config.numerical_columns, state
    )
    X_num_test = _transform_numerical(
        df.iloc[test_indices], dataset_config.numerical_columns, state
    )

    X_train = _combine_features(X_cat_train, X_num_train)
    X_valid = _combine_features(X_cat_valid, X_num_valid)
    X_test = _combine_features(X_cat_test, X_num_test)

    n_cat = len(dataset_config.categorical_columns)
    categorical_cardinalities = tuple(
        len(state.categorical_mappings[column]) + 1
        for column in dataset_config.categorical_columns
    )
    feature_columns = (
        dataset_config.categorical_columns + dataset_config.numerical_columns
    )

    prepared = PreparedClassificationData(
        X_train=X_train,
        X_valid=X_valid,
        X_test=X_test,
        y_train=y[train_indices].astype(np.int64),
        y_valid=y[valid_indices].astype(np.int64),
        y_test=y[test_indices].astype(np.int64),
        numerical_columns=dataset_config.numerical_columns,
        categorical_columns=dataset_config.categorical_columns,
        categorical_indices=tuple(range(n_cat)),
        categorical_cardinalities=categorical_cardinalities,
        class_names=class_names,
        target_name=dataset_config.target_name,
        dataset_name=dataset_config.name,
        X_num_train=X_num_train,
        X_num_valid=X_num_valid,
        X_num_test=X_num_test,
        X_cat_train=X_cat_train,
        X_cat_valid=X_cat_valid,
        X_cat_test=X_cat_test,
        numerical_indices=tuple(
            range(n_cat, n_cat + len(dataset_config.numerical_columns))
        ),
        feature_columns=feature_columns,
        train_indices=train_indices,
        valid_indices=valid_indices,
        test_indices=test_indices,
        positive_class=dataset_config.positive_class,
        preprocessing_state=state,
        split_report={},
    )
    prepared.split_report = validate_prepared_data(prepared, y)
    return prepared


def validate_prepared_data(
    data: PreparedClassificationData,
    all_encoded_targets: np.ndarray | None = None,
    class_tolerance: float = 0.02,
) -> dict[str, Any]:
    """Run lightweight assertions that guard against common leakage mistakes."""

    _assert_disjoint_indices(data.train_indices, data.valid_indices, "train", "valid")
    _assert_disjoint_indices(data.train_indices, data.test_indices, "train", "test")
    _assert_disjoint_indices(data.valid_indices, data.test_indices, "valid", "test")

    all_indices = np.concatenate(
        [data.train_indices, data.valid_indices, data.test_indices]
    )
    if len(np.unique(all_indices)) != len(all_indices):
        raise ValueError("Each source row must belong to exactly one split.")
    if all_encoded_targets is not None:
        expected_indices = np.arange(len(all_encoded_targets), dtype=np.int64)
        if not np.array_equal(np.sort(all_indices), expected_indices):
            raise ValueError("The splits do not cover every source row exactly once.")

    arrays = {
        "X_train": data.X_train,
        "X_valid": data.X_valid,
        "X_test": data.X_test,
        "X_num_train": data.X_num_train,
        "X_num_valid": data.X_num_valid,
        "X_num_test": data.X_num_test,
    }
    for name, array in arrays.items():
        if not np.isfinite(array).all():
            raise ValueError(f"{name} contains NaN or infinite values.")

    expected_feature_width = len(data.categorical_columns) + len(
        data.numerical_columns
    )
    for split_name in ("train", "valid", "test"):
        features = getattr(data, f"X_{split_name}")
        labels = getattr(data, f"y_{split_name}")
        if features.shape != (len(labels), expected_feature_width):
            raise ValueError(
                f"X_{split_name} has shape {features.shape}; expected "
                f"({len(labels)}, {expected_feature_width})."
            )

    for name, labels in {
        "y_train": data.y_train,
        "y_valid": data.y_valid,
        "y_test": data.y_test,
    }.items():
        if labels.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional.")
        if not set(np.unique(labels)).issubset(set(range(len(data.class_names)))):
            raise ValueError(f"{name} contains labels outside class_names.")

    for split_name, encoded in {
        "train": data.X_cat_train,
        "valid": data.X_cat_valid,
        "test": data.X_cat_test,
    }.items():
        if encoded.shape[1] != len(data.categorical_cardinalities):
            raise ValueError(f"{split_name} categorical width is inconsistent.")
        for col_idx, cardinality in enumerate(data.categorical_cardinalities):
            values = encoded[:, col_idx]
            if values.size and (values.min() < 0 or values.max() >= cardinality):
                raise ValueError(
                    f"{split_name} categorical column {col_idx} exceeds cardinality."
                )

    if data.preprocessing_state.fitted_on != "train":
        raise ValueError("Preprocessing state must be fitted on the train split.")
    if data.preprocessing_state.fit_row_count != len(data.y_train):
        raise ValueError("Preprocessing fit row count does not match train size.")

    report = {
        "split_sizes": {
            "train": int(len(data.y_train)),
            "valid": int(len(data.y_valid)),
            "test": int(len(data.y_test)),
        },
        "no_index_overlap": True,
        "all_rows_assigned_once": True,
        "no_nan_after_preprocessing": True,
        "labels_are_valid": True,
        "cardinalities_are_valid": True,
        "preprocessing_fitted_only_on_train": True,
        "test_split_used_for_model_selection": False,
        "split_fingerprint": data.split_fingerprint(),
        "class_distribution": {
            "train": _class_distribution(data.y_train, len(data.class_names)),
            "valid": _class_distribution(data.y_valid, len(data.class_names)),
            "test": _class_distribution(data.y_test, len(data.class_names)),
        },
        "unknown_category_counts": {
            "valid": _unknown_counts(data.X_cat_valid),
            "test": _unknown_counts(data.X_cat_test),
        },
    }

    if all_encoded_targets is not None:
        full_dist = _class_distribution(all_encoded_targets, len(data.class_names))
        report["class_distribution"]["full"] = full_dist
        for split_name in ("train", "valid", "test"):
            deltas = [
                abs(report["class_distribution"][split_name][label] - full_dist[label])
                for label in full_dist
            ]
            if max(deltas, default=0.0) > class_tolerance:
                raise ValueError(
                    f"Class proportions in {split_name} deviate by more than "
                    f"{class_tolerance:.3f}."
                )
        report["class_proportions_preserved"] = True

    return report


def _validate_regression_dataset_columns(
    df: pd.DataFrame,
    config: RegressionDatasetConfig,
) -> None:
    required = (
        set(config.numerical_columns)
        | set(config.categorical_columns)
        | {config.target_name}
        | set(config.excluded_columns)
    )
    missing = sorted(column for column in required if column not in df.columns)
    if missing:
        raise ValueError(f"Missing configured columns: {missing}")

    feature_columns = set(config.numerical_columns) | set(config.categorical_columns)
    overlap = set(config.numerical_columns).intersection(config.categorical_columns)
    if overlap:
        raise ValueError(f"Columns cannot be both numerical and categorical: {overlap}")
    forbidden = feature_columns.intersection(
        set(config.excluded_columns) | {config.target_name}
    )
    if forbidden:
        raise ValueError(
            "The target or excluded columns are still configured as features: "
            f"{forbidden}"
        )
    leakage = feature_columns.intersection(config.leakage_columns)
    if leakage:
        raise ValueError(f"Configured leakage columns are still features: {leakage}")

    for column in config.numerical_columns + (config.target_name,):
        if not is_numeric_dtype(df[column]):
            raise TypeError(
                f"Expected numerical column {column!r}; got {df[column].dtype}."
            )
    for column in config.categorical_columns:
        if not (is_string_dtype(df[column]) or df[column].dtype == object):
            raise TypeError(
                f"Expected categorical/string column {column!r}; "
                f"got {df[column].dtype}."
            )


def _make_regression_split_indices(
    n_rows: int,
    config: RegressionSplitConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if config.strategy != "random":
        raise NotImplementedError(
            "This experiment executes the random strategy only. Grouped, spatial, "
            "or temporal strategies require dataset-specific split definitions."
        )
    indices = np.arange(n_rows, dtype=np.int64)
    train_count = int(round(n_rows * config.train_size))
    valid_count = int(round(n_rows * config.valid_size))
    test_count = n_rows - train_count - valid_count
    if min(train_count, valid_count, test_count) <= 0:
        raise ValueError("The requested proportions leave an empty split.")
    train_indices, temp_indices = train_test_split(
        indices,
        train_size=train_count,
        random_state=config.random_state,
        shuffle=True,
    )
    valid_indices, test_indices = train_test_split(
        temp_indices,
        train_size=valid_count,
        random_state=config.random_state,
        shuffle=True,
    )
    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(valid_indices, dtype=np.int64),
        np.asarray(test_indices, dtype=np.int64),
    )


def _regression_target_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std(ddof=0)),
        "min": float(array.min()),
        "median": float(np.median(array)),
        "max": float(array.max()),
    }


def _validate_dataset_columns(df: pd.DataFrame, config: DatasetConfig) -> None:
    required = (
        set(config.numerical_columns)
        | set(config.categorical_columns)
        | {config.target_name}
        | set(config.excluded_columns)
    )
    missing = sorted(column for column in required if column not in df.columns)
    if missing:
        raise ValueError(f"Missing configured columns: {missing}")

    overlap = set(config.numerical_columns).intersection(config.categorical_columns)
    if overlap:
        raise ValueError(f"Columns cannot be both numerical and categorical: {overlap}")

    feature_columns = set(config.numerical_columns) | set(config.categorical_columns)
    forbidden = feature_columns.intersection(config.excluded_columns)
    if forbidden:
        raise ValueError(
            f"Excluded columns are still configured as features: {forbidden}"
        )

    for column in config.numerical_columns:
        if not is_numeric_dtype(df[column]):
            raise TypeError(
                f"Expected numerical column {column!r}; got {df[column].dtype}."
            )
    for column in config.categorical_columns:
        if not (is_string_dtype(df[column]) or df[column].dtype == object):
            raise TypeError(
                f"Expected categorical/string column {column!r}; "
                f"got {df[column].dtype}."
            )


def _encode_target(
    target: pd.Series,
    positive_class: str | None,
    negative_class: str | None,
) -> tuple[np.ndarray, tuple[str, ...]]:
    values = target.astype("string")
    observed = sorted(values.dropna().unique().astype(str).tolist())
    if len(observed) < 2:
        raise ValueError("Classification requires at least two classes.")

    if len(observed) == 2 and positive_class is not None:
        if positive_class not in observed:
            raise ValueError(f"Positive class {positive_class!r} was not found.")
        if negative_class is None:
            negative_candidates = [
                label for label in observed if label != positive_class
            ]
            negative_class = negative_candidates[0]
        if negative_class not in observed:
            raise ValueError(f"Negative class {negative_class!r} was not found.")
        class_names = (negative_class, positive_class)
    else:
        class_names = tuple(observed)

    mapping = {label: idx for idx, label in enumerate(class_names)}
    encoded = values.map(mapping)
    if encoded.isna().any():
        unknown = sorted(values[encoded.isna()].dropna().unique().astype(str).tolist())
        raise ValueError(f"Target contains classes outside class_names: {unknown}")
    return encoded.to_numpy(dtype=np.int64), class_names


def _make_split_indices(
    df: pd.DataFrame,
    y: np.ndarray,
    config: SplitConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if config.strategy != "stratified_random":
        raise NotImplementedError(
            "The pipeline exposes group and temporal split configuration for "
            "future TLM-UAV work, but this notebook only executes stratified_random."
        )

    indices = np.arange(len(df))
    train_indices, temp_indices, y_train, y_temp = train_test_split(
        indices,
        y,
        train_size=config.train_size,
        random_state=config.random_state,
        stratify=y,
    )
    relative_valid_size = config.valid_size / (config.valid_size + config.test_size)
    valid_indices, test_indices = train_test_split(
        temp_indices,
        train_size=relative_valid_size,
        random_state=config.random_state,
        stratify=y_temp,
    )

    return (
        train_indices.astype(np.int64),
        valid_indices.astype(np.int64),
        test_indices.astype(np.int64),
    )


def _fit_preprocessing(
    df: pd.DataFrame,
    train_indices: np.ndarray,
    numerical_columns: tuple[str, ...],
    categorical_columns: tuple[str, ...],
) -> PreprocessingState:
    train_df = df.iloc[train_indices]

    medians: dict[str, float] = {}
    means: dict[str, float] = {}
    stds: dict[str, float] = {}
    for column in numerical_columns:
        values = pd.to_numeric(train_df[column], errors="coerce")
        median = float(values.median()) if not values.dropna().empty else 0.0
        imputed = values.fillna(median).astype(float)
        mean = float(imputed.mean())
        std = float(imputed.std(ddof=0))
        medians[column] = median
        means[column] = mean
        stds[column] = std if std > 0 else 1.0

    mappings: dict[str, dict[str, int]] = {}
    for column in categorical_columns:
        values = (
            train_df[column].astype("string").fillna(MISSING_CATEGORY).astype(str)
        )
        categories = sorted(set(values.tolist()) | {MISSING_CATEGORY})
        mappings[column] = {
            category: idx + 1 for idx, category in enumerate(categories)
        }

    return PreprocessingState(
        numerical_medians=medians,
        numerical_means=means,
        numerical_stds=stds,
        categorical_mappings=mappings,
        fitted_on="train",
        fit_row_count=int(len(train_indices)),
    )


def _transform_numerical(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    state: PreprocessingState,
) -> np.ndarray:
    transformed: list[np.ndarray] = []
    for column in columns:
        values = pd.to_numeric(df[column], errors="coerce")
        values = values.fillna(state.numerical_medians[column]).astype(float)
        scaled = (values - state.numerical_means[column]) / state.numerical_stds[column]
        transformed.append(scaled.to_numpy(dtype=np.float32))
    if not transformed:
        return np.empty((len(df), 0), dtype=np.float32)
    return np.column_stack(transformed).astype(np.float32)


def _transform_categorical(
    df: pd.DataFrame,
    columns: tuple[str, ...],
    state: PreprocessingState,
) -> np.ndarray:
    transformed: list[np.ndarray] = []
    for column in columns:
        mapping = state.categorical_mappings[column]
        values = df[column].astype("string").fillna(MISSING_CATEGORY).astype(str)
        encoded = values.map(mapping).fillna(0).to_numpy(dtype=np.int64)
        transformed.append(encoded)
    if not transformed:
        return np.empty((len(df), 0), dtype=np.int64)
    return np.column_stack(transformed).astype(np.int64)


def _combine_features(X_cat: np.ndarray, X_num: np.ndarray) -> np.ndarray:
    if X_cat.size == 0:
        return X_num.astype(np.float32)
    if X_num.size == 0:
        return X_cat.astype(np.float32)
    return np.concatenate([X_cat.astype(np.float32), X_num.astype(np.float32)], axis=1)


def _assert_disjoint_indices(
    left: np.ndarray,
    right: np.ndarray,
    left_name: str,
    right_name: str,
) -> None:
    overlap = set(left.tolist()).intersection(right.tolist())
    if overlap:
        raise ValueError(
            f"{left_name} and {right_name} splits share {len(overlap)} indices."
        )


def _class_distribution(labels: np.ndarray, n_classes: int) -> dict[str, float]:
    counts = np.bincount(labels.astype(np.int64), minlength=n_classes).astype(float)
    total = counts.sum()
    if total == 0:
        return {str(idx): 0.0 for idx in range(n_classes)}
    return {str(idx): float(count / total) for idx, count in enumerate(counts)}


def _unknown_counts(encoded_categories: np.ndarray) -> dict[str, int]:
    if encoded_categories.size == 0:
        return {}
    return {
        str(column_idx): int((encoded_categories[:, column_idx] == 0).sum())
        for column_idx in range(encoded_categories.shape[1])
    }
