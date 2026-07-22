"""Leakage-aware flight reconstruction and multisensor alignment for TLM:UAV."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .audit import (
    CLASS_NAMES,
    FUSION_FILE,
    GPS_COMBINED_FILE,
    IMU_COMBINED_FILE,
    RATE_COMBINED_FILE,
    VIBE_COMBINED_FILE,
    VIBE_FILE,
    dataset_manifest,
    resolve_dataset_dir,
)


SOURCE_FILES = {
    "GPS": GPS_COMBINED_FILE,
    "RATE": RATE_COMBINED_FILE,
    "VIBE": VIBE_COMBINED_FILE,
    "IMU": IMU_COMBINED_FILE,
}

RECONSTRUCTION_FILES = (
    GPS_COMBINED_FILE,
    RATE_COMBINED_FILE,
    VIBE_COMBINED_FILE,
    VIBE_FILE,
    IMU_COMBINED_FILE,
    FUSION_FILE,
)

IMU_SIGNAL_COLUMNS = (
    "abGyrX",
    "abGyrY",
    "abGyrZ",
    "abAccX",
    "abAccY",
    "abAccZ",
)

GPS_FEATURE_COLUMNS = (
    "I",
    "Status",
    "NSats",
    "HDop",
    "Lat",
    "Lng",
    "Alt",
    "Spd",
    "GCrs",
    "VZ",
    "Yaw",
    "U",
)

RATE_FEATURE_COLUMNS = (
    "RDes",
    "R",
    "Rout",
    "PDes",
    "P",
    "POut",
    "YDes",
    "Y",
    "YOut",
    "ADes",
    "A",
    "AOut",
)

VIBE_FEATURE_COLUMNS = (
    "IMU",
    "VibeX",
    "VibeY",
    "VibeZ",
    "Clip",
)

IMU_FEATURE_COLUMNS = (
    "abI",
    *IMU_SIGNAL_COLUMNS,
    "abEG",
    "abEA",
    "abT",
    "abGH",
    "abAH",
    "abGHz",
    "abAHz",
)

FEATURE_COLUMNS_BY_SOURCE = {
    "GPS": GPS_FEATURE_COLUMNS,
    "RATE": RATE_FEATURE_COLUMNS,
    "VIBE": VIBE_FEATURE_COLUMNS,
    "IMU": IMU_FEATURE_COLUMNS,
}

SENSOR_CORE_NUMERICAL_COLUMNS = (
    "gps__Spd",
    "gps__VZ",
    "rate__R",
    "rate__P",
    "rate__Y",
    "rate__A",
    "vibe__VibeX",
    "vibe__VibeY",
    "vibe__VibeZ",
    "imu__abGyrX",
    "imu__abGyrY",
    "imu__abGyrZ",
    "imu__abAccX",
    "imu__abAccY",
    "imu__abAccZ",
)

FULL_DIAGNOSTIC_NUMERICAL_COLUMNS = (
    "gps__NSats",
    "gps__Lat",
    "gps__Lng",
    "gps__Alt",
    "gps__Spd",
    "gps__GCrs",
    "gps__VZ",
    "rate__RDes",
    "rate__R",
    "rate__Rout",
    "rate__PDes",
    "rate__P",
    "rate__POut",
    "rate__YDes",
    "rate__Y",
    "rate__YOut",
    "rate__ADes",
    "rate__A",
    "rate__AOut",
    "vibe__VibeX",
    "vibe__VibeY",
    "vibe__VibeZ",
    "imu__abGyrX",
    "imu__abGyrY",
    "imu__abGyrZ",
    "imu__abAccX",
    "imu__abAccY",
    "imu__abAccZ",
    "imu__abT",
)

FULL_DIAGNOSTIC_CATEGORICAL_COLUMNS = (
    "gps__Status",
    "imu__abGHz",
    "imu__abAHz",
)


@dataclass(frozen=True)
class ReconstructionConfig:
    """Fixed rules used to infer flights and align sensor observations."""

    dataset_dir: Path
    gps_gap_threshold_ms: int = 600_000
    gps_alignment_tolerance_us: int = 220_000
    imu_alignment_tolerance_us: int = 45_000
    imu_assignment_padding_us: int = 100_000
    max_rate_line_delta: int = 50
    imu_match_decimals: int = 6
    expected_flights: int = 2
    target_source: str = "VIBE"
    alignment_direction: str = "backward"

    def __post_init__(self) -> None:
        dataset_dir = Path(self.dataset_dir).resolve()
        object.__setattr__(self, "dataset_dir", dataset_dir)
        if not dataset_dir.is_dir():
            raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
        positive_fields = {
            "gps_gap_threshold_ms": self.gps_gap_threshold_ms,
            "gps_alignment_tolerance_us": self.gps_alignment_tolerance_us,
            "imu_alignment_tolerance_us": self.imu_alignment_tolerance_us,
            "imu_assignment_padding_us": self.imu_assignment_padding_us,
            "max_rate_line_delta": self.max_rate_line_delta,
            "expected_flights": self.expected_flights,
        }
        invalid = [name for name, value in positive_fields.items() if value <= 0]
        if invalid:
            raise ValueError(f"Configuration values must be positive: {invalid}")
        if self.imu_match_decimals < 0:
            raise ValueError("imu_match_decimals cannot be negative")
        if self.target_source != "VIBE":
            raise ValueError("This protocol requires VIBE as the target source")
        if self.alignment_direction != "backward":
            raise ValueError("Only causal backward alignment is supported")


@dataclass
class ReconstructedFlights:
    """Sensor tables with an inferred flight identifier and traceability reports."""

    config: ReconstructionConfig
    manifest: dict[str, str]
    frames: dict[str, pd.DataFrame]
    assignment_summary: pd.DataFrame
    imu_anchor_summary: dict[str, Any]
    fingerprint: str


@dataclass
class AlignedMultisensorData:
    """Canonical VIBE-clock table with labels isolated from source features."""

    config: ReconstructionConfig
    frame: pd.DataFrame
    feature_columns: tuple[str, ...]
    label_audit: pd.DataFrame
    alignment_audit: pd.DataFrame
    alignment_summary: pd.DataFrame
    label_agreement: pd.DataFrame
    dropped_rows: pd.DataFrame
    checks: pd.Series
    source_manifest: dict[str, str]
    reconstruction_fingerprint: str
    alignment_fingerprint: str
    class_names: tuple[str, ...] = CLASS_NAMES
    target_name: str = "label"
    target_source: str = "VIBE"

    def metadata_columns(self) -> tuple[str, ...]:
        """Return columns that must never be passed to a model."""

        return ("row_id", "flight_id", "time_us", "episode_id", self.target_name)


@dataclass(frozen=True)
class FeatureView:
    """A semantically frozen set of model inputs."""

    name: str
    numerical_columns: tuple[str, ...]
    categorical_columns: tuple[str, ...]
    description: str

    @property
    def columns(self) -> tuple[str, ...]:
        """Return numerical and categorical columns in deterministic order."""

        return self.numerical_columns + self.categorical_columns


@dataclass(frozen=True)
class FlightSplitConfig:
    """Rules for external flight holdout and internal purged validation."""

    validation_fraction: float = 0.20
    purge_seconds: float = 5.0
    transition_guard_seconds: float = 0.25
    expected_episodes_per_flight: int = 8

    def __post_init__(self) -> None:
        if not 0.0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must lie in (0, 0.5)")
        if self.purge_seconds <= 0:
            raise ValueError("purge_seconds must be positive")
        if self.transition_guard_seconds < 0:
            raise ValueError("transition_guard_seconds cannot be negative")
        if self.expected_episodes_per_flight <= 0:
            raise ValueError("expected_episodes_per_flight must be positive")


@dataclass
class FlightFold:
    """One external leave-one-flight-out direction and its inner roles."""

    name: str
    development_flight: int
    test_flight: int
    inner_train_row_ids: np.ndarray
    valid_row_ids: np.ndarray
    purged_row_ids: np.ndarray
    development_row_ids: np.ndarray
    test_row_ids: np.ndarray
    stable_test_row_ids: np.ndarray
    transition_test_row_ids: np.ndarray
    report: dict[str, Any]
    fingerprint: str


@dataclass
class PreprocessingState:
    """Train-fitted imputation, scaling, and categorical encoding state."""

    view_name: str
    fitted_on: str
    fit_row_ids: tuple[int, ...]
    numerical_columns: tuple[str, ...]
    dropped_numerical_columns: tuple[str, ...]
    numerical_medians: dict[str, float]
    numerical_means: dict[str, float]
    numerical_stds: dict[str, float]
    categorical_columns: tuple[str, ...]
    categorical_modes: dict[str, Any]
    categorical_mappings: dict[str, dict[Any, int]]
    categorical_cardinalities: tuple[int, ...]
    fit_row_count: int
    fingerprint: str


@dataclass
class TransformedFeatures:
    """Numerical and categorical arrays produced by one fitted state."""

    row_ids: np.ndarray
    X_num: np.ndarray
    X_cat: np.ndarray
    unknown_category_counts: dict[str, int]


@dataclass
class PreparedFlightFold:
    """Tuning and final-refit arrays for one feature view and outer fold."""

    fold: FlightFold
    feature_view: FeatureView
    tuning_state: PreprocessingState
    final_state: PreprocessingState
    train: TransformedFeatures
    valid: TransformedFeatures
    development: TransformedFeatures
    test: TransformedFeatures
    y_train: np.ndarray
    y_valid: np.ndarray
    y_development: np.ndarray
    y_test: np.ndarray
    tuning_class_weights: np.ndarray
    final_class_weights: np.ndarray
    checks: pd.Series
    fingerprint: str
    class_names: tuple[str, ...] = CLASS_NAMES


def make_reconstruction_config(
    project_root: Path | str | None = None,
    **overrides: Any,
) -> ReconstructionConfig:
    """Build a validated reconstruction configuration from the project root."""

    return ReconstructionConfig(
        dataset_dir=resolve_dataset_dir(project_root),
        **overrides,
    )


def available_feature_views() -> tuple[FeatureView, ...]:
    """Return the two predeclared feature protocols."""

    return (
        FeatureView(
            name="sensor_core",
            numerical_columns=SENSOR_CORE_NUMERICAL_COLUMNS,
            categorical_columns=(),
            description=(
                "Instantaneous physical measurements without position, control "
                "references, actuator outputs, status flags, or accumulated state."
            ),
        ),
        FeatureView(
            name="full_diagnostic",
            numerical_columns=FULL_DIAGNOSTIC_NUMERICAL_COLUMNS,
            categorical_columns=FULL_DIAGNOSTIC_CATEGORICAL_COLUMNS,
            description=(
                "Nonconstant operational telemetry including navigation context, "
                "control references, outputs, and discrete diagnostic states."
            ),
        ),
    )


def get_feature_view(name: str) -> FeatureView:
    """Resolve one feature view by its stable name."""

    views = {view.name: view for view in available_feature_views()}
    if name not in views:
        raise ValueError(f"Unknown feature view {name!r}; choose from {sorted(views)}")
    return views[name]


def load_reconstruction_frames(dataset_dir: Path) -> dict[str, pd.DataFrame]:
    """Load only source tables required by reconstruction and alignment."""

    frames: dict[str, pd.DataFrame] = {}
    for relative_path in RECONSTRUCTION_FILES:
        path = Path(dataset_dir) / relative_path
        if not path.is_file():
            raise FileNotFoundError(f"Required reconstruction file missing: {path}")
        frames[relative_path] = pd.read_csv(path)
    return frames


def reconstruct_flights(
    config: ReconstructionConfig,
    frames: dict[str, pd.DataFrame] | None = None,
) -> ReconstructedFlights:
    """Infer two flight sessions without consulting any target values."""

    source_frames = (
        load_reconstruction_frames(config.dataset_dir)
        if frames is None
        else {name: frame.copy() for name, frame in frames.items()}
    )
    _validate_required_frames(source_frames)

    gps, gps_report = _assign_gps_flights(
        source_frames[GPS_COMBINED_FILE],
        config,
    )
    vibe, vibe_report = _assign_vibe_flights(
        source_frames[VIBE_COMBINED_FILE],
        source_frames[VIBE_FILE],
        config,
    )
    rate, rate_report = _assign_rate_flights(
        source_frames[RATE_COMBINED_FILE],
        vibe,
        config,
    )
    imu, imu_report, imu_anchor_summary = _assign_imu_flights(
        source_frames[IMU_COMBINED_FILE],
        source_frames[FUSION_FILE],
        vibe,
        config,
    )

    reconstructed = {
        "GPS": gps,
        "RATE": rate,
        "VIBE": vibe,
        "IMU": imu,
    }
    _validate_reconstructed_frames(reconstructed, config)

    manifest = dataset_manifest(config.dataset_dir)
    fingerprint = _reconstruction_fingerprint(reconstructed, manifest, config)
    return ReconstructedFlights(
        config=config,
        manifest=manifest,
        frames=reconstructed,
        assignment_summary=pd.concat(
            [gps_report, vibe_report, rate_report, imu_report],
            ignore_index=True,
        ),
        imu_anchor_summary=imu_anchor_summary,
        fingerprint=fingerprint,
    )


def align_multisensor_data(
    reconstructed: ReconstructedFlights,
) -> AlignedMultisensorData:
    """Causally align GPS, RATE, VIBE, and IMU on the VIBE timeline."""

    config = reconstructed.config
    aligned_parts: list[pd.DataFrame] = []
    dropped_parts: list[pd.DataFrame] = []

    for flight_id in range(config.expected_flights):
        vibe = _canonical_vibe_frame(reconstructed.frames["VIBE"], flight_id)
        rate = _prefixed_source_frame(
            reconstructed.frames["RATE"],
            flight_id,
            source="RATE",
            time_column="TimeUS",
        )
        gps = _prefixed_source_frame(
            reconstructed.frames["GPS"],
            flight_id,
            source="GPS",
            time_column="TimeUS",
        )
        imu = _prefixed_source_frame(
            reconstructed.frames["IMU"],
            flight_id,
            source="IMU",
            time_column="abTimeUS",
        )

        merged = vibe.merge(
            rate,
            left_on=["flight_id", "time_us"],
            right_on=["flight_id", "rate_time_us"],
            how="left",
            validate="one_to_one",
        )
        merged = pd.merge_asof(
            merged.sort_values("time_us", kind="stable"),
            gps.sort_values("gps_time_us", kind="stable"),
            left_on="time_us",
            right_on="gps_time_us",
            by="flight_id",
            direction=config.alignment_direction,
            tolerance=config.gps_alignment_tolerance_us,
        )
        merged = pd.merge_asof(
            merged.sort_values("time_us", kind="stable"),
            imu.sort_values("imu_time_us", kind="stable"),
            left_on="time_us",
            right_on="imu_time_us",
            by="flight_id",
            direction=config.alignment_direction,
            tolerance=config.imu_alignment_tolerance_us,
        )

        required_source_rows = [
            "gps_source_row",
            "rate_source_row",
            "imu_source_row",
        ]
        missing_sources = merged[required_source_rows].isna()
        complete = ~missing_sources.any(axis=1)
        dropped = merged.loc[~complete, [
            "flight_id",
            "time_us",
            "episode_id",
            "label",
        ]].copy()
        dropped["missing_sources"] = missing_sources.loc[~complete].apply(
            lambda row: ",".join(
                column.removesuffix("_source_row").upper()
                for column, missing in row.items()
                if missing
            ),
            axis=1,
        )
        dropped_parts.append(dropped)
        aligned_parts.append(merged.loc[complete].copy())

    merged_all = pd.concat(aligned_parts, ignore_index=True)
    merged_all = merged_all.sort_values(
        ["flight_id", "time_us"],
        kind="stable",
    ).reset_index(drop=True)
    merged_all.insert(0, "row_id", np.arange(len(merged_all), dtype=np.int64))

    alignment_audit = _build_alignment_audit(merged_all)
    label_audit = _build_label_audit(merged_all)
    alignment_summary = _summarize_alignment(alignment_audit)
    label_agreement = _summarize_label_agreement(label_audit)
    dropped_rows = pd.concat(dropped_parts, ignore_index=True)

    feature_columns = tuple(
        f"{source.lower()}__{column}"
        for source, columns in FEATURE_COLUMNS_BY_SOURCE.items()
        for column in columns
    )
    model_frame = merged_all[[
        "row_id",
        "flight_id",
        "time_us",
        "episode_id",
        "label",
        *feature_columns,
    ]].copy()

    alignment_fingerprint = _frame_fingerprint(
        model_frame,
        extra={
            "reconstruction": reconstructed.fingerprint,
            "gps_tolerance": config.gps_alignment_tolerance_us,
            "imu_tolerance": config.imu_alignment_tolerance_us,
            "direction": config.alignment_direction,
            "target_source": config.target_source,
        },
    )
    checks = validate_aligned_data(
        frame=model_frame,
        feature_columns=feature_columns,
        label_audit=label_audit,
        alignment_audit=alignment_audit,
        dropped_rows=dropped_rows,
        reconstructed=reconstructed,
    )
    if not bool(checks.all()):
        failed = checks.index[~checks].tolist()
        raise AssertionError(f"Aligned data failed integrity checks: {failed}")

    return AlignedMultisensorData(
        config=config,
        frame=model_frame,
        feature_columns=feature_columns,
        label_audit=label_audit,
        alignment_audit=alignment_audit,
        alignment_summary=alignment_summary,
        label_agreement=label_agreement,
        dropped_rows=dropped_rows,
        checks=checks,
        source_manifest=reconstructed.manifest,
        reconstruction_fingerprint=reconstructed.fingerprint,
        alignment_fingerprint=alignment_fingerprint,
    )


def reconstruct_and_align(
    project_root: Path | str | None = None,
    **config_overrides: Any,
) -> tuple[ReconstructedFlights, AlignedMultisensorData]:
    """Run the complete immutable reconstruction and causal alignment stage."""

    config = make_reconstruction_config(project_root, **config_overrides)
    reconstructed = reconstruct_flights(config)
    aligned = align_multisensor_data(reconstructed)
    manifest_after = dataset_manifest(config.dataset_dir)
    if reconstructed.manifest != manifest_after:
        raise AssertionError("Source CSV files changed during reconstruction")
    return reconstructed, aligned


def build_leave_one_flight_out_folds(
    aligned: AlignedMultisensorData,
    config: FlightSplitConfig | None = None,
) -> tuple[FlightFold, ...]:
    """Build two external flight folds with purged episode-wise validation."""

    split_config = FlightSplitConfig() if config is None else config
    frame = aligned.frame.sort_values(
        ["flight_id", "time_us"],
        kind="stable",
    )
    flights = sorted(frame["flight_id"].unique().tolist())
    if flights != [0, 1]:
        raise ValueError(f"Expected inferred flights [0, 1], found {flights}")

    folds: list[FlightFold] = []
    for development_flight in flights:
        test_flight = 1 - development_flight
        development = frame.loc[frame["flight_id"].eq(development_flight)]
        test = frame.loc[frame["flight_id"].eq(test_flight)]
        inner_train, valid, purged = _build_purged_inner_roles(
            development,
            split_config,
        )
        stable_test, transition_test = _split_transition_guard(
            test,
            split_config.transition_guard_seconds,
        )
        report = _build_fold_report(
            development=development,
            test=test,
            inner_train_row_ids=inner_train,
            valid_row_ids=valid,
            purged_row_ids=purged,
            stable_test_row_ids=stable_test,
            transition_test_row_ids=transition_test,
            config=split_config,
        )
        fingerprint = _fold_fingerprint(
            alignment_fingerprint=aligned.alignment_fingerprint,
            development_flight=development_flight,
            test_flight=test_flight,
            inner_train=inner_train,
            valid=valid,
            purged=purged,
            stable_test=stable_test,
            config=split_config,
        )
        fold = FlightFold(
            name=f"develop_f{development_flight}_test_f{test_flight}",
            development_flight=development_flight,
            test_flight=test_flight,
            inner_train_row_ids=inner_train,
            valid_row_ids=valid,
            purged_row_ids=purged,
            development_row_ids=development["row_id"].to_numpy(dtype=np.int64),
            test_row_ids=test["row_id"].to_numpy(dtype=np.int64),
            stable_test_row_ids=stable_test,
            transition_test_row_ids=transition_test,
            report=report,
            fingerprint=fingerprint,
        )
        checks = validate_flight_fold(aligned.frame, fold, split_config)
        fold.report["checks"] = checks.to_dict()
        if not bool(checks.all()):
            failed = checks.index[~checks].tolist()
            raise AssertionError(f"Fold {fold.name} failed checks: {failed}")
        folds.append(fold)
    return tuple(folds)


def validate_flight_fold(
    frame: pd.DataFrame,
    fold: FlightFold,
    config: FlightSplitConfig,
) -> pd.Series:
    """Validate external isolation, inner roles, class support, and purge."""

    role_sets = {
        "inner_train": set(fold.inner_train_row_ids.tolist()),
        "valid": set(fold.valid_row_ids.tolist()),
        "purged": set(fold.purged_row_ids.tolist()),
    }
    development = set(fold.development_row_ids.tolist())
    test = set(fold.test_row_ids.tolist())
    inner_union = set().union(*role_sets.values())
    role_pairs = list(role_sets.values())
    roles_disjoint = all(
        role_pairs[left].isdisjoint(role_pairs[right])
        for left in range(len(role_pairs))
        for right in range(left + 1, len(role_pairs))
    )
    train_labels = _targets_for_rows(frame, fold.inner_train_row_ids)
    valid_labels = _targets_for_rows(frame, fold.valid_row_ids)
    test_labels = _targets_for_rows(frame, fold.test_row_ids)
    stable_labels = _targets_for_rows(frame, fold.stable_test_row_ids)
    expected_classes = set(range(len(CLASS_NAMES)))
    minimum_gap = float(fold.report["minimum_train_valid_gap_seconds"])
    return pd.Series(
        {
            "development_and_test_flights_differ": (
                fold.development_flight != fold.test_flight
            ),
            "development_and_test_rows_are_disjoint": development.isdisjoint(test),
            "inner_roles_are_disjoint": roles_disjoint,
            "inner_roles_partition_development": inner_union == development,
            "test_is_not_an_inner_role": test.isdisjoint(inner_union),
            "all_classes_in_inner_train": set(np.unique(train_labels))
            == expected_classes,
            "all_classes_in_validation": set(np.unique(valid_labels))
            == expected_classes,
            "all_classes_in_external_test": set(np.unique(test_labels))
            == expected_classes,
            "all_classes_in_stable_test": set(np.unique(stable_labels))
            == expected_classes,
            "purge_is_at_least_configured_duration": (
                minimum_gap >= config.purge_seconds
            ),
            "stable_and_transition_test_are_disjoint": set(
                fold.stable_test_row_ids.tolist()
            ).isdisjoint(set(fold.transition_test_row_ids.tolist())),
            "test_masks_partition_external_test": set(
                fold.stable_test_row_ids.tolist()
            ).union(set(fold.transition_test_row_ids.tolist()))
            == test,
            "expected_episode_count": (
                fold.report["development_episode_count"]
                == config.expected_episodes_per_flight
            ),
        },
        dtype=bool,
        name="passed",
    )


def fit_preprocessing(
    frame: pd.DataFrame,
    row_ids: np.ndarray,
    feature_view: FeatureView,
    fitted_on: str,
) -> PreprocessingState:
    """Fit imputation, standardization, and encoders on specified rows only."""

    _validate_feature_view(frame, feature_view)
    selected = _select_rows(frame, row_ids)
    if len(selected) == 0:
        raise ValueError("Cannot fit preprocessing on an empty row set")

    numerical = selected.loc[:, feature_view.numerical_columns].apply(
        pd.to_numeric,
        errors="coerce",
    ).replace([np.inf, -np.inf], np.nan)
    medians = numerical.median(axis=0)
    all_missing = medians.index[medians.isna()].tolist()
    if all_missing:
        raise ValueError(f"Numerical columns are entirely missing: {all_missing}")
    imputed = numerical.fillna(medians)
    means = imputed.mean(axis=0)
    stds = imputed.std(axis=0, ddof=0)
    dropped = tuple(stds.index[stds.le(1e-12)].tolist())
    retained = tuple(
        column
        for column in feature_view.numerical_columns
        if column not in dropped
    )

    categorical_modes: dict[str, Any] = {}
    categorical_mappings: dict[str, dict[Any, int]] = {}
    cardinalities: list[int] = []
    for column in feature_view.categorical_columns:
        values = selected[column]
        modes = values.mode(dropna=True)
        if modes.empty:
            raise ValueError(f"Categorical column {column!r} is entirely missing")
        mode = _python_scalar(modes.iloc[0])
        filled = values.where(values.notna(), mode).map(_python_scalar)
        categories = sorted(
            filled.unique().tolist(),
            key=lambda value: (type(value).__name__, repr(value)),
        )
        mapping = {value: index + 1 for index, value in enumerate(categories)}
        categorical_modes[column] = mode
        categorical_mappings[column] = mapping
        cardinalities.append(len(mapping) + 1)

    state_values = {
        "view_name": feature_view.name,
        "fitted_on": fitted_on,
        "fit_row_ids": [int(value) for value in row_ids],
        "numerical_columns": retained,
        "dropped_numerical_columns": dropped,
        "numerical_medians": {column: float(medians[column]) for column in retained},
        "numerical_means": {column: float(means[column]) for column in retained},
        "numerical_stds": {column: float(stds[column]) for column in retained},
        "categorical_columns": feature_view.categorical_columns,
        "categorical_modes": categorical_modes,
        "categorical_mappings": categorical_mappings,
        "categorical_cardinalities": cardinalities,
    }
    fingerprint = sha256(
        json.dumps(
            _preprocessing_json_values(state_values),
            sort_keys=True,
        ).encode("utf-8")
    ).hexdigest()
    return PreprocessingState(
        view_name=feature_view.name,
        fitted_on=fitted_on,
        fit_row_ids=tuple(int(value) for value in row_ids),
        numerical_columns=retained,
        dropped_numerical_columns=dropped,
        numerical_medians=state_values["numerical_medians"],
        numerical_means=state_values["numerical_means"],
        numerical_stds=state_values["numerical_stds"],
        categorical_columns=feature_view.categorical_columns,
        categorical_modes=categorical_modes,
        categorical_mappings=categorical_mappings,
        categorical_cardinalities=tuple(cardinalities),
        fit_row_count=len(row_ids),
        fingerprint=fingerprint,
    )


def transform_features(
    frame: pd.DataFrame,
    row_ids: np.ndarray,
    state: PreprocessingState,
) -> TransformedFeatures:
    """Apply one fitted preprocessing state without any refitting."""

    selected = _select_rows(frame, row_ids)
    if state.numerical_columns:
        numerical = selected.loc[:, state.numerical_columns].apply(
            pd.to_numeric,
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)
        for column in state.numerical_columns:
            numerical[column] = numerical[column].fillna(
                state.numerical_medians[column]
            )
        means = np.asarray(
            [state.numerical_means[column] for column in state.numerical_columns],
            dtype=np.float64,
        )
        stds = np.asarray(
            [state.numerical_stds[column] for column in state.numerical_columns],
            dtype=np.float64,
        )
        X_num = ((numerical.to_numpy(dtype=np.float64) - means) / stds).astype(
            np.float32
        )
    else:
        X_num = np.empty((len(selected), 0), dtype=np.float32)

    categorical_arrays: list[np.ndarray] = []
    unknown_counts: dict[str, int] = {}
    for column in state.categorical_columns:
        mode = state.categorical_modes[column]
        values = selected[column].where(selected[column].notna(), mode)
        values = values.map(_python_scalar)
        encoded = values.map(state.categorical_mappings[column])
        unknown_counts[column] = int(encoded.isna().sum())
        categorical_arrays.append(
            encoded.fillna(0).to_numpy(dtype=np.int64).reshape(-1, 1)
        )
    X_cat = (
        np.concatenate(categorical_arrays, axis=1)
        if categorical_arrays
        else np.empty((len(selected), 0), dtype=np.int64)
    )
    if not np.isfinite(X_num).all():
        raise ValueError("Numerical transformation produced nonfinite values")
    for index, cardinality in enumerate(state.categorical_cardinalities):
        if X_cat[:, index].min(initial=0) < 0:
            raise ValueError("Categorical encoding produced a negative index")
        if X_cat[:, index].max(initial=0) >= cardinality:
            raise ValueError("Categorical encoding exceeds fitted cardinality")
    return TransformedFeatures(
        row_ids=np.asarray(row_ids, dtype=np.int64).copy(),
        X_num=X_num,
        X_cat=X_cat,
        unknown_category_counts=unknown_counts,
    )


def prepare_flight_fold(
    aligned: AlignedMultisensorData,
    fold: FlightFold,
    feature_view: FeatureView,
) -> PreparedFlightFold:
    """Prepare leakage-isolated tuning and external evaluation matrices."""

    frame = aligned.frame
    tuning_state = fit_preprocessing(
        frame,
        fold.inner_train_row_ids,
        feature_view,
        fitted_on="inner_train",
    )
    final_state = fit_preprocessing(
        frame,
        fold.development_row_ids,
        feature_view,
        fitted_on="development_flight",
    )
    train = transform_features(frame, fold.inner_train_row_ids, tuning_state)
    valid = transform_features(frame, fold.valid_row_ids, tuning_state)
    development = transform_features(
        frame,
        fold.development_row_ids,
        final_state,
    )
    test = transform_features(frame, fold.test_row_ids, final_state)
    y_train = _targets_for_rows(frame, train.row_ids)
    y_valid = _targets_for_rows(frame, valid.row_ids)
    y_development = _targets_for_rows(frame, development.row_ids)
    y_test = _targets_for_rows(frame, test.row_ids)
    tuning_weights = _balanced_class_weights(y_train)
    final_weights = _balanced_class_weights(y_development)
    fingerprint = _prepared_fold_fingerprint(
        aligned.alignment_fingerprint,
        fold,
        feature_view,
        tuning_state,
        final_state,
        train,
        valid,
        development,
        test,
    )
    checks = _validate_prepared_fold(
        fold=fold,
        feature_view=feature_view,
        tuning_state=tuning_state,
        final_state=final_state,
        train=train,
        valid=valid,
        development=development,
        test=test,
        y_train=y_train,
        y_valid=y_valid,
        y_development=y_development,
        y_test=y_test,
    )
    if not bool(checks.all()):
        failed = checks.index[~checks].tolist()
        raise AssertionError(
            f"Prepared fold {fold.name}/{feature_view.name} failed: {failed}"
        )
    return PreparedFlightFold(
        fold=fold,
        feature_view=feature_view,
        tuning_state=tuning_state,
        final_state=final_state,
        train=train,
        valid=valid,
        development=development,
        test=test,
        y_train=y_train,
        y_valid=y_valid,
        y_development=y_development,
        y_test=y_test,
        tuning_class_weights=tuning_weights,
        final_class_weights=final_weights,
        checks=checks,
        fingerprint=fingerprint,
    )


def prepare_lofo_data(
    aligned: AlignedMultisensorData,
    split_config: FlightSplitConfig | None = None,
    feature_views: tuple[FeatureView, ...] | None = None,
) -> tuple[tuple[FlightFold, ...], tuple[PreparedFlightFold, ...]]:
    """Prepare both feature views for both leave-one-flight-out directions."""

    folds = build_leave_one_flight_out_folds(aligned, split_config)
    views = available_feature_views() if feature_views is None else feature_views
    prepared = tuple(
        prepare_flight_fold(aligned, fold, view)
        for fold in folds
        for view in views
    )
    return folds, prepared


def verify_external_test_isolation(
    aligned: AlignedMultisensorData,
    folds: tuple[FlightFold, ...],
    feature_views: tuple[FeatureView, ...] | None = None,
) -> bool:
    """Prove that adversarial external features cannot alter fitted development."""

    views = available_feature_views() if feature_views is None else feature_views
    for fold in folds:
        external_mask = aligned.frame["row_id"].isin(fold.test_row_ids)
        for view in views:
            reference = prepare_flight_fold(aligned, fold, view)
            modified_frame = aligned.frame.copy()
            for column in view.numerical_columns:
                modified_frame.loc[external_mask, column] = (
                    modified_frame.loc[external_mask, column].astype(float)
                    + 1_000_000.0
                )
            for column in view.categorical_columns:
                modified_frame.loc[external_mask, column] = -1_000_000
            modified = replace(aligned, frame=modified_frame)
            adversarial = prepare_flight_fold(modified, fold, view)
            if (
                reference.tuning_state.fingerprint
                != adversarial.tuning_state.fingerprint
            ):
                return False
            if reference.final_state.fingerprint != adversarial.final_state.fingerprint:
                return False
            if not np.array_equal(
                reference.tuning_class_weights,
                adversarial.tuning_class_weights,
            ):
                return False
            if not np.array_equal(
                reference.final_class_weights,
                adversarial.final_class_weights,
            ):
                return False
            for original, changed in (
                (reference.train, adversarial.train),
                (reference.valid, adversarial.valid),
                (reference.development, adversarial.development),
            ):
                if not np.array_equal(original.X_num, changed.X_num):
                    return False
                if not np.array_equal(original.X_cat, changed.X_cat):
                    return False
            if view.categorical_columns and not all(
                count == len(fold.test_row_ids)
                for count in adversarial.test.unknown_category_counts.values()
            ):
                return False
    return True


def verify_label_independent_flight_ids(
    reconstructed: ReconstructedFlights,
    seed: int = 42,
) -> bool:
    """Empirically verify that permuting every target leaves flight IDs unchanged."""

    frames = load_reconstruction_frames(reconstructed.config.dataset_dir)
    rng = np.random.default_rng(seed)
    for frame in frames.values():
        for column in ("labels", "lables", "labels.1"):
            if column in frame:
                frame[column] = rng.permutation(frame[column].to_numpy())
    permuted = reconstruct_flights(reconstructed.config, frames=frames)
    for source in SOURCE_FILES:
        original = reconstructed.frames[source].sort_values("_source_row")
        candidate = permuted.frames[source].sort_values("_source_row")
        if not np.array_equal(
            original["flight_id"].to_numpy(),
            candidate["flight_id"].to_numpy(),
        ):
            return False
    return True


def validate_aligned_data(
    frame: pd.DataFrame,
    feature_columns: tuple[str, ...],
    label_audit: pd.DataFrame,
    alignment_audit: pd.DataFrame,
    dropped_rows: pd.DataFrame,
    reconstructed: ReconstructedFlights,
) -> pd.Series:
    """Return executable integrity and conceptual-leakage checks."""

    config = reconstructed.config
    feature_values = frame.loc[:, feature_columns].to_numpy(dtype=float)
    flight_counts = frame["flight_id"].value_counts().sort_index().to_dict()
    source_label_columns = {"gps_label", "rate_label", "imu_label"}
    frame_columns = set(frame.columns)
    all_classes_per_flight = all(
        set(part["label"].unique()) == set(range(len(CLASS_NAMES)))
        for _, part in frame.groupby("flight_id")
    )
    source_rows_remain_in_flight = True
    for source, source_frame in reconstructed.frames.items():
        prefix = source.lower()
        source_to_flight = source_frame.set_index("_source_row")["flight_id"]
        aligned_source_flights = alignment_audit[
            f"{prefix}_source_row"
        ].map(source_to_flight)
        source_rows_remain_in_flight &= bool(
            aligned_source_flights.notna().all()
            and np.array_equal(
                aligned_source_flights.to_numpy(dtype=np.int64),
                alignment_audit["flight_id"].to_numpy(dtype=np.int64),
            )
        )
    return pd.Series(
        {
            "expected_aligned_shape": frame.shape
            == (8634, 5 + len(feature_columns)),
            "expected_rows_by_flight": flight_counts == {0: 4898, 1: 3736},
            "four_uncovered_rows_reported": len(dropped_rows) == 4,
            "row_ids_are_unique": frame["row_id"].is_unique,
            "flight_time_key_is_unique": not frame.duplicated(
                ["flight_id", "time_us"]
            ).any(),
            "target_values_are_valid": set(frame["label"].unique())
            == set(range(len(CLASS_NAMES))),
            "all_classes_exist_in_each_flight": all_classes_per_flight,
            "features_are_finite": bool(np.isfinite(feature_values).all()),
            "features_have_no_missing_values": not frame[
                list(feature_columns)
            ].isna().any().any(),
            "source_labels_are_isolated": not bool(
                frame_columns.intersection(source_label_columns)
            ),
            "target_not_in_feature_columns": "label" not in feature_columns,
            "identifiers_not_in_feature_columns": not any(
                token in column.lower()
                for column in feature_columns
                for token in ("timeus", "time_us", "lineno", "flight_id", "gms", "gwk")
            ),
            "source_times_are_never_future": bool(
                (alignment_audit[["gps_lag_us", "rate_lag_us", "imu_lag_us"]]
                 >= 0).all().all()
            ),
            "gps_tolerance_respected": bool(
                alignment_audit["gps_lag_us"].le(
                    config.gps_alignment_tolerance_us
                ).all()
            ),
            "rate_alignment_is_exact": bool(
                alignment_audit["rate_lag_us"].eq(0).all()
            ),
            "imu_tolerance_respected": bool(
                alignment_audit["imu_lag_us"].le(
                    config.imu_alignment_tolerance_us
                ).all()
            ),
            "no_cross_flight_alignment": source_rows_remain_in_flight,
            "label_audit_matches_model_rows": np.array_equal(
                frame["row_id"].to_numpy(),
                label_audit["row_id"].to_numpy(),
            ),
            "source_label_disagreement_below_one_percent": all(
                float((part[source] != part["label"]).mean()) < 0.01
                for _, part in label_audit.groupby("flight_id")
                for source in ("gps_label", "rate_label", "imu_label")
            ),
        },
        dtype=bool,
        name="passed",
    )


def persist_reconstruction_artifacts(
    reconstructed: ReconstructedFlights,
    aligned: AlignedMultisensorData,
    output_dir: Path | str,
) -> dict[str, Path]:
    """Persist transparent reconstruction artifacts as CSV and JSON."""

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "aligned_data": destination / "aligned_multisensor.csv",
        "label_audit": destination / "aligned_label_audit.csv",
        "alignment_audit": destination / "alignment_trace.csv",
        "assignment_summary": destination / "flight_assignment_summary.csv",
        "alignment_summary": destination / "alignment_summary.csv",
        "metadata": destination / "reconstruction_metadata.json",
    }
    aligned.frame.to_csv(paths["aligned_data"], index=False)
    aligned.label_audit.to_csv(paths["label_audit"], index=False)
    aligned.alignment_audit.to_csv(paths["alignment_audit"], index=False)
    reconstructed.assignment_summary.to_csv(
        paths["assignment_summary"],
        index=False,
    )
    aligned.alignment_summary.to_csv(paths["alignment_summary"], index=False)
    metadata = {
        "config": _protocol_config(aligned.config),
        "class_names": list(aligned.class_names),
        "target_name": aligned.target_name,
        "target_source": aligned.target_source,
        "feature_columns": list(aligned.feature_columns),
        "source_manifest": aligned.source_manifest,
        "reconstruction_fingerprint": aligned.reconstruction_fingerprint,
        "alignment_fingerprint": aligned.alignment_fingerprint,
        "imu_anchor_summary": reconstructed.imu_anchor_summary,
        "checks": aligned.checks.to_dict(),
        "label_agreement": aligned.label_agreement.to_dict("records"),
        "dropped_rows": aligned.dropped_rows.to_dict("records"),
    }
    paths["metadata"].write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return paths


def persist_data_protocol(
    aligned: AlignedMultisensorData,
    folds: tuple[FlightFold, ...],
    prepared_folds: tuple[PreparedFlightFold, ...],
    output_dir: Path | str,
) -> dict[str, Path]:
    """Persist split roles, feature views, and train-fitted states transparently."""

    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    paths = {
        "fold_assignments": destination / "fold_assignments.csv",
        "feature_views": destination / "feature_views.json",
        "preprocessing": destination / "preprocessing_metadata.json",
    }
    indexed = aligned.frame.set_index("row_id")
    role_rows: list[dict[str, Any]] = []
    for fold in folds:
        stable = set(fold.stable_test_row_ids.tolist())
        role_arrays = {
            "inner_train": fold.inner_train_row_ids,
            "validation": fold.valid_row_ids,
            "purged": fold.purged_row_ids,
            "external_test": fold.test_row_ids,
        }
        for role, row_ids in role_arrays.items():
            for row_id in row_ids:
                source = indexed.loc[int(row_id)]
                evaluation_region = "not_applicable"
                if role == "external_test":
                    evaluation_region = (
                        "stable" if int(row_id) in stable else "transition_guard"
                    )
                role_rows.append(
                    {
                        "fold_name": fold.name,
                        "development_flight": fold.development_flight,
                        "test_flight": fold.test_flight,
                        "row_id": int(row_id),
                        "flight_id": int(source["flight_id"]),
                        "time_us": int(source["time_us"]),
                        "label": int(source["label"]),
                        "role": role,
                        "evaluation_region": evaluation_region,
                    }
                )
    pd.DataFrame(role_rows).to_csv(paths["fold_assignments"], index=False)

    views_payload = {
        "alignment_fingerprint": aligned.alignment_fingerprint,
        "views": [
            {
                "name": view.name,
                "description": view.description,
                "numerical_columns": list(view.numerical_columns),
                "categorical_columns": list(view.categorical_columns),
            }
            for view in available_feature_views()
        ],
    }
    paths["feature_views"].write_text(
        json.dumps(views_payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    preprocessing_payload = {
        "alignment_fingerprint": aligned.alignment_fingerprint,
        "folds": [
            {
                "name": fold.name,
                "development_flight": fold.development_flight,
                "test_flight": fold.test_flight,
                "fingerprint": fold.fingerprint,
                "report": fold.report,
            }
            for fold in folds
        ],
        "prepared": [
            {
                "fold_name": item.fold.name,
                "feature_view": item.feature_view.name,
                "fingerprint": item.fingerprint,
                "checks": item.checks.to_dict(),
                "tuning_class_weights": item.tuning_class_weights.tolist(),
                "final_class_weights": item.final_class_weights.tolist(),
                "tuning_unknown_categories": {
                    "train": item.train.unknown_category_counts,
                    "validation": item.valid.unknown_category_counts,
                },
                "final_unknown_categories": {
                    "development": item.development.unknown_category_counts,
                    "external_test": item.test.unknown_category_counts,
                },
                "tuning_state": _preprocessing_state_to_json(item.tuning_state),
                "final_state": _preprocessing_state_to_json(item.final_state),
            }
            for item in prepared_folds
        ],
    }
    paths["preprocessing"].write_text(
        json.dumps(
            _preprocessing_json_values(preprocessing_payload),
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    return paths


def _build_purged_inner_roles(
    development: pd.DataFrame,
    config: FlightSplitConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    episodes = sorted(development["episode_id"].unique().tolist())
    if len(episodes) != config.expected_episodes_per_flight:
        raise ValueError(
            f"Expected {config.expected_episodes_per_flight} episodes, "
            f"found {len(episodes)}"
        )
    purge_us = int(round(config.purge_seconds * 1_000_000))
    train_ids: list[int] = []
    valid_ids: list[int] = []
    purged_ids: list[int] = []
    for episode_id in episodes:
        episode = development.loc[
            development["episode_id"].eq(episode_id)
        ].sort_values("time_us", kind="stable")
        valid_size = max(1, int(np.floor(config.validation_fraction * len(episode))))
        valid_start = (len(episode) - valid_size) // 2
        validation = episode.iloc[valid_start : valid_start + valid_size]
        first_valid_time = int(validation["time_us"].min())
        last_valid_time = int(validation["time_us"].max())
        purge_mask = (
            episode["time_us"].between(
                first_valid_time - purge_us,
                first_valid_time - 1,
            )
            | episode["time_us"].between(
                last_valid_time + 1,
                last_valid_time + purge_us,
            )
        )
        purged = episode.loc[purge_mask]
        training = episode.loc[
            ~episode["row_id"].isin(
                set(validation["row_id"]).union(purged["row_id"])
            )
        ]
        if training.empty:
            raise ValueError(f"Episode {episode_id} has no rows left for training")
        train_ids.extend(training["row_id"].astype(int).tolist())
        valid_ids.extend(validation["row_id"].astype(int).tolist())
        purged_ids.extend(purged["row_id"].astype(int).tolist())
    return (
        np.asarray(sorted(train_ids), dtype=np.int64),
        np.asarray(sorted(valid_ids), dtype=np.int64),
        np.asarray(sorted(purged_ids), dtype=np.int64),
    )


def _split_transition_guard(
    test: pd.DataFrame,
    guard_seconds: float,
) -> tuple[np.ndarray, np.ndarray]:
    ordered = test.sort_values("time_us", kind="stable")
    transitions = ordered["label"].ne(ordered["label"].shift())
    transition_times = ordered.loc[transitions, "time_us"].iloc[1:].to_numpy(
        dtype=np.int64
    )
    if len(transition_times) == 0:
        raise ValueError("External flight has no target transitions")
    guard_us = int(round(guard_seconds * 1_000_000))
    times = ordered["time_us"].to_numpy(dtype=np.int64)
    distance = np.min(np.abs(times[:, None] - transition_times[None, :]), axis=1)
    transition_mask = distance <= guard_us
    return (
        ordered.loc[~transition_mask, "row_id"].to_numpy(dtype=np.int64),
        ordered.loc[transition_mask, "row_id"].to_numpy(dtype=np.int64),
    )


def _build_fold_report(
    development: pd.DataFrame,
    test: pd.DataFrame,
    inner_train_row_ids: np.ndarray,
    valid_row_ids: np.ndarray,
    purged_row_ids: np.ndarray,
    stable_test_row_ids: np.ndarray,
    transition_test_row_ids: np.ndarray,
    config: FlightSplitConfig,
) -> dict[str, Any]:
    minimum_gap = _minimum_train_valid_gap_seconds(
        development,
        inner_train_row_ids,
        valid_row_ids,
    )
    return {
        "validation_fraction": config.validation_fraction,
        "purge_seconds": config.purge_seconds,
        "transition_guard_seconds": config.transition_guard_seconds,
        "development_episode_count": int(development["episode_id"].nunique()),
        "sizes": {
            "inner_train": len(inner_train_row_ids),
            "validation": len(valid_row_ids),
            "purged": len(purged_row_ids),
            "development_full": len(development),
            "external_test": len(test),
            "stable_external_test": len(stable_test_row_ids),
            "transition_external_test": len(transition_test_row_ids),
        },
        "class_counts": {
            "inner_train": _class_counts_for_rows(
                development,
                inner_train_row_ids,
            ),
            "validation": _class_counts_for_rows(development, valid_row_ids),
            "development_full": _class_counts_for_rows(
                development,
                development["row_id"].to_numpy(dtype=np.int64),
            ),
            "external_test": _class_counts_for_rows(
                test,
                test["row_id"].to_numpy(dtype=np.int64),
            ),
            "stable_external_test": _class_counts_for_rows(
                test,
                stable_test_row_ids,
            ),
        },
        "minimum_train_valid_gap_seconds": minimum_gap,
        "test_used_for_preprocessing_or_selection": False,
        "validation_purpose": "early_stopping_and_epoch_selection_only",
        "primary_test_region": "all_external_rows",
        "secondary_test_region": "outside_transition_guard",
    }


def _minimum_train_valid_gap_seconds(
    development: pd.DataFrame,
    train_row_ids: np.ndarray,
    valid_row_ids: np.ndarray,
) -> float:
    train_set = set(train_row_ids.tolist())
    valid_set = set(valid_row_ids.tolist())
    minimum = np.inf
    for _, episode in development.groupby("episode_id", sort=True):
        train_times = episode.loc[
            episode["row_id"].isin(train_set),
            "time_us",
        ].to_numpy(dtype=np.int64)
        valid_times = episode.loc[
            episode["row_id"].isin(valid_set),
            "time_us",
        ].to_numpy(dtype=np.int64)
        if len(train_times) == 0 or len(valid_times) == 0:
            continue
        insertion = np.searchsorted(train_times, valid_times)
        left = np.clip(insertion - 1, 0, len(train_times) - 1)
        right = np.clip(insertion, 0, len(train_times) - 1)
        distances = np.minimum(
            np.abs(valid_times - train_times[left]),
            np.abs(valid_times - train_times[right]),
        )
        minimum = min(minimum, float(distances.min()))
    if not np.isfinite(minimum):
        raise ValueError("Could not measure an inner train-validation time gap")
    return minimum / 1_000_000


def _fold_fingerprint(
    alignment_fingerprint: str,
    development_flight: int,
    test_flight: int,
    inner_train: np.ndarray,
    valid: np.ndarray,
    purged: np.ndarray,
    stable_test: np.ndarray,
    config: FlightSplitConfig,
) -> str:
    digest = sha256()
    digest.update(alignment_fingerprint.encode("utf-8"))
    digest.update(f"{development_flight}:{test_flight}".encode("ascii"))
    for values in (inner_train, valid, purged, stable_test):
        digest.update(values.astype(np.int64).tobytes())
    digest.update(json.dumps(asdict(config), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _class_counts_for_rows(
    frame: pd.DataFrame,
    row_ids: np.ndarray,
) -> dict[str, int]:
    values = _targets_for_rows(frame, row_ids)
    return {
        str(label): int((values == label).sum())
        for label in range(len(CLASS_NAMES))
    }


def _targets_for_rows(frame: pd.DataFrame, row_ids: np.ndarray) -> np.ndarray:
    return _select_rows(frame, row_ids)["label"].to_numpy(dtype=np.int64)


def _select_rows(frame: pd.DataFrame, row_ids: np.ndarray) -> pd.DataFrame:
    requested = np.asarray(row_ids, dtype=np.int64)
    if len(requested) != len(np.unique(requested)):
        raise ValueError("Requested row IDs contain duplicates")
    if not frame["row_id"].is_unique:
        raise ValueError("Model frame row IDs are not unique")
    indexed = frame.set_index("row_id", drop=False)
    missing = sorted(set(requested.tolist()).difference(indexed.index.tolist()))
    if missing:
        raise KeyError(f"Requested row IDs are absent: {missing[:10]}")
    return indexed.loc[requested].reset_index(drop=True)


def _validate_feature_view(frame: pd.DataFrame, view: FeatureView) -> None:
    if not view.name:
        raise ValueError("Feature view name cannot be empty")
    if not view.columns:
        raise ValueError(f"Feature view {view.name!r} is empty")
    if len(view.columns) != len(set(view.columns)):
        raise ValueError(f"Feature view {view.name!r} contains duplicate columns")
    overlap = set(view.numerical_columns).intersection(view.categorical_columns)
    if overlap:
        raise ValueError(f"Columns appear in both feature types: {sorted(overlap)}")
    missing = sorted(set(view.columns).difference(frame.columns))
    if missing:
        raise ValueError(f"Feature view {view.name!r} is missing columns: {missing}")
    forbidden_tokens = ("label", "flight_id", "time_us", "lineno", "gms", "gwk")
    forbidden = [
        column
        for column in view.columns
        if any(token in column.lower() for token in forbidden_tokens)
    ]
    if forbidden:
        raise ValueError(f"Feature view contains forbidden metadata: {forbidden}")


def _balanced_class_weights(labels: np.ndarray) -> np.ndarray:
    counts = np.bincount(labels, minlength=len(CLASS_NAMES)).astype(np.float64)
    if (counts == 0).any():
        raise ValueError("Balanced class weights require every class in training")
    return (len(labels) / (len(CLASS_NAMES) * counts)).astype(np.float32)


def _validate_prepared_fold(
    fold: FlightFold,
    feature_view: FeatureView,
    tuning_state: PreprocessingState,
    final_state: PreprocessingState,
    train: TransformedFeatures,
    valid: TransformedFeatures,
    development: TransformedFeatures,
    test: TransformedFeatures,
    y_train: np.ndarray,
    y_valid: np.ndarray,
    y_development: np.ndarray,
    y_test: np.ndarray,
) -> pd.Series:
    expected_classes = set(range(len(CLASS_NAMES)))
    all_features = (train, valid, development, test)
    correct_shapes = all(
        values.X_num.shape[0] == len(values.row_ids)
        and values.X_cat.shape[0] == len(values.row_ids)
        for values in all_features
    )
    finite_arrays = all(np.isfinite(values.X_num).all() for values in all_features)
    tuning_standardized = _is_standardized(train.X_num)
    final_standardized = _is_standardized(development.X_num)
    valid_categories = all(
        _categories_within_cardinality(values.X_cat, state)
        for values, state in (
            (train, tuning_state),
            (valid, tuning_state),
            (development, final_state),
            (test, final_state),
        )
    )
    test_rows = set(fold.test_row_ids.tolist())
    return pd.Series(
        {
            "tuning_state_fitted_only_on_inner_train": set(
                tuning_state.fit_row_ids
            )
            == set(fold.inner_train_row_ids.tolist()),
            "final_state_fitted_only_on_development": set(
                final_state.fit_row_ids
            )
            == set(fold.development_row_ids.tolist()),
            "external_test_absent_from_tuning_fit": test_rows.isdisjoint(
                tuning_state.fit_row_ids
            ),
            "external_test_absent_from_final_fit": test_rows.isdisjoint(
                final_state.fit_row_ids
            ),
            "validation_absent_from_tuning_fit": set(
                fold.valid_row_ids.tolist()
            ).isdisjoint(tuning_state.fit_row_ids),
            "transformed_row_counts_are_correct": correct_shapes,
            "all_numerical_arrays_are_finite": finite_arrays,
            "inner_train_is_standardized": tuning_standardized,
            "development_refit_is_standardized": final_standardized,
            "categorical_indices_are_valid": valid_categories,
            "all_classes_in_tuning_train": set(np.unique(y_train))
            == expected_classes,
            "all_classes_in_validation": set(np.unique(y_valid))
            == expected_classes,
            "all_classes_in_development": set(np.unique(y_development))
            == expected_classes,
            "all_classes_in_external_test": set(np.unique(y_test))
            == expected_classes,
            "feature_view_matches_preprocessors": (
                tuning_state.view_name == feature_view.name
                and final_state.view_name == feature_view.name
            ),
            "unknown_category_index_is_reserved": all(
                cardinality >= 2
                for cardinality in (
                    *tuning_state.categorical_cardinalities,
                    *final_state.categorical_cardinalities,
                )
            ),
        },
        dtype=bool,
        name="passed",
    )


def _is_standardized(values: np.ndarray) -> bool:
    if values.shape[1] == 0:
        return True
    return bool(
        np.allclose(values.mean(axis=0), 0.0, atol=2e-5)
        and np.allclose(values.std(axis=0), 1.0, atol=2e-5)
    )


def _categories_within_cardinality(
    values: np.ndarray,
    state: PreprocessingState,
) -> bool:
    if values.shape[1] != len(state.categorical_cardinalities):
        return False
    return all(
        values[:, index].min(initial=0) >= 0
        and values[:, index].max(initial=0) < cardinality
        for index, cardinality in enumerate(state.categorical_cardinalities)
    )


def _prepared_fold_fingerprint(
    alignment_fingerprint: str,
    fold: FlightFold,
    feature_view: FeatureView,
    tuning_state: PreprocessingState,
    final_state: PreprocessingState,
    train: TransformedFeatures,
    valid: TransformedFeatures,
    development: TransformedFeatures,
    test: TransformedFeatures,
) -> str:
    digest = sha256()
    for value in (
        alignment_fingerprint,
        fold.fingerprint,
        feature_view.name,
        tuning_state.fingerprint,
        final_state.fingerprint,
    ):
        digest.update(value.encode("utf-8"))
    for transformed in (train, valid, development, test):
        for array in (transformed.row_ids, transformed.X_num, transformed.X_cat):
            digest.update(str(array.shape).encode("ascii"))
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(np.ascontiguousarray(array).tobytes())
    return digest.hexdigest()


def _preprocessing_state_to_json(state: PreprocessingState) -> dict[str, Any]:
    return {
        "view_name": state.view_name,
        "fitted_on": state.fitted_on,
        "fit_row_ids": list(state.fit_row_ids),
        "numerical_columns": list(state.numerical_columns),
        "dropped_numerical_columns": list(state.dropped_numerical_columns),
        "numerical_medians": state.numerical_medians,
        "numerical_means": state.numerical_means,
        "numerical_stds": state.numerical_stds,
        "categorical_columns": list(state.categorical_columns),
        "categorical_modes": state.categorical_modes,
        "categorical_mappings": state.categorical_mappings,
        "categorical_cardinalities": list(state.categorical_cardinalities),
        "fit_row_count": state.fit_row_count,
        "fingerprint": state.fingerprint,
    }


def _preprocessing_json_values(value: Any) -> Any:
    if isinstance(value, dict):
        return {
            (
                key
                if isinstance(key, str)
                else f"{type(key).__name__}:{repr(key)}"
            ): _preprocessing_json_values(item)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_preprocessing_json_values(item) for item in value]
    return _json_safe(value)


def _python_scalar(value: Any) -> Any:
    return value.item() if isinstance(value, np.generic) else value


def _assign_gps_flights(
    frame: pd.DataFrame,
    config: ReconstructionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    required = {"LineNo", "TimeUS", "GMS", "GWk"}
    _require_columns(frame, required, "GPS")
    gps = frame.copy().reset_index(names="_source_row")
    gps["gps_absolute_ms"] = (
        gps["GWk"].astype("int64") * 604_800_000
        + gps["GMS"].astype("int64")
    )
    gps = gps.sort_values("gps_absolute_ms", kind="stable").reset_index(drop=True)
    gps["flight_id"] = (
        gps["gps_absolute_ms"]
        .diff()
        .fillna(0)
        .gt(config.gps_gap_threshold_ms)
        .cumsum()
        .astype("int64")
    )
    if gps["flight_id"].nunique() != config.expected_flights:
        raise ValueError("GPS absolute time did not recover exactly two flights")
    report = _source_assignment_report(
        gps,
        source="GPS",
        time_column="TimeUS",
        line_column="LineNo",
        method="absolute_gps_gap",
    )
    return gps, report


def _assign_vibe_flights(
    combined_frame: pd.DataFrame,
    reference_frame: pd.DataFrame,
    config: ReconstructionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    del config
    identity_columns = [
        "LineNo",
        "TimeUS",
        "IMU",
        "VibeX",
        "VibeY",
        "VibeZ",
        "Clip",
    ]
    _require_columns(combined_frame, set(identity_columns), "VIBE combined")
    _require_columns(reference_frame, set(identity_columns), "VIBE reference")
    if reference_frame.duplicated(identity_columns).any():
        raise ValueError("VIBE reference identity rows are not unique")
    if combined_frame.duplicated(identity_columns).any():
        raise ValueError("VIBE combined identity rows are not unique")

    reference_keys = reference_frame[identity_columns].assign(
        _reference_flight=True
    )
    vibe = combined_frame.copy().reset_index(names="_source_row")
    vibe = vibe.merge(
        reference_keys,
        on=identity_columns,
        how="left",
        validate="one_to_one",
    )
    vibe["flight_id"] = np.where(
        vibe["_reference_flight"].fillna(False),
        0,
        1,
    ).astype("int64")
    vibe["_assignment_method"] = np.where(
        vibe["flight_id"].eq(0),
        "exact_nonlabel_reference_match",
        "complement_of_exact_reference",
    )
    vibe = vibe.drop(columns="_reference_flight")
    report = _source_assignment_report(
        vibe,
        source="VIBE",
        time_column="TimeUS",
        line_column="LineNo",
        method="exact_nonlabel_lineage",
    )
    return vibe, report


def _assign_rate_flights(
    frame: pd.DataFrame,
    vibe: pd.DataFrame,
    config: ReconstructionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(frame, {"LineNO", "TimeUS"}, "RATE")
    rate = frame.copy().reset_index(names="_source_row")
    candidates = rate[["_source_row", "TimeUS", "LineNO"]].merge(
        vibe[["flight_id", "TimeUS", "LineNo"]],
        on="TimeUS",
        how="left",
        validate="many_to_many",
    )
    if candidates["flight_id"].isna().any():
        raise ValueError("At least one RATE row has no VIBE timestamp candidate")
    candidates["_line_delta"] = (
        candidates["LineNO"] - candidates["LineNo"]
    ).abs()
    minimum = candidates.groupby("_source_row")["_line_delta"].transform("min")
    tied_minima = candidates.loc[candidates["_line_delta"].eq(minimum)]
    ties = tied_minima.groupby("_source_row").size().gt(1)
    if ties.any():
        raise ValueError(f"RATE flight assignment has {int(ties.sum())} ties")
    selected = tied_minima.set_index("_source_row")[
        ["flight_id", "_line_delta"]
    ]
    rate = rate.join(selected, on="_source_row")
    rate["flight_id"] = rate["flight_id"].astype("int64")
    rate["_assignment_method"] = "exact_time_nearest_vibe_line"
    if int(rate["_line_delta"].max()) > config.max_rate_line_delta:
        raise ValueError("RATE-to-VIBE line distance exceeds configured maximum")
    report = _source_assignment_report(
        rate,
        source="RATE",
        time_column="TimeUS",
        line_column="LineNO",
        method="exact_time_nearest_vibe_line",
        extra={
            "minimum_line_delta": int(rate["_line_delta"].min()),
            "maximum_line_delta": int(rate["_line_delta"].max()),
            "assignment_ties": 0,
        },
    )
    return rate, report


def _assign_imu_flights(
    frame: pd.DataFrame,
    fusion: pd.DataFrame,
    vibe: pd.DataFrame,
    config: ReconstructionConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    required = {"LineNo", "abTimeUS", *IMU_SIGNAL_COLUMNS}
    _require_columns(frame, required, "IMU")
    _require_columns(fusion, set(IMU_SIGNAL_COLUMNS), "Fusion IMU block")
    imu = frame.copy().reset_index(names="_source_row")
    times = imu["abTimeUS"].to_numpy(dtype=float)
    lines = imu["LineNo"].to_numpy(dtype=float)
    costs: list[np.ndarray] = []
    for flight_id in range(config.expected_flights):
        trajectory = vibe.loc[vibe["flight_id"].eq(flight_id)].sort_values(
            "TimeUS",
            kind="stable",
        )
        trajectory_times = trajectory["TimeUS"].to_numpy(dtype=float)
        trajectory_lines = trajectory["LineNo"].to_numpy(dtype=float)
        predicted_lines = np.interp(times, trajectory_times, trajectory_lines)
        cost = np.abs(lines - predicted_lines)
        valid = (
            times >= trajectory_times.min() - config.imu_assignment_padding_us
        ) & (
            times <= trajectory_times.max() + config.imu_assignment_padding_us
        )
        cost[~valid] = np.inf
        costs.append(cost)
    cost_matrix = np.column_stack(costs)
    if np.isinf(cost_matrix).all(axis=1).any():
        raise ValueError("At least one IMU row falls outside every VIBE trajectory")
    initial_assignment = np.argmin(cost_matrix, axis=1).astype("int64")
    sorted_costs = np.sort(cost_matrix, axis=1)
    margin = sorted_costs[:, 1] - sorted_costs[:, 0]

    rounded_imu = imu[["_source_row", "LineNo", "abTimeUS", *IMU_SIGNAL_COLUMNS]].copy()
    rounded_fusion = fusion[list(IMU_SIGNAL_COLUMNS)].copy().reset_index(
        names="_fusion_row"
    )
    rounded_imu[list(IMU_SIGNAL_COLUMNS)] = rounded_imu[
        list(IMU_SIGNAL_COLUMNS)
    ].round(config.imu_match_decimals)
    rounded_fusion[list(IMU_SIGNAL_COLUMNS)] = rounded_fusion[
        list(IMU_SIGNAL_COLUMNS)
    ].round(config.imu_match_decimals)
    candidate_counts = (
        rounded_imu.groupby(list(IMU_SIGNAL_COLUMNS), dropna=False)
        .size()
        .rename("_candidate_count")
        .reset_index()
    )
    unique_imu = rounded_imu.merge(
        candidate_counts,
        on=list(IMU_SIGNAL_COLUMNS),
    ).loc[lambda table: table["_candidate_count"].eq(1)]
    anchors = rounded_fusion.merge(
        unique_imu,
        on=list(IMU_SIGNAL_COLUMNS),
        how="inner",
    ).sort_values("_fusion_row", kind="stable")
    if not anchors["_source_row"].is_unique:
        raise ValueError("Fusion-to-IMU anchor mapping is not one-to-one")
    if not anchors["abTimeUS"].is_monotonic_increasing:
        raise ValueError("Unique Fusion IMU anchors do not form a monotonic path")
    anchor_rows = anchors["_source_row"].to_numpy(dtype=int)
    anchor_initial = initial_assignment[anchor_rows]
    values, counts = np.unique(anchor_initial, return_counts=True)
    anchor_flight = int(values[np.argmax(counts)])
    anchor_purity = float(counts.max() / counts.sum())
    if anchor_purity < 0.999:
        raise ValueError("Fusion IMU anchors do not identify one dominant flight")

    final_assignment = initial_assignment.copy()
    final_assignment[anchor_rows] = anchor_flight
    methods = np.full(len(imu), "vibe_line_trajectory", dtype=object)
    methods[anchor_rows] = "fusion_anchor_confirmed"
    overridden = anchor_rows[initial_assignment[anchor_rows] != anchor_flight]
    methods[overridden] = "fusion_anchor_override"
    imu["flight_id"] = final_assignment
    imu["_assignment_margin"] = margin
    imu["_assignment_method"] = methods

    anchor_summary = {
        "unique_anchor_rows": int(len(anchors)),
        "dominant_flight": anchor_flight,
        "dominant_flight_purity": anchor_purity,
        "overridden_rows": int(len(overridden)),
        "anchor_time_min": int(anchors["abTimeUS"].min()),
        "anchor_time_max": int(anchors["abTimeUS"].max()),
        "anchor_time_monotonic": True,
        "labels_used": False,
    }
    report = _source_assignment_report(
        imu,
        source="IMU",
        time_column="abTimeUS",
        line_column="LineNo",
        method="vibe_trajectory_with_unique_fusion_anchor",
        extra={
            "minimum_assignment_margin": float(np.nanmin(margin)),
            "rows_with_margin_below_100": int((margin < 100).sum()),
            "fusion_anchor_rows": int(len(anchors)),
            "fusion_anchor_overrides": int(len(overridden)),
        },
    )
    return imu, report, anchor_summary


def _canonical_vibe_frame(frame: pd.DataFrame, flight_id: int) -> pd.DataFrame:
    source = frame.loc[frame["flight_id"].eq(flight_id)].sort_values(
        "TimeUS",
        kind="stable",
    ).copy()
    target = source["labels"].to_numpy(dtype=np.int64)
    transitions = np.r_[True, target[1:] != target[:-1]]
    source["episode_id"] = np.cumsum(transitions).astype("int64") - 1
    canonical = pd.DataFrame(
        {
            "flight_id": source["flight_id"].to_numpy(dtype=np.int64),
            "time_us": source["TimeUS"].to_numpy(dtype=np.int64),
            "episode_id": source["episode_id"].to_numpy(dtype=np.int64),
            "label": target,
            "vibe_source_row": source["_source_row"].to_numpy(dtype=np.int64),
            "vibe_time_us": source["TimeUS"].to_numpy(dtype=np.int64),
            "vibe_label": target,
        }
    )
    for column in VIBE_FEATURE_COLUMNS:
        canonical[f"vibe__{column}"] = source[column].to_numpy()
    return canonical


def _prefixed_source_frame(
    frame: pd.DataFrame,
    flight_id: int,
    source: str,
    time_column: str,
) -> pd.DataFrame:
    prefix = source.lower()
    selected = frame.loc[frame["flight_id"].eq(flight_id)].copy()
    columns = FEATURE_COLUMNS_BY_SOURCE[source]
    _require_columns(selected, set(columns), source)
    output = pd.DataFrame(
        {
            "flight_id": selected["flight_id"].to_numpy(dtype=np.int64),
            f"{prefix}_time_us": selected[time_column].to_numpy(dtype=np.int64),
            f"{prefix}_source_row": selected["_source_row"].to_numpy(
                dtype=np.int64
            ),
            f"{prefix}_label": selected["labels"].to_numpy(dtype=np.int64),
        }
    )
    for column in columns:
        output[f"{prefix}__{column}"] = selected[column].to_numpy()
    return output.sort_values(f"{prefix}_time_us", kind="stable")


def _build_alignment_audit(frame: pd.DataFrame) -> pd.DataFrame:
    audit = frame[[
        "row_id",
        "flight_id",
        "time_us",
        "gps_time_us",
        "rate_time_us",
        "vibe_time_us",
        "imu_time_us",
        "gps_source_row",
        "rate_source_row",
        "vibe_source_row",
        "imu_source_row",
    ]].copy()
    audit["gps_lag_us"] = audit["time_us"] - audit["gps_time_us"]
    audit["rate_lag_us"] = audit["time_us"] - audit["rate_time_us"]
    audit["vibe_lag_us"] = audit["time_us"] - audit["vibe_time_us"]
    audit["imu_lag_us"] = audit["time_us"] - audit["imu_time_us"]
    integer_columns = [column for column in audit if column != "row_id"]
    audit[integer_columns] = audit[integer_columns].astype("int64")
    return audit


def _build_label_audit(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[[
        "row_id",
        "flight_id",
        "time_us",
        "episode_id",
        "label",
        "gps_label",
        "rate_label",
        "imu_label",
    ]].astype("int64")


def _summarize_alignment(audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for flight_id, part in audit.groupby("flight_id", sort=True):
        for source in ("GPS", "RATE", "VIBE", "IMU"):
            prefix = source.lower()
            source_rows = part[f"{prefix}_source_row"]
            lag = part[f"{prefix}_lag_us"]
            reuse = source_rows.value_counts()
            rows.append(
                {
                    "flight_id": int(flight_id),
                    "source": source,
                    "aligned_rows": len(part),
                    "unique_source_rows": int(source_rows.nunique()),
                    "maximum_reuse": int(reuse.max()),
                    "lag_min_us": int(lag.min()),
                    "lag_median_us": float(lag.median()),
                    "lag_p99_us": float(lag.quantile(0.99)),
                    "lag_max_us": int(lag.max()),
                }
            )
    return pd.DataFrame(rows)


def _summarize_label_agreement(label_audit: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for flight_id, part in label_audit.groupby("flight_id", sort=True):
        for source in ("GPS", "RATE", "IMU"):
            source_column = f"{source.lower()}_label"
            disagreement = part[source_column].ne(part["label"])
            rows.append(
                {
                    "flight_id": int(flight_id),
                    "target_source": "VIBE",
                    "compared_source": source,
                    "rows": len(part),
                    "disagreements": int(disagreement.sum()),
                    "disagreement_rate": float(disagreement.mean()),
                }
            )
    return pd.DataFrame(rows)


def _source_assignment_report(
    frame: pd.DataFrame,
    source: str,
    time_column: str,
    line_column: str,
    method: str,
    extra: dict[str, Any] | None = None,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for flight_id, part in frame.groupby("flight_id", sort=True):
        ordered = part.sort_values(time_column, kind="stable")
        row: dict[str, Any] = {
            "source": source,
            "flight_id": int(flight_id),
            "method": method,
            "rows": len(part),
            "time_min": int(ordered[time_column].min()),
            "time_max": int(ordered[time_column].max()),
            "line_min": int(ordered[line_column].min()),
            "line_max": int(ordered[line_column].max()),
            "time_is_strictly_increasing": bool(
                ordered[time_column].diff().dropna().gt(0).all()
            ),
            "line_is_strictly_increasing": bool(
                ordered[line_column].diff().dropna().gt(0).all()
            ),
            "labels_used_for_assignment": False,
        }
        if extra:
            row.update(extra)
        rows.append(row)
    return pd.DataFrame(rows)


def _validate_required_frames(frames: dict[str, pd.DataFrame]) -> None:
    missing = sorted(set(RECONSTRUCTION_FILES).difference(frames))
    if missing:
        raise KeyError(f"Missing reconstruction frames: {missing}")


def _validate_reconstructed_frames(
    frames: dict[str, pd.DataFrame],
    config: ReconstructionConfig,
) -> None:
    expected_counts = {
        "GPS": {0: 2450, 1: 1869},
        "RATE": {0: 4900, 1: 3738},
        "VIBE": {0: 4900, 1: 3738},
        "IMU": {0: 12253, 1: 9344},
    }
    time_columns = {
        "GPS": "TimeUS",
        "RATE": "TimeUS",
        "VIBE": "TimeUS",
        "IMU": "abTimeUS",
    }
    line_columns = {
        "GPS": "LineNo",
        "RATE": "LineNO",
        "VIBE": "LineNo",
        "IMU": "LineNo",
    }
    for source, frame in frames.items():
        if not frame["_source_row"].is_unique:
            raise AssertionError(f"{source} source rows are not unique")
        counts = frame["flight_id"].value_counts().sort_index().to_dict()
        if counts != expected_counts[source]:
            raise AssertionError(
                f"Unexpected {source} flight counts: {counts}; "
                f"expected {expected_counts[source]}"
            )
        if set(counts) != set(range(config.expected_flights)):
            raise AssertionError(f"{source} flight IDs are not contiguous")
        for _, part in frame.groupby("flight_id"):
            ordered = part.sort_values(time_columns[source], kind="stable")
            if not ordered[time_columns[source]].diff().dropna().gt(0).all():
                raise AssertionError(f"{source} time is not strictly increasing")
            if not ordered[line_columns[source]].diff().dropna().gt(0).all():
                raise AssertionError(f"{source} line number is not strictly increasing")


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    source: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"{source} is missing required columns: {missing}")


def _reconstruction_fingerprint(
    frames: dict[str, pd.DataFrame],
    manifest: dict[str, str],
    config: ReconstructionConfig,
) -> str:
    digest = sha256()
    for source in sorted(frames):
        ordered = frames[source].sort_values("_source_row")
        digest.update(source.encode("utf-8"))
        digest.update(
            pd.util.hash_pandas_object(
                ordered[["_source_row", "flight_id"]],
                index=False,
            ).to_numpy().tobytes()
        )
    digest.update(json.dumps(manifest, sort_keys=True).encode("utf-8"))
    digest.update(json.dumps(_protocol_config(config), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _frame_fingerprint(frame: pd.DataFrame, extra: dict[str, Any]) -> str:
    digest = sha256()
    digest.update(
        pd.util.hash_pandas_object(frame, index=False).to_numpy().tobytes()
    )
    digest.update(json.dumps(_json_safe(extra), sort_keys=True).encode("utf-8"))
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _protocol_config(config: ReconstructionConfig) -> dict[str, Any]:
    """Return path-independent values that define reconstruction behavior."""

    values = asdict(config)
    values["dataset_dir"] = "dataset"
    return _json_safe(values)
