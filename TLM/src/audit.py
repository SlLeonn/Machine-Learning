"""Reproducible structural audit for the local TLM:UAV release."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from itertools import combinations
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


CLASS_NAMES = (
    "Normal",
    "GPS fault",
    "Accelerometer fault",
    "Engine fault",
    "RC fault",
)
FUSION_FILE = "Fusion_Data.csv"
ATT_FILE = "ATT/ALL_FAIL_LOG_ATT.csv"
MAG_FILE = "MAG/ALL_FAIL_LOG_MAG_0.csv"
GPS_FILE = "GPS/ALL_FAIL_LOG_GPS_0.csv"
GPS_COMBINED_FILE = "AddNum/ALL_FAIL_LOG_GPS_0_Add_Num_Random.csv"
IMU_COMBINED_FILE = "IMU/ALL_FAIL_LOG_IMU_0_Random.csv"
RATE_FILE = "RATE/ALL_FAIL_LOG_RATE.csv"
RATE_COMBINED_FILE = "AddNum/ALL_FAIL_LOG_RATE_Add_Random.csv"
VIBE_FILE = "VIBE/ALL_FAIL_LOG_VIBE_0_Random.csv"
VIBE_COMBINED_FILE = "AddNum/ALL_FAIL_LOG_VIBE_0_Add_Random.csv"

RAW_ALIGNMENT_SOURCES = (
    ("ATT", ATT_FILE),
    ("BARO", "BARO/ALL_FAIL_LOG_BARO.csv"),
    ("BAT", "BAT/ALL_FAIL_LOG_BAT_0.csv"),
    ("CTUN", "CTUN/ALL_FAIL_LOG_CTUN.csv"),
    ("MAG", MAG_FILE),
    ("MOTB", "MOTB/ALL_FAIL_LOG_MOTB.csv"),
    ("PSCD", "PSCD/ALL_FAIL_LOG_PSCD.csv"),
    ("RATE", RATE_FILE),
    ("VIBE", VIBE_FILE),
    ("XKF1", "XKF1/ALL_FAIL_LOG_XKF1_0_Random.csv"),
)

AUGMENTATION_COMPARISONS = (
    (
        "GPS",
        GPS_FILE,
        GPS_COMBINED_FILE,
    ),
    (
        "RATE",
        RATE_FILE,
        RATE_COMBINED_FILE,
    ),
    (
        "VIBE",
        VIBE_FILE,
        VIBE_COMBINED_FILE,
    ),
)

FUSION_INTERPOLATION_MAP = (
    ("timestamp", ATT_FILE, "TimeUS"),
    ("DesRoll", ATT_FILE, "DesRoll"),
    ("Roll", ATT_FILE, "Roll"),
    ("DesPitch", ATT_FILE, "DesPitch"),
    ("Pitch", ATT_FILE, "Pitch"),
    ("DesYaw", ATT_FILE, "DesYaw"),
    ("Yaw", ATT_FILE, "Yaw"),
    ("ErrRP", ATT_FILE, "ErrRP"),
    ("ErrYaw", ATT_FILE, "ErrYaw"),
    ("ErrYaw", ATT_FILE, "ErrRP"),
    ("MagX", MAG_FILE, "MagX"),
    ("MagY", MAG_FILE, "MagY"),
    ("MagZ", MAG_FILE, "MagZ"),
    ("MagZ", MAG_FILE, "MagY"),
)


@dataclass
class DatasetAudit:
    """All tables and checks produced by one immutable dataset snapshot."""

    project_root: Path
    dataset_dir: Path
    manifest: dict[str, str]
    frames: dict[str, pd.DataFrame]
    inventory: pd.DataFrame
    schema: pd.DataFrame
    class_distribution: pd.DataFrame
    temporal_summary: pd.DataFrame
    episodes: pd.DataFrame
    duplicate_columns: pd.DataFrame
    raw_label_summary: dict[str, Any]
    raw_label_disagreements: pd.DataFrame
    fusion_reconstruction: pd.DataFrame
    fusion_label_alignment: pd.DataFrame
    fusion_summary: dict[str, Any]
    flight_summary: pd.DataFrame
    augmentation_overlap: pd.DataFrame
    split_diagnostics: dict[str, Any]
    checks: pd.Series


def resolve_dataset_dir(project_root: Path | str | None = None) -> Path:
    """Resolve and validate the project-local dataset directory."""

    root = (
        Path.cwd().resolve()
        if project_root is None
        else Path(project_root).resolve()
    )
    dataset_dir = root / "dataset"
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"Dataset directory not found: {dataset_dir}")
    if not (dataset_dir / FUSION_FILE).is_file():
        raise FileNotFoundError(f"Missing required file: {dataset_dir / FUSION_FILE}")
    return dataset_dir


def dataset_manifest(dataset_dir: Path) -> dict[str, str]:
    """Return deterministic SHA-256 hashes without modifying source files."""

    return {
        path.relative_to(dataset_dir).as_posix(): sha256(path.read_bytes()).hexdigest()
        for path in sorted(dataset_dir.rglob("*.csv"))
    }


def load_frames(dataset_dir: Path) -> dict[str, pd.DataFrame]:
    """Load every CSV under a stable relative path key."""

    frames = {
        path.relative_to(dataset_dir).as_posix(): pd.read_csv(path)
        for path in sorted(dataset_dir.rglob("*.csv"))
    }
    if not frames:
        raise ValueError(f"No CSV files found under {dataset_dir}")
    return frames


def find_label_column(frame: pd.DataFrame) -> str | None:
    """Find the known target spelling while preserving the original schema."""

    matches = [column for column in frame.columns if "lab" in column.lower()]
    return matches[0] if matches else None


def find_time_column(frame: pd.DataFrame) -> str | None:
    """Find the first explicit time-like column."""

    matches = [column for column in frame.columns if "time" in column.lower()]
    return matches[0] if matches else None


def find_line_column(frame: pd.DataFrame) -> str | None:
    """Find LineNo despite capitalization differences."""

    return next(
        (column for column in frame.columns if column.lower() == "lineno"),
        None,
    )


def build_inventory(
    dataset_dir: Path,
    frames: dict[str, pd.DataFrame],
    manifest: dict[str, str],
) -> pd.DataFrame:
    """Summarize file shape, quality, labels, and temporal key behavior."""

    rows: list[dict[str, Any]] = []
    for relative_path, frame in frames.items():
        label_column = find_label_column(frame)
        time_column = find_time_column(frame)
        time_duplicates = 0
        label_conflicts = 0
        monotonic_time: bool | None = None
        if time_column is not None:
            time_duplicates = int(frame[time_column].duplicated().sum())
            monotonic_time = bool(frame[time_column].is_monotonic_increasing)
            if label_column is not None:
                conflicts = frame.groupby(time_column)[label_column].nunique(
                    dropna=False
                )
                label_conflicts = int((conflicts > 1).sum())
        rows.append(
            {
                "file": relative_path,
                "bytes": (dataset_dir / relative_path).stat().st_size,
                "sha256": manifest[relative_path],
                "rows": len(frame),
                "columns": frame.shape[1],
                "missing_cells": int(frame.isna().sum().sum()),
                "duplicate_rows": int(frame.duplicated().sum()),
                "label_column": label_column,
                "time_column": time_column,
                "line_column": find_line_column(frame),
                "time_is_monotonic": monotonic_time,
                "duplicate_timestamps": time_duplicates,
                "timestamp_label_conflicts": label_conflicts,
            }
        )
    return pd.DataFrame(rows).sort_values("file").reset_index(drop=True)


def build_schema(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return one row per source column."""

    rows: list[dict[str, Any]] = []
    for relative_path, frame in frames.items():
        for column in frame.columns:
            series = frame[column]
            rows.append(
                {
                    "file": relative_path,
                    "column": column,
                    "dtype": str(series.dtype),
                    "non_null": int(series.notna().sum()),
                    "unique": int(series.nunique(dropna=False)),
                    "constant": bool(series.nunique(dropna=False) == 1),
                }
            )
    return pd.DataFrame(rows)


def build_class_distribution(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Count target values independently for every source."""

    rows: list[dict[str, Any]] = []
    for relative_path, frame in frames.items():
        label_column = find_label_column(frame)
        if label_column is None:
            continue
        counts = frame[label_column].value_counts(dropna=False).sort_index()
        for label, count in counts.items():
            label_int = int(label)
            rows.append(
                {
                    "file": relative_path,
                    "label_column": label_column,
                    "label": label_int,
                    "class_name": CLASS_NAMES[label_int],
                    "count": int(count),
                    "fraction": float(count / len(frame)),
                }
            )
    return pd.DataFrame(rows)


def _transition_count(values: pd.Series | np.ndarray) -> int:
    array = np.asarray(values)
    return int(np.count_nonzero(array[1:] != array[:-1])) if len(array) > 1 else 0


def build_temporal_summary(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Describe order, sampling gaps, and target transitions per file."""

    rows: list[dict[str, Any]] = []
    for relative_path, frame in frames.items():
        time_column = find_time_column(frame)
        if time_column is None:
            continue
        label_column = find_label_column(frame)
        time = pd.to_numeric(frame[time_column], errors="coerce")
        delta = time.diff()
        sorted_index = time.sort_values(kind="stable").index
        rows.append(
            {
                "file": relative_path,
                "time_column": time_column,
                "time_min": float(time.min()),
                "time_max": float(time.max()),
                "unique_timestamps": int(time.nunique()),
                "duplicate_timestamps": int(time.duplicated().sum()),
                "negative_deltas_in_row_order": int((delta < 0).sum()),
                "median_positive_delta": (
                    float(delta[delta > 0].median()) if (delta > 0).any() else np.nan
                ),
                "label_transitions_in_row_order": (
                    _transition_count(frame[label_column])
                    if label_column is not None
                    else np.nan
                ),
                "label_transitions_after_time_sort": (
                    _transition_count(frame.loc[sorted_index, label_column])
                    if label_column is not None
                    else np.nan
                ),
            }
        )
    return pd.DataFrame(rows).sort_values("file").reset_index(drop=True)


def _label_runs(frame: pd.DataFrame, source: str) -> list[dict[str, Any]]:
    label_column = find_label_column(frame)
    time_column = find_time_column(frame)
    if label_column is None or time_column is None:
        return []
    labels = frame[label_column].to_numpy()
    starts = np.r_[0, np.flatnonzero(labels[1:] != labels[:-1]) + 1]
    ends = np.r_[starts[1:], len(frame)]
    return [
        {
            "source": source,
            "episode": episode,
            "label": int(labels[start]),
            "class_name": CLASS_NAMES[int(labels[start])],
            "start_row": int(start),
            "end_row": int(end - 1),
            "rows": int(end - start),
            "start_time": float(frame.iloc[start][time_column]),
            "end_time": float(frame.iloc[end - 1][time_column]),
        }
        for episode, (start, end) in enumerate(zip(starts, ends))
    ]


def build_episode_table(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Expose contiguous episodes in the clean ATT source and fusion table."""

    rows = _label_runs(frames[ATT_FILE], "ATT")
    rows.extend(_label_runs(frames[FUSION_FILE], "Fusion_Data"))
    return pd.DataFrame(rows)


def find_exact_duplicate_columns(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Find exact same-valued columns, distinguishing constant coincidences."""

    rows: list[dict[str, Any]] = []
    for relative_path, frame in frames.items():
        for left, right in combinations(frame.columns, 2):
            if frame[left].equals(frame[right]):
                rows.append(
                    {
                        "file": relative_path,
                        "left": left,
                        "right": right,
                        "nonconstant": bool(frame[left].nunique(dropna=False) > 1),
                    }
                )
    return pd.DataFrame(rows, columns=["file", "left", "right", "nonconstant"])


def audit_raw_label_alignment(
    frames: dict[str, pd.DataFrame],
) -> tuple[dict[str, Any], pd.DataFrame]:
    """Align first-flight raw sources on exact TimeUS and compare targets."""

    aligned: pd.DataFrame | None = None
    source_names: list[str] = []
    for source_name, relative_path in RAW_ALIGNMENT_SOURCES:
        frame = frames[relative_path]
        label_column = find_label_column(frame)
        if label_column is None or "TimeUS" not in frame:
            raise ValueError(f"Cannot align labels for {relative_path}")
        selected = frame[["TimeUS", label_column]].rename(
            columns={label_column: source_name}
        )
        aligned = (
            selected
            if aligned is None
            else aligned.merge(
                selected,
                on="TimeUS",
                how="inner",
                validate="one_to_one",
            )
        )
        source_names.append(source_name)
    if aligned is None:
        raise AssertionError("No raw sources were aligned")
    matrix = aligned[source_names].to_numpy()
    agrees = np.all(matrix == matrix[:, [0]], axis=1)
    summary = {
        "sources": source_names,
        "common_timestamps": int(len(aligned)),
        "all_sources_agree": int(agrees.sum()),
        "disagreement_timestamps": int((~agrees).sum()),
        "agreement_rate": float(agrees.mean()),
    }
    return summary, aligned.loc[~agrees].reset_index(drop=True)


def interpolate_three(values: pd.Series | np.ndarray) -> np.ndarray:
    """Reproduce the inclusive three-point interpolation in the source code."""

    array = np.asarray(values, dtype=float)
    if len(array) < 2:
        raise ValueError("At least two samples are required for interpolation")
    result = np.empty(3 * (len(array) - 1), dtype=float)
    result[0::3] = array[:-1]
    result[1::3] = (array[:-1] + array[1:]) / 2
    result[2::3] = array[1:]
    return result


def audit_fusion(
    frames: dict[str, pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Reconstruct fusion columns and compare fused labels with ATT time."""

    fusion = frames[FUSION_FILE]
    rows: list[dict[str, Any]] = []
    for fusion_column, source_file, source_column in FUSION_INTERPOLATION_MAP:
        expected = interpolate_three(frames[source_file][source_column])[: len(fusion)]
        actual = fusion[fusion_column].to_numpy(dtype=float)
        error = np.abs(actual - expected)
        rows.append(
            {
                "fusion_column": fusion_column,
                "source_file": source_file,
                "source_column": source_column,
                "rows_within_1e-9": int((error <= 1e-9).sum()),
                "rows": len(fusion),
                "max_absolute_error": float(error.max()),
                "mean_absolute_error": float(error.mean()),
            }
        )
    reconstruction = pd.DataFrame(rows)

    fusion_for_merge = fusion.copy()
    fusion_for_merge["timestamp"] = fusion_for_merge["timestamp"].astype("int64")
    aligned = pd.merge_asof(
        fusion_for_merge.sort_values("timestamp", kind="stable"),
        frames[ATT_FILE][["TimeUS", "lables"]].sort_values("TimeUS"),
        left_on="timestamp",
        right_on="TimeUS",
        direction="nearest",
    )
    label_alignment = pd.crosstab(
        aligned["labels"],
        aligned["lables"],
        rownames=["fusion_label"],
        colnames=["nearest_ATT_label"],
    )
    duplicate_pairs = {
        tuple(sorted((row["left"], row["right"])))
        for row in find_exact_duplicate_columns(
            {FUSION_FILE: fusion}
        ).to_dict("records")
    }
    class_four_mask = aligned["labels"] == 4

    imu_columns = [
        "abGyrX",
        "abGyrY",
        "abGyrZ",
        "abAccX",
        "abAccY",
        "abAccZ",
    ]
    fusion_imu = fusion[imu_columns].round(6).reset_index(names="fusion_row")
    raw_imu = frames[IMU_COMBINED_FILE][
        imu_columns + ["abTimeUS", "LineNo"]
    ].copy()
    raw_imu[imu_columns] = raw_imu[imu_columns].round(6)
    candidate_counts = (
        raw_imu.groupby(imu_columns, dropna=False)
        .size()
        .rename("candidate_count")
        .reset_index()
    )
    unique_raw_imu = raw_imu.merge(candidate_counts, on=imu_columns).loc[
        lambda table: table["candidate_count"] == 1
    ]
    mapped_imu = (
        fusion_imu.merge(unique_raw_imu, on=imu_columns, how="inner")
        .sort_values("fusion_row")
        .reset_index(drop=True)
    )
    summary = {
        "shape": tuple(fusion.shape),
        "source_ATT_rows": len(frames[ATT_FILE]),
        "ATT_rows_consumed_by_truncation": int((len(fusion) + 2) / 3),
        "fusion_timestamp_max": float(fusion["timestamp"].max()),
        "ATT_class_four_start_time": float(
            frames[ATT_FILE].loc[frames[ATT_FILE]["lables"] == 4, "TimeUS"].min()
        ),
        "class_four_rows": int(class_four_mask.sum()),
        "class_four_rows_nearest_ATT_normal": int(
            (aligned.loc[class_four_mask, "lables"] == 0).sum()
        ),
        "uniquely_mapped_IMU_rows": int(len(mapped_imu)),
        "mapped_IMU_time_min": int(mapped_imu["abTimeUS"].min()),
        "mapped_IMU_time_max": int(mapped_imu["abTimeUS"].max()),
        "mapped_IMU_time_is_monotonic": bool(
            mapped_imu["abTimeUS"].is_monotonic_increasing
        ),
        "exact_nonconstant_duplicate_pairs": sorted(duplicate_pairs),
    }
    return reconstruction, label_alignment, summary


def audit_gps_flights(
    frames: dict[str, pd.DataFrame],
    gap_threshold_ms: int = 600_000,
) -> pd.DataFrame:
    """Recover GPS recording groups from absolute GPS week and milliseconds."""

    frame = frames[GPS_COMBINED_FILE].copy()
    frame["gps_absolute_ms"] = (
        frame["GWk"].astype("int64") * 604_800_000 + frame["GMS"].astype("int64")
    )
    ordered = frame.sort_values("gps_absolute_ms", kind="stable").reset_index(drop=True)
    group = ordered["gps_absolute_ms"].diff().fillna(0).gt(gap_threshold_ms).cumsum()
    rows: list[dict[str, Any]] = []
    for flight_id, part in ordered.groupby(group, sort=True):
        counts = part["labels"].value_counts().sort_index()
        row: dict[str, Any] = {
            "inferred_flight": int(flight_id),
            "rows": len(part),
            "duration_seconds": float(
                (part["gps_absolute_ms"].max() - part["gps_absolute_ms"].min()) / 1000
            ),
            "gps_week": ",".join(map(str, sorted(part["GWk"].unique()))),
            "gps_absolute_start_ms": int(part["gps_absolute_ms"].min()),
            "gps_absolute_end_ms": int(part["gps_absolute_ms"].max()),
            "TimeUS_min": int(part["TimeUS"].min()),
            "TimeUS_max": int(part["TimeUS"].max()),
        }
        row.update({f"label_{label}": int(counts.get(label, 0)) for label in range(5)})
        rows.append(row)
    return pd.DataFrame(rows)


def _rows_present(reference: pd.DataFrame, candidate: pd.DataFrame) -> int:
    reference_hash = pd.util.hash_pandas_object(reference, index=False)
    candidate_hash = set(pd.util.hash_pandas_object(candidate, index=False).tolist())
    return int(reference_hash.isin(candidate_hash).sum())


def audit_augmentation_overlap(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Measure how combined files contain their first-flight counterparts."""

    rows: list[dict[str, Any]] = []
    for source, original_file, combined_file in AUGMENTATION_COMPARISONS:
        original = frames[original_file]
        combined = frames[combined_file]
        common = [
            column
            for column in original
            if column in combined and column != "labels.1"
        ]
        non_target = [column for column in common if "lab" not in column.lower()]
        time_column = find_time_column(combined)
        label_column = find_label_column(combined)
        conflict_count = 0
        if time_column is not None and label_column is not None:
            conflicts = combined.groupby(time_column)[label_column].nunique(
                dropna=False
            )
            conflict_count = int((conflicts > 1).sum())
        rows.append(
            {
                "source": source,
                "original_file": original_file,
                "combined_file": combined_file,
                "original_rows": len(original),
                "combined_rows": len(combined),
                "row_count_difference": len(combined) - len(original),
                "original_rows_present_with_label": _rows_present(
                    original[common], combined[common]
                ),
                "original_rows_present_without_label": _rows_present(
                    original[non_target], combined[non_target]
                ),
                "timestamp_groups_with_label_conflicts": conflict_count,
            }
        )
    return pd.DataFrame(rows)


def _class_counts(labels: np.ndarray, indices: np.ndarray) -> dict[str, int]:
    values, counts = np.unique(labels[indices], return_counts=True)
    return {str(int(value)): int(count) for value, count in zip(values, counts)}


def _nearest_index_distance(query: np.ndarray, reference: np.ndarray) -> np.ndarray:
    reference = np.sort(reference)
    positions = np.searchsorted(reference, query)
    left_positions = np.maximum(positions - 1, 0)
    right_positions = np.minimum(positions, len(reference) - 1)
    left = np.where(positions > 0, np.abs(query - reference[left_positions]), np.inf)
    right = np.where(
        positions < len(reference),
        np.abs(query - reference[right_positions]),
        np.inf,
    )
    return np.minimum(left, right)


def audit_split_protocols(
    fusion: pd.DataFrame,
    seed: int = 42,
) -> dict[str, Any]:
    """Quantify contamination and class support for two naive split choices."""

    indices = np.arange(len(fusion))
    labels = fusion["labels"].to_numpy()
    train_indices, temporary_indices = train_test_split(
        indices,
        test_size=0.30,
        random_state=seed,
        stratify=labels,
    )
    valid_indices, test_indices = train_test_split(
        temporary_indices,
        test_size=0.50,
        random_state=seed,
        stratify=labels[temporary_indices],
    )

    def timestamp_overlap(left: np.ndarray, right: np.ndarray) -> int:
        left_values = set(fusion.iloc[left]["timestamp"])
        right_values = set(fusion.iloc[right]["timestamp"])
        return len(left_values & right_values)

    train_timestamps = set(fusion.iloc[train_indices]["timestamp"])
    distances = _nearest_index_distance(test_indices, train_indices)
    random_summary = {
        "sizes": {
            "train": len(train_indices),
            "valid": len(valid_indices),
            "test": len(test_indices),
        },
        "class_counts": {
            "train": _class_counts(labels, train_indices),
            "valid": _class_counts(labels, valid_indices),
            "test": _class_counts(labels, test_indices),
        },
        "shared_unique_timestamps": {
            "train_valid": timestamp_overlap(train_indices, valid_indices),
            "train_test": timestamp_overlap(train_indices, test_indices),
            "valid_test": timestamp_overlap(valid_indices, test_indices),
        },
        "test_rows_with_timestamp_in_train": int(
            fusion.iloc[test_indices]["timestamp"].isin(train_timestamps).sum()
        ),
        "test_nearest_train_row_distance": {
            "median": float(np.median(distances)),
            "p90": float(np.quantile(distances, 0.90)),
            "p99": float(np.quantile(distances, 0.99)),
            "max": int(distances.max()),
            "adjacent_fraction": float(np.mean(distances == 1)),
            "within_two_rows_fraction": float(np.mean(distances <= 2)),
        },
    }

    train_end = int(np.floor(0.70 * len(fusion)))
    valid_end = int(np.floor(0.85 * len(fusion)))
    chronological_indices = {
        "train": np.arange(0, train_end),
        "valid": np.arange(train_end, valid_end),
        "test": np.arange(valid_end, len(fusion)),
    }
    chronological_summary = {
        name: {
            "size": len(split_indices),
            "class_counts": _class_counts(labels, split_indices),
            "time_min": float(fusion.iloc[split_indices]["timestamp"].min()),
            "time_max": float(fusion.iloc[split_indices]["timestamp"].max()),
        }
        for name, split_indices in chronological_indices.items()
    }
    return {
        "seed": seed,
        "random_stratified": random_summary,
        "chronological_70_15_15": chronological_summary,
    }


def validate_audit(audit: DatasetAudit, manifest_after: dict[str, str]) -> pd.Series:
    """Fail loudly when the audited local release violates core expectations."""

    fusion = audit.frames[FUSION_FILE]
    duplicate_pairs = {
        tuple(sorted((row.left, row.right)))
        for row in audit.duplicate_columns.loc[
            (audit.duplicate_columns["file"] == FUSION_FILE)
            & audit.duplicate_columns["nonconstant"]
        ].itertuples()
    }
    checks = pd.Series(
        {
            "dataset_manifest_unchanged": audit.manifest == manifest_after,
            "seventeen_csv_files_loaded": len(audit.frames) == 17,
            "all_files_have_valid_labels": all(
                set(frame[label].unique()).issubset(set(range(5)))
                for frame in audit.frames.values()
                if (label := find_label_column(frame)) is not None
            ),
            "no_missing_cells": bool((audit.inventory["missing_cells"] == 0).all()),
            "fusion_shape_is_12253_by_19": fusion.shape == (12_253, 19),
            "fusion_timestamp_is_monotonic": bool(
                fusion["timestamp"].is_monotonic_increasing
            ),
            "fusion_interpolation_reproduced": bool(
                (
                    audit.fusion_reconstruction.loc[
                        (audit.fusion_reconstruction["fusion_column"] == "timestamp")
                        & (audit.fusion_reconstruction["source_column"] == "TimeUS"),
                        "rows_within_1e-9",
                    ]
                    == len(fusion)
                ).all()
            ),
            "fusion_duplicate_pairs_confirmed": {
                ("ErrRP", "ErrYaw"),
                ("MagY", "MagZ"),
            }.issubset(duplicate_pairs),
            "two_gps_flights_recovered": len(audit.flight_summary) == 2,
            "raw_label_agreement_above_99_percent": (
                audit.raw_label_summary["agreement_rate"] > 0.99
            ),
            "random_split_timestamp_overlap_detected": (
                audit.split_diagnostics["random_stratified"]
                ["shared_unique_timestamps"]["train_test"]
                > 0
            ),
        },
        name="passed",
        dtype=bool,
    )
    failed = checks.index[~checks].tolist()
    if failed:
        raise AssertionError(f"Audit checks failed: {failed}")
    return checks


def run_full_audit(project_root: Path | str | None = None) -> DatasetAudit:
    """Run the complete read-only audit for the current local release."""

    dataset_dir = resolve_dataset_dir(project_root)
    root = dataset_dir.parent
    manifest_before = dataset_manifest(dataset_dir)
    frames = load_frames(dataset_dir)
    inventory = build_inventory(dataset_dir, frames, manifest_before)
    raw_summary, raw_disagreements = audit_raw_label_alignment(frames)
    reconstruction, fusion_alignment, fusion_summary = audit_fusion(frames)
    audit = DatasetAudit(
        project_root=root,
        dataset_dir=dataset_dir,
        manifest=manifest_before,
        frames=frames,
        inventory=inventory,
        schema=build_schema(frames),
        class_distribution=build_class_distribution(frames),
        temporal_summary=build_temporal_summary(frames),
        episodes=build_episode_table(frames),
        duplicate_columns=find_exact_duplicate_columns(frames),
        raw_label_summary=raw_summary,
        raw_label_disagreements=raw_disagreements,
        fusion_reconstruction=reconstruction,
        fusion_label_alignment=fusion_alignment,
        fusion_summary=fusion_summary,
        flight_summary=audit_gps_flights(frames),
        augmentation_overlap=audit_augmentation_overlap(frames),
        split_diagnostics=audit_split_protocols(frames[FUSION_FILE]),
        checks=pd.Series(dtype=bool),
    )
    audit.checks = validate_audit(audit, dataset_manifest(dataset_dir))
    return audit


def plot_class_distribution(frame: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    """Plot the target distribution of one audited table."""

    label_column = find_label_column(frame)
    if label_column is None:
        raise ValueError("The frame has no recognized label column")
    counts = frame[label_column].value_counts().reindex(range(5), fill_value=0)
    figure, axis = plt.subplots(figsize=(9, 4.5))
    colors = ["#3b82f6", "#ef4444", "#f59e0b", "#10b981", "#8b5cf6"]
    axis.bar(CLASS_NAMES, counts.to_numpy(), color=colors)
    axis.set_ylabel("Rows")
    axis.set_title("Class distribution")
    axis.tick_params(axis="x", rotation=20)
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    return figure, axis


def plot_label_timeline(frame: pd.DataFrame) -> tuple[plt.Figure, plt.Axes]:
    """Plot labels in row order against the explicit time column."""

    label_column = find_label_column(frame)
    time_column = find_time_column(frame)
    if label_column is None or time_column is None:
        raise ValueError("A recognized label and time column are required")
    figure, axis = plt.subplots(figsize=(11, 3.8))
    axis.step(
        frame[time_column].to_numpy() / 1_000_000,
        frame[label_column].to_numpy(),
        where="post",
        color="#2563eb",
        linewidth=1.5,
    )
    axis.set_xlabel("Time (s, source clock)")
    axis.set_ylabel("Label")
    axis.set_yticks(range(5), CLASS_NAMES)
    axis.set_title("Contiguous target episodes")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    return figure, axis
