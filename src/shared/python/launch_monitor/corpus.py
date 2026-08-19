"""Load the private launch-monitor shot corpus as canonical frames.

The D-sorganization data authority (``Launch-Monitor-Flight-Model-Campaign``)
publishes a source-partitioned Parquet corpus of archived launch-monitor
shots at ``data/authority/database/shot_corpus_parquet/``. This module reads
that dataset into the canonical launch-monitor schema (radians, m/s, rad/s,
metres) so corpus shots flow directly into ``flexible_analysis``,
``comparison``, and the analytics workbench alongside user-imported sessions.

Access follows the same convention as ``validation_pkg.kaggle_validation``:
the ``LAUNCH_MONITOR_DATA_ROOT`` environment variable points at an
authorized, commit-pinned checkout of the private repository. There is no
download fallback; the loader fails closed without authorized data.

Reading Parquet requires ``pyarrow`` (the ``data`` extra); the import is
lazy so this module stays importable without it.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

import pandas as pd

from src.shared.python.launch_monitor.importer import _convert

# Corpus native column -> (canonical metric name, source unit). The corpus
# stores source-native imperial units. ``apex_native`` is excluded: its unit
# varies by source, so it cannot be converted safely.
CORPUS_COLUMN_MAP: dict[str, tuple[str, str]] = {
    "club_speed_mph": ("club_speed", "mph"),
    "ball_speed_mph": ("ball_speed", "mph"),
    "smash_factor": ("smash_factor", "1"),
    "launch_angle_deg": ("launch_angle", "deg"),
    "launch_direction_deg": ("launch_direction", "deg"),
    "spin_rate_rpm": ("spin_rate", "rpm"),
    "back_spin_rpm": ("back_spin", "rpm"),
    "side_spin_rpm": ("side_spin", "rpm"),
    "spin_axis_deg": ("spin_axis", "deg"),
    "attack_angle_deg": ("attack_angle", "deg"),
    "club_path_deg": ("club_path", "deg"),
    "face_angle_deg": ("face_angle", "deg"),
    "carry_yd": ("carry_distance", "yd"),
    "total_yd": ("total_distance", "yd"),
    "descent_angle_deg": ("descent_angle", "deg"),
    "lateral_carry_yd": ("lateral_carry", "yd"),
    "flight_time_s": ("flight_time", "s"),
}

# Identity columns carried straight through when the corpus provides them.
# captured_at is what the Trends analysis binds to; a corpus built before the
# data authority added it simply lacks the column.
OPTIONAL_IDENTITY_COLUMNS: tuple[str, ...] = ("captured_at",)


def corpus_dataset_path(root: str | Path | None = None) -> Path:
    """Resolve the Parquet corpus path inside the private checkout."""
    resolved = root if root is not None else os.environ.get("LAUNCH_MONITOR_DATA_ROOT")
    if not resolved:
        raise FileNotFoundError(
            "private launch-monitor authority is unavailable; set "
            "LAUNCH_MONITOR_DATA_ROOT to an authorized, commit-pinned "
            "Launch-Monitor-Flight-Model-Campaign checkout"
        )
    return Path(resolved) / "data" / "authority" / "database" / "shot_corpus_parquet"


def _selected_column_map(metrics: list[str] | None) -> dict[str, tuple[str, str]]:
    """Narrow the corpus column map to an optional canonical-metric allowlist."""
    if metrics is None:
        return dict(CORPUS_COLUMN_MAP)
    unknown = set(metrics) - {name for name, _ in CORPUS_COLUMN_MAP.values()}
    if unknown:
        raise ValueError(f"Unknown corpus metrics requested: {sorted(unknown)}")
    return {
        column: (name, unit)
        for column, (name, unit) in CORPUS_COLUMN_MAP.items()
        if name in metrics
    }


def _source_filter(
    pyarrow_dataset: Any, dataset_dir: Path, sources: list[str] | None
) -> Any:
    """Build the partition filter for a ``source_id`` allowlist, if any."""
    if sources is None:
        return None
    available = {
        entry.name.split("=", 1)[1]
        for entry in dataset_dir.iterdir()
        if entry.is_dir() and entry.name.startswith("source_id=")
    }
    unknown = set(sources) - available
    if unknown:
        raise ValueError(f"Unknown corpus sources requested: {sorted(unknown)}")
    return pyarrow_dataset.field("source_id").isin(sources)


def _canonicalize_metrics(
    frame: pd.DataFrame, selected_map: dict[str, tuple[str, str]]
) -> pd.DataFrame:
    """Convert native corpus columns to canonical units and metric names."""
    from src.shared.python.launch_monitor.schema import METRICS

    present = {
        column: value
        for column, value in selected_map.items()
        if column in frame.columns
    }
    for column, (name, unit) in present.items():
        frame[column] = _convert(frame[column], unit, METRICS[name].canonical_unit)
    return frame.rename(columns={column: name for column, (name, _) in present.items()})


def _apply_identity(frame: pd.DataFrame) -> pd.DataFrame:
    """Derive ``shot_id`` and rename corpus identity columns to the schema."""
    identity = (
        frame["source_id"].astype(str)
        + "\x1f"
        + frame["file"].astype(str)
        + "\x1f"
        + frame["row_index"].astype(str)
    )
    frame["shot_id"] = identity.map(
        lambda value: hashlib.sha256(value.encode()).hexdigest()[:20]
    )
    return frame.rename(
        columns={
            "source_id": "session_id",
            "monitor": "monitor_vendor",
            "row_index": "source_row",
        }
    ).drop(columns=["file"])


def load_private_corpus(
    root: str | Path | None = None,
    sources: list[str] | None = None,
    metrics: list[str] | None = None,
) -> pd.DataFrame:
    """Load corpus shots as one canonical-schema DataFrame.

    Args:
        root: Private checkout root; defaults to ``LAUNCH_MONITOR_DATA_ROOT``.
        sources: Optional ``source_id`` allowlist; ``None`` loads everything.
        metrics: Optional canonical metric-name allowlist; pruning is pushed
            down to the Parquet reader.

    Returns:
        DataFrame with canonical metric columns, identity columns
        (``shot_id``, ``session_id`` carrying the corpus ``source_id``,
        ``source_row``, ``monitor_vendor``, ``club``), and
        ``observation_kind`` fixed to ``"shot"``.
    """
    try:
        import pyarrow.dataset as pyarrow_dataset
    except ImportError as exc:  # pragma: no cover - environment-dependent
        raise ImportError(
            "loading the private corpus requires pyarrow; install the "
            "'data' extra: pip install 'upstream-drift[data]'"
        ) from exc

    dataset_dir = corpus_dataset_path(root)
    if not dataset_dir.is_dir():
        raise FileNotFoundError(
            f"shot corpus dataset not found at {dataset_dir}; the checkout "
            "may predate the Parquet corpus - sync it to a newer commit"
        )

    selected_map = _selected_column_map(metrics)
    dataset = pyarrow_dataset.dataset(
        dataset_dir, format="parquet", partitioning="hive"
    )
    # A corpus pinned before a column was introduced simply lacks it; select
    # what the dataset actually has rather than failing the whole read.
    available_columns = set(dataset.schema.names)
    requested = [
        "source_id",
        "monitor",
        "club",
        "file",
        "row_index",
        *OPTIONAL_IDENTITY_COLUMNS,
        *selected_map,
    ]
    table = dataset.to_table(
        columns=[name for name in requested if name in available_columns],
        filter=_source_filter(pyarrow_dataset, dataset_dir, sources),
    )

    frame = _canonicalize_metrics(table.to_pandas(), selected_map)
    frame = _apply_identity(frame)
    frame["observation_kind"] = "shot"
    return frame
