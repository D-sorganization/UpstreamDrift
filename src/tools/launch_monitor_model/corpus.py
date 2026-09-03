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

Manifest validation (ADR-0048 D30)
----------------------------------
The corpus directory carries a ``_MANIFEST.json`` the data authority
publishes alongside the partitions. This loader **refuses to read a corpus
that manifest does not describe**, and there is no flag to skip the gate: an
unvalidated corpus is not a corpus.

Five fail-closed checks run before a single row is materialised, each
refusing by name:

1. ``missing_manifest`` - no ``_MANIFEST.json`` in the corpus directory
   (:class:`FileNotFoundError`).
2. ``unsupported_schema`` - ``schema_version`` is not
   ``SUPPORTED_MANIFEST_SCHEMA_VERSION``, or ``sources`` is not an object.
3. ``row_cap_exceeded`` - the declared ``total_rows`` is negative or above the
   ``MAX_RETAINED_ROWS`` desktop retained-data limit.
4. ``row_count_mismatch`` - the declared ``total_rows`` disagrees with the rows
   on disk.
5. ``source_set_mismatch`` - the declared ``sources`` disagree with the
   ``source_id=`` partitions on disk.

The check basis is the **whole corpus**, never the caller's selection: the
observed row count comes from ``dataset.count_rows()`` on the *unfiltered*
dataset and the observed source set from the hive partition directory names,
so a ``sources``/``metrics`` pushdown (D31) can neither weaken nor invalidate
the guarantee. The guarantee is about the bytes the authority published, not
about the slice this call happens to want.

This gate closes governance hole **D30** of the ADR-0046 G0.1 cross-stack
measurement (``tests/integration/launch_monitor_drift/test_corpus_drift.py``),
which pinned five corpora that ``rate_of_closure.launch_monitor_private_corpus``
refused and this loader accepted silently, returning the same four rows in all
five. Its canonical resolution is Tools#4907's P19 merge, landed as Tools
``src/shared/python/launch_monitor/corpus.py``; that module is the reference
for the checks, their messages, their order and their check basis, and this
module reimplements them rather than importing it - the vendored Tools package
is a *measurement* dependency of the drift gates, never a runtime dependency of
UpstreamDrift. ``MAX_RETAINED_ROWS`` is likewise redefined here rather than
imported, and pinned equal to the vendored constant by the drift gate.

What is unchanged: the ADR-0031 unit canonicalisation (D29), the 20-hex shot
identity digest, and the source/metric selection pushdown (D31) all keep their
existing behaviour and run *after* the gate, on what survives it.

Reading Parquet requires ``pyarrow`` (the ``data`` extra); the import is
lazy so this module stays importable without it.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from src.tools.launch_monitor_model.importer import _convert

MANIFEST_FILENAME = "_MANIFEST.json"
SUPPORTED_MANIFEST_SCHEMA_VERSION = 1

# The desktop retained-data limit. Redefined here rather than imported from the
# vendored ``rate_of_closure.launch_monitor_linked_scatter``: the vendored Tools
# package is a measurement dependency of the drift gates, not a runtime
# dependency of this package. The drift gate pins the two values equal.
MAX_RETAINED_ROWS = 300_000

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


@dataclass(frozen=True)
class CorpusManifest:
    """The authority's published description of one corpus snapshot."""

    schema_version: int
    sources: dict[str, Any]
    total_rows: int
    manifest_sha256: str

    @property
    def source_count(self) -> int:
        """Return how many source partitions the manifest declares."""
        return len(self.sources)


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


def read_corpus_manifest(dataset_dir: Path) -> CorpusManifest:
    """Read and schema-check the corpus manifest, content-addressing its bytes.

    Refusals 1 (``missing_manifest``) and 2 (``unsupported_schema``) of the
    ADR-0048 D30 gate.

    Args:
        dataset_dir: The ``shot_corpus_parquet`` directory holding the corpus.

    Returns:
        The parsed :class:`CorpusManifest`, carrying the SHA-256 of the exact
        manifest bytes read.

    Raises:
        FileNotFoundError: The corpus directory carries no manifest.
        ValueError: The manifest declares an unsupported schema.
    """
    manifest_path = dataset_dir / MANIFEST_FILENAME
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f"Private corpus manifest not found at {manifest_path}; the loader "
            "refuses an unvalidated corpus (ADR-0048 D30)"
        )
    manifest_bytes = manifest_path.read_bytes()
    payload = json.loads(manifest_bytes)
    if payload.get("schema_version") != SUPPORTED_MANIFEST_SCHEMA_VERSION or not (
        isinstance(payload.get("sources"), dict)
    ):
        raise ValueError("Private corpus manifest schema is unsupported")
    return CorpusManifest(
        schema_version=int(payload["schema_version"]),
        sources=dict(payload["sources"]),
        total_rows=int(payload.get("total_rows", -1)),
        manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),
    )


def validate_corpus_manifest(
    manifest: CorpusManifest,
    *,
    observed_rows: int,
    observed_sources: set[str],
) -> None:
    """Refuse a corpus whose bytes disagree with the manifest that describes it.

    Refusals 3 (``row_cap_exceeded``), 4 (``row_count_mismatch``) and 5
    (``source_set_mismatch``) of the ADR-0048 D30 gate, in that order.

    ``observed_rows`` and ``observed_sources`` must describe the **whole**
    dataset, before any ``sources``/``metrics`` selection is applied - the
    guarantee is about the corpus on disk, not about the caller's slice of it.

    Args:
        manifest: The manifest read by :func:`read_corpus_manifest`.
        observed_rows: Rows in the unfiltered dataset.
        observed_sources: ``source_id`` values the partition tree holds.

    Raises:
        ValueError: The declared row count is outside the desktop retained-data
            limit, disagrees with the rows on disk, or the declared source set
            disagrees with the partitions on disk.
    """
    if not 0 <= manifest.total_rows <= MAX_RETAINED_ROWS:
        raise ValueError(
            "Private corpus manifest row count is outside the desktop retained-"
            f"data limit of {MAX_RETAINED_ROWS}"
        )
    if observed_rows != manifest.total_rows:
        raise ValueError(
            f"Private corpus row count mismatch: expected {manifest.total_rows}, "
            f"loaded {observed_rows}"
        )
    if observed_sources != set(manifest.sources):
        raise ValueError("Private corpus source IDs do not match the manifest")


def _partition_source_ids(dataset_dir: Path) -> set[str]:
    """Return the ``source_id`` values the hive partition tree actually holds."""
    return {
        entry.name.split("=", 1)[1]
        for entry in dataset_dir.iterdir()
        if entry.is_dir() and entry.name.startswith("source_id=")
    }


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
    pyarrow_dataset: Any, available: set[str], sources: list[str] | None
) -> Any:
    """Build the partition filter for a ``source_id`` allowlist, if any."""
    if sources is None:
        return None
    unknown = set(sources) - available
    if unknown:
        raise ValueError(f"Unknown corpus sources requested: {sorted(unknown)}")
    return pyarrow_dataset.field("source_id").isin(sources)


def _canonicalize_metrics(
    frame: pd.DataFrame, selected_map: dict[str, tuple[str, str]]
) -> pd.DataFrame:
    """Convert native corpus columns to canonical units and metric names."""
    from src.tools.launch_monitor_model.schema import METRICS

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
    """Load validated corpus shots as one canonical-schema DataFrame.

    The ADR-0048 D30 manifest gate runs **first and always**, against the
    unfiltered dataset, and only then is the caller's selection applied and the
    surviving rows canonicalised into the ADR-0031 schema. There is no flag to
    skip it.

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

    Raises:
        FileNotFoundError: No authorized root, no corpus directory, or no
            manifest describing the corpus.
        ValueError: The manifest schema, row cap, row count or source set does
            not describe the corpus on disk, or an unknown source or metric was
            requested.
        ImportError: ``pyarrow`` is not installed.
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

    manifest = read_corpus_manifest(dataset_dir)
    dataset = pyarrow_dataset.dataset(
        dataset_dir, format="parquet", partitioning="hive"
    )
    # The gate's basis is the WHOLE corpus: an unfiltered row count and the
    # partition directory names. A selection is applied only after it passes.
    available_sources = _partition_source_ids(dataset_dir)
    validate_corpus_manifest(
        manifest,
        observed_rows=dataset.count_rows(),
        observed_sources=available_sources,
    )

    selected_map = _selected_column_map(metrics)
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
        filter=_source_filter(pyarrow_dataset, available_sources, sources),
    )

    frame = _canonicalize_metrics(table.to_pandas(), selected_map)
    frame = _apply_identity(frame)
    frame["observation_kind"] = "shot"
    return frame


__all__ = [
    "CORPUS_COLUMN_MAP",
    "MANIFEST_FILENAME",
    "MAX_RETAINED_ROWS",
    "OPTIONAL_IDENTITY_COLUMNS",
    "SUPPORTED_MANIFEST_SCHEMA_VERSION",
    "CorpusManifest",
    "corpus_dataset_path",
    "load_private_corpus",
    "read_corpus_manifest",
    "validate_corpus_manifest",
]
