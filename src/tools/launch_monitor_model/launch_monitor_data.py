"""Import public Launch-Monitor-Data canonical exports as provenance-preserving sessions.

The public ``D-sorganization/Launch-Monitor-Data`` repository publishes two
canonical shapes that are not vendor exports and therefore must not go
through the header-fingerprint vendor profiles (#8365):

* **Shot-level frames** written from ``launch_monitor_data.corpus.load_shots``.
  They carry the corpus identity columns (``source_id``, ``monitor``,
  ``club``, ``file``, ``row_index``, ``captured_at``, ``observation_kind``)
  and either canonical SI metric columns (``club_speed`` in m/s, ...) or the
  source-native columns (``club_speed_mph``, ``carry_yd``, ...). A vendor
  profile would detect the SI frame as a vendor export and convert 45 m/s as
  though it were 45 mph, so units here come only from the column name (native)
  or the canonical registry (SI) — never from a profile default.
* **Aggregate tables** (``upstreamdrift_aggregate_metrics.csv`` /
  ``metric_observations.csv``): one long-format row per published group mean.
  They are pivoted to one row per (source, monitor, model, software,
  environment, cohort, club) observation, stamped ``observation_kind =
  "aggregate"`` so :func:`~shared.python.launch_monitor.flexible_analysis.analyze_variables`
  refuses to fit them at shot level, and every published cell is retained
  verbatim under its metric so nothing is expanded into fabricated shots.

Both paths return an :class:`~shared.python.launch_monitor.schema.ImportedSession`
whose ``session_id`` column carries the corpus ``source_id`` and whose
``shot_id`` is the same 20-hex digest the private corpus loader derives, so a
shot has one identity whichever public surface it arrives through.
"""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal

import numpy as np
import pandas as pd

from shared.python.launch_monitor.corpus import CORPUS_COLUMN_MAP
from shared.python.launch_monitor.importer import import_session
from shared.python.launch_monitor.schema import (
    METRICS,
    ColumnMapping,
    ImportedSession,
    ImportManifest,
    ImportOptions,
)

__all__ = [
    "AGGREGATE_EXPORT_PROFILE_ID",
    "LAUNCH_MONITOR_DATA_PUBLISHER",
    "SHOT_EXPORT_PROFILE_ID",
    "ExportKind",
    "detect_launch_monitor_data_export",
    "import_launch_monitor_data_export",
]

ExportKind = Literal["shots", "aggregates"]

LAUNCH_MONITOR_DATA_PUBLISHER: Final = "Launch-Monitor-Data"
SHOT_EXPORT_PROFILE_ID: Final = "launch_monitor_data_shots"
AGGREGATE_EXPORT_PROFILE_ID: Final = "launch_monitor_data_aggregates"

_SHOT_IDENTITY_COLUMNS: Final = frozenset({"source_id", "monitor", "file", "row_index"})
_SHOT_METRIC_COLUMNS: Final = frozenset(CORPUS_COLUMN_MAP) | frozenset(METRICS)
_OBSERVATION_KINDS: Final = frozenset({"shot", "aggregate"})

# The group an aggregate row describes; every other column is a per-metric
# cell and is retained under ``source::<metric>::<column>``.
_AGGREGATE_GROUP_KEY: Final = (
    "source_id",
    "monitor_vendor",
    "monitor_model",
    "software_version",
    "environment",
    "cohort",
    "club",
)
_AGGREGATE_REQUIRED_COLUMNS: Final = frozenset(
    (
        *_AGGREGATE_GROUP_KEY,
        "observation_id",
        "metric",
        "aggregation_level",
        "observation_kind",
        "sample_count",
        "measurement_status",
        "reported_mean",
        "reported_sd",
        "reported_unit",
        "canonical_mean",
        "canonical_sd",
        "canonical_unit",
    )
)
_SUPPORTED_AGGREGATION_LEVEL: Final = "group_mean"


def detect_launch_monitor_data_export(headers: list[str]) -> ExportKind | None:
    """Classify a header row as a public export shape, or ``None`` if neither.

    Detection is exact-name membership rather than the fuzzy fingerprint the
    vendor profiles use, because the public schema is fixed and a near miss
    is a different file, not a lower-confidence match.
    """
    if not headers:
        raise ValueError("headers must contain at least one column")
    names = {str(header) for header in headers}
    if names >= _AGGREGATE_REQUIRED_COLUMNS:
        return "aggregates"
    if names >= _SHOT_IDENTITY_COLUMNS and names & _SHOT_METRIC_COLUMNS:
        return "shots"
    return None


def import_launch_monitor_data_export(
    source: str | Path, *, session_name: str | None = None
) -> ImportedSession:
    """Import one public Launch-Monitor-Data CSV/TSV export.

    Raises:
        ValueError: The file is missing, is not one of the two public shapes,
            mixes native and canonical columns for one metric, carries a
            column whose unit would have to be assumed, or (aggregates) holds
            a row that is not a ``group_mean`` aggregate, names an unknown
            metric, reports a unit other than the registry unit, has a
            non-numeric mean, or repeats a metric within one group.
    """
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Launch-Monitor-Data export does not exist: {path}")
    headers = [str(column) for column in _read_csv(path, nrows=0).columns]
    kind = detect_launch_monitor_data_export(headers)
    if kind is None:
        raise ValueError(
            f"{path.name} is not a Launch-Monitor-Data canonical export; import "
            "vendor files with import_session instead"
        )
    if kind == "shots":
        return _import_shots(path, headers, session_name)
    return _import_aggregates(path, headers, session_name)


def _read_csv(path: Path, **kwargs: object) -> pd.DataFrame:
    return pd.read_csv(path, sep=None, engine="python", **kwargs)  # type: ignore[arg-type]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _corpus_shot_id(
    source_id: pd.Series, file: pd.Series, row_index: pd.Series
) -> pd.Series:
    """Mirror ``shared.python.launch_monitor.corpus._apply_identity``."""
    identity = (
        source_id.astype(str)
        + "\x1f"
        + file.astype(str)
        + "\x1f"
        + row_index.astype(str)
    )
    return identity.map(lambda value: hashlib.sha256(value.encode()).hexdigest()[:20])


def _shot_metric_mappings(headers: list[str]) -> tuple[ColumnMapping, ...]:
    """Map every metric column with the unit its name or the registry declares."""
    mappings: list[ColumnMapping] = []
    for header in headers:
        if header in CORPUS_COLUMN_MAP:
            target, unit = CORPUS_COLUMN_MAP[header]
            mappings.append(ColumnMapping(header, target, source_unit=unit))
        elif header in METRICS:
            mappings.append(
                ColumnMapping(
                    header, header, source_unit=METRICS[header].canonical_unit
                )
            )
    targets = [mapping.target_column for mapping in mappings]
    mixed = sorted({target for target in targets if targets.count(target) > 1})
    if mixed:
        raise ValueError(
            f"Export carries both native and canonical columns for {mixed}; "
            "the unit of a metric must be declared once"
        )
    return tuple(mappings)


def _import_shots(
    path: Path, headers: list[str], session_name: str | None
) -> ImportedSession:
    metric_mappings = _shot_metric_mappings(headers)
    identity_mappings = tuple(
        ColumnMapping(source, target)
        for source, target in (
            ("monitor", "monitor_vendor"),
            ("club", "club"),
            ("captured_at", "captured_at"),
        )
        if source in headers
    )
    session = import_session(
        path,
        ImportOptions(
            profile_id="generic",
            mappings=(*identity_mappings, *metric_mappings),
            session_name=session_name,
        ),
    )
    manifest = session.manifest
    declared = {mapping.source_column for mapping in metric_mappings}
    assumed = sorted(set(manifest.metric_sources.values()) - declared)
    if assumed:
        raise ValueError(
            f"Columns {assumed} would be mapped with an assumed unit; a "
            "Launch-Monitor-Data export declares units only through its column "
            "names or the canonical registry. Import the file with a vendor "
            "profile instead."
        )

    shots = session.shots
    shots["session_id"] = shots["source::source_id"].astype(str)
    shots["shot_id"] = _corpus_shot_id(
        shots["source::source_id"], shots["source::file"], shots["source::row_index"]
    )
    kinds = (
        shots["source::observation_kind"].astype(str)
        if "source::observation_kind" in shots
        else pd.Series("shot", index=shots.index)
    )
    invalid = sorted(set(kinds) - _OBSERVATION_KINDS)
    if invalid:
        raise ValueError(f"Unknown observation_kind values in export: {invalid}")
    shots["observation_kind"] = kinds
    return ImportedSession(
        session_id=f"{SHOT_EXPORT_PROFILE_ID}-{manifest.file_sha256[:16]}",
        name=session.name,
        shots=shots,
        manifest=replace(
            manifest,
            profile_id=SHOT_EXPORT_PROFILE_ID,
            vendor=LAUNCH_MONITOR_DATA_PUBLISHER,
        ),
        source_path=path,
        metadata={
            **session.metadata,
            "export_kind": "shots",
            "publisher": LAUNCH_MONITOR_DATA_PUBLISHER,
        },
    )


def _validate_aggregate_rows(frame: pd.DataFrame) -> None:
    """Reject rows the aggregate contract cannot represent without inventing data."""
    fabricated = frame.index[frame["observation_kind"].astype(str) != "aggregate"]
    if len(fabricated):
        raise ValueError(
            "Published aggregate rows must not be expanded into shot-level "
            f"observations; rows {list(fabricated)} report a non-aggregate "
            "observation_kind"
        )
    levels = set(frame["aggregation_level"].astype(str)) - {
        _SUPPORTED_AGGREGATION_LEVEL
    }
    if levels:
        raise ValueError(f"Unsupported aggregation_level values: {sorted(levels)}")
    unknown = sorted(set(frame["metric"].astype(str)) - set(METRICS))
    if unknown:
        raise ValueError(f"Unknown canonical metric in export: {unknown}")
    for metric, group in frame.groupby("metric", sort=True):
        expected = METRICS[str(metric)].canonical_unit
        units = set(group["canonical_unit"].astype(str)) - {expected}
        if units:
            raise ValueError(
                f"Export is unit-incompatible with the registry: {metric} reports "
                f"canonical_unit {sorted(units)} but the canonical unit is "
                f"'{expected}'"
            )
    means = pd.to_numeric(frame["canonical_mean"], errors="coerce")
    if means.isna().any():
        rows = list(frame.index[means.isna()])
        raise ValueError(f"Rows {rows} carry a non-numeric canonical_mean")


def _aggregate_record(
    key: tuple[str, ...], group: pd.DataFrame, retained: list[str]
) -> dict[str, object]:
    identity = dict(zip(_AGGREGATE_GROUP_KEY, key, strict=True))
    record: dict[str, object] = {
        "shot_id": hashlib.sha256("\x1f".join(key).encode()).hexdigest()[:20],
        "session_id": identity["source_id"],
        "source_row": int(group["_source_row"].min()),
        "monitor_vendor": identity["monitor_vendor"],
        "monitor_model": identity["monitor_model"],
        "software_version": identity["software_version"],
        "club": identity["club"],
        "observation_kind": "aggregate",
        "source::environment": identity["environment"],
        "source::cohort": identity["cohort"],
    }
    duplicated = sorted(set(group["metric"][group["metric"].duplicated()]))
    if duplicated:
        raise ValueError(
            f"Metrics {duplicated} are reported more than once for group {key}"
        )
    for _, row in group.sort_values("metric").iterrows():
        metric = str(row["metric"])
        record[metric] = float(row["canonical_mean"])
        record[f"status::{metric}"] = row["measurement_status"]
        for column in retained:
            record[f"source::{metric}::{column}"] = row[column]
    return record


def _import_aggregates(
    path: Path, headers: list[str], session_name: str | None
) -> ImportedSession:
    frame = _read_csv(path)
    if frame.empty:
        raise ValueError(f"Launch-Monitor-Data export contains no rows: {path}")
    _validate_aggregate_rows(frame)
    frame["_source_row"] = np.arange(2, len(frame) + 2)
    for column in _AGGREGATE_GROUP_KEY:
        frame[column] = frame[column].fillna("").astype(str)
    retained = [
        column
        for column in headers
        if column not in _AGGREGATE_GROUP_KEY and column != "metric"
    ]
    records = [
        _aggregate_record(tuple(key), group, retained)
        for key, group in frame.groupby(list(_AGGREGATE_GROUP_KEY), sort=True)
    ]
    wide = pd.DataFrame(records)
    wide["captured_at"] = pd.Series(
        pd.NaT, index=wide.index, dtype="datetime64[ns, UTC]"
    )
    metrics = sorted(set(frame["metric"].astype(str)))
    ordered = [
        *(
            column
            for column in wide.columns
            if "::" not in column and column not in metrics
        ),
        *metrics,
        *sorted(column for column in wide.columns if column.startswith("status::")),
        *sorted(column for column in wide.columns if column.startswith("source::")),
    ]
    digest = _sha256(path)
    manifest = ImportManifest(
        source_path=str(path),
        file_sha256=digest,
        profile_id=AGGREGATE_EXPORT_PROFILE_ID,
        vendor=LAUNCH_MONITOR_DATA_PUBLISHER,
        imported_at=datetime.now(UTC).isoformat(),
        row_count=len(wide),
        source_columns=tuple(headers),
        metric_sources=dict.fromkeys(metrics, "canonical_mean"),
        source_units={metric: METRICS[metric].canonical_unit for metric in metrics},
        unit_evidence=dict.fromkeys(metrics, "canonical_unit"),
    )
    return ImportedSession(
        session_id=f"{AGGREGATE_EXPORT_PROFILE_ID}-{digest[:16]}",
        name=session_name or path.stem,
        shots=wide[ordered],
        manifest=manifest,
        source_path=path,
        metadata={
            "imported_at": manifest.imported_at,
            "export_kind": "aggregates",
            "publisher": LAUNCH_MONITOR_DATA_PUBLISHER,
        },
    )
