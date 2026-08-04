"""Multi-format launch-monitor session importer."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.shared.python.launch_monitor.profiles import PROFILES, detect_profile
from src.shared.python.launch_monitor.schema import (
    IDENTITY_COLUMNS,
    METRICS,
    ColumnMapping,
    ImportedSession,
    ImportManifest,
    ImportOptions,
)

_UNIT_ALIASES = {
    "mph": "mph",
    "mi/h": "mph",
    "kph": "km/h",
    "kmh": "km/h",
    "km/h": "km/h",
    "mps": "m/s",
    "m/s": "m/s",
    "yd": "yd",
    "yds": "yd",
    "yard": "yd",
    "yards": "yd",
    "ft": "ft",
    "feet": "ft",
    "foot": "ft",
    "in": "in",
    "inch": "in",
    "inches": "in",
    "mm": "mm",
    "cm": "cm",
    "m": "m",
    "deg": "deg",
    "degree": "deg",
    "degrees": "deg",
    "rad": "rad",
    "radian": "rad",
    "radians": "rad",
    "rpm": "rpm",
    "rps": "rps",
    "s": "s",
    "sec": "s",
    "second": "s",
    "seconds": "s",
    "1": "1",
}


def _read_source(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path, sep=None, engine="python")
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(path, sep="\t")
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, list):
            return pd.json_normalize(payload, sep=".")
        if isinstance(payload, dict):
            for key in ("shots", "Shots", "records", "Records"):
                records = payload.get(key)
                if isinstance(records, list):
                    return pd.json_normalize(records, sep=".")
            return pd.json_normalize([payload], sep=".")
        raise ValueError("JSON source must contain an object or list of shot records")
    raise ValueError(
        f"Unsupported launch-monitor file type '{suffix}'. "
        "Supported: CSV, TSV, XLSX, XLS, JSON."
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _unit_from_header(header: str) -> str | None:
    candidates = re.findall(r"\(([^)]+)\)|\[([^]]+)\]", header.lower())
    flattened = [part for pair in candidates for part in pair if part]
    flattened.extend(
        re.findall(
            r"(?:^|\s)(mph|kph|kmh|m/s|mps|rpm|rps|yds?|ft|mm|cm|deg|rad|sec|s)$",
            header.lower(),
        )
    )
    for candidate in reversed(flattened):
        normalized = candidate.strip().lower()
        if normalized in _UNIT_ALIASES:
            return _UNIT_ALIASES[normalized]
    return None


def _convert(values: pd.Series, source_unit: str, canonical_unit: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    unit = _UNIT_ALIASES.get(source_unit.lower(), source_unit.lower())
    if canonical_unit == "1":
        if unit != "1":
            raise ValueError(f"Cannot convert unit '{source_unit}' to dimensionless")
        return numeric
    if canonical_unit == "m/s":
        factors = {"m/s": 1.0, "mph": 0.44704, "km/h": 1 / 3.6}
    elif canonical_unit == "m":
        factors = {
            "m": 1.0,
            "yd": 0.9144,
            "ft": 0.3048,
            "in": 0.0254,
            "cm": 0.01,
            "mm": 0.001,
        }
    elif canonical_unit == "rad":
        factors = {"rad": 1.0, "deg": np.pi / 180.0}
    elif canonical_unit == "rad/s":
        factors = {"rad/s": 1.0, "rpm": 2 * np.pi / 60.0, "rps": 2 * np.pi}
    elif canonical_unit == "s":
        factors = {"s": 1.0}
    else:
        raise ValueError(f"Unsupported canonical unit: {canonical_unit}")
    if unit not in factors:
        raise ValueError(f"Cannot convert unit '{source_unit}' to '{canonical_unit}'")
    return numeric * factors[unit]


def _combine_timestamp(raw: pd.DataFrame, mapped: dict[str, pd.Series]) -> pd.Series:
    if "captured_at" in mapped:
        return pd.to_datetime(mapped["captured_at"], errors="coerce", utc=True)
    if "date" in mapped and "time" in mapped:
        combined = mapped["date"].astype(str) + " " + mapped["time"].astype(str)
        return pd.to_datetime(combined, errors="coerce", utc=True)
    if "date" in mapped:
        return pd.to_datetime(mapped["date"], errors="coerce", utc=True)
    return pd.Series(pd.NaT, index=raw.index, dtype="datetime64[ns, UTC]")


def _resolve_mappings(
    headers: list[str], options: ImportOptions
) -> tuple[str, tuple[ColumnMapping, ...]]:
    profile_id = options.profile_id or detect_profile(headers).profile_id
    if profile_id not in PROFILES:
        raise ValueError(f"Unknown import profile: {profile_id}")
    auto = list(PROFILES[profile_id].mappings_for(headers))
    overrides = {mapping.source_column: mapping for mapping in options.mappings}
    mappings = [overrides.pop(mapping.source_column, mapping) for mapping in auto]
    mappings.extend(overrides.values())
    targets: set[str] = set()
    deduplicated: list[ColumnMapping] = []
    for mapping in reversed(mappings):
        if mapping.target_column in targets:
            continue
        targets.add(mapping.target_column)
        deduplicated.append(mapping)
    return profile_id, tuple(reversed(deduplicated))


def import_session(
    source: str | Path,
    options: ImportOptions | None = None,
) -> ImportedSession:
    """Import one launch-monitor export into canonical shot records."""
    path = Path(source).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Launch-monitor source does not exist: {path}")
    config = options or ImportOptions()
    raw = _read_source(path)
    if raw.empty:
        raise ValueError(f"Launch-monitor source contains no rows: {path}")
    headers = [str(column) for column in raw.columns]
    profile_id, mappings = _resolve_mappings(headers, config)
    profile = PROFILES[profile_id]
    digest = _sha256(path)
    session_id = f"{profile_id}-{digest[:16]}"
    output = pd.DataFrame(index=raw.index)
    output["session_id"] = session_id
    output["source_row"] = np.arange(2, len(raw) + 2)
    output["monitor_vendor"] = profile.vendor
    output["monitor_model"] = config.monitor_model or ""
    output["software_version"] = config.software_version or ""
    output["player"] = config.player or ""
    output["tags"] = ", ".join(config.tags)
    mapped_identity: dict[str, pd.Series] = {}
    metric_sources: dict[str, str] = {}
    source_units: dict[str, str] = {}
    unit_evidence: dict[str, str] = {}
    warnings: list[str] = []
    for mapping in mappings:
        if mapping.source_column not in raw.columns:
            warnings.append(f"Mapped source column not found: {mapping.source_column}")
            continue
        source_values = raw[mapping.source_column]
        target = mapping.target_column
        if target in METRICS:
            definition = METRICS[target]
            header_unit = _unit_from_header(mapping.source_column)
            source_unit = (
                mapping.source_unit or header_unit or profile.default_units[target]
            )
            evidence = (
                "mapping"
                if mapping.source_unit
                else "header"
                if header_unit
                else "profile-default"
            )
            try:
                converted = _convert(
                    source_values, source_unit, definition.canonical_unit
                )
            except ValueError as exc:
                warnings.append(f"{target}: {exc}")
                continue
            output[target] = converted * mapping.multiplier
            output[f"status::{target}"] = mapping.measurement_status
            metric_sources[target] = mapping.source_column
            source_units[target] = source_unit
            unit_evidence[target] = evidence
        else:
            mapped_identity[target] = source_values
            if target in IDENTITY_COLUMNS and target not in {
                "session_id",
                "source_row",
            }:
                output[target] = source_values
    output["captured_at"] = _combine_timestamp(raw, mapped_identity)
    if "shot_id" not in output or output["shot_id"].isna().all():
        output["shot_id"] = [f"{session_id}:{row}" for row in output["source_row"]]
    else:
        output["shot_id"] = output["shot_id"].astype(str)
    for source_column in headers:
        output[f"source::{source_column}"] = raw[source_column].to_numpy(copy=True)
    imported_at = datetime.now(UTC).isoformat()
    manifest = ImportManifest(
        source_path=str(path),
        file_sha256=digest,
        profile_id=profile_id,
        vendor=profile.vendor,
        imported_at=imported_at,
        row_count=len(output),
        source_columns=tuple(headers),
        metric_sources=metric_sources,
        source_units=source_units,
        unit_evidence=unit_evidence,
        warnings=tuple(warnings),
    )
    return ImportedSession(
        session_id=session_id,
        name=config.session_name or path.stem,
        shots=output.reset_index(drop=True),
        manifest=manifest,
        source_path=path,
        metadata={"imported_at": imported_at},
    )
