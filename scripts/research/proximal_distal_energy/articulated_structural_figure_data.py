"""Build deterministic reviewer-facing data for the structural figure."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.articulated_structural_cell_evidence import (
    Array,
    validate_structural_cell_evidence,
)
from scripts.research.proximal_distal_energy.articulated_structural_result import (
    AXIS_PATHWAYS,
    CORNER_PATHWAYS,
)

PackKey = tuple[str, str]
SCHEMA_VERSION = "articulated-structural-figure-data/v1"
INTERPRETATION = (
    "synthetic engineering sensitivity; no causal, population, human, or "
    "coaching inference"
)


def _serialized(record: dict[str, Any]) -> bytes:
    return (json.dumps(record, indent=2, allow_nan=False) + "\n").encode("utf-8")


def _digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _valid_sha256(value: Any) -> bool:
    text = str(value)
    try:
        return len(text) == 64 and int(text, 16) >= 0
    except ValueError:
        return False


def _validate_support_rows(rows: Any) -> dict[PackKey, dict[str, Any]]:
    if not isinstance(rows, list):
        raise ValueError("figure support must be a list")
    observed = [(row.get("corner_id"), row.get("pathway")) for row in rows]
    if observed != list(CORNER_PATHWAYS):
        raise ValueError("figure support rows must retain registered order")
    required = {
        "planned_cell_count",
        "feasible_cell_count",
        "executed_cell_count",
        "matched_cell_count",
        "common_executed_cell_count",
        "nominal_only_executed_cell_count",
        "corner_only_executed_cell_count",
        "persistent_cell_count",
        "entered_cell_count",
        "exited_cell_count",
        "resolved_persistent_cell_count",
    }
    support = dict(zip(CORNER_PATHWAYS, rows, strict=True))
    nominal = {
        pathway: support[("nominal", pathway)] for pathway in ("shaft", "ground")
    }
    for key, row in support.items():
        if not required.issubset(row):
            raise ValueError("figure support row is incomplete")
        try:
            counts = {name: int(row[name]) for name in required}
        except (TypeError, ValueError) as error:
            raise ValueError("figure support counts must be integers") from error
        if any(
            isinstance(row[name], bool) or row[name] != counts[name]
            for name in required
        ):
            raise ValueError("figure support counts must be integers")
        if any(value < 0 for value in counts.values()):
            raise ValueError("figure support counts must be nonnegative")
        if not (
            counts["matched_cell_count"]
            <= counts["executed_cell_count"]
            == counts["feasible_cell_count"]
            <= counts["planned_cell_count"]
        ):
            raise ValueError("figure support denominators do not reconcile")
        nominal_executed = int(nominal[key[1]]["executed_cell_count"])
        if (
            counts["common_executed_cell_count"]
            + counts["nominal_only_executed_cell_count"]
            != nominal_executed
            or counts["common_executed_cell_count"]
            + counts["corner_only_executed_cell_count"]
            != counts["executed_cell_count"]
        ):
            raise ValueError("figure common execution support does not reconcile")
        transitions = (
            counts["persistent_cell_count"]
            + counts["entered_cell_count"]
            + counts["exited_cell_count"]
        )
        if transitions > counts["common_executed_cell_count"] or (
            counts["resolved_persistent_cell_count"] > counts["persistent_cell_count"]
        ):
            raise ValueError("figure support transitions do not reconcile")
    return support


def _validate_outcomes(rows: Any, support: dict[PackKey, dict[str, Any]]) -> None:
    if not isinstance(rows, list):
        raise ValueError("persistent outcomes must be a list")
    order = {key: index for index, key in enumerate(CORNER_PATHWAYS)}
    observed_order: list[int] = []
    identities: set[tuple[str, str, str]] = set()
    counts: Counter[PackKey] = Counter()
    for row in rows:
        key = (str(row.get("corner_id")), str(row.get("pathway")))
        if key not in order or key[0] == "nominal":
            raise ValueError("persistent outcome is not registered")
        identity = str(row.get("cell_identity", ""))
        identity_key = (*key, identity)
        if not identity or identity_key in identities:
            raise ValueError("persistent outcome identities must be unique")
        identities.add(identity_key)
        observed_order.append(order[key])
        try:
            change = float(row["change_m_s"])
            threshold = float(row["resolution_threshold_m_s"])
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("persistent outcome values must be numeric") from error
        if not np.isfinite(change) or not np.isfinite(threshold) or threshold <= 0:
            raise ValueError("persistent outcome values must be finite and resolved")
        if not isinstance(row.get("resolved"), bool) or row["resolved"] != (
            abs(change) > threshold
        ):
            raise ValueError("persistent outcome resolution status does not reproduce")
        counts[key] += 1
    if observed_order != sorted(observed_order):
        raise ValueError("persistent outcomes must retain registered order")
    for key in CORNER_PATHWAYS:
        expected = (
            0 if key[0] == "nominal" else int(support[key]["persistent_cell_count"])
        )
        if counts[key] != expected:
            raise ValueError("persistent outcomes do not match support transitions")


def _validate_axes(rows: Any) -> None:
    if not isinstance(rows, list):
        raise ValueError("axis secants must be a list")
    observed = [(row.get("axis_name"), row.get("pathway")) for row in rows]
    if observed != list(AXIS_PATHWAYS):
        raise ValueError("axis secants must retain registered order")
    secant_names = (
        "low_to_nominal_secant_m_s_per_unit_scale",
        "nominal_to_high_secant_m_s_per_unit_scale",
        "low_to_nominal_secant_range_m_s_per_unit_scale",
        "nominal_to_high_secant_range_m_s_per_unit_scale",
    )
    classification_priority = (
        ("resolved_opposing", "resolved_opposing_on_shared_support"),
        (
            "resolved_materially_unequal",
            "resolved_materially_unequal_on_shared_support",
        ),
        ("resolution_limited", "resolution_limited_on_shared_support"),
        (
            "resolved_direction_consistent",
            "resolved_direction_consistent_on_shared_support",
        ),
    )
    for row in rows:
        scales = np.asarray(
            [row.get("low_scale"), row.get("nominal_scale"), row.get("high_scale")],
            dtype=float,
        )
        if not np.all(np.isfinite(scales)) or not scales[0] < scales[1] < scales[2]:
            raise ValueError("axis secant scales must be finite and ordered")
        raw_support = row.get("shared_persistent_cell_count", -1)
        try:
            support = int(raw_support)
        except (TypeError, ValueError) as error:
            raise ValueError("axis shared support must be an integer") from error
        if isinstance(raw_support, bool) or raw_support != support:
            raise ValueError("axis shared support must be an integer")
        values = [row.get(name) for name in secant_names]
        if support < 0 or (support == 0) != all(value is None for value in values):
            raise ValueError("axis secant support and nullability do not agree")
        counts = row.get("cell_classification_counts")
        try:
            invalid_counts = not isinstance(counts, dict) or any(
                name not in dict(classification_priority)
                or isinstance(value, bool)
                or int(value) != value
                or value < 0
                for name, value in counts.items()
            )
        except (TypeError, ValueError):
            invalid_counts = True
        if invalid_counts:
            raise ValueError("axis cell classifications are not registered")
        if sum(counts.values()) != support:
            raise ValueError("axis classification counts do not match shared support")
        expected_classification = "insufficient_shared_persistent_support"
        for name, classification in classification_priority:
            if counts.get(name, 0):
                expected_classification = classification
                break
        if row.get("nonmonotonic_classification") != expected_classification:
            raise ValueError("axis nonmonotonic classification does not reproduce")
        if support:
            medians = np.asarray(values[:2], dtype=float)
            ranges = np.asarray(values[2:], dtype=float)
            if (
                not np.all(np.isfinite(medians))
                or ranges.shape != (2, 2)
                or not np.all(np.isfinite(ranges))
                or np.any(ranges[:, 0] > medians)
                or np.any(medians > ranges[:, 1])
            ):
                raise ValueError("axis secants and ranges must be finite and ordered")


def _validate_failures(rows: Any, support: dict[PackKey, dict[str, Any]]) -> None:
    if not isinstance(rows, list):
        raise ValueError("retained failures must be a list")
    counts: Counter[PackKey] = Counter()
    identities: set[tuple[str, str, int, int]] = set()
    for row in rows:
        key = (str(row.get("corner_id")), str(row.get("pathway")))
        try:
            identity = (*key, int(row["case_index"]), int(row["phase_index"]))
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError("retained failure identity is incomplete") from error
        if key not in support or identity in identities or not row.get("failure_class"):
            raise ValueError("retained failure is duplicated or unregistered")
        identities.add(identity)
        counts[key] += 1
    for key, row in support.items():
        missing_cells = int(row["planned_cell_count"]) - int(row["feasible_cell_count"])
        if missing_cells % 32 or counts[key] != missing_cells // 32:
            raise ValueError("retained failures do not reconcile with feasibility")


def validate_structural_figure_data_record(record: dict[str, Any]) -> None:
    """Reject a digest-valid figure record with inconsistent scientific semantics."""

    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("structural figure data schema is not registered")
    if not _valid_sha256(record.get("result_sha256")):
        raise ValueError("structural figure result binding must be SHA-256")
    if record.get("interpretation") != INTERPRETATION:
        raise ValueError("structural figure interpretation boundary is missing")
    support = _validate_support_rows(record.get("support"))
    _validate_outcomes(record.get("persistent_outcomes"), support)
    _validate_axes(record.get("axis_secants"))
    _validate_failures(record.get("retained_failures"), support)
    ground = support[("nominal", "ground")]
    if (
        record.get("nominal_ground_matched_cell_count") != 0
        or ground["matched_cell_count"] != 0
        or ground["planned_cell_count"] != 384
    ):
        raise ValueError("nominal ground support must remain explicit at 0/384")
    payload = {
        key: value for key, value in record.items() if key != "figure_data_sha256"
    }
    if not _valid_sha256(record.get("figure_data_sha256")) or record.get(
        "figure_data_sha256"
    ) != _digest(payload):
        raise ValueError("structural figure data digest does not reproduce")


def _pack_identities(pack: dict[str, Array]) -> set[str]:
    return set(np.asarray(pack["cell_identity"], dtype=str).tolist())


def _support_row(
    corner: dict[str, Any], pack: dict[str, Array], nominal_ids: set[str]
) -> dict[str, Any]:
    identities = _pack_identities(pack)
    status = np.asarray(pack["comparison_status"], dtype=str)
    counts = Counter(status.tolist())
    return {
        "corner_id": corner["corner_id"],
        "pathway": corner["pathway"],
        "planned_cell_count": corner["planned_headline_cell_count"],
        "feasible_cell_count": corner["feasible_headline_cell_count"],
        "executed_cell_count": corner["executed_headline_cell_count"],
        "matched_cell_count": corner["matched_cell_count"],
        "common_executed_cell_count": len(identities.intersection(nominal_ids)),
        "nominal_only_executed_cell_count": len(nominal_ids - identities),
        "corner_only_executed_cell_count": len(identities - nominal_ids),
        "persistent_cell_count": counts["persistent_resolved"]
        + counts["persistent_unresolved"],
        "entered_cell_count": counts["entered_support"],
        "exited_cell_count": counts["exited_support"],
        "resolved_persistent_cell_count": counts["persistent_resolved"],
    }


def _outcome_rows(
    corner: dict[str, Any], pack: dict[str, Array]
) -> list[dict[str, Any]]:
    if corner["corner_id"] == "nominal":
        return []
    status = np.asarray(pack["comparison_status"], dtype=str)
    persistent = np.isin(status, ("persistent_resolved", "persistent_unresolved"))
    identities = np.asarray(pack["cell_identity"], dtype=str)
    change = np.asarray(pack["corner_minus_nominal_speed_difference_m_s"], dtype=float)
    threshold = np.asarray(pack["resolution_threshold_m_s"], dtype=float)
    resolved = np.asarray(pack["resolved_outcome_change"], dtype=bool)
    return [
        {
            "corner_id": corner["corner_id"],
            "pathway": corner["pathway"],
            "cell_identity": str(identities[index]),
            "change_m_s": float(change[index]),
            "resolution_threshold_m_s": float(threshold[index]),
            "resolved": bool(resolved[index]),
        }
        for index in np.flatnonzero(persistent)
    ]


def _validate_pack_bindings(
    result: dict[str, Any], packs: dict[PackKey, dict[str, Array]]
) -> None:
    if set(packs) != set(CORNER_PATHWAYS):
        raise ValueError("figure data requires exactly 14 corner-pathway packs")
    for corner in result["corners"]:
        key = (corner["corner_id"], corner["pathway"])
        pack = packs[key]
        validate_structural_cell_evidence(pack)
        if (
            str(pack["pathway"].item()) != corner["pathway"]
            or str(pack["evidence_sha256"].item()) != corner["cell_evidence_sha256"]
        ):
            raise ValueError("figure cell pack does not agree with result binding")


def build_structural_figure_data(
    result: dict[str, Any], packs: dict[PackKey, dict[str, Array]]
) -> dict[str, Any]:
    """Assemble all preregistered figure panels without favorable selection."""

    _validate_pack_bindings(result, packs)
    nominal_ids = {
        pathway: _pack_identities(packs[("nominal", pathway)])
        for pathway in ("shaft", "ground")
    }
    support = []
    outcomes = []
    failures = []
    for corner in result["corners"]:
        key = (corner["corner_id"], corner["pathway"])
        pack = packs[key]
        support.append(_support_row(corner, pack, nominal_ids[corner["pathway"]]))
        outcomes.extend(_outcome_rows(corner, pack))
        failures.extend(
            {
                "corner_id": corner["corner_id"],
                "pathway": corner["pathway"],
                **failure,
            }
            for failure in corner["retained_failures"]
        )
    nominal_ground = next(
        row
        for row in support
        if row["corner_id"] == "nominal" and row["pathway"] == "ground"
    )
    payload = {
        "schema_version": SCHEMA_VERSION,
        "result_sha256": result["result_sha256"],
        "support": support,
        "persistent_outcomes": outcomes,
        "axis_secants": [dict(value) for value in result["axes"]],
        "retained_failures": failures,
        "nominal_ground_matched_cell_count": nominal_ground["matched_cell_count"],
        "interpretation": INTERPRETATION,
    }
    record = {**payload, "figure_data_sha256": _digest(payload)}
    validate_structural_figure_data_record(record)
    return record


def write_structural_figure_data(record: dict[str, Any], output: Path) -> None:
    """Write prevalidated figure data atomically as strict JSON."""

    validate_structural_figure_data_record(record)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(_serialized(record))
    temporary.replace(output)


def validate_structural_figure_data(path: Path, result_sha256: str) -> dict[str, Any]:
    """Validate semantics, result binding, digest, and canonical bytes."""

    raw = path.read_bytes()
    try:
        record = json.loads(
            raw.decode("utf-8"),
            parse_constant=lambda value: (_ for _ in ()).throw(
                ValueError(f"nonfinite JSON constant: {value}")
            ),
        )
        validate_structural_figure_data_record(record)
    except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as error:
        raise RuntimeError("structural figure data is invalid") from error
    if record["result_sha256"] != result_sha256:
        raise RuntimeError("structural figure result binding does not agree")
    if raw != _serialized(record):
        raise RuntimeError("structural figure data does not retain exact bytes")
    return record


__all__ = [
    "build_structural_figure_data",
    "validate_structural_figure_data",
    "validate_structural_figure_data_record",
    "write_structural_figure_data",
]
