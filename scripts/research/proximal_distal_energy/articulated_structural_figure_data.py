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
    CORNER_PATHWAYS,
)

PackKey = tuple[str, str]


def _digest(record: dict[str, Any]) -> str:
    return hashlib.sha256(
        json.dumps(record, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


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
        "schema_version": "articulated-structural-figure-data/v1",
        "result_sha256": result["result_sha256"],
        "support": support,
        "persistent_outcomes": outcomes,
        "axis_secants": [dict(value) for value in result["axes"]],
        "retained_failures": failures,
        "nominal_ground_matched_cell_count": nominal_ground["matched_cell_count"],
        "interpretation": "synthetic engineering sensitivity; no causal, population, human, or coaching inference",
    }
    return {**payload, "figure_data_sha256": _digest(payload)}


def write_structural_figure_data(record: dict[str, Any], output: Path) -> None:
    """Write prevalidated figure data atomically as strict JSON."""

    if record.get("figure_data_sha256") != _digest(
        {key: value for key, value in record.items() if key != "figure_data_sha256"}
    ):
        raise RuntimeError("structural figure data digest does not reproduce")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    temporary.replace(output)


__all__ = ["build_structural_figure_data", "write_structural_figure_data"]
