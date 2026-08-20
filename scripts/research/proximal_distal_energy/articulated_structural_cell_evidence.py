"""Build atomic, digest-bound per-cell articulated structural evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_structural_common_support import (
    CommonSupportComparison,
    HeadlineCells,
)

Array = NDArray[Any]
SCHEMA_VERSION = "articulated-structural-cell-evidence/v1"
REQUIRED_CELL_FIELDS = {
    "cell_identity",
    "matched_load_work",
    "matched_final_speed_difference_m_s",
    "load_match_relative_error",
    "work_match_relative_error",
    "gate_status",
    "failure_class",
    "two_engine_speed_difference_discrepancy_m_s",
    "time_step_speed_difference_discrepancy_m_s",
    "resolution_threshold_m_s",
    "resolved_outcome_change",
    "comparison_status",
}


def _digest(arrays: dict[str, Array]) -> str:
    digest = hashlib.sha256()
    for name in sorted(arrays):
        if name == "evidence_sha256":
            continue
        value = np.ascontiguousarray(arrays[name])
        if value.dtype.hasobject:
            raise ValueError("cell evidence may not contain object arrays")
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.tobytes())
    return digest.hexdigest()


def _encoded_identity(identity: tuple[int, int, float, float, str, float]) -> str:
    return json.dumps(identity, separators=(",", ":"), ensure_ascii=True)


def build_structural_cell_evidence(
    cells: HeadlineCells,
    *,
    gate_status: NDArray[np.bool_],
    failure_class: NDArray[np.str_],
    comparison: CommonSupportComparison | None = None,
) -> dict[str, Array]:
    """Build one pathway/corner cell pack without inventing paired outcomes."""

    size = len(cells.identities)
    gates = np.asarray(gate_status, dtype=bool)
    failures = np.asarray(failure_class, dtype=str)
    if gates.shape != (size,) or failures.shape != (size,):
        raise ValueError("gate and failure arrays must align with cell identities")
    missing_failure = (~gates) & np.isin(failures, ("", "none", "feasible"))
    if np.any(missing_failure):
        raise ValueError("failed gates require a failure class")
    if np.any(gates & ~np.isin(failures, ("none", "feasible"))):
        raise ValueError("passing gates cannot retain a failure class")

    threshold = np.full(size, np.nan)
    resolved = np.zeros(size, dtype=bool)
    status = np.full(size, "not_compared", dtype="U32")
    if comparison is not None:
        if comparison.pathway != cells.pathway:
            raise ValueError("comparison pathway must match cell evidence")
        index = {identity: slot for slot, identity in enumerate(cells.identities)}
        persistent_index = {
            identity: slot
            for slot, identity in enumerate(comparison.persistent_identities)
        }
        for identity in comparison.persistent_identities:
            slot = index[identity]
            comparison_slot = persistent_index[identity]
            threshold[slot] = comparison.resolution_threshold_m_s[comparison_slot]
            resolved[slot] = comparison.resolved_outcome_change[comparison_slot]
            status[slot] = (
                "persistent_resolved" if resolved[slot] else "persistent_unresolved"
            )
        for identity in comparison.entered_identities:
            status[index[identity]] = "entered_support"
        for identity in comparison.exited_identities:
            status[index[identity]] = "exited_support"
        status[status == "not_compared"] = "common_unmatched"

    arrays: dict[str, Array] = {
        "schema_version": np.asarray(SCHEMA_VERSION),
        "pathway": np.asarray(cells.pathway),
        "cell_identity": np.asarray(
            [_encoded_identity(value) for value in cells.identities]
        ),
        "matched_load_work": np.asarray(cells.matched, dtype=bool),
        "matched_final_speed_difference_m_s": np.asarray(
            cells.final_speed_difference_m_s, dtype=float
        ),
        "load_match_relative_error": np.asarray(
            cells.load_match_relative_error, dtype=float
        ),
        "work_match_relative_error": np.asarray(
            cells.work_match_relative_error, dtype=float
        ),
        "gate_status": gates,
        "failure_class": failures,
        "two_engine_speed_difference_discrepancy_m_s": np.asarray(
            cells.two_engine_speed_difference_discrepancy_m_s, dtype=float
        ),
        "time_step_speed_difference_discrepancy_m_s": np.asarray(
            cells.time_step_speed_difference_discrepancy_m_s, dtype=float
        ),
        "resolution_threshold_m_s": threshold,
        "resolved_outcome_change": resolved,
        "comparison_status": status,
    }
    arrays["evidence_sha256"] = np.asarray(_digest(arrays))
    validate_structural_cell_evidence(arrays)
    return arrays


def validate_structural_cell_evidence(arrays: dict[str, Array]) -> None:
    """Validate schema, alignment, classifications, and content digest."""

    if not REQUIRED_CELL_FIELDS.issubset(arrays) or {
        "schema_version",
        "pathway",
        "evidence_sha256",
    } - set(arrays):
        raise ValueError("cell evidence is missing required arrays")
    if str(np.asarray(arrays["schema_version"]).item()) != SCHEMA_VERSION:
        raise ValueError("cell evidence schema is not registered")
    pathway = str(np.asarray(arrays["pathway"]).item())
    if pathway not in ("shaft", "ground"):
        raise ValueError("cell evidence pathway is not registered")
    size = np.asarray(arrays["cell_identity"]).size
    for name in REQUIRED_CELL_FIELDS:
        if np.asarray(arrays[name]).shape != (size,):
            raise ValueError("cell evidence arrays must share one cell dimension")
    identities = np.asarray(arrays["cell_identity"], dtype=str)
    if len(set(identities.tolist())) != size:
        raise ValueError("cell evidence identities must be unique")
    gates = np.asarray(arrays["gate_status"], dtype=bool)
    failures = np.asarray(arrays["failure_class"], dtype=str)
    if np.any((~gates) & np.isin(failures, ("", "none", "feasible"))):
        raise ValueError("failed gates require a failure class")
    observed = str(np.asarray(arrays["evidence_sha256"]).item())
    if len(observed) != 64 or observed != _digest(arrays):
        raise RuntimeError("cell evidence digest does not reproduce")


def write_structural_cell_evidence(arrays: dict[str, Array], output: Path) -> None:
    """Write validated NPZ evidence atomically without pickle arrays."""

    validate_structural_cell_evidence(arrays)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(output)


def load_structural_cell_evidence(path: Path) -> dict[str, Array]:
    """Load and validate a cell pack with pickle disabled."""

    with np.load(path, allow_pickle=False) as source:
        arrays = {name: np.asarray(source[name]).copy() for name in source.files}
    validate_structural_cell_evidence(arrays)
    return arrays


__all__ = [
    "REQUIRED_CELL_FIELDS",
    "SCHEMA_VERSION",
    "build_structural_cell_evidence",
    "load_structural_cell_evidence",
    "validate_structural_cell_evidence",
    "write_structural_cell_evidence",
]
