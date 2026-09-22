"""Independent fit-metric scoring for club-only matrix review (CO-08 #10612).

Path-anchor facade over :mod:`club_only.matrix_qualification`. Compares common
observables across model complexities; never invents native G1 success.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.shared.python.motion_matching.club_only.matrix_qualification import (
    MATRIX_SCHEMA,
    ExportedCandidatePackage,
    MatrixCellResult,
    MatrixQualificationReport,
    build_matrix_qualification_report,
    compare_common_observables,
    matrix_qualification_evidence_payload,
)

__all__ = [
    "MATRIX_SCHEMA",
    "ExportedCandidatePackage",
    "MatrixCellResult",
    "MatrixQualificationReport",
    "build_matrix_qualification_report",
    "common_observable_intersection",
    "compare_common_observables",
    "matrix_cell_original_3d_rmse",
    "matrix_qualification_evidence_payload",
]


def common_observable_intersection(
    packages: Sequence[ExportedCandidatePackage],
) -> frozenset[str]:
    """Return observables supported by every package (not weighted objectives)."""
    return compare_common_observables(packages)


def matrix_cell_original_3d_rmse(
    cell: MatrixCellResult | Mapping[str, Any],
) -> float | None:
    """Extract original 3D RMSE from a scored matrix cell; None when blocked."""
    if isinstance(cell, MatrixCellResult):
        return cell.original_3d_rmse_m
    value = cell.get("original_3d_rmse_m")
    if value is None:
        return None
    return float(value)
