"""Canonical ``ClubTarget`` dataclass and validation.

Mirrors the MATLAB target struct defined in CLUB_IK_SPEC.md. Frozen so that
loaders are forced to produce a fully-formed, validated artifact rather than
mutating one in place.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from src.shared.python.core.vector_math import row_euclidean_norm

# Validation tolerances (per CLUB_IK_SPEC § "Validation rules").
QUAT_NORM_TOL = 1.0e-6
MAX_POSITION_NORM_M = 5.0
TIME_EPS = 1.0e-9

ValidAlignment = Literal["impact", "address", "none"]


@dataclass(frozen=True)
class SourceProvenance:
    """File-level metadata for traceability.

    Attributes:
        filename: Bare filename (no directory) of the source artifact.
        format:   ``"excel"``, ``"c3d"``, or ``"synthetic"``.
        subject_id: Free-form subject identifier (e.g. ``"TW"``, ``"GW"``).
        trial_id:  Free-form trial identifier (sheet name or trial label).
        sha256:    Hex sha256 digest of the source file's bytes.
    """

    filename: str
    format: str
    subject_id: str
    trial_id: str
    sha256: str


@dataclass(frozen=True)
class AlignOptions:
    """Time-alignment and resampling options for the loaders.

    Attributes:
        sample_rate_hz:    Output timegrid rate. Default 1 kHz.
        simulation_time_s: Total simulation duration. Default 0.3 s.
        time_alignment:    ``"impact"`` (default), ``"address"``, or ``"none"``.
        impact_target_t_s: Where the measured impact lands on the sim grid.
    """

    sample_rate_hz: float = 1000.0
    simulation_time_s: float = 0.3
    time_alignment: ValidAlignment = "impact"
    impact_target_t_s: float = 0.25


@dataclass(frozen=True)
class ClubTarget:
    """Canonical 6-DOF club trajectory.

    Validated at construction; any violation of the CLUB_IK_SPEC validation
    rules raises ``ValueError``.
    """

    time: np.ndarray
    butt: np.ndarray
    clubhead: np.ndarray
    club_quat: np.ndarray
    impact_idx: int
    source: SourceProvenance
    _validated: bool = field(default=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        """Run all postcondition checks at construction."""
        _validate_clubtarget(self)


def _validate_time(time: np.ndarray) -> int:
    """Check the time vector and return its length ``N``."""
    if time.ndim != 1:
        raise ValueError(f"time must be 1-D, got shape {time.shape}")
    n = time.shape[0]
    if n < 2:
        raise ValueError(f"time must have at least 2 samples (got {n})")
    if abs(float(time[0])) > TIME_EPS:
        raise ValueError(f"time[0] must be 0, got {time[0]!r}")
    if not np.all(np.diff(time) > 0):
        raise ValueError("time must be strictly increasing")
    return n


def _validate_shapes(t: ClubTarget, n: int) -> None:
    """Verify trajectory arrays match ``time``'s row count."""
    for name, arr, cols in (
        ("butt", t.butt, 3),
        ("clubhead", t.clubhead, 3),
        ("club_quat", t.club_quat, 4),
    ):
        if arr.shape != (n, cols):
            raise ValueError(f"{name} must have shape ({n}, {cols}), got {arr.shape}")


def _validate_positions(t: ClubTarget) -> None:
    """Reject NaN/Inf and implausibly large position vectors."""
    for name, arr in (("butt", t.butt), ("clubhead", t.clubhead)):
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} contains NaN or Inf")
        norms = row_euclidean_norm(arr)
        if np.any(norms >= MAX_POSITION_NORM_M):
            raise ValueError(
                f"{name} has |r| >= {MAX_POSITION_NORM_M} m "
                f"(max {float(norms.max()):.3f})"
            )


def _validate_clubtarget(t: ClubTarget) -> None:
    """Enforce the CLUB_IK_SPEC validation rules."""
    n = _validate_time(t.time)
    _validate_shapes(t, n)
    _validate_positions(t)
    qnorms = row_euclidean_norm(t.club_quat)
    if np.any(np.abs(qnorms - 1.0) > QUAT_NORM_TOL):
        max_dev = float(np.abs(qnorms - 1.0).max())
        raise ValueError(
            "club_quat rows must be unit-norm to within "
            f"{QUAT_NORM_TOL} (max deviation {max_dev:.2e})"
        )
    if not (1 <= int(t.impact_idx) <= n):
        raise ValueError(f"impact_idx must be in [1, {n}], got {t.impact_idx}")
    if not isinstance(t.source, SourceProvenance):
        raise TypeError("source must be a SourceProvenance instance")
