"""Determinate net club wrench parity between two engine runs (HRG-GS, #13306, #13334).

Per GRIP_PARITY_DECISIONS.md Section 19, club kinematics identify only the net
wrench the hands apply to the club. The per-hand split and the internal squeeze
of a two-hand grip are indeterminate, so engine parity for the grip-contact
adapters is accepted on the net club force and the net club moment about the
club centre of mass. This module is the one implementation of that comparison;
the MuJoCo and Simscape parity tests both use it.
"""

from __future__ import annotations

import difflib
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.grip_contact.club_dynamics import ClubDynamics
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.parity import (
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
    GripKineticsSeries,
    QuantityError,
)

logger = logging.getLogger(__name__)

__all__ = [
    "failing_net_wrench",
    "net_club_wrench",
    "net_club_wrench_errors",
    "report_generated_drift",
]


def report_generated_drift(
    path: Path,
    current: str | None,
    expected: str,
    *,
    max_lines: int = 80,
) -> None:
    """Log which generated file failed a ``--check`` and a bounded unified diff (#13544).

    Diagnostics only: callers still return their non-zero drift code. ``current``
    is ``None`` when the committed file is missing. At most ``max_lines`` diff
    lines are logged (``-`` committed, ``+`` freshly rendered).
    """
    if max_lines < 1:
        raise ValueError(f"max_lines must be >= 1, got {max_lines!r}")
    if current is None:
        logger.warning("generated file missing: %s", path)
        return
    diff = list(
        difflib.unified_diff(
            current.splitlines(),
            expected.splitlines(),
            fromfile=f"{path} (committed)",
            tofile=f"{path} (rendered now)",
            n=1,
            lineterm="",
        )
    )
    shown = diff[:max_lines] or ["(no line diff: trailing newline or line endings)"]
    if len(diff) > max_lines:
        shown.append(f"... diff truncated ({len(diff) - max_lines} more lines)")
    logger.warning("generated file drifted: %s\n%s", path, "\n".join(shown))


def net_club_wrench(
    series: GripKineticsSeries, club: ClubDynamics, interface: GripInterface
) -> tuple[np.ndarray, np.ndarray]:
    """Net hand force on the club and its moment about the club CoM, per sample.

    Postconditions:
        Both arrays have shape ``(n, 3)`` in world axes; the moment is taken
        about the club centre of mass located from the lead grip point.
    """
    if not isinstance(series, GripKineticsSeries):
        raise TypeError(
            f"series must be GripKineticsSeries, got {type(series).__name__}"
        )
    dp_l = np.asarray(club.com_m, float) - np.asarray(interface.left.position_m, float)
    f_l, f_r = series.force_on_club_n["L"], series.force_on_club_n["R"]
    p_l, p_r = series.grip_point_m["L"], series.grip_point_m["R"]
    p_com = p_l + np.einsum("nij,j->ni", series.club_rotation, dp_l)
    m_com = (
        series.torque_on_club_nm["L"]
        + series.torque_on_club_nm["R"]
        + np.cross(p_l - p_com, f_l)
        + np.cross(p_r - p_com, f_r)
    )
    return f_l + f_r, m_com


def net_club_wrench_errors(
    candidate: GripKineticsSeries,
    reference: GripKineticsSeries,
    club: ClubDynamics,
    interface: GripInterface,
) -> dict[str, QuantityError]:
    """Peak and RMS errors of the candidate net club wrench against the reference.

    Errors are normalised by the reference peak magnitude, as in
    :func:`~src.shared.python.grip_contact.parity.parity_errors`.

    Preconditions:
        Both series share their sample count, and the reference peak of each
        quantity is positive.
    """
    f_cand, m_cand = net_club_wrench(candidate, club, interface)
    f_ref, m_ref = net_club_wrench(reference, club, interface)
    if f_cand.shape != f_ref.shape:
        raise ValueError(
            f"candidate {f_cand.shape} and reference {f_ref.shape} must share samples"
        )
    out: dict[str, QuantityError] = {}
    for name, x, x_ref in (
        ("net_force", f_cand, f_ref),
        ("net_moment_com", m_cand, m_ref),
    ):
        mag, mag_ref = np.linalg.norm(x, axis=1), np.linalg.norm(x_ref, axis=1)
        peak_ref = float(mag_ref.max())
        if not peak_ref > 0.0:
            raise ValueError(f"reference {name} peak must be positive, got {peak_ref}")
        rms = float(np.sqrt(np.mean(np.sum((x - x_ref) ** 2, axis=1))))
        out[name] = QuantityError(
            peak_reference=peak_ref,
            peak_candidate=float(mag.max()),
            peak_error=abs(float(mag.max()) - peak_ref) / peak_ref,
            rms_error=rms / peak_ref,
        )
    return out


def failing_net_wrench(errors: dict[str, QuantityError]) -> dict[str, Any]:
    """The quantities that miss the 5 % peak / 2 % RMS parity tolerances."""
    return {
        k: v.to_dict()
        for k, v in errors.items()
        if not (
            v.passes()
            and v.peak_error <= PEAK_TOLERANCE
            and v.rms_error <= RMS_TOLERANCE
        )
    }
