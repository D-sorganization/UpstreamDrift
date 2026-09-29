"""Shared finite-difference preconditions (issue #7146).

Several inverse-dynamics paths estimate ``qdot``/``qddot`` from a sampled ``q``
trajectory by finite differencing. For too-short trajectories the estimators
return all-zero derivatives (1 frame ⇒ qdot and qddot zero; 2 frames ⇒ qddot
zero). That silently turns inverse dynamics into *statics* — the velocity and
acceleration terms vanish without any error — which is the textbook case for an
explicit precondition.

This module centralizes the one contract shared by every Python entry point
(``pose_interchange`` reference adapter and ``motion_pipeline`` solver) and the
Rust kernel so the minimum-frame rule cannot drift between implementations
(DbC + DRY + K/reproducibility).
"""

from __future__ import annotations

# Centred first difference needs a neighbour on at least one side.
MIN_FRAMES_FOR_QDOT = 2
# The three-point second difference needs a neighbour on both sides.
MIN_FRAMES_FOR_QDDOT = 3


def require_enough_frames_for_finite_diff(
    *,
    n_frames: int,
    need_qdot: bool,
    need_qddot: bool,
) -> None:
    """Raise ``ValueError`` if ``n_frames`` is too small to finite-difference.

    Args:
        n_frames: Number of trajectory samples.
        need_qdot: Whether ``qdot`` must be estimated (no caller override).
        need_qddot: Whether ``qddot`` must be estimated (no caller override).

    Raises:
        ValueError: If a required derivative cannot be estimated without
            silently returning zeros. The message names the minimum frame
            count and the override that would bypass the check.
    """

    if need_qddot and n_frames < MIN_FRAMES_FOR_QDDOT:
        raise ValueError(
            "finite-difference qddot requires at least "
            f"{MIN_FRAMES_FOR_QDDOT} frames, got {n_frames}; supply explicit "
            "qddot (and qdot) to run inverse dynamics on a shorter trajectory "
            "instead of silently using zero acceleration (statics)."
        )
    if need_qdot and n_frames < MIN_FRAMES_FOR_QDOT:
        raise ValueError(
            "finite-difference qdot requires at least "
            f"{MIN_FRAMES_FOR_QDOT} frames, got {n_frames}; supply explicit "
            "qdot to run inverse dynamics on a single-frame trajectory instead "
            "of silently using zero velocity (statics)."
        )


__all__ = [
    "MIN_FRAMES_FOR_QDOT",
    "MIN_FRAMES_FOR_QDDOT",
    "require_enough_frames_for_finite_diff",
]
