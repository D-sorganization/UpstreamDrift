"""Balance a bar-loaded lift's IK over mid-foot (OSL-15, #13093).

The Movement Optimizer balances its planar squat with the bar at its shoulder
joint (``full_squat`` bar model, zero ``squat_bar_depth``/``squat_bar_height``;
the Tools API accepts only a bar behind or below the shoulder).  The full-body
back bar sits on the torso shelf of :mod:`.back_bar`, fitted to the parity
standard's start-pose bar: 0.20 m above and 0.06 m in front of the AN-1
shoulder joints.  Retargeted rigidly with the trunk, that shelf carries the
bar 0.21 m ahead of the shoulders at the squat bottom, so with the 100 kg bar
the combined centre of mass of body and bar sat 0.15-0.19 m ahead of mid-foot
and the loaded re-track pivoted forward on the toes (#13072, cause B).

:func:`balance_lift` closes that gap in the IK.  The lift is solved on the
lift body with the bar welded rigidly to the hands
(:func:`~src.shared.python.lifting.barbell_runs.rigid_bar_spec`, the same
model the loaded re-track controller uses), and each frame's trunk angle is
corrected until the centre of mass of body plus bar lies over mid-foot.  This
is the rule the optimizer applies to its own squat bottom
(``balance_pose(..., adjust_joint=2)``: the torso angle that puts the centre
of mass at the base-of-support centre), applied with the full-body masses and
the bar where the shelf holds it.  The bar stays on the shelf: its target
rides the corrected trunk exactly as before.

Mid-foot is :func:`~src.shared.python.lifting.bundles.stance_mid_foot` (the
mean floor point of the contact spheres), the origin the shelf was fitted
from.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass, replace
from typing import Any, Protocol

import numpy as np
from numpy.typing import NDArray

from src.shared.python.lifting.barbell_bushing import Barbell
from src.shared.python.lifting.sagittal_retarget import SEGMENTS, SagittalSolution

Array = NDArray[np.float64]

__all__ = [
    "LoadBalance",
    "balance_lift",
    "combined_com_offsets",
    "load_balance_from_record",
    "loaded_com_offsets",
]

METHOD = (
    "IK on the rigid-bar lift body (barbell_runs.rigid_bar_spec); per-frame "
    "secant correction of the sagittal trunk angle until the body-plus-bar "
    "centre of mass lies over mid-foot (mean contact-sphere floor point)"
)
#: Largest trunk correction applied in one secant step, radians.
_MAX_STEP_RAD = 0.25


class _Solved(Protocol):
    """What :func:`balance_lift` needs of a solved lift: its joint rows."""

    @property
    def q(self) -> Array: ...


@dataclass(frozen=True)
class LoadBalance:
    """Bar load the lift IK balances over mid-foot (#13093).

    Attributes:
        barbell: The bar carried on the back (mass and inertia; the same
            :class:`Barbell` the loaded re-track attaches).
        segment: Sagittal segment whose angle is corrected (``"trunk"``, the
            optimizer's own squat balance joint).
        tolerance_m: Largest accepted forward distance of the combined centre
            of mass from mid-foot at any frame.
        max_iterations: Trajectory IK solves allowed after the first.
        probe_rad: Size of the first correction, which measures the slope.
    """

    barbell: Barbell
    segment: str = "trunk"
    tolerance_m: float = 5e-4
    max_iterations: int = 20
    probe_rad: float = 0.05

    def __post_init__(self) -> None:
        if not isinstance(self.barbell, Barbell):
            raise TypeError("barbell must be a Barbell")
        if self.segment not in SEGMENTS:
            raise ValueError(f"segment must be one of {SEGMENTS}: {self.segment!r}")
        if not (np.isfinite(self.tolerance_m) and self.tolerance_m > 0.0):
            raise ValueError("tolerance_m must be positive and finite")
        if int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be at least 1")
        if not (np.isfinite(self.probe_rad) and self.probe_rad > 0.0):
            raise ValueError("probe_rad must be positive and finite")

    def to_record(self) -> dict[str, Any]:
        """JSON-serialisable settings (:func:`load_balance_from_record` reads it)."""
        return {**asdict(self), "barbell": asdict(self.barbell)}


def load_balance_from_record(record: Mapping[str, Any]) -> LoadBalance:
    """The :class:`LoadBalance` of a :meth:`LoadBalance.to_record` record."""
    fields = dict(record)
    return LoadBalance(**{**fields, "barbell": Barbell(**dict(fields["barbell"]))})


def combined_com_offsets(kin: Any, q_rows: Array, forward: Array, ground: Any) -> Array:
    """Forward distance (m) of ``kin``'s centre of mass from mid-foot per row.

    ``kin`` is a :class:`FullBodyMarkerKinematics`; built on the rigid-bar spec
    its centre of mass is that of body plus bar.  Mid-foot is the mean floor
    point of the contact spheres of the same row.
    """
    basis = kin._plane_basis(ground)  # noqa: SLF001 - the IK's own plane basis
    axis = basis @ np.asarray(forward, dtype=float)
    out = []
    for q in np.asarray(q_rows, dtype=float):
        mid = np.mean(list(kin.sphere_ground_points(q, ground).values()), axis=0)
        out.append(float((kin.com_plane_position(q, ground) - basis @ mid) @ axis))
    return np.asarray(out)


def loaded_com_offsets(spec_bytes: bytes, q_rows: Array, load: LoadBalance) -> Array:
    """Body-plus-bar centre-of-mass offsets from mid-foot of ``q_rows``.

    The forward axis is the stance frame of the first row; the ground is the
    lift IK's (horizontal, height 0).
    """
    from src.shared.python.lifting.barbell_runs import rigid_bar_spec
    from src.shared.python.lifting.bundles import back_bar_kinematics, stance_frame
    from src.shared.python.motion_matching.contact_law import GroundPlane

    rows = np.asarray(q_rows, dtype=float)
    kin = back_bar_kinematics(rigid_bar_spec(bytes(spec_bytes), load.barbell))
    ground = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)
    forward = np.asarray(stance_frame(kin, rows[0], ground).forward)
    return combined_com_offsets(kin, rows, forward, ground)


def _corrected(
    solution: SagittalSolution, column: int, delta: Array
) -> SagittalSolution:
    angles = solution.segment_angles_rad.copy()
    angles[:, column] += delta
    return replace(solution, segment_angles_rad=angles)


def _secant_step(
    off: Array, delta: Array, previous: tuple[Array, Array] | None, slope: Array
) -> tuple[Array, Array]:
    """Next corrections and the slope estimate, frame by frame."""
    if previous is not None:
        d_off, d_delta = off - previous[0], delta - previous[1]
        moved = np.abs(d_delta) > 1e-12
        measured = np.divide(d_off, d_delta, out=np.zeros_like(off), where=moved)
        slope = np.where(moved & (measured > 0.0), measured, slope)
    step = np.clip(-off / slope, -_MAX_STEP_RAD, _MAX_STEP_RAD)
    return delta + step, slope


def balance_lift(
    solve: Callable[[SagittalSolution], _Solved],
    solution: SagittalSolution,
    offsets: Callable[[Array], Array],
    load: LoadBalance,
) -> tuple[Any, dict[str, Any]]:
    """Solve ``solution`` with each frame's ``load.segment`` angle balanced.

    Args:
        solve: Maps a planar solution to its full-body lift (``.q`` rows).
        solution: The optimizer's planar solution.
        offsets: Forward offset of the combined centre of mass from mid-foot
            for each row of a lift's ``q`` (positive: ahead of mid-foot).
        load: Bar load and iteration settings.

    Returns:
        The balanced lift and its record: the per-frame correction, the final
        and the unbalanced largest offsets and the iterations used.

    Postcondition: every frame's offset is within ``load.tolerance_m``.

    Raises:
        ValueError: If ``offsets`` does not return one value per frame.
        RuntimeError: If a frame is not balanced within ``load.max_iterations``.
    """
    column = SEGMENTS.index(load.segment)
    n = solution.time_s.size
    delta = np.zeros(n)

    def measure(correction: Array) -> tuple[Any, Array]:
        lift = solve(_corrected(solution, column, correction))
        off = np.asarray(offsets(lift.q), dtype=float)
        if off.shape != (n,) or not np.isfinite(off).all():
            raise ValueError(f"offsets must return one offset per frame ({n})")
        return lift, off

    lift, off = measure(delta)
    unbalanced = float(np.abs(off).max())
    previous: tuple[Array, Array] | None = None
    # Leaning the segment forward moves the centre of mass forward.
    slope = np.full(n, 1.0)
    iterations = 0
    while float(np.abs(off).max()) > load.tolerance_m:
        if iterations == load.max_iterations:
            raise RuntimeError(
                f"the {load.segment} cannot put the loaded centre of mass over "
                f"mid-foot: {float(np.abs(off).max()):.4f} m after {iterations} "
                "corrections"
            )
        if previous is None:
            new_delta = delta - load.probe_rad * np.sign(off)
        else:
            new_delta, slope = _secant_step(off, delta, previous, slope)
        previous = (off, delta)
        delta = new_delta
        lift, off = measure(delta)
        iterations += 1
    return lift, {
        "method": METHOD,
        "load": load.to_record(),
        "correction_rad": delta.tolist(),
        "max_offset_m": float(np.abs(off).max()),
        "unbalanced_max_offset_m": unbalanced,
        "iterations": iterations,
    }
