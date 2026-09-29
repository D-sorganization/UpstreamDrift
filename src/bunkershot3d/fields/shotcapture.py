"""Keeping the sand field of a whole marched shot (issue #8729).

What this adds over :mod:`.capture`
-----------------------------------

:func:`~.capture.capture_f1_field` marches the **declared straight-line
constant-velocity approach** to a queried pose. That was the only
kinematics F1 had when issue #8710 was written, and the field it stores
says so in its provenance.

Issue #8733 gave F1 a whole-shot march: the head's real trajectory,
integrated against the sand it is pushing, decelerating as it goes. The
field of *that* is a different field, and the difference is exactly the
part a viewer would otherwise read into the picture for free -- sand
thrown by a decelerating head is not sand thrown by one driven through at
constant speed.

So this module captures against the march itself rather than against a
declared approach, and stamps the provenance accordingly. An animation of
an approach and an animation of a shot look identical on screen, which is
precisely why which one it is has to be written down.

How it gets at the field
------------------------

:func:`~bunkershot3d.solvers.mpm.wholeshot.simulate_f1_shot` advances one
particle bed in place, so the sand field of a whole shot only exists
*during* the march. Reconstructing it afterwards is not possible: the run
carries the final bed, which is one instant and not a record.

The march therefore takes a
:class:`~bunkershot3d.solvers.mpm.wholeshot.ShotFieldRecorder`, and
:class:`WholeShotRecorder` here implements it. It is shown every instant
and keeps one in ``stride``, sampling onto the grid with the solver's own
transfer operators -- the same
:func:`~.capture.sample_grid_field` the declared-approach capture uses,
so the two fields are the same quantities formed the same way and are
directly comparable.

Nothing here changes the trajectory. The recorder is only ever *shown*
the state; it is never asked about it, and it holds no reference to the
live arrays.
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from ..exceptions import BunkerShot3DValueError
from ..solvers.envelope import RefusalPolicy, ValidityVerdict
from ..solvers.mpm.body import RigidSection
from ..solvers.mpm.envelope import RefusedQuantity
from ..solvers.mpm.grid import PlaneStrainGrid
from ..solvers.mpm.solver import PlaneStrainMPMSolver
from ..solvers.mpm.state import ParticleState
from ..solvers.mpm.wholeshot import F1ShotResult, F1ShotSettings, simulate_f1_shot
from ..solvers.protocol import IntrusionState
from .capture import GridFieldSample, sample_grid_field
from .schema import FieldLayout, GridGeometry, SandFieldSeries
from .standing import (
    FieldProvenance,
    OccupancyRule,
    RetentionPolicy,
    RetentionRecord,
)

__all__ = [
    "WHOLE_SHOT_KINEMATICS_NOTE",
    "WholeShotRecorder",
    "capture_f1_shot_field",
]

WHOLE_SHOT_KINEMATICS_NOTE = (
    "whole-shot march: the head's own trajectory, integrated against the "
    "sand it displaces and decelerating as it goes (issue #8733), not a "
    "declared constant-velocity approach"
)
"""How the body's motion was supplied, in words, for the provenance record.

Stated rather than implied because a field animated from a declared
approach and a field animated from a marched shot are indistinguishable
on screen and are not the same claim."""

_DIMENSION = 2


class WholeShotRecorder:
    """Keeps one instant in ``stride`` as a whole shot marches past.

    Implements
    :class:`~bunkershot3d.solvers.mpm.wholeshot.ShotFieldRecorder`
    structurally; the solver package does not import this one.
    """

    def __init__(self, policy: RetentionPolicy) -> None:
        """Start an empty record.

        Args:
            policy: What to keep. The stride is derived from it and the
                march's own step count in :meth:`begin`, so a longer shot
                gets a coarser stride rather than a truncated tail.
        """
        self._policy = policy
        self._grid: PlaneStrainGrid | None = None
        self._time_step_s = 0.0
        self._n_steps = 0
        self._stride = 1
        self._seen = -1
        self.times_s: list[float] = []
        self.fields: list[GridFieldSample] = []
        self.outlines_m: list[NDArray[np.float64]] = []

    @property
    def grid(self) -> PlaneStrainGrid:
        """The lattice the march ran on.

        Raises:
            BunkerShot3DValueError: If the march never started.
        """
        if self._grid is None:
            raise BunkerShot3DValueError(
                "this recorder was never handed a march, so it has no lattice; "
                "pass it to simulate_f1_shot rather than reading it first"
            )
        return self._grid

    @property
    def n_frames(self) -> int:
        """Instants kept."""
        return len(self.times_s)

    @property
    def stride(self) -> int:
        """Steps between kept instants."""
        return self._stride

    @property
    def steps_marched(self) -> int:
        """Steps the march was budgeted for."""
        return self._n_steps

    @property
    def time_step_s(self) -> float:
        """The CFL step the march ran at [s]."""
        return self._time_step_s

    def begin(self, grid: PlaneStrainGrid, *, time_step_s: float, n_steps: int) -> None:
        """Told the march's shape before the first step.

        Args:
            grid: The lattice, which does not move.
            time_step_s: The CFL step [s].
            n_steps: The step budget.
        """
        self._grid = grid
        self._time_step_s = float(time_step_s)
        self._n_steps = int(n_steps)
        self._stride = self._policy.stride_for(max(int(n_steps), 1))

    def sample(
        self, time_s: float, particles: ParticleState, section: RigidSection
    ) -> None:
        """Offered one instant; keep it if the stride says so.

        The undisturbed bed at ``t = 0`` is always kept: it is the
        reference every later frame is read against.

        The intruder's own outline is kept alongside each frame, because
        a velocity field with no body in it cannot answer the question
        the field was computed for.

        Args:
            time_s: Elapsed time [s].
            particles: The live bed. Sampled onto the grid here and not
                retained -- the march advances these arrays in place, so
                a stored reference would silently become the last frame
                repeated.
            section: The head's pose.
        """
        self._seen += 1
        if self._seen % self._stride != 0:
            return
        self.times_s.append(float(time_s))
        self.fields.append(
            sample_grid_field(
                self.grid,
                particles,
                include_shear_rate=self._policy.include_shear_rate,
            )
        )
        self.outlines_m.append(np.array(section.vertices_m, dtype=np.float64))


def capture_f1_shot_field(
    solver: PlaneStrainMPMSolver,
    state: IntrusionState,
    *,
    settings: F1ShotSettings,
    policy: RetentionPolicy | None = None,
    refusal_policy: RefusalPolicy | None = None,
) -> tuple[SandFieldSeries, F1ShotResult]:
    """March a whole shot and keep the sand field as it goes.

    Args:
        solver: The F1 solver.
        state: The delivery -- the head's attitude, velocity and spin at
            entry. Nothing here prescribes where the head goes.
        settings: How the shot is set up and when it stops.
        policy: What to keep. Defaults to
            :class:`~.standing.RetentionPolicy`'s own defaults.
        refusal_policy: Overrides the solver's policy for this capture.
            A field of a refused query is legitimate to *look* at while
            being illegitimate to quote, and the refusal travels in the
            provenance either way.

    Returns:
        ``(series, shot)`` -- the stored field and the whole-shot result,
        so a caller gets the trajectory and the sand from one solve. They
        are the same solve, which is the point: pairing a field from one
        run with a trajectory from another is the substitution the 3-D
        view refuses at the other end.

    Raises:
        BunkerShot3DValueError: If ``solver`` is not an F1 solver, or the
            march kept no frames.
        OutOfEnvelopeError: If the verdict refuses under the policy in
            force.
    """
    if not isinstance(solver, PlaneStrainMPMSolver):
        raise BunkerShot3DValueError(
            "capture_f1_shot_field needs a PlaneStrainMPMSolver; other tiers "
            f"write their own capture against the same schema, got "
            f"{type(solver).__name__}"
        )
    keep = RetentionPolicy() if policy is None else policy
    verdict = solver.envelope(state)
    verdict.require_usable(
        solver.refusal_policy if refusal_policy is None else refusal_policy
    )

    recorder = WholeShotRecorder(keep)
    shot = simulate_f1_shot(solver, state, settings=settings, recorder=recorder)
    if recorder.n_frames == 0:  # pragma: no cover - the march always samples
        raise BunkerShot3DValueError(
            "the march kept no field frames, so there is nothing to animate"
        )

    grid = recorder.grid
    geometry = GridGeometry(
        origin_m=grid.origin_m,
        cell_size_m=grid.cell_size_m,
        shape=(grid.node_counts[0], grid.node_counts[1]),
        axis_names=("x", "z"),
    )
    return (
        _series(solver, state, settings, verdict, keep, geometry, recorder),
        shot,
    )


def _series(
    solver: PlaneStrainMPMSolver,
    state: IntrusionState,
    settings: F1ShotSettings,
    verdict: ValidityVerdict,
    policy: RetentionPolicy,
    geometry: GridGeometry,
    recorder: WholeShotRecorder,
) -> SandFieldSeries:
    """Stack the recorded instants into the stored series."""
    store = np.dtype(policy.store_dtype)

    def stacked(values: list[NDArray[np.float64]]) -> NDArray[np.float64]:
        return np.stack(values, axis=0).astype(store, copy=False)

    frames = recorder.fields
    return SandFieldSeries(
        time_s=np.asarray(recorder.times_s, dtype=np.float64),
        velocity_m_s=stacked([sample.velocity_m_s for sample in frames]),
        density_kg_m3=stacked([sample.density_kg_m3 for sample in frames]),
        shear_rate_1_s=(
            None
            if not policy.include_shear_rate
            else stacked([_require_shear(sample.shear_rate_1_s) for sample in frames])
        ),
        positions_m=None,
        layout=FieldLayout.GRID,
        geometry=geometry,
        provenance=_provenance(solver, state, settings, verdict, recorder),
        retention=_retention(policy, geometry, recorder),
        occupancy=OccupancyRule(
            reference_density_kg_m3=float(solver.material.density_kg_m3),
            max_admissible_density_kg_m3=float(
                solver.material.density_kg_m3
                * math.exp(-solver.material.cap_volumetric_strain)
            ),
        ),
        body_outline_m=stacked(recorder.outlines_m),
    )


def _require_shear(shear: NDArray[np.float64] | None) -> NDArray[np.float64]:
    """Unwrap a shear array the policy said would be there."""
    if shear is None:  # pragma: no cover - guarded by the policy flag
        raise BunkerShot3DValueError(
            "the retention policy asked for the shear rate but none was formed"
        )
    return shear


def _retention(
    policy: RetentionPolicy, geometry: GridGeometry, recorder: WholeShotRecorder
) -> RetentionRecord:
    """What the policy actually cost on this march."""
    stride = recorder.stride
    dropped: tuple[str, ...] = ()
    if stride > 1:
        dropped = (
            f"temporal: kept 1 step in {stride} of {recorder.steps_marched}, so "
            f"{recorder.steps_marched - recorder.n_frames} steps are not stored "
            f"({stride * recorder.time_step_s * 1e6:.3g} us between frames)",
        )
    return RetentionRecord(
        policy=policy,
        steps_marched=recorder.steps_marched,
        time_stride=stride,
        frames_kept=recorder.n_frames,
        time_step_s=recorder.time_step_s,
        samples_in_domain=geometry.n_samples,
        samples_kept=geometry.n_samples,
        dropped=dropped,
    )


def _provenance(
    solver: PlaneStrainMPMSolver,
    state: IntrusionState,
    settings: F1ShotSettings,
    verdict: ValidityVerdict,
    recorder: WholeShotRecorder,
) -> FieldProvenance:
    """Everything needed to trace this field to its shot and regenerate it.

    The kinematics line is the load-bearing entry. Everything else here
    is settings a reader could in principle recover; which trajectory the
    sand was thrown by is not recoverable from the arrays at all.
    """
    material = solver.material
    entries: dict[str, float | int | str] = {
        "cell_size_m": float(solver.cell_size_m),
        "effective_width_m": float(solver.effective_width_m),
        "bed_depth_m": float(solver.bed_depth_m),
        "cfl_number": float(solver.cfl_number),
        "gravity_m_s2": float(solver.gravity_m_s2),
        "contact_friction": float(solver.contact_friction),
        "time_step_s": float(recorder.time_step_s),
        "n_steps": int(recorder.steps_marched),
        "free_surface_height_m": float(settings.free_surface_height_m),
        "head_mass_kg": float(settings.head_mass_kg),
        "max_time_s": float(settings.max_time_s),
        "include_gravity": int(settings.include_gravity),
        "sand_density_kg_m3": float(material.density_kg_m3),
        "sand_friction_angle_deg": float(material.friction_angle_deg),
        "sand_cohesion_pa": float(material.cohesion_pa),
        "sand_grain_diameter_m": float(material.grain_diameter_m),
    }
    return FieldProvenance(
        fidelity_tier=solver.fidelity_tier,
        envelope_status=verdict.status,
        solver_name=f"{type(solver).__module__}.{type(solver).__name__}",
        kinematics=WHOLE_SHOT_KINEMATICS_NOTE,
        peak_speed_m_s=float(state.speed_m_s),
        caveats=tuple(caveat.value for caveat in verdict.caveats),
        reasons=tuple(verdict.reasons),
        refused=tuple(quantity.value for quantity in RefusedQuantity),
        settings=entries,
        seeds=(),
    )
