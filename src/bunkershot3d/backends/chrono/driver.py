"""
Project Chrono backend driver for BunkerShot3D.

Per ADR-0032 this is an **F3 (grain-scale DEM)** tier and explicit accepted
debt: ``pychrono`` is not a declared dependency and the backend is only ever
exercised against a mock. Issue #8612 fixed four defects here: the reported
wrench was ``GetAppliedForce()`` — the externally applied load, ~0 by
construction, not the contact reaction (B3); the swing trajectory was ignored
in favour of a hard-coded 5 m/s and 200 fixed steps (B9); ``1 / output_rate_hz``
was used as the *integrator* step, ~11 900x the Rayleigh limit (B30); and
grains were stacked on ``np.linspace`` z-layers (B16).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from ...config import BunkerShotConfig
from ...io.schema import BunkerShotResultWriter
from ...kinematics.trajectory import SwingTrajectory
from ...provenance import RunManifest, Validity
from ..packing import lattice_positions
from ..prescribed_motion import load_trajectory
from ..run_provenance import (
    GRAIN_POSITION_SEED,
    GRAIN_RADII_SEED,
    dem_run_manifest,
    fixed_seed_record,
)
from ..stability import (
    DEFAULT_MAX_STEPS,
    StepPlan,
    largest_grain_radius,
    plan_from_config,
    smallest_grain_radius,
    validate_contact_model,
)

try:
    import pychrono as chrono  # type: ignore[import-untyped]

    _HAS_CHRONO = True
except ImportError:
    _HAS_CHRONO = False


from ...exceptions import BackendNotImplementedError

#: Structural relaxation steps run before the swing, at the integration
#: timestep. At the Rayleigh limit these cover a fraction of a millisecond;
#: they are not a physical consolidation time.
SETTLE_STEPS = 500

#: Why a Chrono result is never labelled valid (ADR-0032).
_F3_VERDICT = (
    "Chrono is an F3 grain-scale DEM tier. A USGA bunker base at true scale "
    "needs 2.1e8 grains and days per shot, so any tractable run here is a "
    "coarse-grained proxy, not the bunker. Use it for grain-scale studies "
    "only, never as a design answer."
)


class ChronoDriver:
    """Driver for running the bunker shot simulation using Project Chrono."""

    def __init__(self, config_path: Path | str) -> None:
        self.config_path = Path(config_path)
        self.config = BunkerShotConfig.from_yaml(self.config_path)
        self._system: Any = None
        self._clubhead_body: Any = None

    def _make_contact_material(self) -> Any:
        """Build a ``ChContactMaterialSMC`` from the flat contact params.

        Single source of truth for SMC material configuration (issue #6936):
        walls, grains, and the clubhead all call this so a new contact
        property is wired once rather than in three drifting copies. Reads
        the assembled ``config.to_contact_material()`` value object
        (issue #8608) rather than reaching into ``config.contact_model.*``.
        """
        params = self.config.to_contact_material()
        material = chrono.ChContactMaterialSMC()
        material.SetFriction(params.friction)
        material.SetRestitution(params.restitution)
        material.SetYoungModulus(params.youngs_modulus_pa)
        material.SetPoissonRatio(params.poisson_ratio)
        return material

    def _grain_radii(self, count: int) -> np.ndarray:
        """Log-normal grain radii, bounded at +/- ``SIGMA_SPAN`` in log-space.

        The bound is what makes the lattice packing guarantee hold: the
        placement uses the largest admissible diameter as its pitch.
        """
        rng = np.random.default_rng(seed=GRAIN_RADII_SEED)
        grains = self.config.to_grain_population()
        r_mean = grains.radius_mean_m
        sigma = grains.diameter_sigma_log
        radii = rng.lognormal(mean=np.log(r_mean), sigma=sigma, size=count)
        return np.clip(
            radii,
            smallest_grain_radius(self.config),
            largest_grain_radius(self.config),
        )

    def _build_system(self) -> Any:
        """Construct and return a populated ChSystemSMC."""
        system = chrono.ChSystemSMC()
        system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.80665))

        lx, ly, lz = self.config.to_domain_box().extents_m

        wall_material = self._make_contact_material()

        def _add_fixed_box(
            sys: Any,
            mat: Any,
            half_extents: tuple[float, float, float],
            pos: tuple[float, float, float],
        ) -> None:
            body = chrono.ChBody()
            body.SetFixed(True)
            body.SetPos(chrono.ChVector3d(*pos))
            shape = chrono.ChCollisionShapeBox(
                mat, half_extents[0], half_extents[1], half_extents[2]
            )
            body.AddCollisionShape(shape)
            body.EnableCollision(True)
            sys.Add(body)

        # Floor and four walls
        _add_fixed_box(system, wall_material, (lx / 2, ly / 2, 0.005), (0, 0, -0.005))
        _add_fixed_box(
            system, wall_material, (lx / 2, 0.005, lz / 2), (0, -ly / 2, lz / 2)
        )
        _add_fixed_box(
            system, wall_material, (lx / 2, 0.005, lz / 2), (0, ly / 2, lz / 2)
        )
        _add_fixed_box(
            system, wall_material, (0.005, ly / 2, lz / 2), (-lx / 2, 0, lz / 2)
        )
        _add_fixed_box(
            system, wall_material, (0.005, ly / 2, lz / 2), (lx / 2, 0, lz / 2)
        )

        # Grain contact material
        grain_mat = self._make_contact_material()

        grains = self.config.to_grain_population()
        density = grains.density_kg_m3
        cgf = grains.coarse_graining_factor
        effective_count = grains.effective_count

        radii = self._grain_radii(effective_count)
        positions = lattice_positions(
            count=effective_count,
            extents=(lx, ly, lz),
            diameter=2.0 * largest_grain_radius(self.config),
            rng=np.random.default_rng(seed=GRAIN_POSITION_SEED),
        )

        for i in range(effective_count):
            ri = float(radii[i])
            mass = density * (4.0 / 3.0) * np.pi * ri**3 * cgf
            grain = chrono.ChBody()
            grain.SetMass(mass)
            grain.SetPos(chrono.ChVector3d(*(float(v) for v in positions[i])))
            shape = chrono.ChCollisionShapeSphere(grain_mat, ri)
            grain.AddCollisionShape(shape)
            grain.EnableCollision(True)
            system.Add(grain)

        # Clubhead body
        clubhead = self.config.clubhead
        ch_w = clubhead.width
        ch_h = clubhead.height
        ch_mass = clubhead.mass

        clubhead_mat = self._make_contact_material()

        self._clubhead_body = chrono.ChBody()
        self._clubhead_body.SetMass(ch_mass)
        self._clubhead_body.SetPos(chrono.ChVector3d(-lx / 2, 0, lz))
        ch_shape = chrono.ChCollisionShapeBox(
            clubhead_mat, ch_w / 2, ch_w / 2, ch_h / 2
        )
        self._clubhead_body.AddCollisionShape(ch_shape)
        self._clubhead_body.EnableCollision(True)
        system.Add(self._clubhead_body)

        return system

    def setup(self) -> None:
        """Setup the Chrono system (grains, clubhead, constraints).

        Raises:
            BackendNotImplementedError: pychrono is not installed.
            ContactStiffnessError: The configured stiffness cannot resolve a
                tour-speed impact without gross grain interpenetration.
        """
        if not _HAS_CHRONO:
            raise BackendNotImplementedError(
                "pychrono is not installed. "
                "Install with: pip install upstream-drift[chrono]"
            )
        validate_contact_model(self.config)
        self._system = self._build_system()

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _prescribe_clubhead(
        self, trajectory: SwingTrajectory, time: float, *, moving: bool = True
    ) -> None:
        """Write the clubhead pose and velocity for ``time``.

        Chrono's contact solver, like MuJoCo's, works from body velocities, so
        prescribing the pose alone would present a stationary clubhead to the
        sand however fast the trajectory says it is travelling.
        """
        position, quaternion, lin_vel, ang_vel = trajectory.interpolate(time)
        body = self._clubhead_body
        body.SetPos(chrono.ChVector3d(*(float(v) for v in position)))
        body.SetRot(chrono.ChQuaterniond(*(float(q) for q in quaternion)))
        if moving:
            body.SetPosDt(chrono.ChVector3d(*(float(v) for v in lin_vel)))
            body.SetAngVelParent(chrono.ChVector3d(*(float(w) for w in ang_vel)))
        else:
            body.SetPosDt(chrono.ChVector3d(0.0, 0.0, 0.0))
            body.SetAngVelParent(chrono.ChVector3d(0.0, 0.0, 0.0))

    def _contact_wrench(self) -> tuple[np.ndarray, np.ndarray]:
        """Contact reaction on the clubhead, in the absolute frame.

        ``GetContactForce``/``GetContactTorque`` accumulate the contact wrench
        about the body's centre of mass. The previous code read
        ``GetAppliedForce``/``GetAppliedTorque``, which report *externally
        applied* loads and are therefore ~0 regardless of the sand (#8612, B3).
        """
        force = self._clubhead_body.GetContactForce()
        torque = self._clubhead_body.GetContactTorque()
        return (
            np.array([force.x, force.y, force.z], dtype=float),
            np.array([torque.x, torque.y, torque.z], dtype=float),
        )

    def run_manifest(self) -> RunManifest:
        """Provenance for a run of this driver (issue #8608, finding B18).

        Returns:
            A manifest naming the configuration hashes, both fixed grain seeds
            and the F3 verdict, ready to attach to the result file.
        """
        return dem_run_manifest(
            self.config,
            solver="chrono",
            seeds=(
                fixed_seed_record("grain-radii", GRAIN_RADII_SEED),
                fixed_seed_record("grain-positions", GRAIN_POSITION_SEED),
            ),
            validity=Validity.OUT_OF_ENVELOPE,
            validity_reason=_F3_VERDICT,
        )

    def _plan(self, trajectory: SwingTrajectory, max_steps: int) -> StepPlan:
        """Resolve a Rayleigh- and Courant-stable integration schedule."""
        return plan_from_config(
            self.config,
            max_speed=trajectory.max_linear_speed(),
            max_steps=max_steps,
            extra_steps=SETTLE_STEPS,
        )

    def run(
        self, output_path: Path | str, *, max_steps: int = DEFAULT_MAX_STEPS
    ) -> None:
        """Run the simulation and write HDF5 output.

        Args:
            output_path: Destination path for the HDF5 result file.
            max_steps: Ceiling on integration steps before the configuration is
                refused as intractable at this grain scale.

        Raises:
            BackendNotImplementedError: pychrono is not installed or
                setup() has not been called.
            TrajectoryUnavailableError: The configured swing does not resolve.
            StepBudgetExceededError: A stable run exceeds ``max_steps``.
        """
        if not _HAS_CHRONO:
            raise BackendNotImplementedError(
                "pychrono is not installed. "
                "Install with: pip install upstream-drift[chrono]"
            )
        if self._system is None:
            raise BackendNotImplementedError("Call setup() before run().")

        system = self._system
        trajectory = load_trajectory(self.config_path, self.config)
        plan = self._plan(trajectory, max_steps)
        start_time = float(trajectory.time[0])

        writer = BunkerShotResultWriter(output_path, manifest=self.run_manifest())
        try:
            # Settle phase: relax the packing with the clubhead parked.
            self._prescribe_clubhead(trajectory, start_time, moving=False)
            for _ in range(SETTLE_STEPS):
                system.DoStepDynamics(plan.dt)

            for step in range(plan.n_steps):
                time = start_time + step * plan.dt
                self._prescribe_clubhead(trajectory, time)
                system.DoStepDynamics(plan.dt)

                if step % plan.output_every:
                    continue
                ch_pos = self._clubhead_body.GetPos()
                ch_rot = self._clubhead_body.GetRot()
                writer.write_clubhead_state(
                    time,
                    np.array([ch_pos.x, ch_pos.y, ch_pos.z], dtype=float),
                    np.array([ch_rot.e0, ch_rot.e1, ch_rot.e2, ch_rot.e3], dtype=float),
                )
                force_arr, torque_arr = self._contact_wrench()
                writer.write_contact_wrench(time, force_arr, torque_arr)
        finally:
            writer.close()
