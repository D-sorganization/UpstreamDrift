"""
MuJoCo backend driver for BunkerShot3D.
Uses discrete spheres as an approximation for granular media if MPM is unavailable.

Per ADR-0032 this is an **F3 (grain-scale DEM proxy)** tier, not the supported
design path. Issue #8612 fixed four defects here: the contact wrench had no
sign convention (B5) and no moment arm (B5b), the clubhead was teleported via
``qpos`` with a stale ``qvel`` (B4), and a missing trajectory silently became a
5 m/s swing (B10). Grain initialisation now uses a lattice (B16) and the
integration timestep follows a Courant criterion rather than the authored
1 ms (B13).
"""

from pathlib import Path

import typing

try:
    import mujoco
except (ImportError, OSError):
    # OSError covers a *present but unloadable* native library: on Windows a
    # MuJoCo built against a newer MSVC runtime than the host raises
    # ``OSError(1114, "DLL initialization routine failed")`` rather than
    # ImportError. Catching only ImportError let that escape and take the
    # whole launcher down at import time (#8084).
    mujoco = None

import numpy as np

from ...config import BunkerShotConfig
from ...io.schema import BunkerShotResultWriter
from ...kinematics.trajectory import SwingTrajectory
from ..packing import lattice_positions
from ..prescribed_motion import load_trajectory
from ..stability import (
    DEFAULT_MAX_STEPS,
    StepPlan,
    plan_from_config,
    validate_contact_model,
)
from .contact import contact_wrench_on_body

#: Structural relaxation steps run before the swing. These are integration
#: steps, not a physical settling time: at the Courant-limited timestep they
#: cover well under a millisecond. They exist to let the lattice relax, not to
#: consolidate a bed.
SETTLE_STEPS = 500

#: Draft cap on the number of MuJoCo spheres. The configured grain count is
#: routinely 50 000, which MuJoCo cannot carry at interactive cost.
MAX_SPHERES = 1000


class MPMDriver:
    """Driver for running the bunker shot simulation using MuJoCo."""

    def __init__(self, config_path: Path | str) -> None:
        self.config_path = Path(config_path)
        self.config = BunkerShotConfig.from_yaml(self.config_path)

        self.model: typing.Any = None
        self.data: typing.Any = None

    # ------------------------------------------------------------------
    # Model construction
    # ------------------------------------------------------------------

    def _grain_positions(self) -> np.ndarray:
        """Non-overlapping grain centres for the configured population.

        Replaces the previous ``rng.uniform`` draw over the whole domain, which
        initialised grains interpenetrating (#8612, B16).
        """
        extents = self.config.domain_extents()
        count = min(MAX_SPHERES, self.config.grain_count)
        return lattice_positions(
            count=count,
            extents=extents,
            diameter=self.config.grain_diameter_mean,
            rng=np.random.default_rng(seed=42),
        )

    def _generate_xml(self) -> str:
        """Generate the MJCF XML string for the bunker and clubhead."""
        lx, ly, lz = self.config.domain_extents()
        radius = self.config.grain_diameter_mean / 2.0
        positions = self._grain_positions()

        ch_w = self.config.clubhead_width
        ch_h = self.config.clubhead_height

        # The authored timestep is a placeholder: run() overwrites
        # ``model.opt.timestep`` with the Courant-stable value for the actual
        # swing speed before stepping (#8612, B13).
        xml = f"""
        <mujoco model="bunkershot">
            <option timestep="0.001" gravity="0 0 -9.80665" />
            <worldbody>
                <light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>
                <!-- Container -->
                <geom type="box" size="{lx / 2} {ly / 2} 0.01" pos="0 0 -0.01" rgba="0.8 0.8 0.8 1"/>
                <geom type="box" size="{lx / 2} 0.01 {lz / 2}" pos="0 {-ly / 2} {lz / 2}" rgba="0.8 0.8 0.8 0.5"/>
                <geom type="box" size="{lx / 2} 0.01 {lz / 2}" pos="0 {ly / 2} {lz / 2}" rgba="0.8 0.8 0.8 0.5"/>
                <geom type="box" size="0.01 {ly / 2} {lz / 2}" pos="{-lx / 2} 0 {lz / 2}" rgba="0.8 0.8 0.8 0.5"/>
                <geom type="box" size="0.01 {ly / 2} {lz / 2}" pos="{lx / 2} 0 {lz / 2}" rgba="0.8 0.8 0.8 0.5"/>

                <!-- Clubhead -->
                <body name="clubhead" pos="{-lx / 2} 0 {lz}">
                    <freejoint/>
                    <geom name="clubface" type="box" size="{ch_w / 2} {ch_w / 2} {ch_h / 2}" rgba="0.2 0.2 0.2 1"/>
                </body>
        """

        # Accumulate the per-grain blocks in a list and ``"".join`` once instead
        # of ``xml += ...`` in the loop, which is O(N^2) for up to 1000 grains.
        parts = [xml]
        for index, (px, py, pz) in enumerate(positions):
            parts.append(f"""
                <body name="g{index}" pos="{px} {py} {pz}">
                    <freejoint/>
                    <geom type="sphere" size="{radius}" rgba="0.9 0.8 0.5 1"/>
                </body>
            """)

        parts.append("""
            </worldbody>
        </mujoco>
        """)
        return "".join(parts)

    def setup(self) -> None:
        """Build the MuJoCo model.

        Raises:
            ContactStiffnessError: The configured stiffness cannot resolve a
                tour-speed impact without gross grain interpenetration.
        """
        validate_contact_model(self.config)
        xml_string = self._generate_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)

    def _require_model(self) -> tuple[typing.Any, typing.Any]:
        """Return ``(model, data)``, or raise if ``setup()`` has not run.

        A plain ``raise``: ``assert`` is stripped by ``python -O``.
        """
        if self.model is None or self.data is None:
            raise RuntimeError("MPMDriver.setup() must be called before run()")
        return self.model, self.data

    # ------------------------------------------------------------------
    # Kinematics
    # ------------------------------------------------------------------

    def _clubhead_addresses(self, clubhead_id: int) -> tuple[int, int]:
        """``(qpos_adr, dof_adr)`` of the clubhead free joint."""
        model, _data = self._require_model()
        for joint_id in range(model.njnt):
            if model.jnt_bodyid[joint_id] == clubhead_id:
                return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])
        raise RuntimeError("the clubhead body has no joint to prescribe")

    def _prescribe_clubhead(
        self,
        trajectory: SwingTrajectory,
        time: float,
        dt: float,
        qpos_adr: int,
        dof_adr: int,
        *,
        moving: bool = True,
    ) -> None:
        """Write the clubhead pose **and** velocity for ``time``.

        Writing ``qpos`` alone (the previous behaviour, #8612 B4) leaves
        ``qvel`` at whatever the free body accumulated under gravity, so the
        constraint solver — which works at the velocity level — never sees the
        swing and the sand is never struck at speed.

        A mocap body or a weld would not fix this: MuJoCo treats a mocap body
        as having zero velocity, and a weld lets the 0.3 kg head be pushed by
        the constraint solver. Setting both fields from the trajectory, with
        ``mj_differentiatePos`` supplying the free-joint velocity convention
        (linear in world, angular in the body frame), is the only option that
        is self-consistent by construction.
        """
        model, data = self._require_model()

        position, quaternion, _lin_vel, _ang_vel = trajectory.interpolate(time)
        data.qpos[qpos_adr : qpos_adr + 3] = position
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = quaternion

        if not moving:
            data.qvel[dof_adr : dof_adr + 6] = 0.0
            return

        next_position, next_quaternion, _lv, _av = trajectory.interpolate(time + dt)
        qpos_now = np.asarray(data.qpos, dtype=float).copy()
        qpos_next = qpos_now.copy()
        qpos_next[qpos_adr : qpos_adr + 3] = next_position
        qpos_next[qpos_adr + 3 : qpos_adr + 7] = next_quaternion

        qvel = np.zeros(model.nv)
        mujoco.mj_differentiatePos(model, qvel, dt, qpos_now, qpos_next)
        data.qvel[dof_adr : dof_adr + 6] = qvel[dof_adr : dof_adr + 6]

    def _extract_contact_wrench(
        self, clubhead_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Total contact force and moment acting on the clubhead.

        See :func:`..mpm.contact.contact_wrench_on_body` for the sign
        convention and the moment-arm term.

        Args:
            clubhead_id: MuJoCo body id of the clubhead.

        Returns:
            force (3,), moment about the clubhead CoM (3,).
        """
        model, data = self._require_model()
        return contact_wrench_on_body(model, data, clubhead_id, mujoco=mujoco)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------

    def _plan(self, trajectory: SwingTrajectory, max_steps: int) -> StepPlan:
        """Resolve the integration schedule for this swing.

        The Rayleigh criterion is **not** applied here: MuJoCo resolves
        contacts implicitly at the velocity level with soft constraints rather
        than as Hertzian soft spheres, so the grain wave-speed limit does not
        govern. The Courant traversal limit does — at 25 m/s the authored 1 ms
        step moves the clubhead 25 mm, 62 diameters of 0.4 mm sand, so contacts
        are never generated at all.
        """
        return plan_from_config(
            self.config,
            max_speed=trajectory.max_linear_speed(),
            max_steps=max_steps,
            extra_steps=SETTLE_STEPS,
            enforce_rayleigh=False,
        )

    def run(
        self, output_path: Path | str, *, max_steps: int = DEFAULT_MAX_STEPS
    ) -> None:
        """Run the simulation and write HDF5 output.

        Args:
            output_path: Destination HDF5 file.
            max_steps: Ceiling on integration steps before the configuration is
                refused as intractable.

        Raises:
            TrajectoryUnavailableError: The configured swing does not resolve.
            ContactStiffnessError: The stiffness cannot resolve the impact.
            StepBudgetExceededError: A stable run exceeds ``max_steps``.
        """
        if self.model is None or self.data is None:
            self.setup()
        model, data = self._require_model()

        trajectory = load_trajectory(self.config_path, self.config)
        plan = self._plan(trajectory, max_steps)
        model.opt.timestep = plan.dt

        clubhead_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "clubhead")
        qpos_adr, dof_adr = self._clubhead_addresses(clubhead_id)
        start_time = float(trajectory.time[0])

        writer = BunkerShotResultWriter(output_path)
        try:
            # Relax the lattice with the clubhead parked at its start pose.
            for _ in range(SETTLE_STEPS):
                self._prescribe_clubhead(
                    trajectory, start_time, plan.dt, qpos_adr, dof_adr, moving=False
                )
                mujoco.mj_step(model, data)

            for step in range(plan.n_steps):
                time = start_time + step * plan.dt
                self._prescribe_clubhead(trajectory, time, plan.dt, qpos_adr, dof_adr)
                mujoco.mj_step(model, data)

                if step % plan.output_every:
                    continue
                writer.write_clubhead_state(
                    time,
                    np.asarray(data.xpos[clubhead_id], dtype=float).copy(),
                    np.asarray(data.xquat[clubhead_id], dtype=float).copy(),
                )
                force, torque = self._extract_contact_wrench(clubhead_id)
                writer.write_contact_wrench(time, force, torque)
        finally:
            writer.close()
