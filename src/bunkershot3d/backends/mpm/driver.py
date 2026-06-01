"""
MuJoCo backend driver for BunkerShot3D.
Uses discrete spheres as an approximation for granular media if MPM is unavailable.
"""

from pathlib import Path

import typing

try:
    import mujoco
except ImportError:
    mujoco = None

import numpy as np

from bunkershot3d.config import BunkerShotConfig
from bunkershot3d.io.schema import BunkerShotResultWriter
from bunkershot3d.kinematics.trajectory import SwingTrajectory


class MPMDriver:
    """Driver for running the bunker shot simulation using MuJoCo."""

    def __init__(self, config_path: Path | str) -> None:
        self.config_path = Path(config_path)
        self.config = BunkerShotConfig.from_yaml(self.config_path)

        self.model: typing.Any = None
        self.data: typing.Any = None

    def _generate_xml(self) -> str:
        """Generate the MJCF XML string for the bunker and clubhead."""
        lx, ly, lz = self.config.domain_extents()

        # We will use a smaller number of spheres for draft simulation performance
        num_grains = min(1000, self.config.grain_count)
        r = self.config.grain_diameter_mean / 2.0

        # Simple clubhead block
        ch_w = self.config.clubhead_width
        ch_h = self.config.clubhead_height

        xml = f"""
        <mujoco model="bunkershot">
            <option timestep="0.001" gravity="0 0 -9.81" />
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

        # Add grains — seeded for reproducibility (mirrors Chrono backend seed=42)
        rng = np.random.default_rng(seed=42)
        for i in range(num_grains):
            px = rng.uniform(-lx / 2 + r, lx / 2 - r)
            py = rng.uniform(-ly / 2 + r, ly / 2 - r)
            pz = rng.uniform(r, lz - r)
            xml += f"""
                <body name="g{i}" pos="{px} {py} {pz}">
                    <freejoint/>
                    <geom type="sphere" size="{r}" rgba="0.9 0.8 0.5 1"/>
                </body>
            """

        xml += """
            </worldbody>
        </mujoco>
        """
        return xml

    def setup(self) -> None:
        """Setup the MuJoCo model."""
        xml_string = self._generate_xml()
        self.model = mujoco.MjModel.from_xml_string(xml_string)
        self.data = mujoco.MjData(self.model)

    def _load_trajectory(self) -> SwingTrajectory | None:
        """Load the swing trajectory from the configured file, or return None if unavailable."""
        traj_file = Path(self.config.trajectory_file)
        if traj_file.is_absolute() and traj_file.exists():
            return SwingTrajectory.from_csv(traj_file)
        # Try relative to config directory
        candidate = self.config_path.parent / traj_file
        if candidate.exists():
            return SwingTrajectory.from_csv(candidate)
        return None

    def _extract_contact_wrench(
        self, clubhead_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Aggregate contact forces/torques for contacts involving the clubhead body.

        Iterates over ``data.contact`` entries and sums the 6-DOF contact force
        (expressed in the world frame via ``mj_contactForce``) whenever one of
        the two geoms in the contact pair belongs to the clubhead body.

        Args:
            clubhead_id: MuJoCo body id of the clubhead.

        Returns:
            force (3,), torque (3,) summed over all relevant contacts.
        """
        assert self.model is not None and self.data is not None

        force_total = np.zeros(3)
        torque_total = np.zeros(3)

        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1_id = contact.geom1
            geom2_id = contact.geom2

            body1 = self.model.geom_bodyid[geom1_id]
            body2 = self.model.geom_bodyid[geom2_id]

            if body1 == clubhead_id or body2 == clubhead_id:
                # Extract the 6-DOF contact wrench in the contact frame
                raw = np.zeros(6)
                mujoco.mj_contactForce(self.model, self.data, i, raw)
                # raw[:3] is force, raw[3:] is torque in the contact frame.
                # MuJoCo's contact.frame stores the contact-frame basis vectors
                # as ROWS in world coordinates, i.e. it maps world->contact.
                # Converting a contact-frame vector to world therefore needs the
                # TRANSPOSE (#6639 F3): world = frame.T @ contact.
                frame = contact.frame.reshape(3, 3)
                force_total += frame.T @ raw[:3]
                torque_total += frame.T @ raw[3:]

        return force_total, torque_total

    def run(self, output_path: Path | str) -> None:
        """Run the simulation and write HDF5 output."""
        if self.model is None or self.data is None:
            self.setup()

        assert self.model is not None and self.data is not None

        writer = BunkerShotResultWriter(output_path)

        # Settle the grains first
        for _ in range(500):
            mujoco.mj_step(self.model, self.data)

        # Impact phase
        clubhead_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "clubhead"
        )

        dt = self.model.opt.timestep
        # Derive step count from configured trajectory duration
        n_steps = int(round(self.config.trajectory_duration / dt))

        # Load swing trajectory if available
        trajectory = self._load_trajectory()

        # Determine the qpos offset for the clubhead freejoint (first body after world)
        # A freejoint contributes 7 qpos values (pos xyz + quat wxyz).
        # The clubhead is the first free body; its qpos start index is 0.
        clubhead_qpos_start = 0
        # Walk bodies to find the correct qpos address for the clubhead freejoint
        for jid in range(self.model.njnt):
            if self.model.jnt_bodyid[jid] == clubhead_id:
                clubhead_qpos_start = self.model.jnt_qposadr[jid]
                break

        for i in range(n_steps):
            time = i * dt

            if trajectory is not None:
                # Interpolate prescribed position and orientation from trajectory
                pos, quat, _lv, _av = trajectory.interpolate(time)
                # Override clubhead qpos: positions then quaternion (wxyz)
                self.data.qpos[clubhead_qpos_start : clubhead_qpos_start + 3] = pos
                self.data.qpos[clubhead_qpos_start + 3 : clubhead_qpos_start + 7] = quat
            else:
                # Fallback: advance position along x at a fixed velocity
                vel = 5.0  # m/s
                self.data.qpos[clubhead_qpos_start] += vel * dt

            mujoco.mj_step(self.model, self.data)

            # Extract state
            pos = self.data.xpos[clubhead_id]
            quat = self.data.xquat[clubhead_id]
            writer.write_clubhead_state(time, pos, quat)

            # Proper contact wrench aggregation over clubhead contacts
            force, torque = self._extract_contact_wrench(clubhead_id)
            writer.write_contact_wrench(time, force, torque)

        writer.close()
