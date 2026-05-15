"""
MuJoCo backend driver for BunkerShot3D.
Uses discrete spheres as an approximation for granular media if MPM is unavailable.
"""

from pathlib import Path
import numpy as np
import mujoco
from bunkershot3d.io.schema import BunkerShotResultWriter
from bunkershot3d.config import BunkerShotConfig


class MPMDriver:
    """Driver for running the bunker shot simulation using MuJoCo."""

    def __init__(self, config_path: Path | str) -> None:
        self.config_path = Path(config_path)
        self.config = BunkerShotConfig.from_yaml(self.config_path)

        self.model: mujoco.MjModel | None = None
        self.data: mujoco.MjData | None = None

    def _generate_xml(self) -> str:
        """Generate the MJCF XML string for the bunker and clubhead."""
        domain = self.config.bunker_bed.domain
        lx, ly, lz = domain.length_x, domain.width_y, domain.depth_z

        # We will use a smaller number of spheres for draft simulation performance
        num_grains = min(1000, self.config.grain_population.count)
        r = self.config.grain_population.diameter_mean / 2.0

        # Simple clubhead block
        ch_w = self.config.clubhead.width
        ch_h = self.config.clubhead.height

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

        # Add grains
        for i in range(num_grains):
            px = np.random.uniform(-lx / 2 + r, lx / 2 - r)
            py = np.random.uniform(-ly / 2 + r, ly / 2 - r)
            pz = np.random.uniform(r, lz - r)
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
        for i in range(200):
            time = i * dt

            # Kinematic override (move clubhead forward)
            vel = 5.0  # m/s
            self.data.qpos[0] += vel * dt  # Move x

            mujoco.mj_step(self.model, self.data)

            # Extract state
            pos = self.data.xpos[clubhead_id]
            quat = self.data.xquat[clubhead_id]

            writer.write_clubhead_state(time, pos, quat)

            # Wrench (mock extraction for draft)
            cfrc_ext = self.data.cfrc_ext[clubhead_id]
            writer.write_contact_wrench(time, cfrc_ext[:3], cfrc_ext[3:])

        writer.close()
