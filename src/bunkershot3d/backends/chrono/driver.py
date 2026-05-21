"""
Project Chrono backend driver for BunkerShot3D.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from bunkershot3d.config import BunkerShotConfig
from bunkershot3d.io.schema import BunkerShotResultWriter

try:
    import pychrono as chrono  # type: ignore[import-untyped]

    _HAS_CHRONO = True
except ImportError:
    _HAS_CHRONO = False


from bunkershot3d.exceptions import BackendNotImplementedError


class ChronoDriver:
    """Driver for running the bunker shot simulation using Project Chrono."""

    def __init__(self, config_path: Path | str) -> None:
        self.config_path = Path(config_path)
        self.config = BunkerShotConfig.from_yaml(self.config_path)
        self._system: Any = None
        self._clubhead_body: Any = None

    def _build_system(self) -> Any:
        """Construct and return a populated ChSystemSMC."""
        system = chrono.ChSystemSMC()
        system.SetGravitationalAcceleration(chrono.ChVector3d(0, 0, -9.81))

        domain = self.config.bunker_bed.domain
        lx = domain.length_x
        ly = domain.width_y
        lz = domain.depth_z

        wall_material = chrono.ChContactMaterialSMC()
        wall_material.SetFriction(self.config.contact_model.friction_coefficient)
        wall_material.SetRestitution(self.config.contact_model.restitution_coefficient)
        wall_material.SetYoungModulus(self.config.contact_model.youngs_modulus)
        wall_material.SetPoissonRatio(self.config.contact_model.poisson_ratio)

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
        grain_mat = chrono.ChContactMaterialSMC()
        grain_mat.SetFriction(self.config.contact_model.friction_coefficient)
        grain_mat.SetRestitution(self.config.contact_model.restitution_coefficient)
        grain_mat.SetYoungModulus(self.config.contact_model.youngs_modulus)
        grain_mat.SetPoissonRatio(self.config.contact_model.poisson_ratio)

        rng = np.random.default_rng(seed=42)
        count = self.config.grain_population.count
        r_mean = self.config.grain_population.diameter_mean / 2.0
        r_sigma = self.config.grain_population.diameter_sigma_log
        density = self.config.grain_population.density
        cgf = self.config.grain_population.coarse_graining_factor

        # Effective particle count after coarse-graining
        effective_count = max(1, int(count / cgf))

        radii = rng.lognormal(mean=np.log(r_mean), sigma=r_sigma, size=effective_count)
        px = rng.uniform(-lx / 2 + r_mean, lx / 2 - r_mean, effective_count)
        py = rng.uniform(-ly / 2 + r_mean, ly / 2 - r_mean, effective_count)
        pz_layers = np.linspace(r_mean, lz * 0.9, effective_count)

        for i in range(effective_count):
            ri = float(radii[i])
            mass = density * (4.0 / 3.0) * np.pi * ri**3 * cgf
            grain = chrono.ChBody()
            grain.SetMass(mass)
            grain.SetPos(
                chrono.ChVector3d(float(px[i]), float(py[i]), float(pz_layers[i]))
            )
            shape = chrono.ChCollisionShapeSphere(grain_mat, ri)
            grain.AddCollisionShape(shape)
            grain.EnableCollision(True)
            system.Add(grain)

        # Clubhead body
        ch_w = self.config.clubhead.width
        ch_h = self.config.clubhead.height
        ch_mass = self.config.clubhead.mass

        clubhead_mat = chrono.ChContactMaterialSMC()
        clubhead_mat.SetFriction(self.config.contact_model.friction_coefficient)
        clubhead_mat.SetRestitution(self.config.contact_model.restitution_coefficient)
        clubhead_mat.SetYoungModulus(self.config.contact_model.youngs_modulus)
        clubhead_mat.SetPoissonRatio(self.config.contact_model.poisson_ratio)

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
        """
        if not _HAS_CHRONO:
            raise BackendNotImplementedError(
                "pychrono is not installed. "
                "Install with: pip install upstream-drift[chrono]"
            )
        self._system = self._build_system()

    def run(self, output_path: Path | str) -> None:
        """Run the simulation and write HDF5 output.

        Args:
            output_path: Destination path for the HDF5 result file.

        Raises:
            BackendNotImplementedError: pychrono is not installed or
                setup() has not been called.
        """
        if not _HAS_CHRONO:
            raise BackendNotImplementedError(
                "pychrono is not installed. "
                "Install with: pip install upstream-drift[chrono]"
            )
        if self._system is None:
            raise BackendNotImplementedError("Call setup() before run().")

        system = self._system
        writer = BunkerShotResultWriter(output_path)

        dt = 1.0 / self.config.output.rate_hz

        # Settle phase: let grains come to rest before the clubhead swing
        for _ in range(500):
            system.DoStepDynamics(dt)

        # Impact phase: kinematically drive the clubhead in +x at 5 m/s
        impact_velocity = 5.0  # m/s
        for i in range(200):
            time = float(i) * dt

            pos_now = self._clubhead_body.GetPos()
            self._clubhead_body.SetPos(
                chrono.ChVector3d(
                    pos_now.x + impact_velocity * dt,
                    pos_now.y,
                    pos_now.z,
                )
            )

            system.DoStepDynamics(dt)

            ch_pos = self._clubhead_body.GetPos()
            ch_rot = self._clubhead_body.GetRot()
            pos_arr = np.array([ch_pos.x, ch_pos.y, ch_pos.z])
            quat_arr = np.array([ch_rot.e0, ch_rot.e1, ch_rot.e2, ch_rot.e3])
            writer.write_clubhead_state(time, pos_arr, quat_arr)

            ch_force = self._clubhead_body.GetAppliedForce()
            ch_torque = self._clubhead_body.GetAppliedTorque()
            force_arr = np.array([ch_force.x, ch_force.y, ch_force.z])
            torque_arr = np.array([ch_torque.x, ch_torque.y, ch_torque.z])
            writer.write_contact_wrench(time, force_arr, torque_arr)

        writer.close()
