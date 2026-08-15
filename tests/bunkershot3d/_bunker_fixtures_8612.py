"""Shared helpers for the W5 backend-correctness regression tests (#8612).

Kept in one place so the MuJoCo, Chrono and LIGGGHTS regression suites agree on
what a physically admissible configuration looks like.

``QUARTZ_*`` are the measured properties of silica sand used by
``calibration/configs/canonical.yaml`` after #8612: a Young's modulus of
1e7 Pa gives 47 % Hertzian grain interpenetration at 25 m/s, so the soft-DEM
folklore value cannot be used for an impact problem.
"""

from __future__ import annotations

import math
import textwrap
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

QUARTZ_YOUNGS_MODULUS_PA = 7.0e10
QUARTZ_POISSON_RATIO = 0.17
QUARTZ_DENSITY = 2650.0

#: Tour-professional clubhead speed at impact (m/s), per ADR-0032.
IMPACT_SPEED_MPS = 25.0


def hertz_overlap_ratio(
    impact_speed: float,
    density: float,
    youngs_modulus: float,
    poisson_ratio: float,
) -> float:
    """Peak Hertzian overlap of two equal spheres, as a fraction of diameter.

    Written out longhand here (rather than imported from the implementation)
    so the regression tests pin the *physics*, not the code under test.

    Energy balance for a binary impact of two identical spheres::

        (1/2) m* v^2 = (8/15) E* sqrt(R*) delta_max^(5/2)

    with ``m* = m/2``, ``R* = R/2`` and ``E* = E / (2 (1 - nu^2))``. Because
    ``m*/sqrt(R*)`` scales as ``R^(5/2)``, ``delta_max`` scales as ``R`` and the
    ratio ``delta_max / d`` is independent of grain size — coarse-graining
    cannot fix an over-soft stiffness, only raising ``E`` can.
    """
    e_star = youngs_modulus / (2.0 * (1.0 - poisson_ratio**2))
    radius = 1.0  # scale-invariant; any radius gives the same ratio
    m_star = 0.5 * density * (4.0 / 3.0) * math.pi * radius**3
    r_star = radius / 2.0
    delta_max = (
        (15.0 / 16.0) * m_star * impact_speed**2 / (e_star * math.sqrt(r_star))
    ) ** 0.4
    return float(delta_max / (2.0 * radius))


def rayleigh_time(
    radius: float, density: float, youngs_modulus: float, poisson_ratio: float
) -> float:
    """Rayleigh surface-wave transit time for a sphere (seconds).

    ``t_R = pi R sqrt(rho / G) / (0.1631 nu + 0.8766)``; DEM integrators are
    normally run at 0.1-0.2 t_R.
    """
    shear_modulus = youngs_modulus / (2.0 * (1.0 + poisson_ratio))
    return float(
        math.pi
        * radius
        * math.sqrt(density / shear_modulus)
        / (0.1631 * poisson_ratio + 0.8766)
    )


def config_yaml(
    *,
    grain_count: int = 100,
    diameter_mean: float = 0.01,
    diameter_sigma_log: float = 0.0,
    youngs_modulus: float = QUARTZ_YOUNGS_MODULUS_PA,
    poisson_ratio: float = QUARTZ_POISSON_RATIO,
    duration: float = 0.005,
    rate_hz: float = 1000.0,
    trajectory_file: str = "swing.csv",
    length_x: float = 2.0,
    width_y: float = 1.0,
    depth_z: float = 0.5,
) -> str:
    """Render a ``BunkerShotConfig`` YAML document with overridable fields."""
    return textwrap.dedent(f"""\
        bunker_bed:
          domain:
            length_x: {length_x}
            width_y: {width_y}
            depth_z: {depth_z}
          boundary: "fixed"
        grain_population:
          count: {grain_count}
          diameter_mean: {diameter_mean}
          diameter_sigma_log: {diameter_sigma_log}
          density: {QUARTZ_DENSITY}
          coarse_graining_factor: 1.0
        contact_model:
          friction_coefficient: 0.5
          restitution_coefficient: 0.3
          youngs_modulus: {youngs_modulus}
          poisson_ratio: {poisson_ratio}
        clubhead:
          loft_deg: 56.0
          bounce_deg: 10.0
          width: 0.1
          height: 0.05
          mass: 0.3
        trajectory:
          file: "{trajectory_file}"
          duration: {duration}
        output:
          downsample_grains: 1
          rate_hz: {rate_hz}
        """)


def write_config(path: Path, **kwargs: object) -> Path:
    """Write a config YAML to *path* and return it."""
    path.write_text(config_yaml(**kwargs), encoding="utf-8")  # type: ignore[arg-type]
    return path


def write_straight_trajectory(
    path: Path,
    *,
    speed: float = 1.0,
    duration: float = 0.01,
    n_samples: int = 21,
    start: tuple[float, float, float] = (-0.5, 0.0, 0.05),
    direction: tuple[float, float, float] = (1.0, 0.0, 0.0),
) -> Path:
    """Write a constant-velocity straight-line trajectory CSV.

    Slow by default so the derived CFL timestep keeps regression tests cheap;
    the physics under test does not depend on the speed being tour-realistic.
    """
    time = np.linspace(0.0, duration, n_samples)
    unit = np.asarray(direction, dtype=float)
    unit = unit / np.linalg.norm(unit)
    velocity = speed * unit
    positions = (
        np.asarray(start, dtype=float)[None, :] + velocity[None, :] * time[:, None]
    )
    frame = pd.DataFrame(
        {
            "time": time,
            "px": positions[:, 0],
            "py": positions[:, 1],
            "pz": positions[:, 2],
            "qw": np.ones(n_samples),
            "qx": np.zeros(n_samples),
            "qy": np.zeros(n_samples),
            "qz": np.zeros(n_samples),
            "vx": np.full(n_samples, velocity[0]),
            "vy": np.full(n_samples, velocity[1]),
            "vz": np.full(n_samples, velocity[2]),
            "wx": np.zeros(n_samples),
            "wy": np.zeros(n_samples),
            "wz": np.zeros(n_samples),
        }
    )
    frame.to_csv(path, index=False)
    return path


def make_mock_chrono() -> MagicMock:
    """Build a ``MagicMock`` that quacks like ``pychrono``.

    ``pychrono`` is not, and per ADR-0032 will not become, a declared
    dependency of this repository, so the Chrono driver is only ever exercised
    against this mock. Every body the driver creates is recorded on
    ``created_bodies`` in creation order (five walls, then the grains, then the
    clubhead) so tests can inspect grain placement.
    """
    chrono = MagicMock(name="pychrono")
    created: list[MagicMock] = []

    def make_vec(x: float = 0.0, y: float = 0.0, z: float = 0.0) -> MagicMock:
        vec = MagicMock()
        vec.x, vec.y, vec.z = float(x), float(y), float(z)
        return vec

    chrono.ChVector3d.side_effect = make_vec

    def make_body() -> MagicMock:
        body = MagicMock()
        body.GetPos.return_value = make_vec(0.0, 0.0, 0.0)
        rotation = MagicMock()
        rotation.e0, rotation.e1, rotation.e2, rotation.e3 = 1.0, 0.0, 0.0, 0.0
        body.GetRot.return_value = rotation
        body.GetContactForce.return_value = make_vec(0.0, 0.0, 0.0)
        body.GetContactTorque.return_value = make_vec(0.0, 0.0, 0.0)
        body.GetAppliedForce.return_value = make_vec(0.0, 0.0, 0.0)
        body.GetAppliedTorque.return_value = make_vec(0.0, 0.0, 0.0)
        created.append(body)
        return body

    chrono.ChBody.side_effect = make_body
    chrono.created_bodies = created
    return chrono
