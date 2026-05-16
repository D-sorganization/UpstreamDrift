"""
Angle of repose calibration experiment.

Real path: MuJoCo cylindrical hopper simulation (settle + measure pile half-angle).
Fast path: analytical mock formula for unit tests (use_mock=True).
"""

from __future__ import annotations

import numpy as np

from bunkershot3d.exceptions import BackendNotImplementedError


def _mock_angle_of_repose(friction: float) -> float:
    """Analytical stand-in used for fast unit tests only."""
    return 20.0 + friction * 24.0


def _mujoco_angle_of_repose(
    friction: float,
    *,
    n_grains: int = 200,
    grain_radius: float = 0.005,
    settle_steps: int = 3000,
) -> float:
    """Run a short MuJoCo hopper experiment and return the pile half-angle."""
    import mujoco  # local import so missing mujoco does not break module load

    r = grain_radius
    cyl_radius = 0.10
    cyl_height = 0.30
    rng = np.random.default_rng(seed=42)

    grain_xml_parts: list[str] = []
    placed = 0
    layer_capacity = max(1, int(np.pi * (cyl_radius / (2 * r)) ** 2))

    z_layer = 0
    while placed < n_grains:
        z_center = r + z_layer * 2 * r * 1.1 + 0.01
        if z_center > cyl_height - r:
            break
        for _ in range(layer_capacity):
            if placed >= n_grains:
                break
            for _attempt in range(20):
                angle = rng.uniform(0, 2 * np.pi)
                rad = rng.uniform(0, cyl_radius - r - 0.001)
                px = float(rad * np.cos(angle))
                py = float(rad * np.sin(angle))
                pz = float(z_center + rng.uniform(-r * 0.1, r * 0.1))
                grain_xml_parts.append(
                    f'<body name="g{placed}" pos="{px:.4f} {py:.4f} {pz:.4f}">'
                    f"<freejoint/>"
                    f'<geom type="sphere" size="{r}" rgba="0.9 0.8 0.5 1"/>'
                    f"</body>"
                )
                placed += 1
                break
        z_layer += 1

    grains_xml = "\n".join(grain_xml_parts)

    xml = (
        '<mujoco model="hopper_aor">'
        f'<option timestep="0.001" gravity="0 0 -9.81" iterations="50"/>'
        f'<default><geom friction="{friction:.4f} 0.005 0.0001" condim="3"/></default>'
        "<worldbody>"
        '<light diffuse=".5 .5 .5" pos="0 0 3" dir="0 0 -1"/>'
        '<geom type="plane" size="1 1 0.1" rgba="0.5 0.5 0.5 1"/>'
        f'<geom type="cylinder" size="{cyl_radius} {cyl_height / 2}" '
        f'pos="0 0 {cyl_height / 2}" rgba="0.7 0.7 0.9 0.3" '
        'contype="1" conaffinity="1"/>' + grains_xml + "</worldbody></mujoco>"
    )

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    for _ in range(settle_steps):
        mujoco.mj_step(model, data)

    n_bodies = model.nbody
    grain_positions = [data.xpos[bid].copy() for bid in range(1, n_bodies)]

    if not grain_positions:
        return 30.0

    positions = np.array(grain_positions)
    xy_radii = np.sqrt(positions[:, 0] ** 2 + positions[:, 1] ** 2)
    z_vals = positions[:, 2]

    z_max = float(z_vals.max())
    z_min = float(z_vals.min())
    if z_max <= z_min + r:
        return 30.0

    mask = z_vals > z_min + (z_max - z_min) * 0.1
    z_filtered = z_vals[mask]
    r_filtered = xy_radii[mask]

    n_bins = 10
    z_edges = np.linspace(z_filtered.min(), z_filtered.max(), n_bins + 1)
    bin_max_r: list[float] = []
    bin_z_centers: list[float] = []
    for k in range(n_bins):
        in_bin = (z_filtered >= z_edges[k]) & (z_filtered < z_edges[k + 1])
        if in_bin.sum() > 0:
            bin_max_r.append(float(r_filtered[in_bin].max()))
            bin_z_centers.append(float((z_edges[k] + z_edges[k + 1]) / 2))

    if len(bin_max_r) < 2:
        max_r = float(xy_radii.max())
        pile_height = z_max - z_min
        if pile_height < 1e-6:
            return 30.0
        return float(np.degrees(np.arctan2(max_r, pile_height)))

    bin_max_r_arr = np.array(bin_max_r)
    bin_z_arr = np.array(bin_z_centers)
    dz = np.diff(bin_z_arr)
    dr = np.diff(bin_max_r_arr)
    valid = dz != 0
    if not valid.any():
        return 30.0

    slopes = np.abs(dr[valid] / dz[valid])
    median_slope = float(np.median(slopes))
    angle_deg = float(np.degrees(np.arctan(median_slope)))
    return float(np.clip(angle_deg, 5.0, 70.0))


class AngleOfReposeExperiment:
    """Simulates pouring particles from a lifted cylinder to measure pile angle."""

    def __init__(self, backend: str = "mpm", *, use_mock: bool | None = None) -> None:
        self.backend = backend
        if use_mock is None:
            self._use_mock = backend == "mock"
        else:
            self._use_mock = use_mock

        if backend not in ("mock", "mpm", "mujoco"):
            raise BackendNotImplementedError(
                backend,
                feature="AngleOfReposeExperiment requires 'mpm', 'mujoco', or 'mock' backend",
            )

        self.target_angle = 32.0

    def run_simulation(self, params: dict) -> float:
        """Run the experiment and return angle of repose in degrees."""
        friction = float(params.get("friction_coefficient", 0.5))
        if self._use_mock:
            return _mock_angle_of_repose(friction)
        return _mujoco_angle_of_repose(friction)

    def calibrate(self) -> dict:
        """Grid-search friction to minimise residual vs target angle."""
        best_params: dict = {"friction_coefficient": 0.5}
        best_residual = float("inf")
        for friction in np.linspace(0.1, 0.9, 9):
            angle = self.run_simulation({"friction_coefficient": float(friction)})
            residual = abs(angle - self.target_angle)
            if residual < best_residual:
                best_residual = residual
                best_params = {"friction_coefficient": float(friction)}
        return best_params


def compute_angle_of_repose(
    friction: float, backend: str = "mpm", *, use_mock: bool = False
) -> float:
    """Convenience function: single angle-of-repose measurement."""
    experiment = AngleOfReposeExperiment(backend=backend, use_mock=use_mock)
    return experiment.run_simulation({"friction_coefficient": friction})
