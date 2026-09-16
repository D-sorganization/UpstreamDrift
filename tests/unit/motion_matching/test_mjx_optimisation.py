"""Tests for the differentiable MJX trajectory optimiser and JAX contact physics (HO-3 #10157).

Validates:
1. Linear-interpolation knot basis properties (partition of unity, endpoint mapping, knot counts).
2. Cost-horizon knot masking (tail coordinate freezing).
3. Contact-law numerical parity: pure JAX contact_force_jax matches the reference
   sphere_ground_contact Hunt-Crossley + regularised Coulomb law to < 1e-6 N across
   200 random states.
4. Weld spring-damper reaction wrenches: equal and opposite on the two bodies; zero at coincidence.
5. initial_state foot preload: lowest sphere sits at -m*g / (k*n_spheres).
6. Differentiable rollout gradient finiteness on a 12-frame truncation of the driver package.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
from typing import Any

import numpy as np
import pytest

try:
    import jax
    import jax.numpy as jnp
    from mujoco import mjx

    # Enable float64 mode in JAX during tests for exact double-precision parity
    jax.config.update("jax_enable_x64", True)
    JAX_AVAILABLE = True
except ImportError:
    jax = None  # type: ignore[assignment]
    jnp = None  # type: ignore[assignment]
    mjx = None  # type: ignore[assignment]
    JAX_AVAILABLE = False

pytestmark = [
    pytest.mark.unit,
    pytest.mark.requires_jax,
    pytest.mark.skipif(not JAX_AVAILABLE, reason="requires jax and mujoco.mjx"),
]

from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    GroundPlane,
    random_contact_states,
    sphere_ground_contact,
)

SCRIPT_PATH = (
    Path(__file__).resolve().parents[3]
    / "docs"
    / "development"
    / "full_body_models"
    / "evidence"
    / "ground_support"
    / "mjx_trajectory_optimisation.py"
)

mjx_opt: Any = None
if JAX_AVAILABLE:
    spec = importlib.util.spec_from_file_location(
        "mjx_trajectory_optimisation", SCRIPT_PATH
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load spec for {SCRIPT_PATH}")
    mjx_opt = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mjx_opt)

DRIVER_PKG_DIR = SCRIPT_PATH.parent / "anthro_driver"


def test_knot_basis_properties() -> None:
    """Rows sum to 1, endpoints map to single knots, spacing gives expected counts."""
    # 1. Grid where times coincide with knot boundaries [0.0, ..., 1.80] (46 knots)
    endpoint_times = np.linspace(0.0, 1.80, 46)
    basis_aligned = mjx_opt.knot_basis(endpoint_times, 0.04)
    assert basis_aligned.shape == (46, 46)
    # Rows sum to 1
    np.testing.assert_allclose(basis_aligned.sum(axis=1), 1.0, atol=1e-12)
    # First frame maps to first knot
    np.testing.assert_allclose(basis_aligned[0], [1.0] + [0.0] * 45, atol=1e-12)
    # Last frame maps to last knot
    np.testing.assert_allclose(basis_aligned[-1], [0.0] * 45 + [1.0], atol=1e-12)

    # 2. Driver package times (654 frames, t=0.0 to 1.81389 s)
    pkg = dict(np.load(DRIVER_PKG_DIR / "mjx_package.npz"))
    driver_times = pkg["time_s"]
    basis_driver = mjx_opt.knot_basis(driver_times, 0.04)
    # Rows sum to 1
    np.testing.assert_allclose(basis_driver.sum(axis=1), 1.0, atol=1e-12)
    # First frame maps to single knot 0
    assert basis_driver[0, 0] == 1.0
    assert np.all(basis_driver[0, 1:] == 0.0)
    # Spacing 0.04 on driver times gives 47 knots on full capture (REVIEW.md 17 receipted)
    # and 46 knots when restricted to the 1.80 s horizon
    assert basis_driver.shape[1] == 47
    basis_180 = mjx_opt.knot_basis(driver_times[driver_times <= 1.80], 0.04)
    assert basis_180.shape[1] == 46


def test_knot_mask_horizon_freezes_tail() -> None:
    """With --horizon 1.45 every knot beyond 1.45 s is frozen."""
    knot_times = np.arange(0.0, 1.81389 + 0.04, 0.04)
    horizon = 1.45
    mask = mjx_opt.compute_knot_mask(knot_times, horizon)
    mask_np = np.asarray(mask)

    # Knots <= 1.45 s are 1.0, knots > 1.45 s are 0.0
    assert np.all(mask_np[knot_times <= horizon] == 1.0)
    assert np.all(mask_np[knot_times > horizon] == 0.0)
    assert np.sum(mask_np == 0.0) > 0

    # Reference with non-zero delta: coordinates beyond the horizon match q_track
    q_track = np.zeros((10, 5))
    act = np.array([1, 2, 3])
    basis = (
        np.eye(10)[:, : len(knot_times)]
        if len(knot_times) <= 10
        else np.zeros((10, len(knot_times)))
    )
    basis[9, -1] = 1.0  # last frame only depends on last knot
    delta = jnp.ones((len(knot_times), len(act)))
    ref = mjx_opt.build_reference(q_track, act, basis, delta, mask)
    # The last frame has mask 0, so correction is 0
    np.testing.assert_allclose(ref[9], 0.0)


def test_contact_force_parity_with_shared_law() -> None:
    """JAX contact_force_jax equals sphere_ground_contact to < 1e-6 N for 200 random states."""
    # Driver contact parameters from mjx_package.json
    params = ContactParameters(
        stiffness_n_m=50000.0,
        dissipation_s_m=2.0,
        static_friction=0.9,
        dynamic_friction=0.8,
        viscous_friction=0.0,
        transition_velocity_m_s=0.05,
    )
    ground = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)
    radius = 0.03

    states = random_contact_states(seed=42, count=200, radius=radius)
    max_diff = 0.0

    for centre_np, vel_np in states:
        sample = sphere_ground_contact(centre_np, vel_np, radius, ground, params)
        expected_force = sample.normal_force_n + sample.friction_force_n

        c_jax = jnp.asarray(centre_np, dtype=jnp.float64)
        v_jax = jnp.asarray(vel_np, dtype=jnp.float64)
        f_jax = mjx_opt.contact_force_jax(
            c_jax,
            v_jax,
            radius,
            params,
            ground_normal=jnp.asarray(ground.normal, dtype=jnp.float64),
            ground_height_m=ground.height_m,
        )
        f_np = np.asarray(f_jax)

        diff = float(np.linalg.norm(f_np - expected_force))
        if diff > max_diff:
            max_diff = diff

    # Parity across all 200 states within 1e-6 N
    assert max_diff < 1e-6, f"Max difference was {max_diff:.6e} N (expected < 1e-6 N)"


def test_weld_wrench_equal_and_opposite_and_zero_at_coincidence() -> None:
    """Weld spring-damper generates equal & opposite wrenches; zero when coincident at rest."""
    p_a = jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64)
    v_a = jnp.array([0.01, -0.02, 0.03], dtype=jnp.float64)
    r_a = jnp.eye(3, dtype=jnp.float64)
    w_a = jnp.array([0.1, 0.0, -0.1], dtype=jnp.float64)
    com_a = jnp.array([0.05, 0.1, 0.15], dtype=jnp.float64)
    com_b = jnp.array([0.15, 0.25, 0.35], dtype=jnp.float64)

    site_a = mjx_opt.SiteState(p=p_a, v=v_a, r=r_a, w=w_a, com=com_a)
    site_b_coincident = mjx_opt.SiteState(p=p_a, v=v_a, r=r_a, w=w_a, com=com_b)

    # 1. Coincident sites with zero relative velocity and identity orientation
    wa_zero, wb_zero = mjx_opt.weld_wrench_jax(site_a, site_b_coincident)
    np.testing.assert_allclose(np.asarray(wa_zero), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(wb_zero), 0.0, atol=1e-12)

    # 2. Displaced site and relative velocities: equal and opposite forces
    p_b = p_a + jnp.array([0.005, -0.002, 0.001], dtype=jnp.float64)
    v_b = v_a + jnp.array([-0.01, 0.02, 0.0], dtype=jnp.float64)
    r_b = r_a
    w_b = w_a
    site_b = mjx_opt.SiteState(p=p_b, v=v_b, r=r_b, w=w_b, com=com_b)

    wa, wb = mjx_opt.weld_wrench_jax(site_a, site_b)
    wa_np, wb_np = np.asarray(wa), np.asarray(wb)

    # Linear forces on body a and b are strictly equal and opposite
    np.testing.assert_allclose(wa_np[:3], -wb_np[:3], atol=1e-12)
    # Total linear force is zero
    np.testing.assert_allclose(wa_np[:3] + wb_np[:3], 0.0, atol=1e-12)


def test_initial_state_preload_lowest_sphere() -> None:
    """initial_state preloads the feet: lowest sphere sits at -m*g / (k*n_spheres)."""
    meta, pkg, model = mjx_opt.load_package(DRIVER_PKG_DIR)
    sim = mjx_opt.build(meta, pkg, model, substeps=1)
    spheres = int(pkg["sphere_site_ids"].size)
    mass = float(meta["mass_kg"])

    q0 = pkg["q_track"][0]
    v0 = np.zeros(len(pkg["dof_adr"]))
    d0 = mjx_opt.initial_state(sim, q0, v0, mass, spheres)

    # Heights of all spheres at the preloaded state
    heights = (
        np.asarray(d0.site_xpos[sim["sphere_sites"]]) @ np.asarray(sim["n_ground"])
        - np.asarray(sim["sphere_radii"])
        - sim["ground_h"]
    )
    lowest = float(np.min(heights))
    expected_depth = mass * 9.81 / (sim["k_n"] * spheres)
    expected_height = -expected_depth

    np.testing.assert_allclose(lowest, expected_height, atol=1e-6)


def test_gradient_finiteness_on_12_frame_truncation() -> None:
    """jax.grad of the rollout cost is finite on a 12-frame truncation with NaN targets."""
    meta, pkg, model = mjx_opt.load_package(DRIVER_PKG_DIR)
    substeps = 1
    sim = mjx_opt.build(meta, pkg, model, substeps=substeps)

    N = 12
    times = pkg["time_s"][:N]
    q_track = pkg["q_track"][:N]
    targets = jnp.asarray(np.nan_to_num(pkg["targets_m"][:N]))
    valid = jnp.asarray(
        pkg["valid"][:N] & np.isfinite(pkg["targets_m"][:N]).all(axis=2)
    )
    count = float(valid.sum())
    spacing = 0.04
    basis = jnp.asarray(mjx_opt.knot_basis(times, spacing))
    act = np.where(~pkg["root_mask"])[0]
    n_knots = basis.shape[1]
    spheres = int(pkg["sphere_site_ids"].size)
    mass = float(meta["mass_kg"])

    def reference(delta: jnp.ndarray) -> jnp.ndarray:
        q = jnp.asarray(q_track)
        return q.at[:, act].add(basis @ delta)

    def replay(delta: jnp.ndarray) -> jnp.ndarray:
        q_np = reference(delta)
        dt = float(np.mean(np.diff(times)))
        v_np = jnp.gradient(q_np, axis=0) / dt
        a_np = jnp.gradient(v_np, axis=0) / dt
        fractions = jnp.arange(substeps) / substeps

        def table(x: jnp.ndarray) -> jnp.ndarray:
            nxt = jnp.vstack([x[1:], x[-1:]])
            return x[:, None, :] + fractions[None, :, None] * (nxt - x)[:, None, :]

        d0 = mjx_opt.initial_state(sim, q_np[0], v_np[0], mass, spheres)
        return sim["rollout"](d0, table(q_np), table(v_np), table(a_np))

    def cost(delta: jnp.ndarray) -> jnp.ndarray:
        m = replay(delta)
        err2 = jnp.sum((m - targets) ** 2, axis=2)
        marker_cost = jnp.sum(jnp.where(valid, err2, 0.0)) / count
        reg = 1e-3 * jnp.mean(delta**2)
        return marker_cost + reg

    delta = jnp.zeros((n_knots, act.size))
    grad = jax.grad(cost)(delta)

    # Verify finite gradients on all coordinates and knots
    assert bool(jnp.all(jnp.isfinite(grad))), "Expected all gradients to be finite"
