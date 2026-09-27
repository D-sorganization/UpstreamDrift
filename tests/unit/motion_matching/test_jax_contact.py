"""Tests for the differentiable JAX contact law and grip weld in src (epic #11006, #11037).

Validates:
1. Parity: JAX contact law matches the NumPy reference across 2000 states:
   - Normal force difference <= 1e-9 N
   - Friction force difference <= 1e-6 N
2. Gradient:
   - jax.grad of normal-force magnitude matches central finite difference of the
     NumPy reference to 1e-6 relative.
   - Gradient is finite at zero tangential velocity.
3. Weld:
   - Reaction forces are equal and opposite.
   - Reaction wrench is zero when sites coincide at rest with aligned frames.
   - Net moment about the world origin satisfies (b.p - a.p) x F_b.
4. Contracts:
   - Non-positive sphere radius raises PreconditionError / ContractViolationError.
   - Non-positive stiffness or negative damping raises PreconditionError / ContractViolationError.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402

from src.shared.python.core.contracts import (  # noqa: E402
    ContractViolationError,
    PreconditionError,
)
from src.shared.python.motion_matching.contact_law import (  # noqa: E402
    ContactParameters,
    GroundPlane,
    contact_parity_report,
    random_contact_states,
    sphere_ground_contact,
)
from src.shared.python.motion_matching.jax_contact import (  # noqa: E402
    SiteState,
    WeldGains,
    sphere_ground_contact_jax,
    weld_wrench_jax,
)

pytestmark = pytest.mark.unit


class _AdapterSample(NamedTuple):
    """Minimal adapter output satisfying contact_parity_report schema."""

    penetration_m: float
    normal_force_n: np.ndarray
    friction_force_n: np.ndarray


def _make_test_parameters() -> ContactParameters:
    """Standard representative contact parameters for parity and gradient tests."""
    return ContactParameters(
        stiffness_n_m=50000.0,
        dissipation_s_m=2.0,
        static_friction=0.9,
        dynamic_friction=0.8,
        viscous_friction=0.05,
        transition_velocity_m_s=0.05,
    )


def test_contact_law_parity_with_reference_numpy() -> None:
    """JAX contact law achieves required parity across 2000 states (normal <= 1e-9 N, friction <= 1e-6 N)."""
    ground = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)
    params = _make_test_parameters()
    radius = 0.03
    states = random_contact_states(seed=42, count=2000, radius=radius)

    def ref_adapter(
        center: np.ndarray, velocity: np.ndarray, r: float
    ) -> _AdapterSample:
        sample = sphere_ground_contact(center, velocity, r, ground, params)
        return _AdapterSample(
            penetration_m=sample.penetration_m,
            normal_force_n=sample.normal_force_n,
            friction_force_n=sample.friction_force_n,
        )

    def jax_adapter(
        center: np.ndarray, velocity: np.ndarray, r: float
    ) -> _AdapterSample:
        c_jax = jnp.asarray(center, dtype=jnp.float64)
        v_jax = jnp.asarray(velocity, dtype=jnp.float64)
        fn_jax, ft_jax = sphere_ground_contact_jax(c_jax, v_jax, r, ground, params)
        signed = float(np.dot(ground.normal, center)) - ground.height_m - r
        pen = max(0.0, -signed)
        return _AdapterSample(
            penetration_m=pen,
            normal_force_n=np.asarray(fn_jax, dtype=np.float64),
            friction_force_n=np.asarray(ft_jax, dtype=np.float64),
        )

    adapters: dict[str, Any] = {"numpy": ref_adapter, "jax": jax_adapter}
    report = contact_parity_report(
        adapters,  # type: ignore[arg-type]
        states,
        radius=radius,
    )

    max_fn = report["max_normal_force_difference_n"]["jax"]
    max_ft = report["max_friction_force_difference_n"]["jax"]

    assert max_fn <= 1e-9, (
        f"Measured max normal-force difference was {max_fn:.6e} N (required <= 1e-9 N)"
    )
    assert max_ft <= 1e-6, (
        f"Measured max friction-force difference was {max_ft:.6e} N (required <= 1e-6 N)"
    )


def test_contact_law_normal_force_gradient_matches_finite_difference() -> None:
    """jax.grad of normal-force magnitude matches central finite difference of NumPy law to 1e-6 relative."""
    ground = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)
    params = _make_test_parameters()
    radius = 0.03
    step_h = 1e-7

    # Penetrating states where rate * dissipation keeps magnitude > 0
    test_cases = [
        (np.array([0.1, -0.2, 0.015]), np.array([0.2, -0.3, -0.1])),
        (np.array([-0.05, 0.08, 0.025]), np.array([0.0, 0.0, -0.5])),
        (np.array([0.0, 0.0, 0.005]), np.array([0.1, -0.1, 0.05])),
    ]

    def normal_magnitude_jax(c: Any, v: Any) -> Any:
        fn, _ = sphere_ground_contact_jax(c, v, radius, ground, params)
        return jnp.sqrt(jnp.dot(fn, fn))

    grad_fn_jax = jax.grad(normal_magnitude_jax, argnums=0)

    for center_np, velocity_np in test_cases:
        c_jax = jnp.asarray(center_np, dtype=jnp.float64)
        v_jax = jnp.asarray(velocity_np, dtype=jnp.float64)

        analytical_grad = np.asarray(grad_fn_jax(c_jax, v_jax), dtype=np.float64)
        fd_grad = np.zeros(3, dtype=np.float64)

        for axis in range(3):
            shift = np.zeros(3, dtype=np.float64)
            shift[axis] = step_h

            c_plus = center_np + shift
            c_minus = center_np - shift

            sample_plus = sphere_ground_contact(
                c_plus, velocity_np, radius, ground, params
            )
            sample_minus = sphere_ground_contact(
                c_minus, velocity_np, radius, ground, params
            )

            norm_plus = float(np.linalg.norm(sample_plus.normal_force_n))
            norm_minus = float(np.linalg.norm(sample_minus.normal_force_n))

            fd_grad[axis] = (norm_plus - norm_minus) / (2.0 * step_h)

        grad_norm = float(np.linalg.norm(analytical_grad))
        assert grad_norm > 0.0, "Expected non-zero normal force gradient"

        rel_diff = float(np.linalg.norm(analytical_grad - fd_grad)) / grad_norm
        assert rel_diff <= 1e-6, (
            f"Normal force gradient relative error was {rel_diff:.6e} (required <= 1e-6)"
        )


def test_contact_law_gradient_finite_at_zero_tangential_velocity() -> None:
    """Gradients of contact force remain finite when tangential velocity is exactly zero."""
    ground = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)
    params = _make_test_parameters()
    radius = 0.03

    # Pure normal velocity along z -> tangential velocity is exactly zero
    center = jnp.array([0.05, 0.02, 0.01], dtype=jnp.float64)
    velocity = jnp.array([0.0, 0.0, -0.2], dtype=jnp.float64)

    def total_force_component(c: Any, v: Any, axis: int) -> Any:
        fn, ft = sphere_ground_contact_jax(c, v, radius, ground, params)
        total = fn + ft
        return total[axis]

    for axis in range(3):
        grad_c = jax.grad(total_force_component, argnums=0)(center, velocity, axis)
        grad_v = jax.grad(total_force_component, argnums=1)(center, velocity, axis)

        assert bool(jnp.all(jnp.isfinite(grad_c))), (
            f"Non-finite grad_c at zero tangential velocity for axis {axis}"
        )
        assert bool(jnp.all(jnp.isfinite(grad_v))), (
            f"Non-finite grad_v at zero tangential velocity for axis {axis}"
        )


def test_weld_wrench_equal_and_opposite_forces() -> None:
    """Forces exerted on body a and body b are strictly equal and opposite."""
    gains = WeldGains(k=2.0e5, c=400.0, rot_k=2.0e3, rot_c=4.0)

    site_a = SiteState(
        p=jnp.array([0.1, 0.2, 0.3], dtype=jnp.float64),
        v=jnp.array([0.01, -0.02, 0.03], dtype=jnp.float64),
        r=jnp.eye(3, dtype=jnp.float64),
        w=jnp.array([0.1, 0.0, -0.1], dtype=jnp.float64),
        com=jnp.array([0.05, 0.1, 0.15], dtype=jnp.float64),
    )
    site_b = SiteState(
        p=jnp.array([0.105, 0.198, 0.301], dtype=jnp.float64),
        v=jnp.array([0.0, 0.0, 0.03], dtype=jnp.float64),
        r=jnp.eye(3, dtype=jnp.float64),
        w=jnp.array([0.05, -0.02, 0.0], dtype=jnp.float64),
        com=jnp.array([0.15, 0.25, 0.35], dtype=jnp.float64),
    )

    wa, wb = weld_wrench_jax(site_a, site_b, gains)
    wa_np = np.asarray(wa)
    wb_np = np.asarray(wb)

    np.testing.assert_allclose(wa_np[:3], -wb_np[:3], atol=1e-12)
    np.testing.assert_allclose(wa_np[:3] + wb_np[:3], 0.0, atol=1e-12)


def test_weld_wrench_zero_at_coincidence_at_rest() -> None:
    """Wrenches are identically zero when sites coincide at rest with aligned frames."""
    gains = WeldGains(k=2.0e5, c=400.0, rot_k=2.0e3, rot_c=4.0)

    p = jnp.array([0.12, -0.05, 0.85], dtype=jnp.float64)
    v = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64)
    r = jnp.eye(3, dtype=jnp.float64)
    w = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float64)

    site_a = SiteState(
        p=p, v=v, r=r, w=w, com=jnp.array([0.1, 0.0, 0.8], dtype=jnp.float64)
    )
    site_b = SiteState(
        p=p, v=v, r=r, w=w, com=jnp.array([0.15, -0.1, 0.9], dtype=jnp.float64)
    )

    wa, wb = weld_wrench_jax(site_a, site_b, gains)
    np.testing.assert_allclose(np.asarray(wa), 0.0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(wb), 0.0, atol=1e-12)


def test_weld_wrench_net_moment_about_world_origin_identity() -> None:
    """Net moment about the world origin equals (b.p - a.p) x F_b, an exact identity."""
    gains = WeldGains(k=1.5e5, c=350.0, rot_k=1.8e3, rot_c=3.5)

    # Use general non-trivial positions, rotations and velocities
    theta_a = 0.3
    ca, sa = math.cos(theta_a), math.sin(theta_a)
    rot_a = jnp.array(
        [[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=jnp.float64
    )

    theta_b = -0.2
    cb, sb = math.cos(theta_b), math.sin(theta_b)
    rot_b = jnp.array(
        [[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]], dtype=jnp.float64
    )

    site_a = SiteState(
        p=jnp.array([0.25, -0.15, 0.45], dtype=jnp.float64),
        v=jnp.array([0.1, -0.05, 0.2], dtype=jnp.float64),
        r=rot_a,
        w=jnp.array([0.3, -0.2, 0.1], dtype=jnp.float64),
        com=jnp.array([0.2, -0.1, 0.4], dtype=jnp.float64),
    )
    site_b = SiteState(
        p=jnp.array([0.28, -0.14, 0.47], dtype=jnp.float64),
        v=jnp.array([-0.05, 0.1, -0.15], dtype=jnp.float64),
        r=rot_b,
        w=jnp.array([-0.1, 0.2, -0.3], dtype=jnp.float64),
        com=jnp.array([0.35, -0.2, 0.5], dtype=jnp.float64),
    )

    wa, wb = weld_wrench_jax(site_a, site_b, gains)
    wa_np = np.asarray(wa)
    wb_np = np.asarray(wb)

    fa_np, tau_a_np = wa_np[:3], wa_np[3:]
    fb_np, tau_b_np = wb_np[:3], wb_np[3:]

    com_a_np = np.asarray(site_a.com)
    com_b_np = np.asarray(site_b.com)

    # Moments about origin O: tau_O = tau_com + com x F
    tau_o_a = tau_a_np + np.cross(com_a_np, fa_np)
    tau_o_b = tau_b_np + np.cross(com_b_np, fb_np)
    net_moment = tau_o_a + tau_o_b

    p_a_np = np.asarray(site_a.p)
    p_b_np = np.asarray(site_b.p)
    expected_moment = np.cross(p_b_np - p_a_np, fb_np)

    np.testing.assert_allclose(net_moment, expected_moment, atol=1e-12)


def test_contact_law_contracts() -> None:
    """Preconditions on static inputs (radius > 0) are enforced."""
    ground = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)
    params = _make_test_parameters()
    c = jnp.array([0.0, 0.0, 0.01])
    v = jnp.array([0.0, 0.0, 0.0])

    with pytest.raises((ContractViolationError, PreconditionError)):
        sphere_ground_contact_jax(c, v, 0.0, ground, params)

    with pytest.raises((ContractViolationError, PreconditionError)):
        sphere_ground_contact_jax(c, v, -0.01, ground, params)


def test_weld_contracts() -> None:
    """Preconditions on weld gains (stiffness > 0, damping >= 0) are enforced."""
    site = SiteState(
        p=jnp.zeros(3),
        v=jnp.zeros(3),
        r=jnp.eye(3),
        w=jnp.zeros(3),
        com=jnp.zeros(3),
    )

    with pytest.raises((ContractViolationError, PreconditionError)):
        weld_wrench_jax(site, site, WeldGains(k=0.0, c=1.0, rot_k=1.0, rot_c=1.0))

    with pytest.raises((ContractViolationError, PreconditionError)):
        weld_wrench_jax(site, site, WeldGains(k=1.0, c=-0.1, rot_k=1.0, rot_c=1.0))

    with pytest.raises((ContractViolationError, PreconditionError)):
        weld_wrench_jax(site, site, WeldGains(k=1.0, c=1.0, rot_k=0.0, rot_c=1.0))

    with pytest.raises((ContractViolationError, PreconditionError)):
        weld_wrench_jax(site, site, WeldGains(k=1.0, c=1.0, rot_k=1.0, rot_c=-0.1))
