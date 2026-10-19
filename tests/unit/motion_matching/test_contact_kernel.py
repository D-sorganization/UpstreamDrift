"""Tests for the shared array-namespace contact kernel (HRG-GS-M14, #13336).

Acceptance criteria:
- One pure array-namespace kernel in ``src/shared/python/motion_matching/contact_kernel.py``
  shared by ``contact_law.sphere_ground_contact``, ``contact_law.torsional_friction_moment``,
  and ``jax_contact.sphere_ground_contact_jax``, plus ``grip_contact.pair_contact`` and
  ``grip_contact.pad_contact`` through ``contact_law``.
- DbC validation (``ValueError``/``TypeError`` on shapes, signs, parameter ranges)
  happens outside the traced kernel in thin public wrappers.
- Importing ``motion_matching``, ``contact_law``, ``contact_kernel``, or ``grip_contact``
  in a cold process never imports ``jax``.
- Refactored ``sphere_ground_contact`` and ``torsional_friction_moment`` match a frozen
  copy of the pre-refactor formulas to <= 1e-12 over 500 seeded random states,
  including zero-penetration, zero-speed, and liftoff edges.
- Float64 value error between JAX and NumPy <= 1e-10 over 200 seeded random states.
- Smooth-region gradient (``jax.grad``) vs central differences relative error <= 1e-5.
- Non-differentiable / non-smooth regimes (contact onset at zero penetration,
  stick/slip transition, zero tangential speed) are tested and documented.
- Smooth variant ``smooth_contact_law_kernel`` is separately named and never the default.
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.interface import GripInterface
from src.shared.python.grip_contact.pad_contact import build_pad_model, pad_wrench
from src.shared.python.grip_contact.pair_contact import (
    ContactCapsule,
    ContactSphere,
    capsule_capsule,
    sphere_capsule,
)
from src.shared.python.motion_matching.contact_kernel import (
    ContactKernelParams,
    contact_law_kernel,
    evaluate_contact_kernel,
    evaluate_contact_kernel_jax,
    jax_contact_kernel,
    smooth_contact_law_kernel,
    torsional_moment_kernel,
    validate_kernel_params,
)
from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    ContactSample,
    GroundPlane,
    sphere_ground_contact,
    torsional_friction_moment,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"


def _frozen_pre_refactor_sphere_ground_contact(
    center_m: np.ndarray,
    velocity_m_s: np.ndarray,
    radius_m: float,
    ground: GroundPlane,
    parameters: ContactParameters,
) -> ContactSample:
    """Frozen pre-refactor oracle of contact_law.sphere_ground_contact (kept only in test)."""
    center = np.asarray(center_m, dtype=float)
    velocity = np.asarray(velocity_m_s, dtype=float)
    normal = np.asarray(ground.normal)
    signed_distance = float(normal @ center) - ground.height_m - radius_m
    penetration = max(0.0, -signed_distance)
    rate = float(-(normal @ velocity))
    point = center - normal * radius_m
    zero = np.zeros(3)
    if penetration <= 0.0:
        return ContactSample(0.0, rate, point, zero, zero)
    magnitude = (
        parameters.stiffness_n_m
        * penetration
        * (1.0 + parameters.dissipation_s_m * rate)
    )
    magnitude = max(0.0, magnitude)
    normal_force = normal * magnitude
    tangential = velocity - normal * float(normal @ velocity)
    speed = float(math.sqrt(tangential.dot(tangential)))
    if speed <= 0.0 or magnitude <= 0.0:
        return ContactSample(penetration, rate, point, normal_force, zero)
    ratio = speed / parameters.transition_velocity_m_s
    mu = (
        parameters.dynamic_friction
        + (parameters.static_friction - parameters.dynamic_friction)
        * math.exp(-(ratio**2))
        + parameters.viscous_friction * speed
    )
    friction = -tangential / speed * mu * magnitude * math.tanh(ratio)
    return ContactSample(penetration, rate, point, normal_force, friction)


def _frozen_pre_refactor_torsional_friction_moment(
    normal_force_n: np.ndarray,
    angular_velocity_rad_s: np.ndarray,
    ground: GroundPlane,
    *,
    patch_radius_m: float,
    friction: float,
    transition_rad_s: float,
) -> np.ndarray:
    """Frozen pre-refactor oracle of contact_law.torsional_friction_moment (kept only in test)."""
    force = np.asarray(normal_force_n, dtype=float)
    omega = np.asarray(angular_velocity_rad_s, dtype=float)
    normal = np.asarray(ground.normal)
    load = max(0.0, float(normal @ force))
    spin = float(normal @ omega)
    limit = friction * load * patch_radius_m
    return -normal * limit * math.tanh(spin / transition_rad_s)


@pytest.fixture(scope="module")
def default_law() -> ContactParameters:
    return ContactParameters(
        stiffness_n_m=1.5e5,
        dissipation_s_m=0.25,
        static_friction=0.9,
        dynamic_friction=0.65,
        viscous_friction=0.05,
        transition_velocity_m_s=1.0e-3,
    )


@pytest.fixture(scope="module")
def default_params(default_law: ContactParameters) -> ContactKernelParams:
    return ContactKernelParams.from_contact_parameters(
        default_law,
        patch_radius_m=0.004,
        spin_transition_rad_s=1.0,
    )


def test_cold_process_import_does_not_import_jax() -> None:
    """Importing motion_matching, contact_law, contact_kernel, and grip_contact never imports jax."""
    code = (
        "import sys; "
        "import src.shared.python.motion_matching; "
        "import src.shared.python.motion_matching.contact_law; "
        "import src.shared.python.motion_matching.contact_kernel; "
        "import src.shared.python.grip_contact; "
        "loaded = [m for m in sys.modules if m == 'jax' or m.startswith('jax.')]; "
        "assert not loaded, f'jax unexpectedly imported: {loaded}'"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        cwd=str(ROOT),
        env={"PYTHONPATH": f"{ROOT}:{ROOT / 'src'}", "PATH": ""},
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, f"stderr: {proc.stderr}\nstdout: {proc.stdout}"


def test_dbc_validation_rejects_invalid_parameters_and_inputs(
    default_params: ContactKernelParams,
) -> None:
    """DbC wrapper raises TypeError/ValueError on invalid parameters, shapes, or NaNs."""
    with pytest.raises(TypeError, match="ContactKernelParams or ContactParameters"):
        validate_kernel_params("not_params")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="finite"):
        validate_kernel_params(
            ContactKernelParams(
                stiffness_n_m=float("nan"),
                dissipation_s_m=0.1,
                static_friction=0.8,
                dynamic_friction=0.6,
                viscous_friction=0.0,
                transition_velocity_m_s=1e-3,
            )
        )

    with pytest.raises(ValueError, match="positive"):
        validate_kernel_params(
            ContactKernelParams(
                stiffness_n_m=0.0,
                dissipation_s_m=0.1,
                static_friction=0.8,
                dynamic_friction=0.6,
                viscous_friction=0.0,
                transition_velocity_m_s=1e-3,
            )
        )

    with pytest.raises(ValueError, match="nonnegative"):
        validate_kernel_params(
            ContactKernelParams(
                stiffness_n_m=1e5,
                dissipation_s_m=-0.1,
                static_friction=0.8,
                dynamic_friction=0.6,
                viscous_friction=0.0,
                transition_velocity_m_s=1e-3,
            )
        )

    with pytest.raises(ValueError, match="Static friction"):
        validate_kernel_params(
            ContactKernelParams(
                stiffness_n_m=1e5,
                dissipation_s_m=0.1,
                static_friction=0.5,
                dynamic_friction=0.7,
                viscous_friction=0.0,
                transition_velocity_m_s=1e-3,
            )
        )

    with pytest.raises(ValueError, match="patch_radius_m"):
        validate_kernel_params(
            ContactKernelParams(
                stiffness_n_m=1e5,
                dissipation_s_m=0.1,
                static_friction=0.8,
                dynamic_friction=0.6,
                viscous_friction=0.0,
                transition_velocity_m_s=1e-3,
                patch_radius_m=-1e-3,
            )
        )

    with pytest.raises(ValueError, match="spin_transition_rad_s"):
        validate_kernel_params(
            ContactKernelParams(
                stiffness_n_m=1e5,
                dissipation_s_m=0.1,
                static_friction=0.8,
                dynamic_friction=0.6,
                viscous_friction=0.0,
                transition_velocity_m_s=1e-3,
                spin_transition_rad_s=0.0,
            )
        )

    with pytest.raises(TypeError, match="penetration_m"):
        evaluate_contact_kernel(None, 0.0, 0.0, default_params)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="finite"):
        evaluate_contact_kernel(float("nan"), 0.0, 0.0, default_params)

    with pytest.raises(ValueError, match="shape"):
        evaluate_contact_kernel(
            np.zeros(3),
            np.zeros(4),
            np.zeros(3),
            default_params,
        )

    with pytest.raises(ValueError, match="tangential_velocity_m_s"):
        evaluate_contact_kernel(
            np.zeros(3),
            np.zeros(3),
            np.zeros((3, 2)),
            default_params,
        )

    with pytest.raises(ValueError, match="spin_rate_rad_s"):
        evaluate_contact_kernel(
            np.zeros(3),
            np.zeros(3),
            np.zeros(3),
            default_params,
            spin_rate_rad_s=np.zeros(5),
        )


def test_all_callers_delegate_to_single_shared_kernel(
    default_law: ContactParameters,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """contact_law, jax_contact, pair_contact, and pad_contact all call the single kernel."""
    import src.shared.python.motion_matching.contact_law as law_mod

    kernel_calls = 0
    spin_calls = 0

    def wrapped_kernel(*args: object, **kwargs: object):
        nonlocal kernel_calls
        kernel_calls += 1
        return contact_law_kernel(*args, **kwargs)  # type: ignore[arg-type]

    def wrapped_spin(*args: object, **kwargs: object):
        nonlocal spin_calls
        spin_calls += 1
        return torsional_moment_kernel(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(law_mod, "contact_law_kernel", wrapped_kernel)
    monkeypatch.setattr(law_mod, "torsional_moment_kernel", wrapped_spin)

    ground = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)
    sample = sphere_ground_contact(
        np.array([0.0, 0.0, 0.01]),
        np.array([0.1, 0.0, -0.2]),
        0.03,
        ground,
        default_law,
    )
    assert sample.normal_force_n[2] > 0.0
    assert kernel_calls == 1

    moment = torsional_friction_moment(
        sample.normal_force_n,
        np.array([0.0, 0.0, 1.5]),
        ground,
        patch_radius_m=0.01,
        friction=0.8,
        transition_rad_s=0.5,
    )
    assert moment[2] < 0.0
    assert spin_calls == 1

    sph = ContactSphere(center_m=np.zeros(3), radius_m=0.02)
    cap_a = ContactCapsule(
        p0_m=np.array([0.0, 0.0, -0.05]),
        p1_m=np.array([0.0, 0.0, 0.05]),
        radius_m=0.02,
    )
    cap_b = ContactCapsule(
        p0_m=np.array([0.03, 0.0, -0.05]),
        p1_m=np.array([0.03, 0.0, 0.05]),
        radius_m=0.02,
    )
    w_sc = sphere_capsule(sph, cap_b, law=default_law, patch_radius_m=0.003)
    assert w_sc.normal_force_mag_n > 0.0
    assert kernel_calls == 2
    assert spin_calls == 2

    w_cc = capsule_capsule(cap_a, cap_b, law=default_law, patch_radius_m=0.003)
    assert w_cc.normal_force_mag_n > 0.0
    assert kernel_calls == 3
    assert spin_calls == 3

    interface = GripInterface.from_spec(json.loads(SPEC.read_text()))
    pad_model = build_pad_model(interface, squeeze_per_hand_n=1000.0)
    hand = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
    club = BushingState(np.eye(3), np.zeros(3), np.zeros(3), np.zeros(3))
    w_pad = pad_wrench(pad_model, "R", hand, club)
    assert w_pad.normal_force_n.sum() > 0.0
    assert kernel_calls > 3
    assert spin_calls > 3

    jax = pytest.importorskip("jax")
    import jax.numpy as jnp
    import src.shared.python.motion_matching.jax_contact as jax_mod

    jax_calls = 0

    def wrapped_jax_kernel(*args: object, **kwargs: object):
        nonlocal jax_calls
        jax_calls += 1
        return contact_law_kernel(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(jax_mod, "contact_law_kernel", wrapped_jax_kernel)
    fn_j, ft_j = jax_mod.sphere_ground_contact_jax(
        jnp.array([0.0, 0.0, 0.01]),
        jnp.array([0.1, 0.0, -0.2]),
        0.03,
        ground,
        default_law,
    )
    assert float(fn_j[2]) > 0.0
    assert float(ft_j[0]) < 0.0
    assert jax_calls == 1


def test_refactored_contact_law_matches_frozen_pre_refactor_oracle_500_states(
    default_law: ContactParameters,
) -> None:
    """Refactored sphere_ground_contact and torsional_friction_moment match frozen pre-refactor formulas to <= 1e-12 over 500 states."""
    ground = GroundPlane(normal=(0.1, -0.2, 0.95), height_m=-0.01)
    n_vec = np.asarray(ground.normal)
    tangent_dir = np.array([n_vec[2], 0.0, -n_vec[0]])
    tangent_dir = tangent_dir / np.linalg.norm(tangent_dir)
    radius = 0.03

    rng = np.random.default_rng(13336)
    n_states = 500
    centers = rng.uniform(-0.2, 0.2, size=(n_states, 3))
    velocities = rng.normal(scale=1.0, size=(n_states, 3))
    omegas = rng.normal(scale=2.0, size=(n_states, 3))

    # Explicit edge states:
    # 0: Exact zero penetration (signed_distance == 0)
    centers[0] = n_vec * (ground.height_m + radius)
    velocities[0] = -0.3 * n_vec + 0.1 * tangent_dir
    # 1: Separated state (penetration < 0)
    centers[1] = n_vec * (ground.height_m + 1.5 * radius)
    velocities[1] = -0.5 * n_vec + 0.2 * tangent_dir
    # 2: Penetrating with exact zero tangential speed (v_t == 0)
    centers[2] = n_vec * (ground.height_m + 0.5 * radius)
    velocities[2] = -0.4 * n_vec
    # 3: Penetrating at exact damping liftoff threshold (1 + c * rate == 0 => v_n = 1 / c)
    centers[3] = n_vec * (ground.height_m + 0.5 * radius)
    velocities[3] = (1.0 / default_law.dissipation_s_m) * n_vec + 0.15 * tangent_dir
    # 4: Penetrating beyond damping liftoff (1 + c * rate < 0 => v_n > 1 / c)
    centers[4] = n_vec * (ground.height_m + 0.5 * radius)
    velocities[4] = (
        1.0 / default_law.dissipation_s_m + 0.5
    ) * n_vec + 0.15 * tangent_dir
    # 5: Zero spin rate with positive normal load
    centers[5] = n_vec * (ground.height_m + 0.4 * radius)
    velocities[5] = -0.2 * n_vec + 0.05 * tangent_dir
    omegas[5] = np.zeros(3)

    max_pen_err = 0.0
    max_rate_err = 0.0
    max_pt_err = 0.0
    max_fn_err = 0.0
    max_ft_err = 0.0
    max_tfm_err = 0.0

    for i in range(n_states):
        c = centers[i]
        v = velocities[i]
        w = omegas[i]

        expected = _frozen_pre_refactor_sphere_ground_contact(
            c, v, radius, ground, default_law
        )
        actual = sphere_ground_contact(c, v, radius, ground, default_law)

        max_pen_err = max(
            max_pen_err, abs(actual.penetration_m - expected.penetration_m)
        )
        max_rate_err = max(
            max_rate_err,
            abs(actual.penetration_rate_m_s - expected.penetration_rate_m_s),
        )
        max_pt_err = max(
            max_pt_err,
            float(np.max(np.abs(actual.contact_point_m - expected.contact_point_m))),
        )
        max_fn_err = max(
            max_fn_err,
            float(np.max(np.abs(actual.normal_force_n - expected.normal_force_n))),
        )
        max_ft_err = max(
            max_ft_err,
            float(np.max(np.abs(actual.friction_force_n - expected.friction_force_n))),
        )

        m_expected = _frozen_pre_refactor_torsional_friction_moment(
            expected.normal_force_n,
            w,
            ground,
            patch_radius_m=0.015,
            friction=default_law.static_friction,
            transition_rad_s=0.5,
        )
        m_actual = torsional_friction_moment(
            actual.normal_force_n,
            w,
            ground,
            patch_radius_m=0.015,
            friction=default_law.static_friction,
            transition_rad_s=0.5,
        )
        max_tfm_err = max(max_tfm_err, float(np.max(np.abs(m_actual - m_expected))))

    assert max_pen_err <= 1e-12
    assert max_rate_err <= 1e-12
    assert max_pt_err <= 1e-12
    assert max_fn_err <= 1e-12
    assert max_ft_err <= 1e-12
    assert max_tfm_err <= 1e-12


def test_jax_vs_numpy_float64_value_parity_200_random_states(
    default_params: ContactKernelParams,
) -> None:
    """Float64 value error between JAX and NumPy <= 1e-10 over 200 seeded random states."""
    jax = pytest.importorskip("jax", reason="JAX required for array-namespace parity")
    jax.config.update("jax_enable_x64", True)

    rng = np.random.default_rng(13336)
    n_states = 200

    # Include penetrating, separated, zero-penetration, zero-slip, and liftoff states
    penetration = rng.uniform(-0.002, 0.006, size=n_states)
    penetration[0] = 0.0
    penetration[1] = -0.001
    normal_vel = rng.uniform(-1.5, 6.0, size=n_states)
    tangential_vel_scalar = rng.uniform(-0.2, 0.2, size=n_states)
    tangential_vel_scalar[2] = 0.0
    tangential_vel_vec = rng.normal(scale=0.05, size=(n_states, 3))
    tangential_vel_vec[2] = 0.0
    spin_rate = rng.uniform(-4.0, 4.0, size=n_states)
    spin_rate[3] = 0.0

    # 1. Batched evaluation (scalar tangential velocity)
    np_batch = evaluate_contact_kernel(
        penetration, normal_vel, tangential_vel_scalar, default_params, spin_rate
    )
    jax_batch = evaluate_contact_kernel_jax(
        penetration, normal_vel, tangential_vel_scalar, default_params, spin_rate
    )

    err_fn = float(
        np.max(np.abs(np.asarray(jax_batch.normal_force_n) - np_batch.normal_force_n))
    )
    err_ft = float(
        np.max(
            np.abs(
                np.asarray(jax_batch.tangential_force_n) - np_batch.tangential_force_n
            )
        )
    )
    err_ms = float(
        np.max(np.abs(np.asarray(jax_batch.spin_moment_nm) - np_batch.spin_moment_nm))
    )
    assert max(err_fn, err_ft, err_ms) <= 1e-10

    # 2. Batched evaluation (3-vector tangential velocity)
    np_vec = evaluate_contact_kernel(
        penetration, normal_vel, tangential_vel_vec, default_params, spin_rate
    )
    jax_vec = evaluate_contact_kernel_jax(
        penetration, normal_vel, tangential_vel_vec, default_params, spin_rate
    )
    err_vec_ft = float(
        np.max(
            np.abs(np.asarray(jax_vec.tangential_force_n) - np_vec.tangential_force_n)
        )
    )
    assert err_vec_ft <= 1e-10

    # 3. Per-state scalar evaluation over all 200 states
    max_scalar_err = 0.0
    for i in range(n_states):
        res_np = evaluate_contact_kernel(
            float(penetration[i]),
            float(normal_vel[i]),
            tangential_vel_vec[i],
            default_params,
            float(spin_rate[i]),
        )
        res_jax = evaluate_contact_kernel_jax(
            float(penetration[i]),
            float(normal_vel[i]),
            tangential_vel_vec[i],
            default_params,
            float(spin_rate[i]),
        )
        max_scalar_err = max(
            max_scalar_err,
            abs(float(res_jax.normal_force_n) - float(res_np.normal_force_n)),
            float(
                np.max(
                    np.abs(
                        np.asarray(res_jax.tangential_force_n)
                        - res_np.tangential_force_n
                    )
                )
            ),
            abs(float(res_jax.spin_moment_nm) - float(res_np.spin_moment_nm)),
        )
    assert max_scalar_err <= 1e-10


def test_jax_grad_matches_central_differences_in_smooth_region(
    default_params: ContactKernelParams,
) -> None:
    """In the smooth region, jax.grad agrees with central differences to relative error <= 1e-5."""
    jax = pytest.importorskip(
        "jax", reason="JAX required for smooth-region gradient test"
    )
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    rng = np.random.default_rng(13337)
    eps = 1.0e-7

    def fn_normal(x: jax.Array) -> jax.Array:
        return jax_contact_kernel(x[0], x[1], x[2], default_params, x[3]).normal_force_n

    def fn_tangential(x: jax.Array) -> jax.Array:
        return jax_contact_kernel(
            x[0], x[1], x[2], default_params, x[3]
        ).tangential_force_n

    def fn_spin(x: jax.Array) -> jax.Array:
        return jax_contact_kernel(x[0], x[1], x[2], default_params, x[3]).spin_moment_nm

    grad_fns = (
        (fn_normal, jax.grad(fn_normal)),
        (fn_tangential, jax.grad(fn_tangential)),
        (fn_spin, jax.grad(fn_spin)),
    )

    for _ in range(25):
        d0 = float(rng.uniform(0.001, 0.005))
        vn0 = float(rng.uniform(-0.8, 0.5))
        vt0 = float(rng.uniform(0.0004, 0.02)) * float(rng.choice([-1.0, 1.0]))
        w0 = float(rng.uniform(-2.5, 2.5))
        if abs(w0) < 0.2:
            w0 = 0.5

        x0 = jnp.array([d0, vn0, vt0, w0], dtype=jnp.float64)
        for fn, grad_fn in grad_fns:
            grad_ad = np.asarray(grad_fn(x0), dtype=float)
            grad_fd = np.zeros(4, dtype=float)
            for k in range(4):
                dx = np.zeros(4, dtype=float)
                dx[k] = eps
                f_plus = float(fn(jnp.array(np.asarray(x0) + dx, dtype=jnp.float64)))
                f_minus = float(fn(jnp.array(np.asarray(x0) - dx, dtype=jnp.float64)))
                grad_fd[k] = (f_plus - f_minus) / (2.0 * eps)

            denom = max(float(np.linalg.norm(grad_fd)), 1.0e-12)
            rel_err = float(np.linalg.norm(grad_ad - grad_fd)) / denom
            assert rel_err <= 1.0e-5, (
                f"rel_err={rel_err}, grad_ad={grad_ad}, grad_fd={grad_fd}"
            )

    vt_vec0 = jnp.array([0.003, -0.004, 0.002], dtype=jnp.float64)
    jac_ad = np.asarray(
        jax.jacrev(
            lambda v: (
                jax_contact_kernel(
                    0.002, -0.1, v, default_params, 0.5
                ).tangential_force_n
            )
        )(vt_vec0),
        dtype=float,
    )
    jac_fd = np.zeros((3, 3), dtype=float)
    for k in range(3):
        dv = np.zeros(3, dtype=float)
        dv[k] = eps
        f_p = np.asarray(
            jax_contact_kernel(
                0.002, -0.1, jnp.array(np.asarray(vt_vec0) + dv), default_params, 0.5
            ).tangential_force_n
        )
        f_m = np.asarray(
            jax_contact_kernel(
                0.002, -0.1, jnp.array(np.asarray(vt_vec0) - dv), default_params, 0.5
            ).tangential_force_n
        )
        jac_fd[:, k] = (f_p - f_m) / (2.0 * eps)
    rel_jac_err = float(np.linalg.norm(jac_ad - jac_fd) / np.linalg.norm(jac_fd))
    assert rel_jac_err <= 1.0e-5


def test_non_differentiable_points_contact_onset_stick_slip_and_zero_tangential_speed(
    default_params: ContactKernelParams,
) -> None:
    """Demonstrate the 3 non-smooth loci: contact onset (delta=0), zero tangential speed (v_t=0), and stick/slip transition."""
    jax = pytest.importorskip("jax", reason="JAX required for non-smooth point test")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    eps = 1.0e-7
    vn = -0.2
    vt = 0.01
    w = 0.5

    # 1. Contact onset at zero penetration (delta = 0):
    f_0 = float(evaluate_contact_kernel(0.0, vn, vt, default_params, w).normal_force_n)
    f_left = float(
        evaluate_contact_kernel(-eps, vn, vt, default_params, w).normal_force_n
    )
    f_right = float(
        evaluate_contact_kernel(eps, vn, vt, default_params, w).normal_force_n
    )
    d_left_onset = (f_0 - f_left) / eps
    d_right_onset = (f_right - f_0) / eps
    expected_right = default_params.stiffness_n_m * (
        1.0 - default_params.dissipation_s_m * vn
    )
    assert d_left_onset == 0.0
    assert d_right_onset == pytest.approx(expected_right, rel=1e-7)
    assert abs(d_right_onset - d_left_onset) > 1.0e4

    # 2. Zero tangential speed (v_t = 0):
    delta = 0.002
    fn_val = float(
        evaluate_contact_kernel(delta, vn, 0.0, default_params, w).normal_force_n
    )
    ft_0 = float(
        evaluate_contact_kernel(delta, vn, 0.0, default_params, w).tangential_force_n
    )
    ft_minus = float(
        evaluate_contact_kernel(delta, vn, -eps, default_params, w).tangential_force_n
    )
    ft_plus = float(
        evaluate_contact_kernel(delta, vn, eps, default_params, w).tangential_force_n
    )
    slope_left_vt0 = (ft_0 - ft_minus) / eps
    slope_right_vt0 = (ft_plus - ft_0) / eps
    expected_stick_slope = (
        default_params.static_friction * fn_val / default_params.transition_velocity_m_s
    )
    assert ft_0 == 0.0
    assert slope_left_vt0 == pytest.approx(-expected_stick_slope, rel=1e-4)
    assert slope_right_vt0 == pytest.approx(expected_stick_slope, rel=1e-4)
    assert slope_left_vt0 * slope_right_vt0 < 0.0

    # Also verify that 3-vector tangential force at v_t = [0, 0, 0] has finite jax.jacrev (no NaN from 0/0)
    vt_zero_vec = jnp.zeros(3, dtype=jnp.float64)
    jac_at_zero = np.asarray(
        jax.jacrev(
            lambda v: (
                jax_contact_kernel(delta, vn, v, default_params, w).tangential_force_n
            )
        )(vt_zero_vec)
    )
    assert np.isfinite(jac_at_zero).all()

    # 3. Stick/slip transition (|v_t| ~ v_transition):
    v_tr = default_params.transition_velocity_m_s
    grad_ft_fn = jax.grad(
        lambda s: jax_contact_kernel(delta, vn, s, default_params, w).tangential_force_n
    )
    slope_stick = float(grad_ft_fn(jnp.asarray(0.05 * v_tr, dtype=jnp.float64)))
    slope_full_slip = float(grad_ft_fn(jnp.asarray(10.0 * v_tr, dtype=jnp.float64)))
    assert slope_stick > 0.5 * expected_stick_slope
    assert slope_full_slip == pytest.approx(
        default_params.viscous_friction * fn_val, rel=1e-3
    )
    assert slope_stick / slope_full_slip > 1.0e3

    slope_knee_exact = float(grad_ft_fn(jnp.asarray(v_tr, dtype=jnp.float64)))
    ft_knee_plus = float(
        evaluate_contact_kernel(
            delta, vn, 2.0 * v_tr, default_params, w
        ).tangential_force_n
    )
    ft_knee_minus = float(
        evaluate_contact_kernel(delta, vn, 0.0, default_params, w).tangential_force_n
    )
    slope_knee_coarse_fd = (ft_knee_plus - ft_knee_minus) / (2.0 * v_tr)
    knee_rel_err = abs(slope_knee_coarse_fd - slope_knee_exact) / abs(slope_knee_exact)
    assert knee_rel_err > 0.1

    vn_liftoff = 1.0 / default_params.dissipation_s_m
    ft_before_liftoff = float(
        evaluate_contact_kernel(
            delta, vn_liftoff - eps, vt, default_params, w
        ).tangential_force_n
    )
    ft_after_liftoff = float(
        evaluate_contact_kernel(
            delta, vn_liftoff + eps, vt, default_params, w
        ).tangential_force_n
    )
    assert ft_before_liftoff > 0.0
    assert ft_after_liftoff == 0.0


def test_smooth_contact_law_kernel_is_separate_and_c_infinity_at_kinks(
    default_params: ContactKernelParams,
) -> None:
    """smooth_contact_law_kernel is separately named, smooth at delta=0 and v_t=0, and never the default."""
    jax = pytest.importorskip("jax", reason="JAX required for smooth variant test")
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    qual_at_zero = contact_law_kernel(0.0, 0.0, 0.01, default_params)
    smooth_at_zero = smooth_contact_law_kernel(
        0.0, 0.0, 0.01, default_params, smoothing_m=1e-5, velocity_eps_m_s=1e-5
    )
    assert float(qual_at_zero.normal_force_n) == 0.0
    assert float(smooth_at_zero.normal_force_n) > 0.0

    eps = 1.0e-8
    for d_test, vt_test in ((0.0, 0.005), (0.002, 0.0)):
        x0 = jnp.array([d_test, -0.1, vt_test], dtype=jnp.float64)

        def fn(x: jax.Array) -> jax.Array:
            return smooth_contact_law_kernel(
                x[0],
                x[1],
                x[2],
                default_params,
                smoothing_m=1e-4,
                velocity_eps_m_s=1e-4,
                xp=jnp,
            ).tangential_force_n

        grad_ad = np.asarray(jax.grad(fn)(x0), dtype=float)
        grad_fd = np.zeros(3, dtype=float)
        for k in range(3):
            dx = np.zeros(3, dtype=float)
            dx[k] = eps
            f_p = float(fn(jnp.array(np.asarray(x0) + dx, dtype=jnp.float64)))
            f_m = float(fn(jnp.array(np.asarray(x0) - dx, dtype=jnp.float64)))
            grad_fd[k] = (f_p - f_m) / (2.0 * eps)
        np.testing.assert_allclose(grad_ad, grad_fd, rtol=1e-5, atol=1e-7)

    qual = contact_law_kernel(0.003, -0.2, 0.01, default_params, 0.5)
    smooth_tight = smooth_contact_law_kernel(
        0.003,
        -0.2,
        0.01,
        default_params,
        0.5,
        smoothing_m=1e-12,
        velocity_eps_m_s=1e-12,
    )
    assert float(smooth_tight.normal_force_n) == pytest.approx(
        float(qual.normal_force_n), rel=1e-9
    )
    assert float(smooth_tight.tangential_force_n) == pytest.approx(
        float(qual.tangential_force_n), rel=1e-9
    )
    assert float(smooth_tight.spin_moment_nm) == pytest.approx(
        float(qual.spin_moment_nm), rel=1e-9
    )
