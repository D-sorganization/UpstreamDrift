"""Unit tests for NSX-02 canonical grip-closure Jacobian conventions (#12551)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace.closure_convention import (
    CANONICAL_CLOSURE_CONVENTION,
    DRAKE_CLOSURE_CONVENTION,
    MUJOCO_CLOSURE_CONVENTION,
    OPENSIM_CLOSURE_CONVENTION,
    PINOCCHIO_CLOSURE_CONVENTION,
    NativeClosureConvention,
    dual_wrench_to_canonical,
    dual_wrench_to_native,
    from_canonical_closure,
    to_canonical_closure,
)
from src.shared.python.motion_matching.same_input.closure import project_to_closure
from src.shared.python.pose_interchange.se3 import euler_xyz_deg_to_matrix

pytestmark = pytest.mark.unit


def _skew(v: np.ndarray) -> np.ndarray:
    x, y, z = float(v[0]), float(v[1]), float(v[2])
    return np.array(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )


def _analytic_native_closure(
    j_world_at_grip: np.ndarray,
    drift_world_at_grip: np.ndarray,
    *,
    convention: NativeClosureConvention,
    rot_wg: np.ndarray,
    grip_origin: np.ndarray,
    ref_point: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Synthesize an engine's native closure (J, drift) from world-frame kinematics at grip_origin."""
    # Shift from grip_origin p_G to ref_point p_P:
    # v_P = v_G + omega x (p_P - p_G) = v_G - [p_P - p_G]_x omega
    lever = ref_point - grip_origin
    eye3 = np.eye(3)
    zero3 = np.zeros((3, 3))
    shift_to_ref = np.block([[eye3, -_skew(lever)], [zero3, eye3]])
    j_ref_world = shift_to_ref @ j_world_at_grip
    drift_ref_world = shift_to_ref @ drift_world_at_grip

    if convention.frame == "local":
        rot_gw = rot_wg.T
        frame_map = np.block([[rot_gw, zero3], [zero3, rot_gw]])
    else:
        frame_map = np.eye(6)

    j_framed = frame_map @ j_ref_world
    drift_framed = frame_map @ drift_ref_world

    if convention.row_order == "ang_lin":
        perm = np.block([[zero3, eye3], [eye3, zero3]])
    else:
        perm = np.eye(6)

    sign = float(convention.relative_sign)
    return sign * (perm @ j_framed), sign * (perm @ drift_framed)


def test_all_four_engine_conventions_yield_identical_canonical_closure() -> None:
    """Identical analytic motion fed through Pinocchio, Drake, MuJoCo, and OpenSim
    native conventions yields equal canonical (J_c, Jdot_v) to < 1e-12."""
    rng = np.random.default_rng(12551)
    nv = 14
    j_world_grip = rng.normal(size=(6, nv))
    drift_world_grip = rng.normal(size=(6,))
    rot_wg = euler_xyz_deg_to_matrix([25.0, -18.0, 42.0])
    grip_origin = np.array([0.35, -0.12, 1.05], dtype=np.float64)
    placement_a = np.array([0.31, -0.09, 1.02], dtype=np.float64)
    site_xpos = np.array([0.38, -0.15, 1.08], dtype=np.float64)

    conventions_and_points = [
        (PINOCCHIO_CLOSURE_CONVENTION, placement_a),
        (DRAKE_CLOSURE_CONVENTION, placement_a),
        (MUJOCO_CLOSURE_CONVENTION, site_xpos),
        (OPENSIM_CLOSURE_CONVENTION, placement_a),
        (CANONICAL_CLOSURE_CONVENTION, grip_origin),
    ]

    canonical_results: list[tuple[np.ndarray, np.ndarray]] = []
    for conv, ref_pt in conventions_and_points:
        j_nat, d_nat = _analytic_native_closure(
            j_world_grip,
            drift_world_grip,
            convention=conv,
            rot_wg=rot_wg,
            grip_origin=grip_origin,
            ref_point=ref_pt,
        )
        j_c, d_c = to_canonical_closure(
            j_nat,
            d_nat,
            native=conv,
            grip_frame_rotation=rot_wg,
            grip_origin=grip_origin,
            reference_point=ref_pt,
        )
        assert not j_c.flags.writeable
        assert not d_c.flags.writeable
        canonical_results.append((j_c, d_c))

    ref_j, ref_d = canonical_results[0]
    for j_c, d_c in canonical_results[1:]:
        np.testing.assert_allclose(j_c, ref_j, atol=1e-12, rtol=1e-12)
        np.testing.assert_allclose(d_c, ref_d, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "convention",
    [
        PINOCCHIO_CLOSURE_CONVENTION,
        DRAKE_CLOSURE_CONVENTION,
        MUJOCO_CLOSURE_CONVENTION,
        OPENSIM_CLOSURE_CONVENTION,
    ],
)
def test_wrench_power_invariance_and_generalized_torque_duality(
    convention: NativeClosureConvention,
) -> None:
    """Wrench power lambda . (J_c v) and tree generalized load J_c^T lambda are invariant."""
    rng = np.random.default_rng(42)
    nv = 10
    j_nat = rng.normal(size=(6, nv))
    d_nat = rng.normal(size=(6,))
    v = rng.normal(size=(nv,))
    lambda_nat = rng.normal(size=(6,))
    rot_wg = euler_xyz_deg_to_matrix([-15.0, 33.0, -60.0])
    grip_origin = np.array([0.4, 0.2, 0.9], dtype=np.float64)
    ref_point = np.array([0.36, 0.22, 0.87], dtype=np.float64)

    j_c, _ = to_canonical_closure(
        j_nat,
        d_nat,
        native=convention,
        grip_frame_rotation=rot_wg,
        grip_origin=grip_origin,
        reference_point=ref_point,
    )
    lambda_c = dual_wrench_to_canonical(
        lambda_nat,
        native=convention,
        grip_frame_rotation=rot_wg,
        grip_origin=grip_origin,
        reference_point=ref_point,
    )

    power_native = float(lambda_nat @ (j_nat @ v))
    power_canonical = float(lambda_c @ (j_c @ v))
    np.testing.assert_allclose(power_canonical, power_native, atol=1e-12, rtol=1e-12)

    tau_native = j_nat.T @ lambda_nat
    tau_canonical = j_c.T @ lambda_c
    np.testing.assert_allclose(tau_canonical, tau_native, atol=1e-12, rtol=1e-12)

    lambda_recovered = dual_wrench_to_native(
        lambda_c,
        native=convention,
        grip_frame_rotation=rot_wg,
        grip_origin=grip_origin,
        reference_point=ref_point,
    )
    np.testing.assert_allclose(lambda_recovered, lambda_nat, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize(
    "convention",
    [
        PINOCCHIO_CLOSURE_CONVENTION,
        DRAKE_CLOSURE_CONVENTION,
        MUJOCO_CLOSURE_CONVENTION,
        OPENSIM_CLOSURE_CONVENTION,
    ],
)
def test_round_trip_inverse_from_canonical_closure(
    convention: NativeClosureConvention,
) -> None:
    """Exact round-trip inverse from_canonical_closure(to_canonical_closure(...))."""
    rng = np.random.default_rng(99)
    j_nat = rng.normal(size=(6, 12))
    d_nat = rng.normal(size=(6,))
    rot_wg = euler_xyz_deg_to_matrix([12.0, 45.0, -30.0])
    grip_origin = np.array([0.2, -0.1, 1.1], dtype=np.float64)
    ref_point = np.array([0.25, -0.05, 1.05], dtype=np.float64)

    j_c, d_c = to_canonical_closure(
        j_nat,
        d_nat,
        native=convention,
        grip_frame_rotation=rot_wg,
        grip_origin=grip_origin,
        reference_point=ref_point,
    )
    j_rec, d_rec = from_canonical_closure(
        j_c,
        d_c,
        native=convention,
        grip_frame_rotation=rot_wg,
        grip_origin=grip_origin,
        reference_point=ref_point,
    )
    np.testing.assert_allclose(j_rec, j_nat, atol=1e-12, rtol=1e-12)
    np.testing.assert_allclose(d_rec, d_nat, atol=1e-12, rtol=1e-12)


def test_projected_manifold_state_preserves_canonical_closure_equality() -> None:
    """On a state projected with project_to_closure, all four native conventions
    agree on the canonical Jacobian and tangent rate residual."""
    rng = np.random.default_rng(77)
    nv = 8
    d_true = rng.normal(size=(6, nv))
    q_star = np.ones(nv, dtype=np.float64)

    class _AnalyticClosurePlant:
        def closure_pose_residual(self, q: np.ndarray) -> np.ndarray:
            return d_true @ (q - q_star)

        def closure_rate_matrix(self, q: np.ndarray) -> np.ndarray:
            return d_true

    plant = _AnalyticClosurePlant()
    proj = project_to_closure(
        plant,
        q=q_star + 0.02 * rng.normal(size=nv),
        v=rng.normal(size=nv),
    )
    assert proj.pose_residual_after <= 1e-13
    assert proj.rate_residual_after <= 1e-12

    rot_wg = euler_xyz_deg_to_matrix([10.0, -20.0, 35.0])
    grip_origin = np.array([0.3, 0.0, 1.0], dtype=np.float64)
    ref_point = np.array([0.28, 0.02, 0.98], dtype=np.float64)
    drift = np.zeros(6, dtype=np.float64)

    for conv in (
        PINOCCHIO_CLOSURE_CONVENTION,
        DRAKE_CLOSURE_CONVENTION,
        MUJOCO_CLOSURE_CONVENTION,
        OPENSIM_CLOSURE_CONVENTION,
    ):
        j_nat, d_nat = _analytic_native_closure(
            d_true,
            drift,
            convention=conv,
            rot_wg=rot_wg,
            grip_origin=grip_origin,
            ref_point=ref_point,
        )
        j_c, _ = to_canonical_closure(
            j_nat,
            d_nat,
            native=conv,
            grip_frame_rotation=rot_wg,
            grip_origin=grip_origin,
            reference_point=ref_point,
        )
        np.testing.assert_allclose(j_c @ proj.v, np.zeros(6), atol=1e-12, rtol=1e-12)


def test_engine_adapter_constants_are_declared() -> None:
    """Engine adapter modules expose their NATIVE_CLOSURE_CONVENTION constants."""
    from src.engines.physics_engines.drake.python.full_body_model import (
        FullBodyDrakeModel,
    )
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )
    from src.engines.physics_engines.mujoco.python.native_model import (
        NATIVE_CLOSURE_CONVENTION as MUJOCO_MODULE_CONV,
        NativeMujocoModel,
    )
    from src.engines.physics_engines.opensim.python.full_body_parity import (
        OpenSimFullBodyParityAdapter,
    )
    from src.engines.physics_engines.pinocchio.python.force_adapter import (
        PinocchioForceAdapter,
    )
    from src.engines.physics_engines.pinocchio.python.native_model import (
        NativePinocchioModel,
    )
    from src.shared.python.motion_matching.multi_engine_torque_allocator import (
        MujocoForceAdapter,
    )

    assert (
        NativePinocchioModel.NATIVE_CLOSURE_CONVENTION == PINOCCHIO_CLOSURE_CONVENTION
    )
    assert (
        PinocchioForceAdapter.NATIVE_CLOSURE_CONVENTION == PINOCCHIO_CLOSURE_CONVENTION
    )
    assert FullBodyDrakeModel.NATIVE_CLOSURE_CONVENTION == DRAKE_CLOSURE_CONVENTION
    assert MUJOCO_MODULE_CONV == MUJOCO_CLOSURE_CONVENTION
    assert NativeMujocoModel.NATIVE_CLOSURE_CONVENTION == MUJOCO_CLOSURE_CONVENTION
    assert (
        NativeMujocoFullBodyModel.NATIVE_CLOSURE_CONVENTION == MUJOCO_CLOSURE_CONVENTION
    )
    assert MujocoForceAdapter.NATIVE_CLOSURE_CONVENTION == MUJOCO_CLOSURE_CONVENTION
    assert (
        OpenSimFullBodyParityAdapter.NATIVE_CLOSURE_CONVENTION
        == OPENSIM_CLOSURE_CONVENTION
    )


def test_dbc_preconditions_reject_invalid_inputs() -> None:
    """DbC preconditions reject non-orthonormal rotations, wrong row counts, and NaNs."""
    j_ok = np.zeros((6, 8), dtype=np.float64)
    d_ok = np.zeros(6, dtype=np.float64)
    rot_ok = np.eye(3, dtype=np.float64)
    pt_ok = np.zeros(3, dtype=np.float64)

    with pytest.raises(ValueError, match="shape \\(6, nv\\)"):
        to_canonical_closure(
            np.zeros((5, 8)),
            d_ok,
            native=PINOCCHIO_CLOSURE_CONVENTION,
            grip_frame_rotation=rot_ok,
            grip_origin=pt_ok,
        )

    with pytest.raises(ValueError, match="shape \\(6,\\)"):
        to_canonical_closure(
            j_ok,
            np.zeros(5),
            native=PINOCCHIO_CLOSURE_CONVENTION,
            grip_frame_rotation=rot_ok,
            grip_origin=pt_ok,
        )

    with pytest.raises(ValueError, match="SO\\(3\\)"):
        to_canonical_closure(
            j_ok,
            d_ok,
            native=PINOCCHIO_CLOSURE_CONVENTION,
            grip_frame_rotation=2.0 * rot_ok,
            grip_origin=pt_ok,
        )

    with pytest.raises(ValueError, match="SO\\(3\\)"):
        # Improper reflection det = -1
        refl = np.diag([1.0, 1.0, -1.0])
        to_canonical_closure(
            j_ok,
            d_ok,
            native=PINOCCHIO_CLOSURE_CONVENTION,
            grip_frame_rotation=refl,
            grip_origin=pt_ok,
        )

    j_nan = j_ok.copy()
    j_nan[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        to_canonical_closure(
            j_nan,
            d_ok,
            native=PINOCCHIO_CLOSURE_CONVENTION,
            grip_frame_rotation=rot_ok,
            grip_origin=pt_ok,
        )

    with pytest.raises(TypeError, match="NativeClosureConvention"):
        to_canonical_closure(
            j_ok,
            d_ok,
            native="pinocchio",  # type: ignore[arg-type]
            grip_frame_rotation=rot_ok,
            grip_origin=pt_ok,
        )
