"""Self-contained pytest verification suite for UpstreamDrift Putting Dynamics (v3.2/v3.3).

Verifies:
1. UpstreamDrift Architecture Budgets:
   - <= 500 LOC per file
   - <= 100 LOC per function
   - <= 8 parameters per function
2. Issue D-11803 Zero-Allocation @precondition Signature Caching:
   - Signature resolved once at decoration time
   - Zero inspect.signature() calls at function execution time
   - Contract enforcement for invalid arguments
3. turf_properties.py:
   - Corrected USGA Stimpmeter rolling friction: mu_roll = 0.560 / stimp_rating
   - Oblong elliptical green boundary containment and orientation invariance
   - Holmes/Penner chord-dependent cup capture: v_crit(b) = 1.38 * (1 - (b/R_h)^2)^0.75
4. grip_nullspace_stabilization.py:
   - 4-point dual-hand grasp matrix A_couple (both 3x4 modal couple and 3x12 spatial)
   - Moore-Penrose pseudoinverse minimum-norm Moment-of-Force drive u_task = A_couple^+ tau_task
   - Orthogonal Null-Space counter-torque preload u_null = N_null tau_preload
   - ZTCF / ZVCF counterfactual acceleration decomposition and superposition identities
"""

from __future__ import annotations

import ast
from contextlib import contextmanager
import inspect
import math
from pathlib import Path
import re
import sys
from typing import Any, Callable, Iterable, Iterator, Sequence, Tuple
from unittest.mock import patch

import numpy as np

try:
    import pytest  # type: ignore[import-not-found]
except ImportError:

    class _Approx:
        """Fallback pytest.approx implementation for environments without pytest."""

        def __init__(self, expected: float, rel: float = 1e-6, abs_tol: float = 1e-12) -> None:
            self.expected = float(expected)
            self.rel = float(rel)
            self.abs_tol = float(abs_tol)

        def __eq__(self, actual: object) -> bool:
            if not isinstance(actual, (int, float, np.floating)):
                return False
            return math.isclose(
                float(actual), self.expected, rel_tol=self.rel, abs_tol=self.abs_tol
            )

    class _Mark:
        @staticmethod
        def parametrize(
            argnames: str | Sequence[str], argvalues: Iterable[Any]
        ) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
            def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
                setattr(func, "_parametrize", (argnames, list(argvalues)))
                return func

            return decorator

    class _PytestFallback:
        mark = _Mark()

        @staticmethod
        def approx(expected: float, rel: float = 1e-6, abs: float = 1e-12) -> _Approx:
            return _Approx(expected, rel=rel, abs_tol=abs)

        @staticmethod
        @contextmanager
        def raises(
            exc_type: type[BaseException], match: str | None = None
        ) -> Iterator[None]:
            try:
                yield
            except exc_type as exc:
                if match is not None and not re.search(match, str(exc)):
                    raise AssertionError(
                        f"Exception '{exc}' did not match pattern '{match}'"
                    ) from exc
                return
            raise AssertionError(f"Expected exception {exc_type.__name__} was not raised")

    pytest = _PytestFallback()  # type: ignore[assignment]

MODULE_DIR = Path(__file__).resolve().parent
if str(MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(MODULE_DIR))

import grip_nullspace_stabilization as gns  # noqa: E402
import turf_properties as tp  # noqa: E402

MAX_FILE_LOC = 500
MAX_FUNC_LOC = 100
MAX_FUNC_PARAMS = 8


# ============================================================================
# 1. UpstreamDrift Architecture Budget Verification
# ============================================================================


@pytest.mark.parametrize(
    "filename",
    [
        "turf_properties.py",
        "grip_nullspace_stabilization.py",
        "test_putting_dynamics_suite.py",
    ],
)
def test_architecture_budgets_ast(filename: str) -> None:
    """Enforce <=500 LOC/file, <=100 LOC/function, and <=8 parameters/function."""
    path = MODULE_DIR / filename
    source = path.read_text(encoding="utf-8")
    lines = source.splitlines()
    assert len(lines) <= MAX_FILE_LOC, (
        f"{filename} has {len(lines)} lines, exceeding limit of {MAX_FILE_LOC}"
    )

    tree = ast.parse(source, filename=filename)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end_lineno = getattr(node, "end_lineno", node.lineno)
            func_loc = end_lineno - node.lineno + 1
            assert func_loc <= MAX_FUNC_LOC, (
                f"{filename}:{node.name} is {func_loc} LOC (limit {MAX_FUNC_LOC})"
            )

            args = node.args
            total_params = (
                len(args.posonlyargs)
                + len(args.args)
                + len(args.kwonlyargs)
                + (1 if args.vararg else 0)
                + (1 if args.kwarg else 0)
            )
            assert total_params <= MAX_FUNC_PARAMS, (
                f"{filename}:{node.name} has {total_params} parameters "
                f"(limit {MAX_FUNC_PARAMS})"
            )


# ============================================================================
# 2. Issue D-11803: Zero-Allocation @precondition Signature Caching
# ============================================================================


def test_precondition_zero_allocation_signature_caching() -> None:
    """Verify inspect.signature is cached at decoration time and never called at runtime."""
    assert tp.compute_rolling_friction.__wrapped__ in tp._SIGNATURE_CACHE
    assert gns.solve_grip_nullspace_stabilization.__wrapped__ in gns._SIGNATURE_CACHE

    with patch.object(
        inspect,
        "signature",
        side_effect=AssertionError("inspect.signature called at runtime!"),
    ):
        for stimp in (8.5, 10.0, 11.5, 13.0):
            mu = tp.compute_rolling_friction(stimp)
            assert mu == pytest.approx(0.560 / stimp, rel=1e-12)
            mu_kw = tp.compute_rolling_friction(stimp_rating=stimp)
            assert mu_kw == pytest.approx(0.560 / stimp, rel=1e-12)

        v_c = tp.critical_capture_velocity(0.01)
        assert v_c > 0.0

        a_mat = gns.build_grasp_matrix_a_couple()
        tau_task = np.array([0.0, 1.25, 0.08], dtype=np.float64)
        tau_pre = np.array([4.0, -4.0, 3.5, -3.5], dtype=np.float64)
        sol = gns.solve_grip_nullspace_stabilization(a_mat, tau_task, tau_pre)
        assert sol.null_leakage_norm < 1e-12


def test_precondition_contract_violations_raise_value_error() -> None:
    """Verify @precondition raises ValueError on invalid inputs."""
    with pytest.raises(ValueError, match="stimp_rating"):
        tp.compute_rolling_friction(0.0)
    with pytest.raises(ValueError, match="stimp_rating"):
        tp.compute_rolling_friction(-10.5)
    with pytest.raises(ValueError, match="stimp_rating"):
        tp.compute_rolling_friction(float("nan"))

    with pytest.raises(ValueError, match="cup_radius_m"):
        tp.critical_capture_velocity(0.01, cup_radius_m=0.0)


# ============================================================================
# 3. Turf Properties & Holmes/Penner Cup Capture Verification
# ============================================================================


@pytest.mark.parametrize(
    ("stimp_rating", "expected_mu"),
    [
        (8.0, 0.560 / 8.0),
        (10.0, 0.0560),
        (11.2, 0.0500),
        (12.5, 0.560 / 12.5),
        (14.0, 0.0400),
    ],
)
def test_usga_stimpmeter_rolling_friction_formula(
    stimp_rating: float, expected_mu: float
) -> None:
    """Verify corrected USGA Stimpmeter formula mu_roll = 0.560 / stimp_rating."""
    turf = tp.TurfProperties(stimp_rating=stimp_rating)
    assert turf.mu_roll == pytest.approx(expected_mu, rel=1e-14)
    assert tp.compute_rolling_friction(stimp_rating) == pytest.approx(
        expected_mu, rel=1e-14
    )


def test_oblong_green_boundary_containment() -> None:
    """Verify oblong elliptical green containment and rotated frame handling."""
    boundary = tp.OblongGreenBoundary(
        semi_major_a=20.0,
        semi_minor_b=10.0,
        center_x=2.0,
        center_y=-3.0,
        heading_rad=math.pi / 4.0,
    )
    assert tp.is_within_oblong_green(2.0, -3.0, boundary)

    dx_major = 19.5 * math.cos(math.pi / 4.0)
    dy_major = 19.5 * math.sin(math.pi / 4.0)
    assert tp.is_within_oblong_green(2.0 + dx_major, -3.0 + dy_major, boundary)

    dx_minor = -11.0 * math.sin(math.pi / 4.0)
    dy_minor = 11.0 * math.cos(math.pi / 4.0)
    assert not tp.is_within_oblong_green(2.0 + dx_minor, -3.0 + dy_minor, boundary)


def test_holmes_penner_chord_capture_velocity() -> None:
    """Verify v_crit(b) = 1.38 * (1 - (b/R_h)^2)^0.75 across impact parameter b."""
    r_h = tp.USGA_CUP_RADIUS_M
    assert tp.critical_capture_velocity(0.0, r_h) == pytest.approx(1.38, rel=1e-14)

    for frac in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99):
        b = frac * r_h
        expected = 1.38 * ((1.0 - frac**2) ** 0.75)
        actual = tp.critical_capture_velocity(b, r_h)
        assert actual == pytest.approx(expected, rel=1e-14)
        assert tp.critical_capture_velocity(-b, r_h) == pytest.approx(expected, rel=1e-14)

    assert tp.critical_capture_velocity(r_h, r_h) == 0.0
    assert tp.critical_capture_velocity(1.2 * r_h, r_h) == 0.0

    res_center = tp.evaluate_cup_capture(ball_speed_m_s=1.20, impact_offset_b_m=0.0)
    assert res_center.is_captured is True
    assert res_center.outcome == "CAPTURED_CENTER"

    res_chord = tp.evaluate_cup_capture(
        ball_speed_m_s=0.65, impact_offset_b_m=0.5 * r_h
    )
    assert res_chord.is_captured is True
    assert res_chord.outcome == "CAPTURED_CHORD"

    res_lipout = tp.evaluate_cup_capture(
        ball_speed_m_s=1.30, impact_offset_b_m=0.6 * r_h
    )
    assert res_lipout.is_captured is False
    assert res_lipout.outcome == "LIP_OUT_FAST"

    res_miss = tp.evaluate_cup_capture(
        ball_speed_m_s=0.20, impact_offset_b_m=1.05 * r_h
    )
    assert res_miss.is_captured is False
    assert res_miss.outcome == "MISS_OUTSIDE_CUP"


def test_turf_rolling_acceleration_physics() -> None:
    """Verify 5/7 rolling sphere slope factor and opposing Stimp friction vector."""
    turf = tp.TurfProperties(stimp_rating=11.2)
    vel = np.array([2.0, 0.0], dtype=np.float64)
    grad = np.array([0.02, -0.01], dtype=np.float64)
    acc = tp.compute_turf_rolling_acceleration(vel, grad, turf)

    expected_x = -(5.0 / 7.0) * turf.gravity_m_s2 * 0.02 - 0.05 * turf.gravity_m_s2
    expected_y = -(5.0 / 7.0) * turf.gravity_m_s2 * (-0.01)
    np.testing.assert_allclose(acc, np.array([expected_x, expected_y]), rtol=1e-13)


# ============================================================================
# 4. Grip Null-Space Stabilization & ZTCF/ZVCF Decomposition Verification
# ============================================================================


def test_grip_nullspace_stabilization_modal_and_spatial() -> None:
    """Verify u_task = A_couple^+ tau_task and u_null = N_null tau_preload properties."""
    a_modal = gns.build_grasp_matrix_a_couple(spatial_3d=False)
    assert a_modal.shape == (3, 4)
    assert np.linalg.matrix_rank(a_modal) == 3

    tau_task = np.array([0.0, 1.40, 0.05], dtype=np.float64)
    tau_preload = np.array([12.0, -8.0, 9.0, -11.0], dtype=np.float64)

    sol = gns.solve_grip_nullspace_stabilization(a_modal, tau_task, tau_preload)

    np.testing.assert_allclose(sol.realized_tau_task, tau_task, atol=1e-12)
    assert sol.null_leakage_norm < 1e-12
    assert abs(sol.orthogonality_inner_product) < 1e-12
    np.testing.assert_allclose(sol.n_null @ sol.n_null, sol.n_null, atol=1e-12)
    np.testing.assert_allclose(sol.n_null.T, sol.n_null, atol=1e-12)
    assert np.linalg.norm(sol.u_total) ** 2 == pytest.approx(
        np.linalg.norm(sol.u_task) ** 2 + np.linalg.norm(sol.u_null) ** 2,
        rel=1e-12,
    )

    a_spatial = gns.build_grasp_matrix_a_couple(spatial_3d=True)
    assert a_spatial.shape == (3, 12)
    assert np.linalg.matrix_rank(a_spatial) == 3

    tau_3d = np.array([0.15, 1.65, -0.08], dtype=np.float64)
    preload_12d = np.linspace(-5.0, 5.0, 12, dtype=np.float64)
    sol_3d = gns.solve_grip_nullspace_stabilization(a_spatial, tau_3d, preload_12d)
    np.testing.assert_allclose(sol_3d.realized_tau_task, tau_3d, atol=1e-12)
    assert sol_3d.null_leakage_norm < 1e-12
    assert abs(sol_3d.orthogonality_inner_product) < 1e-12


def test_ztcf_zvcf_counterfactual_acceleration_decomposition() -> None:
    """Verify ZTCF and ZVCF counterfactual decomposition and exact linear superposition."""
    m_mat = np.array(
        [
            [0.32, 0.08, 0.03],
            [0.08, 0.14, 0.04],
            [0.03, 0.04, 0.06],
        ],
        dtype=np.float64,
    )
    c_mat = np.array(
        [
            [0.00, -0.05, -0.02],
            [0.05, 0.00, -0.01],
            [0.02, 0.01, 0.00],
        ],
        dtype=np.float64,
    )
    g_vec = np.array([0.45, 0.18, 0.06], dtype=np.float64)
    q_dot = np.array([1.20, -0.45, 0.80], dtype=np.float64)
    tau_applied = np.array([1.85, 0.42, -0.12], dtype=np.float64)

    decomp = gns.decompose_ztcf_zvcf_accelerations(
        m_mat, c_mat, g_vec, q_dot, tau_applied
    )

    assert decomp.superposition_residual_norm < 1e-14
    np.testing.assert_allclose(
        decomp.q_ddot_actual,
        decomp.q_ddot_ztcf + decomp.q_ddot_zvcf - decomp.q_ddot_gravity,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        decomp.q_ddot_actual - decomp.q_ddot_ztcf,
        decomp.q_ddot_torque,
        atol=1e-14,
    )
    np.testing.assert_allclose(
        decomp.q_ddot_actual - decomp.q_ddot_zvcf,
        decomp.q_ddot_velocity,
        atol=1e-14,
    )

    decomp_zero_tau = gns.decompose_ztcf_zvcf_accelerations(
        m_mat, c_mat, g_vec, q_dot, np.zeros(3)
    )
    np.testing.assert_allclose(
        decomp_zero_tau.q_ddot_actual, decomp_zero_tau.q_ddot_ZTCF if hasattr(decomp_zero_tau, "q_ddot_ZTCF") else decomp_zero_tau.q_ddot_ztcf, atol=1e-14
    )

    decomp_zero_vel = gns.decompose_ztcf_zvcf_accelerations(
        m_mat, c_mat, g_vec, np.zeros(3), tau_applied
    )
    np.testing.assert_allclose(
        decomp_zero_vel.q_ddot_actual, decomp_zero_vel.q_ddot_zvcf, atol=1e-14
    )


def _run_suite_standalone() -> Tuple[int, int]:
    """Execute all test_* functions (including parametrized cases) standalone."""
    passed = 0
    for name, obj in list(globals().items()):
        if not name.startswith("test_") or not callable(obj):
            continue
        param_spec = getattr(obj, "_parametrize", None)
        if param_spec is None:
            obj()
            passed += 1
        else:
            _, values = param_spec
            for case in values:
                if isinstance(case, tuple):
                    obj(*case)
                else:
                    obj(case)
                passed += 1
    return passed, 0


if __name__ == "__main__":
    num_passed, _ = _run_suite_standalone()
    print(f"All {num_passed} verification checks passed.")
