"""Tests for ``simulate_with_coefficients`` and the polynomial-torque driver.

Covers:

- Recovery / sanity: zero-torque rollout falls under gravity.
- Determinism: identical inputs -> identical outputs (no global-state leakage).
- Postcondition shape checks: every output array matches the canonical
  ``N = round(T_s * output_rate_hz) + 1`` row count.

Marked ``requires_mujoco``; the entire module is skipped if the ``mujoco``
package is unavailable (see ``conftest.py``).
"""

from __future__ import annotations

import time

import numpy as np
import pytest
from src.engines.physics_engines.mujoco.python.motion_matching import (
    simulate as simulate_module,
)
from src.engines.physics_engines.mujoco.python.motion_matching.fit_swing import (
    FitOptions,
    MinimizerOptions,
    fit_swing_mujoco,
)
from src.engines.physics_engines.mujoco.python.motion_matching.simulate import (
    SimOptions,
    SimOut,
    simulate_with_coefficients,
    synthesize_target_from_coefficients,
)
from src.engines.physics_engines.mujoco.python.motion_matching.torque_driver import (
    POLY_BOUNDS,
    PolynomialTorqueDriver,
    polynomial_torque_bounds,
)

pytestmark = [pytest.mark.requires_mujoco, pytest.mark.unit]


# --- Helpers ----------------------------------------------------------------


def _expected_n(opts: SimOptions) -> int:
    return int(round(opts.T_s * opts.output_rate_hz)) + 1


def _upper_body_nu() -> int:
    import mujoco
    from src.engines.physics_engines.mujoco._golf_swing_upper_body_xml import (
        UPPER_BODY_GOLF_SWING_XML,
    )

    return int(mujoco.MjModel.from_xml_string(UPPER_BODY_GOLF_SWING_XML).nu)


# --- Recovery / sanity ------------------------------------------------------


def test_simulate_exports_synthesize_recovery_oracle() -> None:
    """The simulate module exports the synthesize -> fit -> recover oracle."""
    from src.engines.physics_engines.mujoco.python.motion_matching import (
        synthesize as mj_synthesize,
    )

    assert "synthesize_target_from_coefficients" in simulate_module.__all__
    assert (
        synthesize_target_from_coefficients
        is mj_synthesize.synthesize_target_from_coefficients
    )

    opts = SimOptions(
        variant="upper",
        T_s=0.05,
        output_rate_hz=100.0,
        clip_torque_to_ctrlrange=False,
    )
    theta_truth = np.zeros(_upper_body_nu() * 7, dtype=np.float64)
    target = synthesize_target_from_coefficients(theta_truth, sim_options=opts)

    result = fit_swing_mujoco(
        target,
        FitOptions(
            sim=opts,
            minimizer=MinimizerOptions(
                maxiter=1,
                theta0=theta_truth,
                warm_start_scale=0.01,
            ),
            rng_seed=42,
        ),
    )

    assert np.isfinite(result.final_rmse_m)
    assert result.final_rmse_m < 1e-8
    np.testing.assert_allclose(result.theta_optimal, theta_truth, atol=1e-10)


def test_zero_torque_falls_under_gravity_full() -> None:
    """With theta=0 the grip Z must drop monotonically over the swing."""
    opts = SimOptions(variant="full", T_s=0.3, output_rate_hz=1000.0)
    import mujoco  # local probe for nu
    from src.engines.physics_engines.mujoco._golf_swing_full_body_xml import (
        FULL_BODY_GOLF_SWING_XML,
    )

    nu = mujoco.MjModel.from_xml_string(FULL_BODY_GOLF_SWING_XML).nu
    theta = np.zeros(nu * 7, dtype=np.float64)

    out = simulate_with_coefficients(theta, opts)

    assert out.solver_status == "ok"
    # Grip Z at the end is strictly below grip Z at the start under gravity.
    assert out.grip[-1, 2] < out.grip[0, 2] - 1e-3, (
        f"grip Z did not drop under gravity: "
        f"start={out.grip[0, 2]:.4f}, end={out.grip[-1, 2]:.4f}"
    )


def test_zero_torque_falls_under_gravity_upper() -> None:
    """Same as above, exercising the upper-body variant."""
    opts = SimOptions(variant="upper", T_s=0.3, output_rate_hz=500.0)
    import mujoco
    from src.engines.physics_engines.mujoco._golf_swing_upper_body_xml import (
        UPPER_BODY_GOLF_SWING_XML,
    )

    nu = mujoco.MjModel.from_xml_string(UPPER_BODY_GOLF_SWING_XML).nu
    theta = np.zeros(nu * 7, dtype=np.float64)
    out = simulate_with_coefficients(theta, opts)
    assert out.solver_status == "ok"
    assert out.clubhead[-1, 2] < out.clubhead[0, 2]


# --- Determinism ------------------------------------------------------------


def test_determinism_back_to_back() -> None:
    """Two consecutive runs with identical theta produce identical outputs."""
    opts = SimOptions(variant="full", T_s=0.2, output_rate_hz=500.0)
    import mujoco
    from src.engines.physics_engines.mujoco._golf_swing_full_body_xml import (
        FULL_BODY_GOLF_SWING_XML,
    )

    nu = mujoco.MjModel.from_xml_string(FULL_BODY_GOLF_SWING_XML).nu
    rng = np.random.default_rng(0)
    theta = rng.uniform(-0.5, 0.5, size=nu * 7).astype(np.float64)

    out_a = simulate_with_coefficients(theta, opts)
    out_b = simulate_with_coefficients(theta, opts)

    np.testing.assert_array_equal(out_a.q, out_b.q)
    np.testing.assert_array_equal(out_a.qd, out_b.qd)
    np.testing.assert_array_equal(out_a.tau, out_b.tau)
    np.testing.assert_array_equal(out_a.grip, out_b.grip)
    np.testing.assert_array_equal(out_a.clubhead, out_b.clubhead)


def test_callback_uninstalls_cleanly_between_runs() -> None:
    """The second of two runs with different theta must use the new theta.

    This guards against global-state leakage of ``mjcb_control`` across
    ``simulate_with_coefficients`` calls (the driver is process-global).
    """
    opts = SimOptions(variant="full", T_s=0.1, output_rate_hz=1000.0)
    import mujoco
    from src.engines.physics_engines.mujoco._golf_swing_full_body_xml import (
        FULL_BODY_GOLF_SWING_XML,
    )

    nu = mujoco.MjModel.from_xml_string(FULL_BODY_GOLF_SWING_XML).nu
    theta_zero = np.zeros(nu * 7, dtype=np.float64)
    theta_const = np.zeros((nu, 7), dtype=np.float64)
    theta_const[:, 6] = 5.0  # constant torque on every joint

    out_zero = simulate_with_coefficients(theta_zero, opts)
    out_const = simulate_with_coefficients(theta_const.flatten(), opts)

    # Different inputs must produce different trajectories. (Specifically,
    # theta_const drives the joints; theta_zero lets gravity dominate.)
    assert not np.allclose(out_zero.q, out_const.q), (
        "second run produced identical state — likely callback leakage"
    )
    # The downstream test ``test_no_residual_global_callback_after_run``
    # additionally probes that ``mjcb_control`` is cleared globally.


def test_no_residual_global_callback_after_run() -> None:
    """``mjcb_control`` must be None after ``simulate_with_coefficients`` returns."""
    import mujoco

    opts = SimOptions(variant="upper", T_s=0.05, output_rate_hz=1000.0)
    from src.engines.physics_engines.mujoco._golf_swing_upper_body_xml import (
        UPPER_BODY_GOLF_SWING_XML,
    )

    nu = mujoco.MjModel.from_xml_string(UPPER_BODY_GOLF_SWING_XML).nu
    simulate_with_coefficients(np.zeros(nu * 7), opts)

    # Probe by running a step on a fresh model+data: if a leftover callback
    # were active, it would write to data.ctrl. Compare ctrl before/after.
    from src.engines.physics_engines.mujoco._golf_swing_full_body_xml import (
        FULL_BODY_GOLF_SWING_XML,
    )

    m = mujoco.MjModel.from_xml_string(FULL_BODY_GOLF_SWING_XML)
    d = mujoco.MjData(m)
    d.ctrl[:] = 0.0
    mujoco.mj_step(m, d)
    assert np.all(d.ctrl == 0.0), (
        "data.ctrl was modified by a residual global mjcb_control"
    )


# --- Postcondition shape checks --------------------------------------------


def test_output_shapes_match_canonical_grid() -> None:
    """Every output array must have ``N = round(T_s*output_rate_hz)+1`` rows."""
    opts = SimOptions(variant="full", T_s=0.3, output_rate_hz=1000.0)
    n_expected = _expected_n(opts)

    import mujoco
    from src.engines.physics_engines.mujoco._golf_swing_full_body_xml import (
        FULL_BODY_GOLF_SWING_XML,
    )

    m = mujoco.MjModel.from_xml_string(FULL_BODY_GOLF_SWING_XML)
    nq, nv, nu = m.nq, m.nv, m.nu

    out = simulate_with_coefficients(np.zeros(nu * 7), opts)
    assert isinstance(out, SimOut)
    assert out.time.shape == (n_expected,)
    assert out.q.shape == (n_expected, nq)
    assert out.qd.shape == (n_expected, nv)
    assert out.qdd.shape == (n_expected, nv)
    assert out.tau.shape == (n_expected, nu)
    assert out.grip.shape == (n_expected, 3)
    assert out.grip_quat.shape == (n_expected, 4)
    assert out.clubhead.shape == (n_expected, 3)
    assert out.club_quat.shape == (n_expected, 4)
    assert out.solver_status == "ok"
    assert out.wall_clock_s > 0.0


def test_output_unit_quaternions() -> None:
    """``grip_quat`` and ``club_quat`` must be unit-norm at every frame."""
    opts = SimOptions(variant="full", T_s=0.2, output_rate_hz=500.0)
    import mujoco
    from src.engines.physics_engines.mujoco._golf_swing_full_body_xml import (
        FULL_BODY_GOLF_SWING_XML,
    )

    nu = mujoco.MjModel.from_xml_string(FULL_BODY_GOLF_SWING_XML).nu
    out = simulate_with_coefficients(np.zeros(nu * 7), opts)

    grip_norms = np.linalg.norm(out.grip_quat, axis=1)
    head_norms = np.linalg.norm(out.club_quat, axis=1)
    np.testing.assert_allclose(grip_norms, 1.0, atol=1e-6)
    np.testing.assert_allclose(head_norms, 1.0, atol=1e-6)


def test_time_grid_is_monotonic_and_inclusive() -> None:
    """The time grid is a strictly monotonic linspace ending at exactly T_s."""
    opts = SimOptions(variant="upper", T_s=0.25, output_rate_hz=400.0)
    import mujoco
    from src.engines.physics_engines.mujoco._golf_swing_upper_body_xml import (
        UPPER_BODY_GOLF_SWING_XML,
    )

    nu = mujoco.MjModel.from_xml_string(UPPER_BODY_GOLF_SWING_XML).nu
    out = simulate_with_coefficients(np.zeros(nu * 7), opts)
    assert out.time[0] == 0.0
    assert out.time[-1] == pytest.approx(opts.T_s, abs=1e-12)
    assert np.all(np.diff(out.time) > 0.0)


# --- Performance ------------------------------------------------------------


def test_perf_under_100ms_per_swing() -> None:
    """Empirical perf target from MUJOCO_PARITY_SPEC.md §6: <100 ms/swing.

    Allow a 3x slack to keep this stable across CI hosts.
    """
    opts = SimOptions(variant="full", T_s=0.3, output_rate_hz=1000.0)
    import mujoco
    from src.engines.physics_engines.mujoco._golf_swing_full_body_xml import (
        FULL_BODY_GOLF_SWING_XML,
    )

    nu = mujoco.MjModel.from_xml_string(FULL_BODY_GOLF_SWING_XML).nu
    theta = np.zeros(nu * 7)

    # Warm-up (compile + JIT-ish) does not count.
    simulate_with_coefficients(theta, opts)

    t0 = time.perf_counter()
    out = simulate_with_coefficients(theta, opts)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    assert out.solver_status == "ok"
    # The spec target is <100 ms; allow 300 ms as a stable upper bound.
    assert elapsed_ms < 300.0, (
        f"forward-sim wall-clock {elapsed_ms:.1f} ms exceeds 300 ms "
        f"(spec target: <100 ms)"
    )


# --- Polynomial driver (white-box) ------------------------------------------


def test_polynomial_evaluate_matches_handcomputed() -> None:
    """Evaluate the polynomial directly and check against a hand-computed value."""
    import mujoco
    from src.engines.physics_engines.mujoco._golf_swing_upper_body_xml import (
        UPPER_BODY_GOLF_SWING_XML,
    )

    m = mujoco.MjModel.from_xml_string(UPPER_BODY_GOLF_SWING_XML)
    nu = m.nu
    theta = np.zeros((nu, 7), dtype=np.float64)
    # First joint: tau_0(t) = 1*t^6 + 2*t^5 + 3*t^4 + 4*t^3 + 5*t^2 + 6*t + 7
    theta[0] = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
    drv = PolynomialTorqueDriver(m, theta, t0=0.0, clip_to_ctrlrange=False)

    t = 0.1
    expected = (
        1.0 * t**6 + 2.0 * t**5 + 3.0 * t**4 + 4.0 * t**3 + 5.0 * t**2 + 6.0 * t + 7.0
    )
    got = drv.evaluate(t)
    assert got[0] == pytest.approx(expected, rel=1e-12, abs=1e-12)
    # Other joints are zero.
    np.testing.assert_array_equal(got[1:], np.zeros(nu - 1))


def test_polynomial_bounds_respect_spec() -> None:
    """``polynomial_torque_bounds`` reflects |A,B|<=1000, |C,D|<=500, etc."""
    nj = 4
    lb, ub = polynomial_torque_bounds(nj)
    assert lb.shape == (nj * 7,)
    assert ub.shape == (nj * 7,)
    np.testing.assert_array_equal(ub, -lb)

    expected_per_joint = np.array(POLY_BOUNDS, dtype=np.float64)
    for j in range(nj):
        np.testing.assert_array_equal(ub[j * 7 : (j + 1) * 7], expected_per_joint)

    # Spot-check: A,B = 1000; C,D = 500; E,F = 100; G = 25.
    assert POLY_BOUNDS[0] == 1000.0  # A
    assert POLY_BOUNDS[1] == 1000.0  # B
    assert POLY_BOUNDS[2] == 500.0  # C
    assert POLY_BOUNDS[3] == 500.0  # D
    assert POLY_BOUNDS[4] == 100.0  # E
    assert POLY_BOUNDS[5] == 100.0  # F
    assert POLY_BOUNDS[6] == 25.0  # G


def test_invalid_theta_raises() -> None:
    """A misshapen ``theta`` raises ``ValueError``."""
    import mujoco
    from src.engines.physics_engines.mujoco._golf_swing_upper_body_xml import (
        UPPER_BODY_GOLF_SWING_XML,
    )

    m = mujoco.MjModel.from_xml_string(UPPER_BODY_GOLF_SWING_XML)
    with pytest.raises(ValueError):
        PolynomialTorqueDriver(m, np.zeros(m.nu * 7 + 1))
    with pytest.raises(ValueError):
        PolynomialTorqueDriver(m, np.zeros((m.nu, 6)))
    with pytest.raises(ValueError):
        bad = np.zeros((m.nu, 7))
        bad[0, 0] = np.nan
        PolynomialTorqueDriver(m, bad)


def test_invalid_options_raise() -> None:
    """Bad ``T_s`` or ``output_rate_hz`` raise ``ValueError``."""
    import mujoco
    from src.engines.physics_engines.mujoco._golf_swing_upper_body_xml import (
        UPPER_BODY_GOLF_SWING_XML,
    )

    nu = mujoco.MjModel.from_xml_string(UPPER_BODY_GOLF_SWING_XML).nu
    theta = np.zeros(nu * 7)
    with pytest.raises(ValueError):
        simulate_with_coefficients(theta, SimOptions(variant="upper", T_s=0.0))
    with pytest.raises(ValueError):
        simulate_with_coefficients(
            theta, SimOptions(variant="upper", output_rate_hz=0.0)
        )
    with pytest.raises(ValueError):
        simulate_with_coefficients(theta, SimOptions(variant="upper", dt=0.0))


def test_list_theta_accepted_by_dbc_precondition() -> None:
    """Plain Python list ``theta`` must not crash the DbC precondition.

    Regression for issue #4271 / #4272: the precondition added in PR
    #4268 dereferenced ``theta.size`` directly, which raised
    ``AttributeError`` for list-shaped coefficient vectors before the
    function could normalise inputs via ``np.asarray``. The
    historically-supported list contract must still work.
    """
    import mujoco
    from src.engines.physics_engines.mujoco._golf_swing_upper_body_xml import (
        UPPER_BODY_GOLF_SWING_XML,
    )

    nu = mujoco.MjModel.from_xml_string(UPPER_BODY_GOLF_SWING_XML).nu
    theta_list = [0.0] * (nu * 7)
    # Should not raise AttributeError; rollout completes without
    # touching the simulator beyond the basic shape checks.
    out = simulate_with_coefficients(
        theta_list,
        SimOptions(variant="upper", T_s=0.05, output_rate_hz=200.0),
    )
    assert isinstance(out, SimOut)
    assert out.q.shape[0] > 0
