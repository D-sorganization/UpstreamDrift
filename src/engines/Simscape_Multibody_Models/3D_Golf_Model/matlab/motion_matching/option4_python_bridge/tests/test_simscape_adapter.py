"""Round-trip tests for SimscapeAdapter.

Each test simulates a fixed theta vector through the Python adapter and
compares it against a direct MATLAB-side call to ``simulate_with_coefficients``.
A successful round-trip means the adapter's marshalling is faithful — the
flat-double bridge across ``matlab.engine`` is not corrupting the data.

These tests REQUIRE ``matlab.engine`` to be installable in the active Python
env. On hosts without MATLAB, conftest.py auto-skips with a loud reason.
"""

from __future__ import annotations

import numpy as np
import pytest


@pytest.mark.requires_matlab_engine
def test_engine_starts_and_stops_cleanly():
    from simscape_adapter import SimscapeAdapter

    a = SimscapeAdapter()
    assert a._engine is None  # lazy
    a.start()
    assert a._engine is not None
    a.close()
    assert a._engine is None
    # Idempotent
    a.close()


@pytest.mark.requires_matlab_engine
def test_simulate_with_coefficients_returns_canonical_simout(
    adapter, bounds, has_simscape_multibody
):
    """A zero-coefficient run should still produce a SimOut with consistent shapes."""
    if not has_simscape_multibody:
        pytest.skip(
            "Simscape Multibody license not available — cannot run forward sim."
        )
    lb, _ = bounds
    theta = np.zeros_like(lb)
    sim_out = adapter.simulate_with_coefficients(theta)
    N = sim_out.time.shape[0]
    assert N > 1
    assert sim_out.time[0] == pytest.approx(0.0, abs=1e-9)
    assert sim_out.grip.shape == (N, 3)
    assert sim_out.clubhead.shape == (N, 3)
    assert sim_out.club_quat.shape == (N, 4)
    assert sim_out.q.shape[0] == N
    assert sim_out.tau.shape[0] == N
    if sim_out.solver_status == "success":
        assert np.all(np.isfinite(sim_out.tau))
        assert np.all(np.isfinite(sim_out.q))


@pytest.mark.requires_matlab_engine
def test_round_trip_against_direct_matlab_call(adapter, bounds, has_simscape_multibody):
    """Python adapter and direct MATLAB call must agree to within 1e-6 m."""
    if not has_simscape_multibody:
        pytest.skip(
            "Simscape Multibody license not available — cannot run forward sim."
        )
    import matlab

    lb, _ = bounds
    rng = np.random.default_rng(1234)
    # Small coefficients keep the integrator well-behaved.
    theta = (rng.uniform(-0.05, 0.05, size=lb.size) * np.abs(lb)).astype(np.float64)

    # Path A: through the adapter.
    sim_py = adapter.simulate_with_coefficients(theta)

    # Path B: directly via the engine.
    eng = adapter.engine
    theta_m = matlab.double(theta.reshape(-1, 1).tolist())
    sim_ml = eng.simulate_with_coefficients(theta_m, nargout=1)

    grip_ml = np.array(sim_ml["r_butt"], dtype=np.float64).reshape(-1, 3)
    clubhead_ml = np.array(sim_ml["r_clubhead"], dtype=np.float64).reshape(-1, 3)
    quat_ml = np.array(sim_ml["q_club"], dtype=np.float64).reshape(-1, 4)

    assert sim_py.grip.shape == grip_ml.shape
    diff_grip = sim_py.grip - grip_ml
    rmse_grip = float(np.sqrt(np.vdot(diff_grip, diff_grip) / diff_grip.size))
    diff_ch = sim_py.clubhead - clubhead_ml
    rmse_clubhead = float(np.sqrt(np.vdot(diff_ch, diff_ch) / diff_ch.size))
    diff_q = sim_py.club_quat - quat_ml
    rmse_quat = float(np.sqrt(np.vdot(diff_q, diff_q) / diff_q.size))
    assert rmse_grip < 1e-6, f"grip RMSE {rmse_grip} too large"
    assert rmse_clubhead < 1e-6, f"clubhead RMSE {rmse_clubhead} too large"
    assert rmse_quat < 1e-6, f"quat RMSE {rmse_quat} too large"


@pytest.mark.requires_matlab_engine
def test_get_polynomial_bounds_shape(adapter, n_joints):
    lb, ub = adapter.get_polynomial_bounds(n_joints)
    assert lb.size == n_joints * 7
    assert ub.size == n_joints * 7
    assert np.all(lb < ub)


def test_simulate_rejects_bad_theta_shape():
    """Pre-condition validation does not require a live MATLAB engine."""
    from simscape_adapter import SimscapeAdapter

    a = SimscapeAdapter()
    # length not a multiple of 7
    with pytest.raises(ValueError):
        a.simulate_with_coefficients(np.zeros(6))
    # non-finite
    with pytest.raises(ValueError):
        a.simulate_with_coefficients(np.array([np.nan] * 7))


def test_simout_dataclass_is_frozen():
    from simscape_adapter import SimOut

    out = SimOut(
        time=np.zeros(3),
        grip=np.zeros((3, 3)),
        clubhead=np.zeros((3, 3)),
        club_quat=np.tile([1.0, 0.0, 0.0, 0.0], (3, 1)),
        q=np.zeros((3, 1)),
        qd=np.zeros((3, 1)),
        tau=np.zeros((3, 1)),
        omega=np.zeros((3, 1)),
        joint_names=("j1",),
        solver_status="success",
        impact_idx=0,
    )
    # Frozen dataclasses raise FrozenInstanceError (a TypeError subclass)
    with pytest.raises((AttributeError, TypeError)):
        out.solver_status = "failed"  # type: ignore[misc]
