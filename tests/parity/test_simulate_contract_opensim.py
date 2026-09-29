"""Cross-engine §2.2 contract tests for the OpenSim forward simulator.

Covers the seven gap items from issue #4255 against
``src.engines.physics_engines.opensim.python.motion_matching.simulate.
simulate_with_coefficients``.

Gated on ``@pytest.mark.requires_opensim``; skips cleanly when the
OpenSim Python bindings are not installed.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# Skip the whole module cleanly when OpenSim isn't installed — the
# ``requires_opensim`` marker is informational.
pytest.importorskip("opensim")

from src.engines.physics_engines.opensim.python.motion_matching.simulate import (  # noqa: E402
    COEFFS_PER_JOINT,
    SimOptions,
    SimOut,
    simulate_with_coefficients,
)

pytestmark = [pytest.mark.requires_opensim, pytest.mark.unit]

_OPENSIM_AVAILABLE = importlib.util.find_spec("opensim") is not None
_OSIM_PATH = (
    Path(__file__).resolve().parents[2]
    / "src/engines/physics_engines/opensim/models/golf_humanoid.osim"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _coordinate_actuator_count() -> int:
    """Resolve number of CoordinateActuators in the canonical .osim."""
    if not _OPENSIM_AVAILABLE:
        pytest.skip("opensim not installed")
    if not _OSIM_PATH.exists():
        pytest.skip(f"golf_humanoid.osim not found at {_OSIM_PATH}")
    import opensim as osim
    from src.engines.physics_engines.opensim.python.motion_matching.simulate import (
        _coordinate_actuator_names,
    )

    model = osim.Model(str(_OSIM_PATH))
    model.initSystem()
    return len(_coordinate_actuator_names(model))


def _short_opts() -> SimOptions:
    """5 ms / 1 kHz semi-explicit Euler window for fast contract checks."""
    return SimOptions(t_final=0.005, dt=1e-3, integrator="semi_explicit_euler")


def _has_canonical_theta_validator() -> bool:
    spec = importlib.util.find_spec("src.shared.python.motion_matching.theta_validator")
    return spec is not None


# --------------------------------------------------------------------------- #
# Gap 1 — Happy path.
# --------------------------------------------------------------------------- #


def test_simulate_contract_opensim_happy_path_returns_canonical_simout() -> None:
    """Valid theta -> canonical SimOut with aligned, finite arrays."""
    n_act = _coordinate_actuator_count()
    theta = np.zeros(n_act * COEFFS_PER_JOINT)
    out = simulate_with_coefficients(theta, options=_short_opts())

    assert isinstance(out, SimOut)
    n = out.time.shape[0]
    assert n == out.q.shape[0] == out.qd.shape[0]
    assert out.grip.shape == (n, 3)
    assert out.grip_quat.shape == (n, 4)
    assert out.clubhead.shape == (n, 3)
    assert out.club_quat.shape == (n, 4)
    assert np.all(np.isfinite(out.q))
    assert np.all(np.isfinite(out.qd))
    assert np.all(np.isfinite(out.grip))
    assert np.all(np.isfinite(out.clubhead))
    assert out.solver_status in {"success", "warning", "failed"}


# --------------------------------------------------------------------------- #
# Gap 2 — Zero theta runs without error.
# --------------------------------------------------------------------------- #


def test_zero_theta_runs_without_error() -> None:
    """theta = 0 runs without raising; gravity drives the kinematics."""
    n_act = _coordinate_actuator_count()
    theta = np.zeros(n_act * COEFFS_PER_JOINT)
    out = simulate_with_coefficients(
        theta,
        options=SimOptions(
            t_final=0.05,
            dt=1e-3,
            integrator="semi_explicit_euler",
        ),
    )
    np.testing.assert_allclose(out.tau, 0.0)
    assert np.all(np.isfinite(out.q))
    assert out.solver_status in {"success", "warning"}


# --------------------------------------------------------------------------- #
# Gap 3 — Out-of-bounds theta. Defensive against PR #4252.
# --------------------------------------------------------------------------- #


def test_simulate_contract_opensim_out_of_bounds_theta_rejected_or_handled() -> None:
    n_act = _coordinate_actuator_count()
    theta = np.zeros(n_act * COEFFS_PER_JOINT)
    theta[0] = 1.0e9

    if _has_canonical_theta_validator():
        with pytest.raises(ValueError):
            simulate_with_coefficients(theta, options=_short_opts())
        return

    try:
        out = simulate_with_coefficients(theta, options=_short_opts())
    except (ValueError, RuntimeError):
        # OpenSim's integrator may also raise SimTK errors under huge torques.
        return
    assert out.solver_status in {"success", "warning", "failed"}


# --------------------------------------------------------------------------- #
# Gap 4 — Wrong-length theta.
# --------------------------------------------------------------------------- #


def test_simulate_contract_opensim_wrong_length_theta_raises() -> None:
    """A theta with the wrong joint count raises ValueError."""
    n_act = _coordinate_actuator_count()
    bad = np.zeros((n_act + 1) * COEFFS_PER_JOINT)
    with pytest.raises(ValueError):
        simulate_with_coefficients(bad, options=_short_opts())


def test_non_multiple_of_seven_theta_raises() -> None:
    """A theta whose length isn't ``n_act*7`` raises ValueError."""
    bad = np.zeros(13)
    with pytest.raises(ValueError):
        simulate_with_coefficients(bad, options=_short_opts())


# --------------------------------------------------------------------------- #
# Gap 5 — NaN / Inf theta.
# --------------------------------------------------------------------------- #


def test_simulate_contract_opensim_nan_theta_raises() -> None:
    n_act = _coordinate_actuator_count()
    bad = np.zeros(n_act * COEFFS_PER_JOINT)
    bad[2] = np.nan
    with pytest.raises(ValueError):
        simulate_with_coefficients(bad, options=_short_opts())


def test_simulate_contract_opensim_inf_theta_raises() -> None:
    n_act = _coordinate_actuator_count()
    bad = np.zeros(n_act * COEFFS_PER_JOINT)
    bad[4] = np.inf
    with pytest.raises(ValueError):
        simulate_with_coefficients(bad, options=_short_opts())


# --------------------------------------------------------------------------- #
# Gap 6 — Time monotonicity, time[0] == 0.
# --------------------------------------------------------------------------- #


def test_simulate_contract_opensim_time_monotonic_starts_at_zero() -> None:
    n_act = _coordinate_actuator_count()
    theta = np.zeros(n_act * COEFFS_PER_JOINT)
    out = simulate_with_coefficients(theta, options=_short_opts())
    assert out.time[0] == 0.0
    assert np.all(np.diff(out.time) > 0)


# --------------------------------------------------------------------------- #
# Gap 7 — q width matches n_coords (model.getNumCoordinates()).
# --------------------------------------------------------------------------- #


def test_q_width_matches_model_n_coords() -> None:
    """SimOut.q has model.getNumCoordinates() columns at every timestep."""
    import opensim as osim

    model = osim.Model(str(_OSIM_PATH))
    model.initSystem()
    n_coords = int(model.getNumCoordinates())

    n_act = _coordinate_actuator_count()
    theta = np.zeros(n_act * COEFFS_PER_JOINT)
    out = simulate_with_coefficients(theta, options=_short_opts())

    assert out.q.shape[1] == n_coords
    assert out.qd.shape[1] == n_coords
