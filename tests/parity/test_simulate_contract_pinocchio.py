"""Cross-engine §2.2 contract tests for the Pinocchio forward simulator.

Covers the seven gap items from issue #4255 against
``src.engines.physics_engines.pinocchio.python.motion_matching.simulate.
simulate_with_coefficients``.

Pinocchio's :class:`SimOut` keeps poses as 3x3 rotation matrices
(``grip_rotation`` / ``clubhead_rotation``); the cross-engine contract
maps those to ``q_club`` and the position fields to
``r_grip`` / ``r_clubhead``. The contract test asserts on the actual
field names (``grip_position`` / ``clubhead_position``) per the
canonical engine SimOut so it sees regressions if anyone renames or
removes a field.

Pinocchio is gated on ``@pytest.mark.requires_pinocchio`` and skips
cleanly when the engine is unavailable.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

# Skip the entire module cleanly when the optional engine isn't installed —
# the ``requires_pinocchio`` marker is informational; the module-level
# ``importorskip`` is what actually drops us off the CI graph.
pytest.importorskip("pinocchio")

from src.engines.physics_engines.pinocchio.python.motion_matching.simulate import (  # noqa: E402
    COEFFS_PER_JOINT,
    SimOptions,
    SimOut,
    simulate_with_coefficients,
)

pytestmark = [pytest.mark.requires_pinocchio, pytest.mark.unit]

_PIN_AVAILABLE = importlib.util.find_spec("pinocchio") is not None
_GOLFER_URDF = (
    Path(__file__).resolve().parents[2]
    / "src/engines/physics_engines/pinocchio/models/generated/golfer.urdf"
)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _model_nv() -> int:
    """Resolve actuated DOF count from the canonical golfer URDF."""
    if not _PIN_AVAILABLE:
        pytest.skip("pinocchio not installed")
    if not _GOLFER_URDF.exists():
        pytest.skip(f"golfer.urdf not found at {_GOLFER_URDF}")
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(_GOLFER_URDF))
    return int(model.nv)


def _short_opts() -> SimOptions:
    """5 ms / 1 kHz window. compute_energy off to keep wall-clock low."""
    return SimOptions(t_final=0.005, dt=1e-3, compute_energy=False)


def _has_canonical_theta_validator() -> bool:
    spec = importlib.util.find_spec("src.shared.python.motion_matching.theta_validator")
    return spec is not None


# --------------------------------------------------------------------------- #
# Gap 1 — Happy path.
# --------------------------------------------------------------------------- #


def test_simulate_contract_pinocchio_happy_path_returns_canonical_simout() -> None:
    """Valid theta -> canonical SimOut with aligned, finite arrays.

    Pinocchio's SimOut uses ``t``, ``q``, ``qd``, ``tau``,
    ``grip_position``, ``grip_rotation``, ``clubhead_position``,
    ``clubhead_rotation``. The cross-engine spec maps those to
    ``r_grip`` / ``r_clubhead`` / ``q_club``.
    """
    nv = _model_nv()
    theta = np.zeros(nv * COEFFS_PER_JOINT)
    out = simulate_with_coefficients(theta, options=_short_opts())

    assert isinstance(out, SimOut)
    n = out.t.shape[0]
    assert n == out.q.shape[0] == out.qd.shape[0] == out.tau.shape[0]
    assert out.grip_position.shape == (n, 3)
    assert out.grip_rotation.shape == (n, 3, 3)
    assert out.clubhead_position.shape == (n, 3)
    assert out.clubhead_rotation.shape == (n, 3, 3)
    assert np.all(np.isfinite(out.q))
    assert np.all(np.isfinite(out.qd))
    assert np.all(np.isfinite(out.grip_position))
    assert np.all(np.isfinite(out.clubhead_position))


# --------------------------------------------------------------------------- #
# Gap 2 — Zero theta runs and produces a non-trivial gravity-only motion.
# --------------------------------------------------------------------------- #


def test_zero_theta_runs_and_is_nontrivial() -> None:
    """theta = 0 runs without error; gravity drives a non-trivial trajectory."""
    nv = _model_nv()
    theta = np.zeros(nv * COEFFS_PER_JOINT)
    out = simulate_with_coefficients(
        theta,
        options=SimOptions(t_final=0.05, dt=1e-3, compute_energy=False),
    )
    np.testing.assert_allclose(out.tau, 0.0)
    assert np.all(np.isfinite(out.q))
    assert np.all(np.isfinite(out.qd))
    # Gravity must move at least one DOF over 50 ms.
    assert not np.allclose(out.q[-1], out.q[0]), (
        "zero-torque rollout did not move at all"
    )


# --------------------------------------------------------------------------- #
# Gap 3 — Out-of-bounds theta. Defensive against PR #4252.
# --------------------------------------------------------------------------- #


def test_simulate_contract_pinocchio_out_of_bounds_theta_rejected_or_handled() -> None:
    nv = _model_nv()
    theta = np.zeros(nv * COEFFS_PER_JOINT)
    theta[0] = 1.0e9

    if _has_canonical_theta_validator():
        with pytest.raises(ValueError):
            simulate_with_coefficients(theta, options=_short_opts())
        return

    try:
        out = simulate_with_coefficients(theta, options=_short_opts())
    except ValueError:
        return  # pinocchio raised on its own DbC (divergence => non-finite)
    # If it didn't raise, the integrator has to have produced finite state.
    assert np.all(np.isfinite(out.q))


# --------------------------------------------------------------------------- #
# Gap 4 — Wrong-length theta.
# --------------------------------------------------------------------------- #


def test_simulate_contract_pinocchio_wrong_length_theta_raises() -> None:
    """Wrong joint-count theta -> ValueError."""
    nv = _model_nv()
    # Off by one joint => length not n_joints*7 for the loaded model.
    bad = np.zeros((nv + 1) * COEFFS_PER_JOINT)
    with pytest.raises(ValueError):
        simulate_with_coefficients(bad, options=_short_opts())


def test_non_multiple_of_seven_theta_raises() -> None:
    """A theta whose length isn't a multiple of 7 raises ValueError."""
    bad = np.zeros(13)  # 13 % 7 != 0; also != n_joints*7 for any model.
    with pytest.raises(ValueError):
        simulate_with_coefficients(bad, options=_short_opts())


# --------------------------------------------------------------------------- #
# Gap 5 — NaN / Inf theta.
# --------------------------------------------------------------------------- #


def test_simulate_contract_pinocchio_nan_theta_raises() -> None:
    nv = _model_nv()
    bad = np.zeros(nv * COEFFS_PER_JOINT)
    bad[1] = np.nan
    with pytest.raises(ValueError):
        simulate_with_coefficients(bad, options=_short_opts())


def test_simulate_contract_pinocchio_inf_theta_raises() -> None:
    nv = _model_nv()
    bad = np.zeros(nv * COEFFS_PER_JOINT)
    bad[6] = np.inf
    with pytest.raises(ValueError):
        simulate_with_coefficients(bad, options=_short_opts())


# --------------------------------------------------------------------------- #
# Gap 6 — Time monotonicity, t[0] == 0.
# --------------------------------------------------------------------------- #


def test_simulate_contract_pinocchio_time_monotonic_starts_at_zero() -> None:
    nv = _model_nv()
    theta = np.zeros(nv * COEFFS_PER_JOINT)
    out = simulate_with_coefficients(theta, options=_short_opts())
    assert out.t[0] == 0.0
    assert np.all(np.diff(out.t) > 0)


# --------------------------------------------------------------------------- #
# Gap 7 — q width matches n_joints / model.nq at every step.
# --------------------------------------------------------------------------- #


def test_q_width_matches_model_nq() -> None:
    """SimOut.q has model.nq columns at every timestep."""
    import pinocchio as pin

    model = pin.buildModelFromUrdf(str(_GOLFER_URDF))
    nq, nv = int(model.nq), int(model.nv)

    theta = np.zeros(nv * COEFFS_PER_JOINT)
    out = simulate_with_coefficients(theta, options=_short_opts())

    assert out.q.shape[1] == nq
    assert out.qd.shape[1] == nv
    assert out.tau.shape[1] == nv
