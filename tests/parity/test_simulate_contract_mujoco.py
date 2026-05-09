"""Cross-engine §2.2 contract tests for the MuJoCo forward simulator.

Covers the seven gap items from issue #4255 against
``src.engines.physics_engines.mujoco.python.motion_matching.simulate.
simulate_with_coefficients``.

Drives the **upper-body** MJCF (smaller / faster than ``full``) so the
contract suite stays inside the 60 s pytest timeout. Skips cleanly when
``mujoco`` is missing, via ``@pytest.mark.requires_mujoco``.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest
from src.engines.physics_engines.mujoco.python.motion_matching.simulate import (
    SimOptions,
    SimOut,
    simulate_with_coefficients,
)

pytestmark = [pytest.mark.requires_mujoco, pytest.mark.unit]

COEFFS_PER_JOINT = 7

_MUJOCO_AVAILABLE = importlib.util.find_spec("mujoco") is not None


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _upper_body_nu() -> int:
    """Return the actuator count of the upper-body MJCF (n_joints proxy)."""
    if not _MUJOCO_AVAILABLE:
        pytest.skip("mujoco not installed")
    import mujoco
    from src.engines.physics_engines.mujoco._golf_swing_upper_body_xml import (
        UPPER_BODY_GOLF_SWING_XML,
    )

    return int(mujoco.MjModel.from_xml_string(UPPER_BODY_GOLF_SWING_XML).nu)


def _short_opts() -> SimOptions:
    """5 ms / 1 kHz upper-body window."""
    return SimOptions(
        variant="upper",
        T_s=0.005,
        output_rate_hz=1000.0,
        clip_torque_to_ctrlrange=False,
    )


def _has_canonical_theta_validator() -> bool:
    spec = importlib.util.find_spec("src.shared.python.motion_matching.theta_validator")
    return spec is not None


# --------------------------------------------------------------------------- #
# Gap 1 — Happy path.
# --------------------------------------------------------------------------- #


def test_simulate_contract_mujoco_happy_path_returns_canonical_simout() -> None:
    """Valid theta -> canonical SimOut with aligned, finite arrays."""
    nu = _upper_body_nu()
    theta = np.zeros(nu * COEFFS_PER_JOINT)
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
# Gap 2 — Zero theta runs and produces a non-trivial gravity-only motion.
# --------------------------------------------------------------------------- #


def test_zero_theta_runs_and_is_nontrivial() -> None:
    """theta = 0 runs without error; mid-hands moves under gravity."""
    nu = _upper_body_nu()
    theta = np.zeros(nu * COEFFS_PER_JOINT)
    out = simulate_with_coefficients(
        theta,
        options=SimOptions(
            variant="upper",
            T_s=0.05,  # long enough for gravity to actually deflect the model
            output_rate_hz=1000.0,
            clip_torque_to_ctrlrange=False,
        ),
    )
    assert out.solver_status in {"success", "warning"}
    # Tau is identically zero by construction.
    np.testing.assert_allclose(out.tau, 0.0)
    # State must remain finite at every frame — gravity-only motion is
    # the engine's no-input case.
    assert np.all(np.isfinite(out.q))
    assert np.all(np.isfinite(out.qd))
    # Non-triviality: at least one joint must move from its rest position
    # under the unactuated upper-body's own weight + the club mass.
    assert not np.allclose(out.q[-1], out.q[0]), (
        "zero-torque rollout did not move at all — model may be over-constrained"
    )


# --------------------------------------------------------------------------- #
# Gap 3 — Out-of-bounds theta. Defensive: PR #4252 may not be in yet.
# --------------------------------------------------------------------------- #


def test_simulate_contract_mujoco_out_of_bounds_theta_rejected_or_handled() -> None:
    """A 1e9 coefficient is either rejected (ValueError) or clamped/handled."""
    nu = _upper_body_nu()
    theta = np.zeros(nu * COEFFS_PER_JOINT)
    theta[0] = 1.0e9

    if _has_canonical_theta_validator():
        with pytest.raises(ValueError):
            simulate_with_coefficients(theta, options=_short_opts())
        return

    # Validator not landed: accept either ValueError or a finite/failed sim.
    try:
        out = simulate_with_coefficients(theta, options=_short_opts())
    except ValueError:
        return
    assert out.solver_status in {"success", "warning", "failed"}


# --------------------------------------------------------------------------- #
# Gap 4 — Wrong-length theta.
# --------------------------------------------------------------------------- #


def test_simulate_contract_mujoco_wrong_length_theta_raises() -> None:
    """A theta with size not a multiple of 7 raises ValueError."""
    bad = np.zeros(13)  # 13 % 7 != 0
    with pytest.raises(ValueError):
        simulate_with_coefficients(bad, options=_short_opts())


def test_mismatched_n_joints_theta_raises() -> None:
    """A theta whose joint count != model.nu raises ValueError."""
    nu = _upper_body_nu()
    # nu+1 joints worth of coeffs -- correct size mod 7 but wrong shape.
    bad = np.zeros((nu + 1) * COEFFS_PER_JOINT)
    with pytest.raises(ValueError):
        simulate_with_coefficients(bad, options=_short_opts())


# --------------------------------------------------------------------------- #
# Gap 5 — NaN / Inf theta.
# --------------------------------------------------------------------------- #


def test_simulate_contract_mujoco_nan_theta_raises() -> None:
    nu = _upper_body_nu()
    bad = np.zeros(nu * COEFFS_PER_JOINT)
    bad[2] = np.nan
    with pytest.raises(ValueError):
        simulate_with_coefficients(bad, options=_short_opts())


def test_simulate_contract_mujoco_inf_theta_raises() -> None:
    nu = _upper_body_nu()
    bad = np.zeros(nu * COEFFS_PER_JOINT)
    bad[5] = np.inf
    with pytest.raises(ValueError):
        simulate_with_coefficients(bad, options=_short_opts())


# --------------------------------------------------------------------------- #
# Gap 6 — Time monotonicity, time[0] == 0.
# --------------------------------------------------------------------------- #


def test_simulate_contract_mujoco_time_monotonic_starts_at_zero() -> None:
    nu = _upper_body_nu()
    theta = np.zeros(nu * COEFFS_PER_JOINT)
    out = simulate_with_coefficients(theta, options=_short_opts())
    assert out.time[0] == 0.0
    assert np.all(np.diff(out.time) > 0)


# --------------------------------------------------------------------------- #
# Gap 7 — q width matches n_joints (model.nq for MuJoCo).
# --------------------------------------------------------------------------- #


def test_q_width_matches_model_nq() -> None:
    """SimOut.q has the same column count as model.nq at every timestep."""
    import mujoco
    from src.engines.physics_engines.mujoco._golf_swing_upper_body_xml import (
        UPPER_BODY_GOLF_SWING_XML,
    )

    model = mujoco.MjModel.from_xml_string(UPPER_BODY_GOLF_SWING_XML)
    nq, nv, nu = int(model.nq), int(model.nv), int(model.nu)

    theta = np.zeros(nu * COEFFS_PER_JOINT)
    out = simulate_with_coefficients(theta, options=_short_opts())

    assert out.q.shape[1] == nq
    assert out.qd.shape[1] == nv
    assert out.tau.shape[1] == nu
    # Every timestep keeps the canonical width (no ragged frames).
    assert all(row.size == nq for row in out.q)
