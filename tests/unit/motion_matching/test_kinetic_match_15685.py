"""Kinetic match: forces and torques for a kinematic best match (#15685, K1)."""

from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.motion_matching import kinetic_match as km

pytestmark = pytest.mark.unit

ORDER = ("rx", "ry", "rz", "r1", "r2", "r3", "a", "b", "c")
NATIVE = ("rx", "ry", "rz", "r1", "r2", "r3", "c", "a", "b")


class _Adapter:
    """Nine coordinates in a native order; a 2-row closure on ``a`` and ``c``."""

    coordinate_order = NATIVE
    nv = 9
    actuated_indices = (6, 7, 8)

    def compute_mass_and_bias(self, q, v):
        return np.diag(np.arange(1.0, 10.0)), np.zeros(9)

    def compute_grip_jacobian(self, q):
        jac = np.zeros((2, 9))
        jac[0, 6], jac[0, 7] = 1.0, -1.0  # c - a
        jac[1, 3] = 1.0
        return jac

    def compute_closure_drift(self, q, v):
        return np.zeros(2)


def _close_pose(q):
    """Fake pose closer for the ``c == a`` closure (bundle columns 6, 8)."""
    closed = np.array(q, dtype=float)
    gap = abs(closed[8] - closed[6])
    closed[8] = closed[6]
    return closed, gap


def _match(bundle, settings, allocate):
    return km.kinetic_match(bundle, settings, _Adapter(), allocate, _close_pose)


def _settings(**over):
    base = km.load_settings()
    return replace(base, **over)


def _bundle(frames: int = 40, dt: float = 0.01):
    t = np.arange(frames) * dt
    q = np.zeros((frames, 9))
    q[:, 6] = np.sin(t)  # a
    q[:, 8] = 0.5 * t  # c, breaks the closure c == a
    return SimpleNamespace(
        coordinate_order=ORDER,
        dt_s=dt,
        reference_q=q,
        efforts=np.zeros((frames - 1, 9)),
        provenance={"impact_time_s": 0.2},
    )


def _fake_allocate(peak_tau: float = 10.0, grip: float = 100.0, root: float = 0.0):
    calls = {}

    def allocate(adapter, t, q, v, a, settings):
        calls.update(q=q, v=v, a=a, t=t)
        n = len(t)
        columns = len(adapter.actuated_indices)
        tau = np.tile([1.0, 2.0, peak_tau][:columns], (n, 1))
        lam = np.zeros((n, 6))
        lam[:, 0] = grip
        rootw = np.zeros((n, 6))
        rootw[n // 2 :, 0] = root
        return SimpleNamespace(
            time_s=t,
            tau_actuated=tau,
            f_ground=np.zeros((n, 3)),
            lambda_grip=lam,
            delta_tau_root=rootw,
            max_equilibrium_residual=1e-9,
            as_dict=lambda: {"success": True},
        )

    return allocate, calls


def test_settings_load_and_validate() -> None:
    settings = km.load_settings()
    assert settings.cutoff_hz > 0 and settings.stride >= 1
    with pytest.raises(ValueError, match="stride"):
        replace(settings, stride=0)
    with pytest.raises(ValueError, match="limit"):
        replace(settings, max_grip_force_n=-1.0)


def test_rates_are_reordered_and_projected_onto_the_closure() -> None:
    allocate, calls = _fake_allocate()
    _match(_bundle(), _settings(), allocate)
    jac = _Adapter().compute_grip_jacobian(None)
    assert np.abs(calls["v"] @ jac.T).max() < 1e-9
    assert np.abs(calls["a"] @ jac.T).max() < 1e-9
    # Native column 7 is bundle column 6 (``a``): reordered, not scrambled.
    np.testing.assert_allclose(calls["q"][:, 7], _bundle().reference_q[:, 6], atol=0.05)


def test_efforts_come_back_in_bundle_order_and_receipt_passes() -> None:
    allocate, _ = _fake_allocate()
    out, receipt = _match(_bundle(), _settings(), allocate)
    # native actuated (c, a, b) carry (1, 2, 10) -> bundle (a, b, c) = (2, 10, 1)
    np.testing.assert_allclose(out.efforts[:, 6:], np.tile([2.0, 10.0, 1.0], (39, 1)))
    assert not out.efforts[:, :6].any()
    assert receipt["verdict"] == "plausible"
    assert receipt["method"] == "inverse_dynamics_allocation"
    assert out.provenance["kinetic_match"] == receipt
    json.dumps(receipt)


def test_implausible_peaks_and_root_residual_are_reported_not_hidden() -> None:
    allocate, _ = _fake_allocate(peak_tau=5000.0, grip=9000.0, root=500.0)
    _, receipt = _match(_bundle(), _settings(), allocate)
    assert receipt["verdict"] == "implausible"
    reasons = " ".join(receipt["reasons"])
    assert "grip" in reasons and "torque" in reasons and "root" in reasons
    assert receipt["root_residual_window_s"][1] < 0.39
    assert receipt["peaks"]["joint_torque_coordinate"] == "b"


def test_preconditions() -> None:
    allocate, _ = _fake_allocate()
    bad = _bundle()
    bad.coordinate_order = ORDER[:-1] + ("zz",)
    with pytest.raises(ValueError, match="coordinate"):
        _match(bad, _settings(), allocate)
    with pytest.raises(ValueError, match="frames"):
        _match(_bundle(frames=3), _settings(), allocate)


def test_pose_is_closed_before_inverse_dynamics_and_gap_reported() -> None:
    """An open weld lets the grip multiplier act as a fake external moment.

    The matched pose leaves a gap at the hand weld; the loop Jacobian then has
    root columns of the gap's size and the allocator buys root moment with
    kN of grip force. The pose is projected onto the closure first and the
    largest gap closed is reported.
    """
    allocate, calls = _fake_allocate()
    _, receipt = _match(_bundle(), _settings(), allocate)
    # native (c, a) = columns 6, 7 must coincide after closing the pose
    np.testing.assert_allclose(calls["q"][:, 6], calls["q"][:, 7], atol=1e-12)
    assert receipt["closure_gap_max_m"] == pytest.approx(
        abs(0.5 * 0.39 - np.sin(0.39)), abs=0.05
    )


def test_closure_tolerance_is_a_setting() -> None:
    assert km.load_settings().closure_tolerance_m > 0
    with pytest.raises(ValueError, match="limit"):
        replace(km.load_settings(), closure_tolerance_m=0.0)


class _PassiveAdapter(_Adapter):
    """Native ``b`` (column 8) is a passive spring: no actuator, no closure."""

    actuated_indices = (6, 7)

    def compute_mass_and_bias(self, q, v):
        mass, bias = super().compute_mass_and_bias(q, v)
        bias[8] = 50.0 * (q[8] - 0.1)
        return mass, bias


def test_passive_rows_are_solved_before_allocation_and_returned() -> None:
    """Z6b (#15150): the shaft is the passive response, not a copied path."""
    allocate, calls = _fake_allocate()
    bundle = _bundle()
    bundle.reference_q[:, 7] = 0.3 * np.sin(5.0 * np.arange(40) * 0.01)
    adapter = _PassiveAdapter()
    bundle.reference_v = np.gradient(bundle.reference_q, 0.01, axis=0)
    bundle.q0, bundle.v0 = bundle.reference_q[0].copy(), bundle.reference_v[0].copy()
    out, receipt = km.kinetic_match(bundle, _settings(), adapter, allocate, _close_pose)
    q, v, a = calls["q"], calls["v"], calls["a"]
    rows = [
        (adapter.compute_mass_and_bias(q[k], v[k])[0] @ a[k])[8]
        + adapter.compute_mass_and_bias(q[k], v[k])[1][8]
        for k in range(q.shape[0])
    ]
    assert np.abs(rows).max() < 1e-12
    assert q[0, 8] == pytest.approx(0.1)  # starts at its spring rest
    np.testing.assert_array_equal(out.reference_q[:, 7], q[:, 8])
    np.testing.assert_array_equal(out.reference_v[:, 7], v[:, 8])
    # The bundle invariant: reference states start at q0, v0.
    np.testing.assert_array_equal(out.q0, out.reference_q[0])
    np.testing.assert_array_equal(out.v0, out.reference_v[0])
    assert receipt["passive_coordinates"] == 1


def test_passive_substep_is_a_setting() -> None:
    assert km.load_settings().passive_substep_s > 0
    with pytest.raises(ValueError, match="limit"):
        replace(km.load_settings(), passive_substep_s=0.0)
