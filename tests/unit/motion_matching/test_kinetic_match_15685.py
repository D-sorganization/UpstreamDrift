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
        tau = np.tile([1.0, 2.0, peak_tau], (n, 1))
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
    km.kinetic_match(_bundle(), _settings(), _Adapter(), allocate)
    jac = _Adapter().compute_grip_jacobian(None)
    assert np.abs(calls["v"] @ jac.T).max() < 1e-9
    assert np.abs(calls["a"] @ jac.T).max() < 1e-9
    # Native column 7 is bundle column 6 (``a``): reordered, not scrambled.
    np.testing.assert_allclose(calls["q"][:, 7], _bundle().reference_q[:, 6], atol=0.05)


def test_efforts_come_back_in_bundle_order_and_receipt_passes() -> None:
    allocate, _ = _fake_allocate()
    out, receipt = km.kinetic_match(_bundle(), _settings(), _Adapter(), allocate)
    # native actuated (c, a, b) carry (1, 2, 10) -> bundle (a, b, c) = (2, 10, 1)
    np.testing.assert_allclose(out.efforts[:, 6:], np.tile([2.0, 10.0, 1.0], (39, 1)))
    assert not out.efforts[:, :6].any()
    assert receipt["verdict"] == "plausible"
    assert receipt["method"] == "inverse_dynamics_allocation"
    assert out.provenance["kinetic_match"] == receipt
    json.dumps(receipt)


def test_implausible_peaks_and_root_residual_are_reported_not_hidden() -> None:
    allocate, _ = _fake_allocate(peak_tau=5000.0, grip=9000.0, root=500.0)
    _, receipt = km.kinetic_match(_bundle(), _settings(), _Adapter(), allocate)
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
        km.kinetic_match(bad, _settings(), _Adapter(), allocate)
    with pytest.raises(ValueError, match="frames"):
        km.kinetic_match(_bundle(frames=3), _settings(), _Adapter(), allocate)
