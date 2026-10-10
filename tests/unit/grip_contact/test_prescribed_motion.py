"""Same-input prescribed motion for the grip parity runs (issue #11739)."""

from __future__ import annotations

from importlib import import_module

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.grip_contact.prescribed_motion import (
    CoordinateSpline,
    RigidBodyState,
    fmm_spline_coefficients,
)

pytestmark = pytest.mark.unit


def test_spline_interpolates_and_reproduces_a_cubic_exactly() -> None:
    t = np.linspace(0.0, 1.0, 9)
    q = np.stack([1 + 2 * t - t**2 + 0.5 * t**3, np.sin(t)], axis=1)
    spline = CoordinateSpline(t, q)
    for k, tk in enumerate(t):
        np.testing.assert_allclose(spline.evaluate(tk)[0], q[k], atol=1e-14)
    value, rate = spline.evaluate(0.37)
    assert value[0] == pytest.approx(1 + 0.74 - 0.37**2 + 0.5 * 0.37**3, abs=1e-12)
    assert rate[0] == pytest.approx(2 - 0.74 + 1.5 * 0.37**2, abs=1e-12)


def test_spline_preconditions() -> None:
    with pytest.raises(ValueError, match="increasing"):
        fmm_spline_coefficients(np.array([0.0, 1.0, 1.0, 2.0]), np.zeros(4))
    with pytest.raises(ValueError, match="finite"):
        fmm_spline_coefficients(np.arange(4.0), np.array([0, np.nan, 0, 0]))
    with pytest.raises(ValueError, match="finite"):
        CoordinateSpline(np.arange(5.0), np.zeros((5, 2))).evaluate(np.nan)


def test_spline_matches_opensim_simm_spline() -> None:
    try:
        osim = import_module("opensim")
    except ImportError:
        pytest.skip("opensim is not installed")
    rng = np.random.default_rng(4)
    t = np.arange(40) * 0.002
    y = np.cumsum(rng.normal(size=40))
    reference = osim.SimmSpline()
    for tk, yk in zip(t, y, strict=True):
        reference.addPoint(float(tk), float(yk))
    spline = CoordinateSpline(t, y[:, None])
    for tq in np.linspace(0.0, t[-1], 97):
        arg = osim.Vector(1, float(tq))
        value, rate = spline.evaluate(float(tq))
        assert value[0] == pytest.approx(reference.calcValue(arg), abs=1e-11)
        deriv = reference.calcDerivative(osim.StdVectorInt([0]), arg)
        assert rate[0] == pytest.approx(deriv, abs=1e-8)


def test_offset_frame_state_is_rigid_body_transport() -> None:
    rot = Rotation.from_rotvec([0.2, 0.1, -0.3]).as_matrix()
    body = RigidBodyState(
        rot, np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 2.0])
    )
    offset = np.eye(4)
    offset[:3, :3] = Rotation.from_rotvec([0.0, 0.5, 0.0]).as_matrix()
    offset[:3, 3] = [0.1, 0.0, 0.0]
    frame = body.frame(offset)
    arm = rot @ offset[:3, 3]
    np.testing.assert_allclose(frame.position_m, body.position_m + arm)
    np.testing.assert_allclose(
        frame.velocity_m_s, body.velocity_m_s + np.cross(body.omega_rad_s, arm)
    )
    np.testing.assert_allclose(frame.rotation, rot @ offset[:3, :3])
    with pytest.raises(ValueError, match="4x4"):
        body.frame(np.eye(3))
