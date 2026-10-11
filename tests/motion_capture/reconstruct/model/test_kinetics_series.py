"""Tests for kinetics_to_force_series (FTO-25, #11310)."""

from __future__ import annotations

import numpy as np
import pytest

from src.motion_capture.reconstruct.model.kinematics import (
    ArticulatedModel,
    Joint,
    ModelSpec,
)
from src.motion_capture.reconstruct.model.kinetics_series import (
    kinetics_to_force_series,
)
from src.shared.python.force_overlay.contracts import WrenchKind

pytestmark = [pytest.mark.unit]


def test_kinetics_to_force_series_two_joint_model() -> None:
    """Test kinetics_to_force_series with a two-joint model.

    tau=[2, 3] gives moments 2*axis0 and 3*axis1 at joint centres;
    ADR-0041 y-up frame is directly labeled adr0041_world;
    joints without axes are reported as unavailable.
    """
    spec = ModelSpec(
        name="two_joint_test",
        joints=(
            Joint("joint0", None, axes="x"),
            Joint("joint1", "joint0", (0.0, 1.0, 0.0), "len1", "y"),
            Joint("tip", "joint1", (0.0, 1.0, 0.0), "len2", ""),
        ),
        lengths_m={"len1": 1.0, "len2": 0.5},
    )
    model = ArticulatedModel(spec)

    kinetics = {"tau": [2.0, 3.0], "fps": 30.0}
    series = kinetics_to_force_series(kinetics, model)

    assert len(series) == 1
    frame = series[0]
    assert frame.world_frame == "adr0041_world"

    wrenches_by_body = {w.body: w for w in frame.wrenches}
    assert "joint0" in wrenches_by_body
    assert "joint1" in wrenches_by_body
    assert "tip" not in wrenches_by_body

    w0 = wrenches_by_body["joint0"]
    assert w0.kind == WrenchKind.JOINT_ACTUATOR
    np.testing.assert_allclose(w0.point_m, (0.0, 0.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(w0.torque_nm, (2.0, 0.0, 0.0), atol=1e-6)

    w1 = wrenches_by_body["joint1"]
    assert w1.kind == WrenchKind.JOINT_ACTUATOR
    np.testing.assert_allclose(w1.point_m, (0.0, 1.0, 0.0), atol=1e-6)
    np.testing.assert_allclose(w1.torque_nm, (0.0, 3.0, 0.0), atol=1e-6)

    # Joint without axes is unavailable
    unavailable = frame.metadata.get("unavailable_labels", ())
    assert any("tip" in label for label in unavailable)


def test_kinetics_to_force_series_multi_frame_trajectory() -> None:
    """Test kinetics_to_force_series with multiple frames of torques."""
    spec = ModelSpec(
        name="hinge_test",
        joints=(Joint("base", None, axes="z"),),
    )
    model = ArticulatedModel(spec)

    tau = np.array([[1.0], [2.5], [4.0]])
    kinetics = {"tau": tau, "fps": 20.0}
    series = kinetics_to_force_series(kinetics, model)

    assert len(series) == 3
    assert series.times_s == pytest.approx((0.0, 0.05, 0.10))
    for f, t_val in zip(series, (1.0, 2.5, 4.0), strict=True):
        assert f.world_frame == "adr0041_world"
        assert len(f.wrenches) == 1
        np.testing.assert_allclose(
            f.wrenches[0].torque_nm, (0.0, 0.0, t_val), atol=1e-6
        )
