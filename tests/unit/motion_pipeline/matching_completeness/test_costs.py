"""Tests for motion-matching cost components."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointStateFrame,
    JointTrajectory,
    Marker,
    MarkerFrame,
    MarkerTrajectory,
    SkeletonRig,
    TorqueFrame,
    TorqueTrajectory,
)
from src.shared.python.motion_pipeline.matching.base import CostWeights
from src.shared.python.motion_pipeline.matching.costs import (
    composite_cost,
    effort_cost,
    joint_tracking_cost,
    marker_tracking_cost,
    residual_cost,
    smoothness_cost,
)


def _make_rig() -> SkeletonRig:
    return SkeletonRig(
        id="rig",
        joints={
            "root": JointDef(name="root", parent=None, children=["seg"], axes=["X"]),
            "seg": JointDef(
                name="seg",
                parent="root",
                children=[],
                tpose_offset=[1.0, 0.0, 0.0],
                axes=["X"],
            ),
        },
        root_joint="root",
    )


def _make_traj(
    rig: SkeletonRig, qs: list[list[float]], dt: float = 0.01
) -> JointTrajectory:
    frames = [
        JointStateFrame(timestamp=i * dt, q=q, frame_index=i) for i, q in enumerate(qs)
    ]
    return JointTrajectory(id="traj", skeleton=rig, frames=frames)


def test_joint_tracking_cost_zero_for_identical():
    rig = _make_rig()
    qs = [[0.0, 0.0], [0.1, 0.2], [0.2, 0.4]]
    a = _make_traj(rig, qs)
    b = _make_traj(rig, qs)
    assert joint_tracking_cost(a, b) == pytest.approx(0.0)


def test_joint_tracking_cost_positive_on_difference():
    rig = _make_rig()
    a = _make_traj(rig, [[0.0, 0.0], [0.0, 0.0]])
    b = _make_traj(rig, [[0.1, 0.1], [0.1, 0.1]])
    assert joint_tracking_cost(a, b) == pytest.approx(0.1)


def test_joint_tracking_cost_with_per_joint_weights():
    rig = _make_rig()
    a = _make_traj(rig, [[0.0, 0.0], [0.0, 0.0]])
    b = _make_traj(rig, [[1.0, 1.0], [1.0, 1.0]])
    # weight=0 nullifies the root contribution
    weighted = joint_tracking_cost(a, b, {"root": 0.0, "seg": 1.0})
    full = joint_tracking_cost(a, b)
    assert weighted < full


def test_joint_tracking_cost_frame_mismatch_raises():
    rig = _make_rig()
    a = _make_traj(rig, [[0.0, 0.0]])
    b = _make_traj(rig, [[0.0, 0.0], [0.1, 0.1]])
    with pytest.raises(ValueError):
        joint_tracking_cost(a, b)


def test_marker_tracking_cost_zero_when_identical():
    frames = [
        MarkerFrame(
            timestamp=0.0,
            markers={"a": Marker(name="a", x=0.0, y=0.0, z=0.0)},
        )
    ]
    t1 = MarkerTrajectory(id="m1", frames=frames)
    t2 = MarkerTrajectory(id="m2", frames=frames)
    assert marker_tracking_cost(t1, t2) == pytest.approx(0.0)


def test_marker_tracking_cost_positive():
    f1 = MarkerFrame(
        timestamp=0.0, markers={"a": Marker(name="a", x=0.0, y=0.0, z=0.0)}
    )
    f2 = MarkerFrame(
        timestamp=0.0, markers={"a": Marker(name="a", x=1.0, y=0.0, z=0.0)}
    )
    t1 = MarkerTrajectory(id="m1", frames=[f1])
    t2 = MarkerTrajectory(id="m2", frames=[f2])
    assert marker_tracking_cost(t1, t2) == pytest.approx(1.0)


def test_smoothness_cost_zero_for_constant():
    rig = _make_rig()
    traj = _make_traj(rig, [[0.0, 0.0]] * 5)
    assert smoothness_cost(traj) == pytest.approx(0.0, abs=1e-9)


def test_smoothness_cost_positive_for_jerky():
    rig = _make_rig()
    traj = _make_traj(rig, [[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]])
    assert smoothness_cost(traj) > 0.0


def _make_torque_traj(taus: list[list[float]], dt: float = 0.01) -> TorqueTrajectory:
    frames = [TorqueFrame(timestamp=i * dt, tau=tau) for i, tau in enumerate(taus)]
    return TorqueTrajectory(frames=frames, rig_joint_names=["root", "seg"])


def test_effort_cost_zero_for_zero_torque():
    traj = _make_torque_traj([[0.0, 0.0], [0.0, 0.0]])
    assert effort_cost(traj) == pytest.approx(0.0)


def test_effort_cost_scales_with_torque():
    small = _make_torque_traj([[1.0, 0.0], [1.0, 0.0]])
    big = _make_torque_traj([[10.0, 0.0], [10.0, 0.0]])
    assert effort_cost(big) > effort_cost(small)


def test_residual_cost_aggregates():
    report = {"mean_residual": 0.1, "max_residual": 0.5, "std_residual": 0.05}
    val = residual_cost(report)
    assert val == pytest.approx(0.1 + 0.25 + 0.0125)


def test_composite_cost_postconditions():
    rig = _make_rig()
    a = _make_traj(rig, [[0.0, 0.0], [0.1, 0.1], [0.2, 0.2]])
    b = _make_traj(rig, [[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    weights = CostWeights()
    out = composite_cost(a, b, weights)
    assert out["total"] >= 0
    for k, v in out.items():
        assert np.isfinite(v), f"{k} not finite"
    assert {"joint_tracking", "smoothness", "effort", "residual", "total"} <= set(out)


def test_composite_cost_zero_weight_nullifies_component():
    rig = _make_rig()
    a = _make_traj(rig, [[0.0, 0.0], [0.0, 0.0]])
    b = _make_traj(rig, [[1.0, 1.0], [1.0, 1.0]])
    weights_full = CostWeights(
        joint_tracking=1.0,
        smoothness=0.0,
        effort=0.0,
        residual=0.0,
        marker_tracking=0.0,
    )
    weights_zero = CostWeights(
        joint_tracking=0.0,
        smoothness=0.0,
        effort=0.0,
        residual=0.0,
        marker_tracking=0.0,
    )
    full = composite_cost(a, b, weights_full)
    zero = composite_cost(a, b, weights_zero)
    assert full["total"] > 0
    assert zero["total"] == pytest.approx(0.0)
