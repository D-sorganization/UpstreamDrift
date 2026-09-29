"""Tests for ground-contact models and phase inference."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.motion_pipeline.contracts import (
    JointDef,
    JointStateFrame,
    JointTrajectory,
    SkeletonRig,
)
from src.shared.python.motion_pipeline.matching.contact import (
    FlatGroundContact,
    NoContactModel,
    infer_contact_phases,
)


def _foot_rig() -> SkeletonRig:
    return SkeletonRig(
        id="foot_rig",
        joints={
            "root": JointDef(name="root", parent=None, axes=["X"]),
            "foot": JointDef(
                name="foot", parent="root", tpose_offset=[0.0, 0.0, 0.5], axes=["X"]
            ),
        },
        root_joint="root",
    )


def test_flat_ground_force_at_penetration():
    cm = FlatGroundContact(points=["foot"], stiffness=1000.0, damping=10.0)
    state = {
        "positions": np.array([[0.0, 0.0, -0.01]]),  # 1 cm penetration
        "velocities": np.zeros((1, 3)),
    }
    f = cm.contact_forces(state, time=0.0)
    assert f.shape == (1, 3)
    assert f[0, 2] > 0
    assert f[0, 0] == 0 and f[0, 1] == 0


def test_flat_ground_no_force_above_ground():
    cm = FlatGroundContact(points=["foot"])
    state = {
        "positions": np.array([[0.0, 0.0, 1.0]]),
        "velocities": np.zeros((1, 3)),
    }
    f = cm.contact_forces(state, time=0.0)
    np.testing.assert_allclose(f, np.zeros((1, 3)))


def test_flat_ground_friction_opposes_motion():
    cm = FlatGroundContact(points=["foot"], stiffness=1000.0, damping=0.0, friction=0.5)
    state = {
        "positions": np.array([[0.0, 0.0, -0.01]]),
        "velocities": np.array([[1.0, 0.0, 0.0]]),
    }
    f = cm.contact_forces(state, time=0.0)
    assert f[0, 0] < 0  # friction opposes +x motion
    assert f[0, 2] > 0


def test_flat_ground_validates_points():
    cm = FlatGroundContact(points=["foot"])
    rig = _foot_rig()
    pts = cm.contact_points(rig)
    assert pts == ["foot"]
    bad = FlatGroundContact(points=["nonexistent"])
    with pytest.raises(ValueError):
        bad.contact_points(rig)


def test_flat_ground_validation_negative_stiffness():
    with pytest.raises(ValueError):
        FlatGroundContact(stiffness=-1.0)


def test_no_contact_model_returns_zero():
    cm = NoContactModel(points=["foot"])
    state = {"positions": np.array([[0.0, 0.0, -10.0]])}
    f = cm.contact_forces(state, time=0.0)
    np.testing.assert_allclose(f, np.zeros((1, 3)))


def test_infer_contact_phases_synthetic():
    rig = _foot_rig()
    # Foot height oscillates: low (stance), high (swing), low (stance)
    times_q = [
        (0.00, [0.0, 0.0]),
        (0.05, [0.0, 0.0]),
        (0.10, [0.0, 0.0]),
        (0.15, [0.0, 0.5]),
        (0.20, [0.0, 0.5]),
        (0.25, [0.0, 0.0]),
        (0.30, [0.0, 0.0]),
    ]
    frames = [
        JointStateFrame(timestamp=t, q=q, frame_index=i)
        for i, (t, q) in enumerate(times_q)
    ]
    traj = JointTrajectory(id="t", skeleton=rig, frames=frames)
    # foot's q index is 1 (root has 1 axis, foot starts at index 1)
    phases = infer_contact_phases(
        traj,
        contact_points=["foot"],
        rig=rig,
        height_threshold=0.05,
        min_phase_duration=0.0,
    )
    assert len(phases) >= 1
    # First phase should start at t=0
    assert phases[0][0] == pytest.approx(0.0)


def test_infer_contact_phases_empty_when_no_points():
    rig = _foot_rig()
    frames = [JointStateFrame(timestamp=0.0, q=[0.0, 0.0])]
    traj = JointTrajectory(id="t", skeleton=rig, frames=frames)
    assert infer_contact_phases(traj, contact_points=[], rig=rig) == []
