"""Regression tests for MyoSuite live-kinematics pose application.

MyoSuite is backed by MuJoCo, so the qpos layout, joint-address
introspection, and simulation calls are identical to the MuJoCo service.
These tests mirror the MuJoCo service tests, exercising
:class:`MyosuiteKinematicsService` with the same fake-mujoco stubs.
"""

from __future__ import annotations

import importlib
import sys
import types
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _install_reference_pose_stub(monkeypatch: pytest.MonkeyPatch) -> str:
    canonical_joint = "trail_elbow"

    motion_matching_package = types.ModuleType("src.shared.python.motion_matching")
    motion_matching_package.__path__ = []
    diagnostics_package = types.ModuleType(
        "src.shared.python.motion_matching.diagnostics"
    )
    diagnostics_package.__path__ = []
    reference_pose_module = types.ModuleType(
        "src.shared.python.motion_matching.diagnostics.reference_pose"
    )
    forward_kinematics_module = types.ModuleType(
        "src.shared.python.motion_matching.diagnostics.forward_kinematics"
    )
    reference_pose_module.REFERENCE_GOLFER_FIELDS = (canonical_joint,)
    reference_pose_module.reference_golfer_setup = lambda: {canonical_joint: 0.0}
    forward_kinematics_module.forward_kinematics = lambda angles: {}

    monkeypatch.setitem(
        sys.modules,
        "src.shared.python.motion_matching",
        motion_matching_package,
    )
    monkeypatch.setitem(
        sys.modules,
        "src.shared.python.motion_matching.diagnostics",
        diagnostics_package,
    )
    monkeypatch.setitem(
        sys.modules,
        "src.shared.python.motion_matching.diagnostics.reference_pose",
        reference_pose_module,
    )
    monkeypatch.setitem(
        sys.modules,
        "src.shared.python.motion_matching.diagnostics.forward_kinematics",
        forward_kinematics_module,
    )
    return canonical_joint


def _import_service_module(
    monkeypatch: pytest.MonkeyPatch,
    fake_mujoco: types.ModuleType,
):
    canonical_joint = _install_reference_pose_stub(monkeypatch)
    monkeypatch.setitem(sys.modules, "mujoco", fake_mujoco)
    monkeypatch.setitem(sys.modules, "myosuite", types.ModuleType("myosuite"))

    for module_name in (
        "src.shared.python.pose_interchange.adapters._base",
        "src.shared.python.pose_interchange.canonical",
        "src.shared.python.pose_interchange.services.myosuite",
    ):
        sys.modules.pop(module_name, None)

    module = importlib.import_module(
        "src.shared.python.pose_interchange.services.myosuite"
    )
    return module, canonical_joint


def _build_fake_mujoco(joint_names: list[str]) -> types.ModuleType:
    module = types.ModuleType("mujoco")
    module.mjtObj = SimpleNamespace(mjOBJ_JOINT=1)
    module.mjtJoint = SimpleNamespace(mjJNT_FREE=0, mjJNT_HINGE=3)
    module.forward_calls: list[np.ndarray] = []

    def mj_id2name(model, object_type, joint_index):
        if object_type != module.mjtObj.mjOBJ_JOINT:
            return None
        return joint_names[joint_index]

    def mj_forward(model, data) -> None:
        module.forward_calls.append(np.asarray(data.qpos, dtype=float).copy())

    module.mj_id2name = mj_id2name
    module.mj_forward = mj_forward
    return module


def test_set_pose_uses_joint_addresses_for_fixed_base_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mujoco = _build_fake_mujoco(["myo_trail_elbow"])
    service_module, canonical_joint = _import_service_module(
        monkeypatch,
        fake_mujoco,
    )

    pose = service_module.CanonicalPose(
        pelvis_translation_m=np.array([1.0, 2.0, 3.0]),
        pelvis_rotation_xyz_deg=np.array([15.0, 25.0, 35.0]),
        joint_angles_deg={canonical_joint: 90.0},
    )
    model = SimpleNamespace(
        nq=1,
        njnt=1,
        jnt_qposadr=np.array([0]),
        jnt_type=np.array([fake_mujoco.mjtJoint.mjJNT_HINGE]),
    )
    data = SimpleNamespace(qpos=np.full(1, np.nan))

    service = service_module.MyosuiteKinematicsService()
    service._model = model
    service._data = data

    service.set_pose(pose)

    np.testing.assert_allclose(data.qpos, [np.pi / 2])
    assert len(fake_mujoco.forward_calls) == 1


def test_set_pose_respects_nonzero_free_joint_address(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_mujoco = _build_fake_mujoco(["myo_trail_elbow", "root_free"])
    service_module, canonical_joint = _import_service_module(
        monkeypatch,
        fake_mujoco,
    )

    pose = service_module.CanonicalPose(
        pelvis_translation_m=np.array([1.0, 2.0, 3.0]),
        pelvis_rotation_xyz_deg=np.array([90.0, 0.0, 0.0]),
        joint_angles_deg={canonical_joint: 45.0},
    )
    model = SimpleNamespace(
        nq=8,
        njnt=2,
        jnt_qposadr=np.array([0, 1]),
        jnt_type=np.array(
            [
                fake_mujoco.mjtJoint.mjJNT_HINGE,
                fake_mujoco.mjtJoint.mjJNT_FREE,
            ]
        ),
    )
    data = SimpleNamespace(qpos=np.full(8, np.nan))

    service = service_module.MyosuiteKinematicsService()
    service._model = model
    service._data = data

    service.set_pose(pose)

    np.testing.assert_allclose(data.qpos[0], np.pi / 4)
    np.testing.assert_allclose(data.qpos[1:4], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(
        data.qpos[4:8],
        [np.sqrt(0.5), np.sqrt(0.5), 0.0, 0.0],
    )
    assert len(fake_mujoco.forward_calls) == 1
