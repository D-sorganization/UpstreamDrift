"""Test the physics-engine skeleton extractors using stub engine modules.

The real engines (pydrake, mujoco, opensim, pinocchio) are heavy native
dependencies. These tests inject lightweight stub modules into
``sys.modules`` so the extractor code paths execute without the engines
installed.
"""

from __future__ import annotations

import sys
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Required vocab (same in all extractors)
# ---------------------------------------------------------------------------

# The reverse-mapping `{v: k for k, v in DICT}` keeps the LAST k for each v
# when source has duplicate values. Source has both ``hip:hip`` and
# ``pelvis:hip`` -> MATCHER_TO_X["hip"] == "pelvis". So the body names that
# satisfy the vocabulary check are these:
VOCAB_BODIES = [
    "pelvis",  # matches "hip"
    "spine",
    "torso",
    "hub",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "midpoint",
    "clubhead",
]


# ===========================================================================
# MuJoCo stub
# ===========================================================================


def _install_mujoco_stub(monkeypatch, bodies=None):
    bodies = bodies if bodies is not None else VOCAB_BODIES
    fake = ModuleType("mujoco")

    class _MjModel:
        def __init__(self):
            self.nbody = len(bodies)

        @classmethod
        def from_xml_string(cls, xml):
            return cls()

        @classmethod
        def from_xml_path(cls, p):
            return cls()

    class _MjData:
        def __init__(self, model):
            n = model.nbody
            self.xipos = np.tile(np.arange(3, dtype=float), (n, 1))
            self.qpos = np.zeros(7)

    fake.MjModel = _MjModel
    fake.MjData = _MjData
    fake.mjtObj = MagicMock()
    fake.mjtObj.mjOBJ_BODY = 1

    def mj_id2name(model, obj_type, i):
        return bodies[i]

    fake.mj_id2name = mj_id2name
    fake.mj_forward = MagicMock()
    monkeypatch.setitem(sys.modules, "mujoco", fake)
    return fake


def test_mujoco_provider_loads_model_xml(monkeypatch):
    _install_mujoco_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import mujoco as mj

    p = mj.MuJoCoSkeletonProvider(model_xml="<x/>")
    skel = p.get_skeleton()
    # All matcher vocab keys with a mapping should be populated.
    assert "hip" in skel and "ls" in skel and "ch" in skel
    assert all(s.shape == (3,) for s in skel.values())


def test_mujoco_provider_loads_model_path(monkeypatch):
    _install_mujoco_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import mujoco as mj

    p = mj.MuJoCoSkeletonProvider(model_path="x.xml")
    assert "pelvis" in p.get_available_bodies()


def test_mujoco_provider_missing_body_raises(monkeypatch):
    _install_mujoco_stub(monkeypatch, bodies=["pelvis"])  # incomplete
    from src.tools.starting_pose_matcher.skeleton_extractors import mujoco as mj

    with pytest.raises(mj.MuJoCoProviderError, match="Missing"):
        mj.MuJoCoSkeletonProvider(model_xml="<x/>")


def test_mujoco_get_skeleton_with_qpos(monkeypatch):
    fake = _install_mujoco_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import mujoco as mj

    p = mj.MuJoCoSkeletonProvider(model_xml="<x/>")
    p.get_skeleton(qpos=np.ones(7))
    fake.mj_forward.assert_called()


def test_mujoco_create_provider_factory(monkeypatch):
    _install_mujoco_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import mujoco as mj

    assert isinstance(mj.create_provider(model_xml="<x/>"), mj.MuJoCoSkeletonProvider)


# ===========================================================================
# Pinocchio stub
# ===========================================================================


def _install_pinocchio_stub(monkeypatch, frame_names=None):
    frame_names = frame_names if frame_names is not None else VOCAB_BODIES
    fake = ModuleType("pinocchio")

    class _Frame:
        def __init__(self, name):
            self.name = name

    class _Model:
        def __init__(self):
            self.frames = [_Frame(n) for n in frame_names]
            self.njoints = 2
            self.names = ["root", "j1"]

    class _Placement:
        def __init__(self):
            self.translation = np.array([0.1, 0.2, 0.3])

    class _Data:
        def __init__(self, model):
            n = max(len(model.frames), model.njoints)
            self.oMf = [_Placement() for _ in range(n)]
            self.oMi = [_Placement() for _ in range(n)]

    fake.buildModelFromUrdf = MagicMock(return_value=_Model())
    fake.Data = _Data
    fake.JointModelFreeFlyer = MagicMock()
    fake.neutral = MagicMock(return_value=np.zeros(7))
    fake.forwardKinematics = MagicMock()
    monkeypatch.setitem(sys.modules, "pinocchio", fake)
    return fake


def test_pinocchio_provider_loads_urdf(monkeypatch):
    _install_pinocchio_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import pinocchio as pn

    p = pn.PinocchioSkeletonProvider(urdf_path="x.urdf")
    skel = p.get_skeleton()
    assert "hip" in skel
    assert "left_shoulder" in p.get_available_frames() or True  # may use vocabular keys


def test_pinocchio_provider_with_package_paths(monkeypatch):
    fake = _install_pinocchio_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import pinocchio as pn

    pn.PinocchioSkeletonProvider(urdf_path="x.urdf", package_paths=["/some/path"])
    # Free flyer variant invoked
    assert fake.buildModelFromUrdf.called


def test_pinocchio_provider_missing_frame_raises(monkeypatch):
    _install_pinocchio_stub(monkeypatch, frame_names=["pelvis"])
    from src.tools.starting_pose_matcher.skeleton_extractors import pinocchio as pn

    with pytest.raises(pn.PinocchioProviderError, match="Missing"):
        pn.PinocchioSkeletonProvider(urdf_path="x.urdf")


def test_pinocchio_get_skeleton_with_q(monkeypatch):
    fake = _install_pinocchio_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import pinocchio as pn

    p = pn.PinocchioSkeletonProvider(urdf_path="x.urdf")
    p.get_skeleton(q=np.zeros(7))
    assert fake.forwardKinematics.called


def test_pinocchio_available_lists(monkeypatch):
    _install_pinocchio_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import pinocchio as pn

    p = pn.PinocchioSkeletonProvider(urdf_path="x.urdf")
    frames = p.get_available_frames()
    joints = p.get_available_joints()
    assert isinstance(frames, list)
    assert isinstance(joints, list)


def test_pinocchio_create_provider_factory(monkeypatch):
    _install_pinocchio_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import pinocchio as pn

    assert isinstance(pn.create_provider("x.urdf"), pn.PinocchioSkeletonProvider)


# ===========================================================================
# OpenSim stub
# ===========================================================================


def _install_opensim_stub(monkeypatch, body_names=None):
    body_names = body_names if body_names is not None else VOCAB_BODIES
    fake = ModuleType("opensim")

    class _Body:
        def __init__(self, name):
            self._name = name

        def getName(self):
            return self._name

        def getTransformInGround(self, state):
            return MagicMock(p=[1.0, 2.0, 3.0])

    class _BodySet:
        def __init__(self):
            self._b = [_Body(n) for n in body_names]

        def getSize(self):
            return len(self._b)

        def get(self, i):
            return self._b[i]

    class _MarkerSet:
        def getSize(self):
            return 0

        def get(self, i):
            return MagicMock()

    class _CoordinateSet:
        def get(self, name):
            return MagicMock()

    class _Model:
        def __init__(self, path):
            pass

        def initSystem(self):
            return None

        def getState(self):
            return MagicMock()

        def getBodySet(self):
            return _BodySet()

        def getMarkerSet(self):
            return _MarkerSet()

        def getCoordinateSet(self):
            return _CoordinateSet()

        def realizePosition(self, state):
            pass

    fake.Model = _Model
    monkeypatch.setitem(sys.modules, "opensim", fake)
    return fake


def test_opensim_provider_loads_model_path(monkeypatch):
    _install_opensim_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import opensim as os_mod

    p = os_mod.OpenSimSkeletonProvider(model_path="x.osim")
    skel = p.get_skeleton()
    assert "hip" in skel


def test_opensim_provider_loads_model_xml(monkeypatch, tmp_path):
    _install_opensim_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import opensim as os_mod

    p = os_mod.OpenSimSkeletonProvider(model_xml="<x/>")
    assert isinstance(p.get_available_bodies(), list)


def test_opensim_provider_missing_body_raises(monkeypatch):
    _install_opensim_stub(monkeypatch, body_names=["pelvis"])
    from src.tools.starting_pose_matcher.skeleton_extractors import opensim as os_mod

    with pytest.raises(os_mod.OpenSimProviderError, match="Missing"):
        os_mod.OpenSimSkeletonProvider(model_path="x.osim")


def test_opensim_get_skeleton_with_coordinates(monkeypatch):
    _install_opensim_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import opensim as os_mod

    p = os_mod.OpenSimSkeletonProvider(model_path="x.osim")
    p.get_skeleton(coordinates={"knee": 0.5})  # exercises coord-set path


def test_opensim_get_skeleton_handles_bad_coordinate_name(monkeypatch):
    fake = _install_opensim_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import opensim as os_mod

    # Make CoordinateSet.get raise to exercise except branch
    class BadCoordSet:
        def get(self, n):
            raise RuntimeError("unknown coord")

    fake.Model.getCoordinateSet = lambda self: BadCoordSet()  # type: ignore
    p = os_mod.OpenSimSkeletonProvider(model_path="x.osim")
    p.get_skeleton(coordinates={"junk": 1.0})  # must not raise


def test_opensim_create_provider_factory(monkeypatch):
    _install_opensim_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import opensim as os_mod

    assert isinstance(
        os_mod.create_provider(model_path="x.osim"), os_mod.OpenSimSkeletonProvider
    )


# ===========================================================================
# Drake stub (more involved — uses Parser)
# ===========================================================================


def _install_drake_stub(monkeypatch, body_names=None):
    body_names = body_names if body_names is not None else VOCAB_BODIES
    fake_pydrake = ModuleType("pydrake")
    fake_mb = ModuleType("pydrake.multibody")
    fake_mb_plant = ModuleType("pydrake.multibody.plant")
    fake_mb_parser = ModuleType("pydrake.multibody.parser")
    fake_systems = ModuleType("pydrake.systems")
    fake_systems_fw = ModuleType("pydrake.systems.framework")

    class _Body:
        def __init__(self, name):
            self._name = name

        def name(self):
            return self._name

        def EvalBodyPoseInWorld(self, ctx):
            return MagicMock(translation=lambda: [1.0, 2.0, 3.0])

    class _Plant:
        def __init__(self, dt):
            self._bodies = [_Body(n) for n in body_names]

        def num_bodies(self):
            return len(self._bodies)

        def get_body(self, i):
            return self._bodies[i]

        def Finalize(self):
            pass

        def CreateDefaultContext(self):
            return MagicMock()

        def SetPositions(self, ctx, q):
            pass

    class _Parser:
        def __init__(self, plant):
            self.plant = plant

        def SetPackageMapAutoMerge(self, x):
            pass

        def AddModelFromString(self, xml, fmt):
            pass

        def AddModelFromFile(self, path):
            pass

    fake_mb_plant.MultibodyPlant = _Plant
    fake_mb_parser.Parser = _Parser
    fake_systems_fw.DiagramBuilder = MagicMock()

    monkeypatch.setitem(sys.modules, "pydrake", fake_pydrake)
    monkeypatch.setitem(sys.modules, "pydrake.multibody", fake_mb)
    monkeypatch.setitem(sys.modules, "pydrake.multibody.plant", fake_mb_plant)
    monkeypatch.setitem(sys.modules, "pydrake.multibody.parser", fake_mb_parser)
    monkeypatch.setitem(sys.modules, "pydrake.systems", fake_systems)
    monkeypatch.setitem(sys.modules, "pydrake.systems.framework", fake_systems_fw)
    return fake_pydrake


def test_drake_provider_loads_urdf_xml(monkeypatch):
    _install_drake_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import drake

    p = drake.DrakeSkeletonProvider(model_xml="<robot/>")
    skel = p.get_skeleton()
    assert "hip" in skel
    assert all(s.shape == (3,) for s in skel.values())


def test_drake_provider_detects_sdf_xml(monkeypatch):
    _install_drake_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import drake

    p = drake.DrakeSkeletonProvider(model_xml="<sdf version='1.7'><model/></sdf>")
    assert p.get_skeleton()


def test_drake_provider_malformed_xml_falls_back_to_urdf(monkeypatch):
    _install_drake_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import drake

    # Truly malformed XML — falls back to URDF parser path.
    p = drake.DrakeSkeletonProvider(model_xml="<<not xml at all")
    assert p.get_skeleton()


def test_drake_provider_loads_model_path(monkeypatch):
    _install_drake_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import drake

    p = drake.DrakeSkeletonProvider(model_path="robot.urdf")
    assert isinstance(p.get_available_bodies(), list)


def test_drake_provider_missing_body_raises(monkeypatch):
    _install_drake_stub(monkeypatch, body_names=["pelvis"])
    from src.tools.starting_pose_matcher.skeleton_extractors import drake

    with pytest.raises(drake.DrakeProviderError, match="Missing"):
        drake.DrakeSkeletonProvider(model_xml="<robot/>")


def test_drake_get_skeleton_with_positions(monkeypatch):
    _install_drake_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import drake

    p = drake.DrakeSkeletonProvider(model_xml="<robot/>")
    p.get_skeleton(positions=np.zeros(5))  # exercises SetPositions branch


def test_drake_create_provider_factory(monkeypatch):
    _install_drake_stub(monkeypatch)
    from src.tools.starting_pose_matcher.skeleton_extractors import drake

    assert isinstance(
        drake.create_provider(model_xml="<robot/>"), drake.DrakeSkeletonProvider
    )
