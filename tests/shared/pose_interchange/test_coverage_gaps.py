"""Coverage-focused tests for ``src.shared.python.pose_interchange``.

These tests fill gaps left by the existing ``tests/unit/pose_interchange``
suite. Scope:

- Adapter ``_base`` error paths (bad quat shape, zero-norm quat).
- Adapter ``_layout_from_model`` rejection paths for every engine.
- Adapter ``from_canonical`` convention-tag rejection paths.
- ``protocol.JointSlot`` validation (units, sign, indices, limits).
- ``canonical.CanonicalPose.from_dict`` / ``from_json`` error branches.
- ``pose_io`` error paths (unsupported engine, missing fields,
  type rejection, missing pinocchio archive, opensim malformed file,
  simscape malformed file).
- Live-kinematics service ``capabilities()`` declarations and
  type-checking branches for every real service class.
- ``services.simscape._matlab_engine_is_importable`` graceful
  error handling.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.motion_matching.diagnostics.reference_pose import (
    REFERENCE_GOLFER_FIELDS,
)
from src.shared.python.pose_interchange import (
    CONVENTION_TAG,
    CanonicalPose,
    CapabilityError,
    JointSlot,
    ServiceCapabilities,
    canonical_from_reference_setup,
    canonical_zero_pose,
)
from src.shared.python.pose_interchange.adapters import (
    ADAPTER_REGISTRY,
    DrakeAdapter,
    MujocoAdapter,
    OpenSimAdapter,
    PinocchioAdapter,
    SimscapeAdapter,
)
from src.shared.python.pose_interchange.adapters._base import (
    build_default_joint_layout,
    decode_joint_angles,
    encode_joint_angles,
    euler_xyz_deg_to_quat_wxyz,
    quat_wxyz_to_euler_xyz_deg,
    quat_wxyz_to_xyzw,
    quat_xyzw_to_wxyz,
)
from src.shared.python.pose_interchange.pose_io import (
    MOTION_MATCH_LANDMARKS,
    SUPPORTED_ENGINES,
    list_saved_reference_poses,
    load_initial_state,
    save_initial_state,
    save_motion_match_target,
)
from src.shared.python.pose_interchange.services import (
    KINEMATICS_SERVICE_REGISTRY,
    MockKinematicsService,
)
from src.shared.python.pose_interchange.services.drake import (
    DrakeKinematicsService,
    _drake_is_importable,
)
from src.shared.python.pose_interchange.services.mujoco import (
    MuJoCoKinematicsService,
    _canonical_pose_to_qpos,
    _mujoco_is_importable,
)
from src.shared.python.pose_interchange.services.opensim import (
    OpenSimKinematicsService,
    _opensim_is_importable,
)
from src.shared.python.pose_interchange.services.pinocchio import (
    PinocchioKinematicsService,
    _pinocchio_is_importable,
)
from src.shared.python.pose_interchange.services.simscape import (
    SimscapeKinematicsService,
    _matlab_engine_is_importable,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# adapters/_base helper error paths
# ---------------------------------------------------------------------------


class TestBaseHelpers:
    def test_quat_wxyz_to_xyzw_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            quat_wxyz_to_xyzw(np.zeros(3))

    def test_quat_xyzw_to_wxyz_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            quat_xyzw_to_wxyz(np.zeros(5))

    def test_quat_round_trip(self) -> None:
        wxyz = np.array([0.5, 0.5, 0.5, 0.5])
        np.testing.assert_allclose(quat_xyzw_to_wxyz(quat_wxyz_to_xyzw(wxyz)), wxyz)

    def test_euler_xyz_deg_to_quat_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            euler_xyz_deg_to_quat_wxyz(np.zeros(2))

    def test_euler_xyz_deg_to_quat_unit_norm(self) -> None:
        q = euler_xyz_deg_to_quat_wxyz(np.array([35.0, 12.0, -23.0]))
        np.testing.assert_allclose(np.linalg.norm(q), 1.0)

    def test_quat_wxyz_to_euler_rejects_wrong_shape(self) -> None:
        with pytest.raises(ValueError, match="shape"):
            quat_wxyz_to_euler_xyz_deg(np.zeros(2))

    def test_quat_wxyz_to_euler_rejects_zero_norm(self) -> None:
        with pytest.raises(ValueError, match="zero-norm"):
            quat_wxyz_to_euler_xyz_deg(np.zeros(4))

    def test_quat_round_trip_via_euler(self) -> None:
        angles = np.array([10.0, 20.0, -30.0])
        q = euler_xyz_deg_to_quat_wxyz(angles)
        out = quat_wxyz_to_euler_xyz_deg(q)
        np.testing.assert_allclose(out, angles, atol=1e-9)

    def test_quat_wxyz_to_euler_gimbal_branch(self) -> None:
        # y near +90 deg engages the cy ~ 0 branch.
        q = euler_xyz_deg_to_quat_wxyz(np.array([0.0, 90.0, 0.0]))
        out = quat_wxyz_to_euler_xyz_deg(q)
        # The gimbal branch returns x=0 and folds the rest into z.
        assert out[0] == pytest.approx(0.0, abs=1e-6)
        assert out[1] == pytest.approx(90.0, abs=1e-6)

    def test_build_default_layout_indexing(self) -> None:
        layout = build_default_joint_layout(
            base_offset=7, units="deg", sign=-1, name_prefix="x_", name_suffix="_y"
        )
        assert len(layout) == len(REFERENCE_GOLFER_FIELDS)
        for i, name in enumerate(REFERENCE_GOLFER_FIELDS):
            slot = layout[name]
            assert slot.start_index == 7 + i
            assert slot.units == "deg"
            assert slot.sign == -1
            assert slot.engine_name == f"x_{name}_y"

    def test_encode_decode_round_trip_with_sign_flip(self) -> None:
        layout = build_default_joint_layout(base_offset=0, units="rad", sign=-1)
        q = np.zeros(len(REFERENCE_GOLFER_FIELDS))
        angles = dict.fromkeys(REFERENCE_GOLFER_FIELDS, 30.0)
        encode_joint_angles(angles, layout, q)
        decoded = decode_joint_angles(q, layout)
        for name in REFERENCE_GOLFER_FIELDS:
            assert decoded[name] == pytest.approx(30.0)


# ---------------------------------------------------------------------------
# protocol.JointSlot validation
# ---------------------------------------------------------------------------


class TestJointSlot:
    def test_default_valid(self) -> None:
        slot = JointSlot(canonical_name="a", engine_name="b", start_index=0)
        assert slot.length == 1
        assert slot.units == "rad"
        assert slot.sign == 1

    def test_length_must_be_positive(self) -> None:
        with pytest.raises(ValueError, match="length"):
            JointSlot(canonical_name="a", engine_name="b", start_index=0, length=0)

    def test_units_must_be_known(self) -> None:
        with pytest.raises(ValueError, match="units"):
            JointSlot(canonical_name="a", engine_name="b", start_index=0, units="turns")

    def test_sign_must_be_pm_one(self) -> None:
        with pytest.raises(ValueError, match="sign"):
            JointSlot(canonical_name="a", engine_name="b", start_index=0, sign=2)

    def test_start_index_nonnegative(self) -> None:
        with pytest.raises(ValueError, match="start_index"):
            JointSlot(canonical_name="a", engine_name="b", start_index=-1)

    def test_lower_limit_cannot_be_plus_inf(self) -> None:
        with pytest.raises(ValueError, match="lower_limit"):
            JointSlot(
                canonical_name="a",
                engine_name="b",
                start_index=0,
                lower_limit=float("inf"),
            )

    def test_upper_limit_cannot_be_minus_inf(self) -> None:
        with pytest.raises(ValueError, match="upper_limit"):
            JointSlot(
                canonical_name="a",
                engine_name="b",
                start_index=0,
                upper_limit=float("-inf"),
            )

    def test_lower_must_be_le_upper(self) -> None:
        with pytest.raises(ValueError, match="lower_limit"):
            JointSlot(
                canonical_name="a",
                engine_name="b",
                start_index=0,
                lower_limit=1.0,
                upper_limit=0.0,
            )


# ---------------------------------------------------------------------------
# CanonicalPose extra error paths
# ---------------------------------------------------------------------------


class TestCanonicalPoseExtras:
    def test_from_dict_rejects_non_mapping(self) -> None:
        with pytest.raises(TypeError, match="Mapping"):
            CanonicalPose.from_dict("not a dict")  # type: ignore[arg-type]

    def test_from_dict_rejects_missing_key(self) -> None:
        with pytest.raises(ValueError, match="missing required key"):
            CanonicalPose.from_dict({"pelvis_translation_m": [0, 0, 0]})

    def test_to_dict_round_trip_via_from_dict(self) -> None:
        pose = canonical_from_reference_setup()
        clone = CanonicalPose.from_dict(pose.to_dict())
        np.testing.assert_allclose(
            clone.pelvis_translation_m, pose.pelvis_translation_m
        )
        assert clone.angles_full_dict_deg() == pose.angles_full_dict_deg()


# ---------------------------------------------------------------------------
# Adapter error paths
# ---------------------------------------------------------------------------


_ADAPTERS = [
    ("drake", DrakeAdapter),
    ("mujoco", MujocoAdapter),
    ("opensim", OpenSimAdapter),
    ("pinocchio", PinocchioAdapter),
    ("simscape", SimscapeAdapter),
]


class TestAdapters:
    def test_registry_keys_match_engine_names(self) -> None:
        for name, cls in _ADAPTERS:
            assert name in ADAPTER_REGISTRY
            assert ADAPTER_REGISTRY[name] is cls

    @pytest.mark.parametrize(("name", "cls"), _ADAPTERS)
    def test_rejects_wrong_convention_tag(self, name: str, cls: type) -> None:
        # Build a pose, then mutate convention_tag via object.__setattr__.
        pose = canonical_zero_pose()
        # CanonicalPose is frozen, so use a fresh dataclass-like with bad tag.
        # We bypass __post_init__ tag check by constructing then patching.
        object.__setattr__(pose, "convention_tag", "bogus-tag")
        adapter = cls()
        with pytest.raises(ValueError, match="convention"):
            adapter.from_canonical(pose)

    @pytest.mark.parametrize(("name", "cls"), _ADAPTERS)
    def test_to_canonical_rejects_short_q(self, name: str, cls: type) -> None:
        adapter = cls()
        with pytest.raises(ValueError, match="expected 1-D q"):
            adapter.to_canonical(np.zeros(2))

    @pytest.mark.parametrize(("name", "cls"), _ADAPTERS)
    def test_to_canonical_rejects_2d(self, name: str, cls: type) -> None:
        adapter = cls()
        with pytest.raises(ValueError, match="expected 1-D q"):
            adapter.to_canonical(np.zeros((2, 8)))

    @pytest.mark.parametrize(("name", "cls"), _ADAPTERS)
    def test_layout_rejects_bad_model(self, name: str, cls: type) -> None:
        adapter = cls()
        with pytest.raises(TypeError, match="model"):
            adapter.joint_layout(model=object())

    @pytest.mark.parametrize(("name", "cls"), _ADAPTERS)
    def test_layout_accepts_mapping_with_layout_key(self, name: str, cls: type) -> None:
        adapter = cls()
        slot = JointSlot(
            canonical_name="LEStartPosition", engine_name="x", start_index=10
        )
        model = {"joint_layout": {"LEStartPosition": slot}}
        layout = adapter.joint_layout(model=model)
        assert "LEStartPosition" in layout

    @pytest.mark.parametrize(("name", "cls"), _ADAPTERS)
    def test_layout_accepts_object_with_layout_attr(self, name: str, cls: type) -> None:
        adapter = cls()
        slot = JointSlot(
            canonical_name="LEStartPosition", engine_name="x", start_index=10
        )
        model = SimpleNamespace(joint_layout={"LEStartPosition": slot})
        layout = adapter.joint_layout(model=model)
        assert "LEStartPosition" in layout


# ---------------------------------------------------------------------------
# pose_io error paths
# ---------------------------------------------------------------------------


class TestPoseIO:
    def test_save_initial_state_rejects_non_pose(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="CanonicalPose"):
            save_initial_state(  # type: ignore[arg-type]
                "not a pose", "drake", tmp_path / "x.pkl"
            )

    def test_save_initial_state_rejects_bad_tag(self, tmp_path: Path) -> None:
        pose = canonical_zero_pose()
        object.__setattr__(pose, "convention_tag", "old-tag")
        with pytest.raises(ValueError, match="convention_tag"):
            save_initial_state(pose, "drake", tmp_path / "x.pkl")

    def test_save_initial_state_rejects_unknown_engine(self, tmp_path: Path) -> None:
        pose = canonical_zero_pose()
        with pytest.raises(ValueError, match="not supported"):
            save_initial_state(pose, "unknown_engine", tmp_path / "x")

    def test_save_initial_state_rejects_non_string_engine(self, tmp_path: Path) -> None:
        pose = canonical_zero_pose()
        with pytest.raises(TypeError, match="engine must be a string"):
            save_initial_state(pose, 123, tmp_path / "x")  # type: ignore[arg-type]

    def test_load_initial_state_rejects_unknown_engine(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="not supported"):
            load_initial_state("unknown_engine", tmp_path / "x")

    def test_load_pinocchio_missing_archive_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="pinocchio archive"):
            load_initial_state("pinocchio", tmp_path / "missing")

    def test_load_drake_rejects_bad_pickle(self, tmp_path: Path) -> None:
        import pickle

        path = tmp_path / "bad.pkl"
        with path.open("wb") as fh:
            pickle.dump({"wrong": True}, fh)
        with pytest.raises(ValueError, match="missing required 'q'"):
            load_initial_state("drake", path)

    def test_load_mujoco_missing_qpos(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text('{"not_qpos": true}', encoding="utf-8")
        with pytest.raises(ValueError, match="missing required 'qpos'"):
            load_initial_state("mujoco", path)

    def test_load_opensim_missing_endheader(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.sto"
        path.write_text("no header here\ntime\tx\n0\t0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="endheader"):
            load_initial_state("opensim", path)

    def test_load_opensim_too_few_body_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.sto"
        path.write_text("endheader\n", encoding="utf-8")
        with pytest.raises(ValueError, match="missing column header"):
            load_initial_state("opensim", path)

    def test_load_opensim_bad_first_column(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.sto"
        path.write_text("endheader\nnottime\tx\n0\t0\n", encoding="utf-8")
        with pytest.raises(ValueError, match="first column must be 'time'"):
            load_initial_state("opensim", path)

    def test_load_opensim_row_width_mismatch(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.sto"
        path.write_text(
            "endheader\ntime\tx\ty\n0\t0\n",
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="row width"):
            load_initial_state("opensim", path)

    def test_load_simscape_missing_required(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(ValueError, match="missing required keys"):
            load_initial_state("simscape", path)

    def test_load_simscape_bad_jointangles_type(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.json"
        payload = {
            "Tx": 0,
            "Ty": 0,
            "Tz": 0,
            "Rx": 0,
            "Ry": 0,
            "Rz": 0,
            "Scale": 1.0,
            "jointAngles": "not a dict",
        }
        import json

        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(ValueError, match="jointAngles"):
            load_initial_state("simscape", path)

    def test_save_motion_match_target_rejects_non_pose(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="CanonicalPose"):
            save_motion_match_target("not a pose", tmp_path / "x.json")  # type: ignore[arg-type]

    def test_save_motion_match_target_creates_file(self, tmp_path: Path) -> None:
        pose = canonical_zero_pose()
        path = tmp_path / "subdir" / "target.json"
        save_motion_match_target(pose, path)
        import json

        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema"] == "body_target_json_v1"
        assert payload["marker_names"] == list(MOTION_MATCH_LANDMARKS)
        assert len(payload["time_s"]) == 2
        assert payload["impact_idx"] == 0

    def test_supported_engines_matches_dispatch(self) -> None:
        assert (
            frozenset({"drake", "mujoco", "pinocchio", "opensim", "simscape"})
            == SUPPORTED_ENGINES
        )

    def test_list_saved_reference_poses_returns_list(self) -> None:
        # The library may or may not exist; either way the function
        # returns a list of strings.
        result = list_saved_reference_poses()
        assert isinstance(result, list)
        for entry in result:
            assert isinstance(entry, str)


# ---------------------------------------------------------------------------
# Service capabilities / availability probes
# ---------------------------------------------------------------------------


class TestServiceCapabilities:
    def test_mock_service_capabilities_all_false(self) -> None:
        svc = MockKinematicsService(engine_name="drake")
        caps = svc.capabilities()
        assert caps == ServiceCapabilities(
            supports_dynamics_step=False,
            supports_collision_query=False,
            supports_realtime=False,
        )

    def test_mock_step_raises_capability_error(self) -> None:
        svc = MockKinematicsService(engine_name="drake")
        with pytest.raises(CapabilityError, match="does not support"):
            svc.step(0.001)

    def test_mock_load_rejects_non_path(self) -> None:
        svc = MockKinematicsService(engine_name="drake")
        with pytest.raises(TypeError, match="model_path"):
            svc.load("not a path")  # type: ignore[arg-type]

    def test_mock_set_pose_rejects_non_pose(self) -> None:
        svc = MockKinematicsService(engine_name="drake")
        with pytest.raises(TypeError, match="CanonicalPose"):
            svc.set_pose("not a pose")  # type: ignore[arg-type]

    def test_mock_engine_name_must_be_string(self) -> None:
        with pytest.raises(TypeError, match="engine_name"):
            MockKinematicsService(engine_name=123)  # type: ignore[arg-type]

    def test_mock_engine_name_must_be_nonempty(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            MockKinematicsService(engine_name="")

    def test_mock_get_link_transforms_without_pose_returns_dict(self) -> None:
        svc = MockKinematicsService(engine_name="mujoco")
        out = svc.get_link_transforms()
        assert isinstance(out, dict)
        assert len(out) > 0  # FK returns landmark dict

    def test_mock_reset_clears_pose(self) -> None:
        svc = MockKinematicsService(engine_name="drake")
        svc.set_pose(canonical_zero_pose())
        svc.reset()
        assert svc._pose is None

    def test_mock_load_stores_path(self, tmp_path: Path) -> None:
        svc = MockKinematicsService(engine_name="drake")
        svc.load(tmp_path / "anything.urdf")
        assert svc._model_path is not None

    @pytest.mark.parametrize(
        ("cls", "expected_engine"),
        [
            (DrakeKinematicsService, "drake"),
            (MuJoCoKinematicsService, "mujoco"),
            (OpenSimKinematicsService, "opensim"),
            (PinocchioKinematicsService, "pinocchio"),
            (SimscapeKinematicsService, "simscape"),
        ],
    )
    def test_real_service_engine_name(self, cls: type, expected_engine: str) -> None:
        assert cls.engine_name == expected_engine

    @pytest.mark.parametrize(
        "cls",
        [
            DrakeKinematicsService,
            MuJoCoKinematicsService,
            OpenSimKinematicsService,
            PinocchioKinematicsService,
            SimscapeKinematicsService,
        ],
    )
    def test_real_service_capabilities_return_descriptor(self, cls: type) -> None:
        svc = cls()
        caps = svc.capabilities()
        assert isinstance(caps, ServiceCapabilities)
        assert isinstance(caps.supports_dynamics_step, bool)

    @pytest.mark.parametrize(
        "cls",
        [
            DrakeKinematicsService,
            MuJoCoKinematicsService,
            OpenSimKinematicsService,
            PinocchioKinematicsService,
        ],
    )
    def test_real_service_load_rejects_non_path(self, cls: type) -> None:
        svc = cls()
        with pytest.raises(TypeError, match="model_path"):
            svc.load("not a path")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "cls",
        [
            DrakeKinematicsService,
            MuJoCoKinematicsService,
            OpenSimKinematicsService,
            PinocchioKinematicsService,
        ],
    )
    def test_real_service_set_pose_rejects_non_pose(self, cls: type) -> None:
        svc = cls()
        with pytest.raises(TypeError, match="CanonicalPose"):
            svc.set_pose("not a pose")  # type: ignore[arg-type]

    @pytest.mark.parametrize(
        "cls",
        [
            MuJoCoKinematicsService,
            OpenSimKinematicsService,
            PinocchioKinematicsService,
        ],
    )
    def test_real_service_step_rejects_nonpositive_dt(self, cls: type) -> None:
        svc = cls()
        with pytest.raises(ValueError, match="dt"):
            svc.step(0.0)
        with pytest.raises(ValueError, match="dt"):
            svc.step(-1.0)

    def test_simscape_step_rejects_nonpositive_dt(self) -> None:
        svc = SimscapeKinematicsService()
        with pytest.raises(ValueError, match="dt"):
            svc.step(-1.0)

    def test_simscape_reset_noop_without_engine(self) -> None:
        svc = SimscapeKinematicsService()
        # Should not raise even when no engine attribute set.
        svc.reset()
        assert svc._pose is None

    def test_simscape_set_pose_noop_without_engine(self) -> None:
        svc = SimscapeKinematicsService()
        svc.set_pose(canonical_zero_pose())
        # Stored pose even when no engine.
        assert svc._pose is not None

    def test_simscape_get_link_transforms_empty_without_engine(self) -> None:
        svc = SimscapeKinematicsService()
        assert svc.get_link_transforms() == {}

    def test_simscape_step_with_nothing_loaded_returns(self) -> None:
        svc = SimscapeKinematicsService()
        # Positive dt + no engine should be a no-op.
        svc.step(0.001)

    @pytest.mark.parametrize(
        ("cls", "op"),
        [
            (MuJoCoKinematicsService, "set_pose"),
            (MuJoCoKinematicsService, "get_link_transforms"),
            (MuJoCoKinematicsService, "step"),
            (OpenSimKinematicsService, "set_pose"),
            (OpenSimKinematicsService, "get_link_transforms"),
            (PinocchioKinematicsService, "set_pose"),
            (PinocchioKinematicsService, "get_link_transforms"),
            (PinocchioKinematicsService, "step"),
        ],
    )
    def test_real_service_unloaded_raises_runtime_error(
        self, cls: type, op: str
    ) -> None:
        svc = cls()
        with pytest.raises(RuntimeError, match="not loaded"):
            if op == "set_pose":
                svc.set_pose(canonical_zero_pose())
            elif op == "get_link_transforms":
                svc.get_link_transforms()
            elif op == "step":
                svc.step(0.001)

    def test_drake_set_pose_unloaded_raises(self) -> None:
        svc = DrakeKinematicsService()
        with pytest.raises(RuntimeError, match="not loaded"):
            svc.set_pose(canonical_zero_pose())

    def test_drake_get_link_transforms_unloaded_raises(self) -> None:
        svc = DrakeKinematicsService()
        with pytest.raises(RuntimeError, match="not loaded"):
            svc.get_link_transforms()

    def test_drake_step_unloaded_raises(self) -> None:
        svc = DrakeKinematicsService()
        with pytest.raises(RuntimeError, match="not loaded"):
            svc.step(0.001)

    def test_drake_step_rejects_nonpositive_dt(self) -> None:
        svc = DrakeKinematicsService()
        with pytest.raises(ValueError, match="dt"):
            svc.step(0.0)

    def test_pinocchio_reset_noop_without_load(self) -> None:
        svc = PinocchioKinematicsService()
        # _neutral_q is None, so reset returns early.
        svc.reset()

    def test_mujoco_reset_noop_without_load(self) -> None:
        svc = MuJoCoKinematicsService()
        svc.reset()

    def test_opensim_reset_noop_without_load(self) -> None:
        svc = OpenSimKinematicsService()
        svc.reset()

    def test_drake_reset_noop_without_load(self) -> None:
        svc = DrakeKinematicsService()
        svc.reset()


# ---------------------------------------------------------------------------
# Importable probes — verify they don't crash and return bool
# ---------------------------------------------------------------------------


class TestImportableProbes:
    def test_drake_probe_returns_bool(self) -> None:
        assert isinstance(_drake_is_importable(), bool)

    def test_mujoco_probe_returns_bool(self) -> None:
        assert isinstance(_mujoco_is_importable(), bool)

    def test_opensim_probe_returns_bool(self) -> None:
        assert isinstance(_opensim_is_importable(), bool)

    def test_pinocchio_probe_returns_bool(self) -> None:
        assert isinstance(_pinocchio_is_importable(), bool)

    def test_matlab_probe_returns_bool(self) -> None:
        assert isinstance(_matlab_engine_is_importable(), bool)

    def test_registry_factories_return_service(self) -> None:
        for name, factory in KINEMATICS_SERVICE_REGISTRY.items():
            svc = factory()
            assert hasattr(svc, "engine_name")
            assert svc.engine_name == name


# ---------------------------------------------------------------------------
# MuJoCo qpos builder — directly cover helper
# ---------------------------------------------------------------------------


class TestMujocoQposBuilder:
    def _fake_mujoco(self) -> SimpleNamespace:
        return SimpleNamespace(
            mjtJoint=SimpleNamespace(mjJNT_FREE=0, mjJNT_HINGE=3),
            mjtObj=SimpleNamespace(mjOBJ_JOINT=1),
            mj_id2name=lambda model, kind, idx: None,
        )

    def test_no_free_joint_no_pelvis_silently_skips(self) -> None:
        fake = self._fake_mujoco()
        model = SimpleNamespace(nq=5, njnt=0, jnt_qposadr=(), jnt_type=())
        pose = canonical_zero_pose()
        qpos = _canonical_pose_to_qpos(model, pose, fake)
        assert qpos.shape == (5,)
        assert np.all(qpos == 0)

    def test_free_joint_overrun_raises(self) -> None:
        fake = self._fake_mujoco()
        # Free joint at index 0 needs 7 qpos slots; nq=3 is too small.
        model = SimpleNamespace(
            nq=3,
            njnt=1,
            jnt_qposadr=(0,),
            jnt_type=(fake.mjtJoint.mjJNT_FREE,),
        )
        pose = canonical_zero_pose()
        with pytest.raises(RuntimeError, match="overruns qpos"):
            _canonical_pose_to_qpos(model, pose, fake)

    def test_free_joint_writes_translation_and_quat(self) -> None:
        fake = self._fake_mujoco()
        model = SimpleNamespace(
            nq=7,
            njnt=1,
            jnt_qposadr=(0,),
            jnt_type=(fake.mjtJoint.mjJNT_FREE,),
        )
        pose = CanonicalPose(
            pelvis_translation_m=np.array([1.0, 2.0, 3.0]),
            pelvis_rotation_xyz_deg=np.zeros(3),
        )
        qpos = _canonical_pose_to_qpos(model, pose, fake)
        np.testing.assert_allclose(qpos[:3], [1.0, 2.0, 3.0])
        # zero rotation -> identity quat [1, 0, 0, 0]
        np.testing.assert_allclose(qpos[3:7], [1.0, 0.0, 0.0, 0.0])

    def test_configured_layout_via_attr(self) -> None:
        fake = self._fake_mujoco()
        # Configured layout takes precedence over discovery.
        slot = JointSlot(
            canonical_name="LEStartPosition",
            engine_name="le",
            start_index=0,
            units="rad",
            sign=1,
        )
        model = SimpleNamespace(
            nq=2,
            njnt=0,
            jnt_qposadr=(),
            jnt_type=(),
            joint_layout={"LEStartPosition": slot},
        )
        pose = CanonicalPose(
            pelvis_translation_m=np.zeros(3),
            pelvis_rotation_xyz_deg=np.zeros(3),
            joint_angles_deg={"LEStartPosition": 90.0},
        )
        qpos = _canonical_pose_to_qpos(model, pose, fake)
        assert qpos[0] == pytest.approx(np.pi / 2)

    def test_configured_layout_via_mapping(self) -> None:
        fake = self._fake_mujoco()
        slot = JointSlot(
            canonical_name="LEStartPosition",
            engine_name="le",
            start_index=0,
            units="rad",
            sign=1,
        )
        model = {
            "nq": 2,
            "njnt": 0,
            "jnt_qposadr": (),
            "jnt_type": (),
            "joint_layout": {"LEStartPosition": slot},
        }

        # _canonical_pose_to_qpos uses attribute access for nq, so wrap it.
        class _ModelView:
            def __init__(self, payload: dict) -> None:
                self._payload = payload

            def __getattr__(self, key: str):
                return self._payload[key]

        # Use dict directly: _configured_joint_layout supports Mapping branch.
        pose = canonical_zero_pose()
        # Direct call would fail (nq via attr); use the view to exercise the
        # Mapping branch on the joint_layout helper alone:
        from src.shared.python.pose_interchange.services.mujoco import (
            _configured_joint_layout,
        )

        assert _configured_joint_layout(model) == {"LEStartPosition": slot}
