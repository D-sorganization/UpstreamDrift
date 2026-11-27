"""Opt-in knee axial (tibial) rotation coordinate, default off (#15296)."""

from __future__ import annotations

import copy
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching import knee_axial_rotation as kar
from src.shared.python.motion_matching.execution import spec_builder as sb
from src.shared.python.motion_matching.execution import subject_preset as sp
from src.shared.python.motion_matching.full_body_spec import (
    validate_full_body_structure as validate,
)
from src.shared.python.motion_matching.pipeline.lane import configure_lane
from tests.helpers.opensim_support import require_opensim

pytestmark = [pytest.mark.unit]

_MODELS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
FK_TOL = 1e-12
#: A flexed, turned leg pose (rad) used to compare engines with the spec.
_POSE = {"knee_angle_r": -0.9, "knee_rotation_r": 0.3, "hip_flexion_r": 0.5} | {
    "knee_angle_l": -0.4,
    "knee_rotation_l": -0.25,
}


@pytest.fixture(scope="module")
def base() -> dict:
    path = _MODELS / "full_body_spec_anthro_driver.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def turned(base) -> dict:
    return kar.apply_knee_axial_rotation(base, (-20.0, 20.0))


@pytest.fixture(scope="module")
def probed(turned) -> dict:
    """The opt-in document with a probe frame on each toe body."""
    doc = copy.deepcopy(turned)
    doc["frames"] += [
        {"name": f"Toe{s}", "body": f"toes_{s}", "placement": np.eye(4).tolist()}
        for s in kar.SIDES
    ]
    return doc


def _knee(document: dict, side: str) -> dict:
    return next(j for j in document["joints"] if j["name"] == f"knee_{side}")


def test_off_is_the_unchanged_document(base) -> None:
    out = kar.apply_knee_axial_rotation(base, None)
    assert out is not base and json.dumps(out) == json.dumps(base)
    assert kar.knee_axial_bounds_rad(base) == {}
    kar.require_hinge_knee(base["coordinate_order"], "any consumer")


def test_on_adds_one_bounded_coordinate_per_knee(base, turned) -> None:
    added = ["knee_rotation_r", "knee_rotation_l"]
    order = turned["coordinate_order"]
    assert [c for c in order if c not in base["coordinate_order"]] == added
    assert len(turned["joints"]) == len(base["joints"])
    for side in kar.SIDES:
        prims = [
            (p["primitive"], p["coordinate"]) for p in _knee(turned, side)["primitives"]
        ]
        assert prims == [("Rz", f"knee_angle_{side}"), ("Rx", f"knee_rotation_{side}")]
        at = order.index(f"knee_angle_{side}")
        assert order[at + 1] == f"knee_rotation_{side}"
        assert turned["coordinate_ranges_deg"][f"knee_rotation_{side}"] == [-20.0, 20.0]
    assert turned[kar.KNEE_AXIAL_KEY]["range"] == [-20.0, 20.0]
    validate(turned)


def test_builder_default_is_off_and_the_setting_turns_it_on() -> None:
    kwargs = sp.load_subject_preset("tour_average_driver").builder_kwargs()
    assert kwargs[kar.KNEE_AXIAL_KEY] is None
    default = sb.compose_anthropometric_document(**kwargs).document
    kwargs.pop(kar.KNEE_AXIAL_KEY)
    plain = sb.compose_anthropometric_document(**kwargs).document
    assert json.dumps(default) == json.dumps(plain) and kar.KNEE_AXIAL_KEY not in plain
    on = sb.compose_anthropometric_document(**kwargs, knee_axial_rotation_deg=15)
    assert on.document[kar.KNEE_AXIAL_KEY]["range"] == [-15.0, 15.0]
    assert len(on.document["coordinate_order"]) == len(plain["coordinate_order"]) + 2
    args = sb.build_parser().parse_args(
        ["--native", "n", "--osim", "o", "--native-candidate", "c", "--stature", "1.8"]
        + ["--mass", "80", "--output", "out", "--knee-axial-rotation"]
    )
    assert args.knee_axial_rotation == kar.DEFAULT_KNEE_AXIAL_RANGE_DEG[1]


def test_zero_rotation_leaves_every_body_where_it_was(base, turned) -> None:
    for pose in (
        {},
        {"knee_angle_r": -1.1, "knee_angle_l": -0.3, "hip_flexion_r": 0.6},
    ):
        before, after = spec_body_poses(base, pose), spec_body_poses(turned, pose)
        for name, expected in before.items():
            np.testing.assert_allclose(after[name], expected, atol=FK_TOL)


@pytest.mark.parametrize("side", kar.SIDES)
def test_rotation_turns_the_foot_inward_about_the_shank(turned, side) -> None:
    angle = np.radians(20.0)
    rest = spec_body_poses(turned, {})
    moved = spec_body_poses(turned, {f"knee_rotation_{side}": angle})
    joint = rest[f"tibia_{side}"] @ np.asarray(_knee(turned, side)["child_to_follower"])
    centre, axis = joint[:3, 3], joint[:3, 0]
    turn = np.eye(4)
    turn[:3, :3] = Rotation.from_rotvec(angle * axis).as_matrix()
    turn[:3, 3] = centre - turn[:3, :3] @ centre
    tibia = f"tibia_{side}"
    np.testing.assert_allclose(moved[tibia], turn @ rest[tibia], atol=FK_TOL)
    shank = rest[f"talus_{side}"][:3, 3] - centre
    cosine = abs(float(axis @ shank)) / float(np.linalg.norm(shank))
    assert np.degrees(np.arccos(cosine)) < kar.MAX_AXIS_OFFSET_DEG
    # Positive is internal rotation on both sides: the toes move medially.
    toes = moved[f"toes_{side}"][:3, 3] - rest[f"toes_{side}"][:3, 3]
    assert (1.0 if side == "r" else -1.0) * toes[1] > 0.01  # world +y is left


@pytest.mark.parametrize(
    "bad",
    [0.0, -5.0, 60.0, float("nan"), (5.0, 20.0), (-20.0, -5.0), (20.0, -20.0),
     (-50.0, 10.0), (-10.0, float("inf")), (1.0, 2.0, 3.0), "wide", True],
)  # fmt: skip
def test_invalid_ranges_raise(base, bad) -> None:
    with pytest.raises((ValueError, TypeError)):
        kar.apply_knee_axial_rotation(base, bad)


def test_contract_violations_raise(base, turned) -> None:
    with pytest.raises(ValueError, match="already"):
        kar.apply_knee_axial_rotation(turned, 20.0)
    tampered = copy.deepcopy(turned)
    tampered["coordinate_ranges_deg"]["knee_rotation_r"] = [-90.0, 90.0]
    with pytest.raises(ValueError, match="knee_rotation_r"):
        validate(tampered)
    with pytest.raises(ValueError, match="muscle graft.*knee axial"):
        kar.require_hinge_knee(turned["coordinate_order"], "muscle graft")


def test_fit_lane_bounds_the_coordinate_from_the_document(base, turned) -> None:
    off, on = SimpleNamespace(bounds={}), SimpleNamespace(bounds={})
    configure_lane(off, base)
    configure_lane(on, turned)
    assert set(on.bounds) - set(off.bounds) == set(kar.knee_axial_coordinates())
    np.testing.assert_allclose(on.bounds["knee_rotation_l"], np.radians([-20.0, 20.0]))


def test_preset_setting_is_optional_and_validated(tmp_path) -> None:
    stock = _MODELS / "evidence/subject_presets/tour_average_driver.json"
    raw = json.loads(stock.read_text(encoding="utf-8"))
    assert kar.KNEE_AXIAL_KEY not in raw["settings"]
    raw["provenance"][kar.KNEE_AXIAL_KEY] = "best estimate; not measured"
    raw["settings"][kar.KNEE_AXIAL_KEY] = 20.0
    path = tmp_path / "turned.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    assert sp.load_subject_preset(path).knee_axial_rotation_deg == 20.0
    raw["settings"][kar.KNEE_AXIAL_KEY] = 60.0
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match=kar.KNEE_AXIAL_KEY):
        sp.load_subject_preset(path)


_ENGINES = {
    "mujoco": ("mujoco.python.full_body_model", "NativeMujocoFullBodyModel"),
    "drake": ("drake.python.full_body_model", "FullBodyDrakeModel"),
    "pinocchio": ("pinocchio.python.native_model", "FullBodyPinocchioModel"),
    "opensim": ("opensim.python.full_body_parity", "OpenSimFullBodyParityAdapter"),
}


@pytest.mark.parametrize(
    "engine",
    [pytest.param(e, marks=getattr(pytest.mark, f"requires_{e}")) for e in _ENGINES],
)
def test_engine_forward_kinematics_matches_the_spec(probed, engine) -> None:
    if engine == "opensim":
        require_opensim()
    else:
        pytest.importorskip("pydrake.all" if engine == "drake" else engine)
    module, name = _ENGINES[engine]
    factory = getattr(
        importlib.import_module(f"src.engines.physics_engines.{module}"), name
    )
    as_bytes = engine in ("mujoco", "opensim")
    model = factory(json.dumps(probed).encode("utf-8") if as_bytes else probed)
    got = model.frame_poses(dict.fromkeys(probed["coordinate_order"], 0.0) | _POSE)
    poses = spec_body_poses(probed, _POSE)
    for frame in probed["frames"]:
        expected = poses[frame["body"]] @ np.asarray(frame["placement"], dtype=float)
        np.testing.assert_allclose(got[frame["name"]], expected, atol=1e-9)


def test_name_mapped_consumers_refuse_the_coordinate(turned) -> None:
    from src.engines.physics_engines.myosuite.python import retarget
    from src.engines.physics_engines.opensim.python import musculoskeletal_graft

    with pytest.raises(ValueError, match="knee axial"):
        musculoskeletal_graft.spec_document(json.dumps(turned).encode("utf-8"))
    with pytest.raises(ValueError, match="knee axial"):
        retarget.source_coordinate_index(
            turned["coordinate_order"], retarget.default_retarget_map()
        )
