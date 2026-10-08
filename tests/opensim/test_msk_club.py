"""OSV-9 (#11756): the improved club in the Rajagopal musculoskeletal models.

Every Rajagopal-based OpenSim golf model carries the shared club (meshes, face
roll, mass and inertia) held by both hands, square at the captured address
pose. Pure-XML checks run everywhere; the OpenSim checks skip without it.
"""

from __future__ import annotations

import importlib.util
import json
import math
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest
from defusedxml import ElementTree as SafeET

from src.engines.physics_engines.opensim.python import club_visuals
from src.engines.physics_engines.opensim.python import msk_club as mc
from src.engines.physics_engines.opensim.python.full_body_osim import (
    _aggregate_body_inertia,
)
from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance.mesh_io import read_stl_bytes
from src.shared.python.motion_matching.club_models import DRIVER, club_solids

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[2]
MODELS = {
    name: mc.MODELS_DIR / f"{name}.osim"
    for name in ("golf_humanoid", "golf_humanoid_scaled")
}
BUILDER = ROOT / "scripts" / "build_humanoid_osim.py"
SQUARE_TOL_DEG = 2.0  # OSV-8 face-square tolerance
CLOSURE_TOL_M = 1e-6
GRIP_GAP_TOL_M = 0.002  # anatomical grip point to the shaft grip point
RAJAGOPAL_BODIES = 22


def _model(name: str):  # noqa: ANN202
    model = SafeET.parse(str(MODELS[name])).getroot().find("Model")
    assert model is not None
    return model


def _named(model, tag: str, name: str):  # noqa: ANN001, ANN202
    return [e for e in model.iter(tag) if e.get("name") == name]


def _club_body(model):  # noqa: ANN001, ANN202
    bodies = [b for b in model.find("BodySet/objects") if b.get("name") == "Club"]
    assert len(bodies) == 1
    return bodies[0]


def _builder() -> ModuleType:
    spec = importlib.util.spec_from_file_location("build_humanoid_osim", BUILDER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# ------------------------------------------------------------------ pure XML
@pytest.mark.parametrize("name", sorted(MODELS))
def test_club_meshes_are_the_shared_assembly_files_on_disk(name: str) -> None:
    meshes = _club_body(_model(name)).findall("attached_geometry/Mesh")
    files = sorted(m.findtext("mesh_file") for m in meshes)
    expected = sorted(p.name for p in club_visuals.asset_paths("driver").values())
    assert files == expected
    for path in club_visuals.asset_paths("driver").values():
        assert path.is_file()
    # no offset frame: the Club body frame is the shared club-body frame
    assert {m.findtext("socket_frame") for m in meshes} == {".."}


def test_committed_head_mesh_is_the_rolled_shared_head() -> None:
    club = mc.load_msk_club("driver")
    assert club.assembly.face_roll_deg == ca.ADDRESS_SQUARE_FACE_ROLL_DEG["driver"]
    stl = read_stl_bytes(club_visuals.asset_paths("driver")["head"].read_bytes())
    head = ca.assembly_meshes(club.assembly)["head"]
    triangles = head.vertices[head.faces].reshape(-1, 3)
    np.testing.assert_allclose(stl.vertices, triangles, atol=1e-6)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_club_mass_and_inertia_equal_the_shared_spec(name: str) -> None:
    body = _club_body(_model(name))
    mass = float(body.findtext("mass"))
    com = np.array(body.findtext("mass_center").split(), float)
    ixx, iyy, izz, ixy, ixz, iyz = map(float, body.findtext("inertia").split())
    inertia = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    # independent composition of the ClubSpec solids (head, shaft, grip)
    ref_mass, ref_com, ref_inertia = _aggregate_body_inertia(
        {"solids": club_solids("club", DRIVER)}
    )
    assert mass == pytest.approx(DRIVER.total_mass_kg, abs=1e-12)
    assert mass == pytest.approx(ref_mass, abs=1e-12)
    np.testing.assert_allclose(com, ref_com, atol=1e-12)
    np.testing.assert_allclose(inertia, ref_inertia, atol=1e-12)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_both_hands_hold_the_club_lead_joint_trail_constraint(name: str) -> None:
    model = _model(name)
    (joint,) = _named(model, "WeldJoint", mc.LEAD_JOINT)
    assert joint.findtext("socket_parent_frame") == "/bodyset/hand_l/hand_l_grip_offset"
    assert joint.findtext("socket_child_frame") == "/bodyset/Club/club_grip_offset"
    (weld,) = _named(model, "WeldConstraint", mc.TRAIL_CONSTRAINT)
    assert weld.findtext("socket_frame1") == "/bodyset/hand_r/hand_r_grip_offset"
    assert weld.findtext("socket_frame2") == "/bodyset/Club/club_trail_grip_offset"
    club_joints = [
        j
        for j in model.find("JointSet/objects")
        if "Club" in (j.findtext("socket_child_frame") or "")
    ]
    assert club_joints == [joint]  # no one-hand weld left over


@pytest.mark.parametrize("name", sorted(MODELS))
def test_body_and_muscle_counts_are_unchanged(name: str) -> None:
    model = _model(name)
    bodies = [b.get("name") for b in model.find("BodySet/objects")]
    assert len(bodies) == RAJAGOPAL_BODIES + 1 and bodies.count("Club") == 1
    forces = list(model.find("ForceSet/objects"))
    assert not [f for f in forces if "Muscle" in f.tag]  # muscle-stripped base
    coords = mc.coordinate_names(model)
    actuators = [f for f in forces if f.tag == "CoordinateActuator"]
    assert len(actuators) == len(coords) == 39


def test_grip_points_are_the_shared_grip_on_the_shaft_axis() -> None:
    club = mc.load_msk_club("driver")
    for side in mc.HAND_BODIES:
        shared = club.interface.frame(side).position_m
        point = club.grip_points[side]
        assert point[1] == pytest.approx(shared[1])  # same place along the shaft
        assert point[0] == 0.0 and point[2] == club.assembly.axis_offset_m
    lead, trail = club.grip_points["L"][1], club.grip_points["R"][1]
    assert -DRIVER.length_m < lead < trail < -(DRIVER.length_m - DRIVER.grip_length_m)


def test_attach_club_contracts() -> None:
    model = _model("golf_humanoid")
    club = mc.load_msk_club("driver")
    calibration = mc.load_calibration("golf_humanoid", "driver")
    with pytest.raises(ValueError, match="grip_model"):
        mc.attach_club(model, club, calibration, grip_model="glue")
    iron = mc.GripCalibration(
        "golf_humanoid",
        "iron7",
        calibration.hand_frames,
        calibration.address_q,
        calibration.club_in_ground,
        {},
    )
    with pytest.raises(ValueError, match="calibration is for"):
        mc.attach_club(model, club, iron)
    with pytest.raises(KeyError, match="no grip calibration"):
        mc.load_calibration("no_such_model")
    with pytest.raises(ValueError, match="club must be"):
        mc.spec_path("putter")


def test_calibration_closes_both_hands_on_the_grip() -> None:
    for name in MODELS:
        report = mc.load_calibration(name).report
        for side, gap in report["hand_grip_point_gap_m"].items():
            assert gap <= GRIP_GAP_TOL_M, (name, side, gap)


def _build_into(tmp_path: Path, grip_model: str) -> Path:
    builder = _builder()
    if not builder.BASE_OSIM.is_file():
        pytest.skip("Rajagopal OpenSense base (opensim-models submodule) missing")
    out = tmp_path / "golf_humanoid.osim"
    return builder.build(output_path=out, grip_model=grip_model)


def test_builder_regenerates_the_committed_model(tmp_path: Path) -> None:
    out = _build_into(tmp_path, "weld")
    assert out.read_bytes() == MODELS["golf_humanoid"].read_bytes()


def test_bushing_grip_model_frees_the_club_with_two_bushings(tmp_path: Path) -> None:
    model = SafeET.parse(str(_build_into(tmp_path, "bushing"))).getroot()
    assert not _named(model, "WeldJoint", mc.LEAD_JOINT)
    assert not _named(model, "WeldConstraint", mc.TRAIL_CONSTRAINT)
    assert len(_named(model, "FreeJoint", mc.FREE_JOINT)) == 1
    bushing = mc.load_msk_club("driver").interface.bushing
    for label, side in (("left", "l"), ("right", "r")):
        (force,) = _named(model, "BushingForce", f"grip_bushing_{label}")
        assert force.findtext("socket_frame1").startswith(f"/bodyset/hand_{side}/")
        assert force.findtext("socket_frame2").startswith("/bodyset/Club/")
        stiff = [float(v) for v in force.findtext("translational_stiffness").split()]
        assert stiff == pytest.approx(bushing.translational_stiffness_n_m)


# ------------------------------------------------------------------ OpenSim
@pytest.fixture(scope="module")
def osim():  # noqa: ANN201
    module = pytest.importorskip("opensim")
    if not hasattr(module, "Model"):
        pytest.skip("opensim in sys.modules is a test double")
    club_visuals.register_geometry_path()
    return module


def _loaded(osim, path: Path):  # noqa: ANN001, ANN202
    model = osim.Model(str(path))
    state = model.initSystem()
    return model, state


def _pose(component, state) -> np.ndarray:  # noqa: ANN001
    from src.engines.physics_engines.opensim.python.msk_club_calibration import (
        transform_in_ground,
    )

    return transform_in_ground(component, state)


@pytest.mark.parametrize("name", sorted(MODELS))
def test_init_system_closes_the_trail_hand_and_places_both_hands(
    osim, name: str
) -> None:  # noqa: ANN001
    model, state = _loaded(osim, MODELS[name])
    model.realizePosition(state)
    hand = _pose(model.getComponent("/bodyset/hand_r/hand_r_grip_offset"), state)
    club = _pose(model.getComponent("/bodyset/Club/club_trail_grip_offset"), state)
    assert np.linalg.norm(hand[:3, 3] - club[:3, 3]) <= CLOSURE_TOL_M
    np.testing.assert_allclose(hand[:3, :3], club[:3, :3], atol=1e-6)
    club_pose = _pose(model.getBodySet().get("Club"), state)
    shaft_dir = club_pose[:3, 1]
    for side, body in mc.HAND_BODIES.items():
        hand_pose = _pose(model.getBodySet().get(body), state)
        point = hand_pose[:3, :3] @ mc.hand_grip_point(side) + hand_pose[:3, 3]
        grip = club_pose[:3, :3] @ mc.load_msk_club().grip_points[side]
        grip = grip + club_pose[:3, 3]
        # hand-to-shaft distance at the grip: the palm grip point on the axis
        offset = point - grip
        radial = offset - (offset @ shaft_dir) * shaft_dir
        assert np.linalg.norm(radial) <= GRIP_GAP_TOL_M, (side, np.linalg.norm(radial))
        assert np.linalg.norm(offset) <= GRIP_GAP_TOL_M


@pytest.mark.parametrize("name", sorted(MODELS))
def test_face_is_square_at_the_captured_address_pose(osim, name: str) -> None:  # noqa: ANN001
    from src.engines.physics_engines.opensim.python.msk_club_calibration import (
        face_angle_deg,
    )

    model, state = _loaded(osim, MODELS[name])
    model.realizePosition(state)
    angle = face_angle_deg(model, state, mc.load_msk_club("driver"))
    assert abs(angle) <= SQUARE_TOL_DEG, angle


@pytest.mark.parametrize("name", sorted(MODELS))
def test_club_mass_properties_load_into_opensim(osim, name: str) -> None:  # noqa: ANN001
    model, _ = _loaded(osim, MODELS[name])
    body = model.getBodySet().get("Club")
    assert body.getMass() == pytest.approx(DRIVER.total_mass_kg)
    assert model.getConstraintSet().getSize() >= 1


def _forward(osim, path: Path, duration: float = 0.01) -> np.ndarray:  # noqa: ANN001
    model, state = _loaded(osim, path)
    manager = osim.Manager(model)
    manager.initialize(state)
    final = manager.integrate(duration)
    model.realizePosition(final)
    q = np.array([final.getQ().get(i) for i in range(final.getNQ())])
    assert final.getTime() == pytest.approx(duration)
    return q


@pytest.mark.parametrize("name", sorted(MODELS))
def test_short_forward_step_stays_finite(osim, name: str) -> None:  # noqa: ANN001
    assert np.isfinite(_forward(osim, MODELS[name])).all()


def test_bushing_model_initialises_and_steps(osim, tmp_path: Path) -> None:  # noqa: ANN001
    path = _build_into(tmp_path, "bushing")
    q = _forward(osim, path)
    assert np.isfinite(q).all()


def test_calibration_round_trips_through_json(tmp_path: Path) -> None:
    calibration = mc.load_calibration("golf_humanoid")
    path = tmp_path / "cal.json"
    mc.store_calibration(calibration, path)
    again = mc.load_calibration("golf_humanoid", path=path)
    for side in mc.HAND_BODIES:
        np.testing.assert_allclose(
            again.hand_frames[side], calibration.hand_frames[side]
        )
    assert again.address_q == calibration.address_q
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert doc["schema"] == mc.CALIBRATION_SCHEMA
    assert math.isfinite(again.report["landmark_rms_m"])


def test_calibration_rejects_bad_frames() -> None:
    good = mc.load_calibration("golf_humanoid")
    with pytest.raises(ValueError, match="L and R"):
        mc.GripCalibration("m", "driver", {"L": np.eye(4)}, {}, np.eye(4), {})
    with pytest.raises(ValueError, match="4x4"):
        mc.GripCalibration(
            "m", "driver", {"L": np.eye(3), "R": np.eye(4)}, {}, np.eye(4), {}
        )
    with pytest.raises(ValueError, match="finite"):
        mc.GripCalibration(
            "m", "driver", good.hand_frames, {"q": math.nan}, np.eye(4), {}
        )


def test_strip_club_removes_every_club_reference() -> None:
    model = _model("golf_humanoid")
    removed = mc.strip_club(model)
    assert removed >= 4  # body, lead joint, trail constraint, hand frames
    text = "".join(e.text or "" for e in model.iter())
    assert "/bodyset/Club" not in text
    assert not [b for b in model.find("BodySet/objects") if b.get("name") == "Club"]


def test_grip_model_of_reads_the_attachment() -> None:
    model = _model("golf_humanoid")
    assert mc.grip_model_of(model) == "weld"
    calibration = mc.load_calibration("golf_humanoid")
    mc.attach_club(model, mc.load_msk_club(), calibration, grip_model="bushing")
    assert mc.grip_model_of(model) == "bushing"


def test_passive_club_coordinates_match_the_free_joint() -> None:
    from src.engines.physics_engines.opensim.python import musculoskeletal_swing as ms

    assert ms.PASSIVE_CLUB_COORDINATES == mc.FREE_COORDINATES


def test_muscle_model_holds_the_golf_humanoid_club(osim) -> None:  # noqa: ANN001
    from src.engines.physics_engines.opensim.python import musculoskeletal_swing as ms

    try:
        ms.resolve_base_model()
    except FileNotFoundError as exc:
        pytest.skip(str(exc))
    model, info = ms.build_musculoskeletal_model(MODELS["golf_humanoid_scaled"])
    assert info["n_muscles"] == 80  # unchanged muscle count
    gaps = info["grip_calibration"]["hand_grip_point_gap_m"]
    assert max(gaps.values()) <= GRIP_GAP_TOL_M
    state = model.initSystem()
    assert model.getBodySet().get("Club").getMass() == pytest.approx(
        DRIVER.total_mass_kg
    )
    assert model.getJointSet().contains(mc.LEAD_JOINT)
    assert model.getConstraintSet().contains(mc.TRAIL_CONSTRAINT)
    meshes = model.getBodySet().get("Club").getPropertyByName("attached_geometry")
    assert meshes.size() >= 1  # shared club meshes
    model.realizePosition(state)
    hand = _pose(model.getComponent("/bodyset/hand_r/hand_r_grip_offset"), state)
    club = _pose(model.getComponent("/bodyset/Club/club_trail_grip_offset"), state)
    assert np.linalg.norm(hand[:3, 3] - club[:3, 3]) <= CLOSURE_TOL_M
    from src.engines.physics_engines.opensim.python.msk_club_calibration import (
        face_angle_deg,
    )

    face = face_angle_deg(model, state, mc.load_msk_club())
    assert abs(face) <= SQUARE_TOL_DEG
