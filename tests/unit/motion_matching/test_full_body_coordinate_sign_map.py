"""Calibrated probe of the 44 full-body v1 coordinate axes and signs (HRG-00, #13161).

Nothing here is read from a comment or a field name. Each coordinate is
perturbed by +10 degrees from q = 0 in the compiled model and two things are
measured:

* the rotation axis of the child body, in the pelvis frame (x forward,
  y left, z up), from the child body's orientation change; and
* the motion of a body-fixed probe point 0.1 m from the joint anchor along a
  pelvis axis that is not parallel to the rotation axis.

The pinned table in ``coordinate_sign_map`` must reproduce both. The same axes
are asserted again from Drake and Pinocchio forward kinematics when those
engines import.

The fixture is the repository's anthropometric driver spec after hip
calibration. It is not the private best-fit spec: the grip rotation differs, but
the family, the wrist basis and the 44-coordinate order are the same.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching import coordinate_sign_map as sign_map
from src.shared.python.motion_matching.anthropometric_geometry import WRIST_ROLL
from src.shared.python.motion_matching.range_of_motion import HUMAN_RANGES_DEG

mujoco = pytest.importorskip("mujoco")

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
SPEC_PATH = (
    MODELS
    / "evidence/ground_support/anthro_driver_seeds/full_body_spec_hipcal_scaled.json"
)
NATIVE_PATH = (
    ROOT
    / "docs/development/simscape_tour_matching/native_evidence"
    / "native_geometry_spec_9967.json"
)
STEP_DEG = 10.0
STEP_M = 0.01
PROBE_M = 0.1
PROBE_BASIS = {"forward": 0, "left": 1, "up": 2}


def _spec(path: Path = SPEC_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _short(name: str) -> str:
    return name.rsplit("/", 1)[-1]


def _compile(path: Path = SPEC_PATH) -> tuple[Any, Any]:
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    xml, _ = export_full_body_mjcf(path.read_bytes())
    model = mujoco.MjModel.from_xml_string(xml)
    return model, mujoco.MjData(model)


def _forward(model: Any, data: Any, q: np.ndarray) -> None:
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)


def _pelvis_rotation(model: Any, data: Any) -> np.ndarray:
    _forward(model, data, np.zeros(model.nq))
    return data.xmat[1].reshape(3, 3).copy()


def _probe_axis(axis: np.ndarray) -> str:
    """First of forward, up, left that is well away from the rotation axis."""
    for name in ("forward", "up", "left"):
        if abs(float(axis[PROBE_BASIS[name]])) < 0.5:
            return name
    raise AssertionError("no probe direction is perpendicular to the axis")


def _measure_mujoco(model: Any, data: Any) -> dict[str, dict[str, Any]]:
    pelvis = _pelvis_rotation(model, data)
    out: dict[str, dict[str, Any]] = {}
    for joint in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint)
        body = int(model.jnt_bodyid[joint])
        hinge = int(model.jnt_type[joint]) == int(mujoco.mjtJoint.mjJNT_HINGE)
        _forward(model, data, np.zeros(model.nq))
        rot0 = data.xmat[body].reshape(3, 3).copy()
        origin0 = data.xpos[body].copy()
        anchor = data.xanchor[joint].copy()
        step = np.radians(STEP_DEG) if hinge else STEP_M
        q = np.zeros(model.nq)
        q[model.jnt_qposadr[joint]] = step
        if not hinge:
            _forward(model, data, q)
            moved = pelvis.T @ (data.xpos[body] - origin0)
            out[name] = {
                "axis": moved / np.linalg.norm(moved),
                "probe": "origin",
                "motion": sign_map.dominant_direction(moved),
            }
        else:
            axis_world = np.asarray(data.xaxis[joint]).copy()
            probe = _probe_axis(pelvis.T @ axis_world)
            probe_dir = pelvis[:, PROBE_BASIS[probe]]
            point0 = anchor + PROBE_M * probe_dir
            point_body = rot0.T @ (point0 - origin0)
            _forward(model, data, q)
            rot1 = data.xmat[body].reshape(3, 3)
            point1 = data.xpos[body] + rot1 @ point_body
            axis, angle = sign_map.axis_angle_from_rotation(
                pelvis.T @ rot1 @ rot0.T @ pelvis
            )
            assert angle == pytest.approx(np.radians(STEP_DEG), abs=1e-9)
            motion = pelvis.T @ (point1 - point0)
            out[name] = {
                "axis": axis,
                "axis_xaxis": pelvis.T @ axis_world,
                "probe": probe,
                "motion": sign_map.dominant_direction(motion),
            }
        out[name]["body"] = _short(
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body)
        )
        out[name]["parent"] = _short(
            mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, model.body_parentid[body]
            )
        )
    return out


@pytest.fixture(scope="module")
def compiled() -> tuple[Any, Any]:
    return _compile()


@pytest.fixture(scope="module")
def measured(compiled: tuple[Any, Any]) -> dict[str, dict[str, Any]]:
    return _measure_mujoco(*compiled)


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def test_table_covers_every_coordinate_in_spec_order() -> None:
    assert tuple(sign_map.COORDINATE_SIGN_MAP) == tuple(_spec()["coordinate_order"])
    assert len(sign_map.COORDINATE_SIGN_MAP) == 44


@pytest.mark.parametrize("coordinate", list(sign_map.COORDINATE_SIGN_MAP))
def test_pinned_row_matches_compiled_mujoco_geometry(
    coordinate: str, measured: dict[str, dict[str, Any]]
) -> None:
    row = sign_map.COORDINATE_SIGN_MAP[coordinate]
    got = measured[coordinate]
    assert (got["parent"], got["body"]) == (row.parent_body, row.body)
    assert (
        _angle_deg(got["axis"], np.asarray(row.axis_model))
        < sign_map.AXIS_TOLERANCE_DEG
    )
    assert got["probe"] == row.probe
    assert got["motion"] == row.probe_motion


def test_mjcf_axis_agrees_with_orientation_change(
    measured: dict[str, dict[str, Any]],
) -> None:
    for name, got in measured.items():
        if "axis_xaxis" in got:
            assert _angle_deg(got["axis"], got["axis_xaxis"]) < 1e-6, name


# --- wrists: which of X and Y is flexion and which is deviation --------------


def test_wrist_x_is_radial_ulnar_deviation_parallel_to_the_elbow(
    measured: dict[str, dict[str, Any]],
) -> None:
    for side in ("L", "R"):
        elbow = measured[f"{side}EInput"]["axis"]
        cock = measured[f"{side}WInputX"]["axis"]
        assert abs(float(np.dot(elbow, cock))) > 1 - 1e-9
        # Positive cock lifts the forward hand the way elbow flexion does, i.e.
        # toward the elbow pit: radial deviation. +elbow moves it the other way.
        assert measured[f"{side}WInputX"]["motion"] == "up"
        assert measured[f"{side}EInput"]["motion"] == "down"
        assert "radial" in sign_map.COORDINATE_SIGN_MAP[f"{side}WInputX"].meaning


def test_wrist_y_is_flexion_extension_with_mirrored_sense(
    measured: dict[str, dict[str, Any]],
) -> None:
    left, right = measured["LWInputY"], measured["RWInputY"]
    forearm = measured["LFInput"]["axis"]
    for got in (left, right):
        assert abs(float(np.dot(got["axis"], forearm))) < 1e-9
        assert got["motion"] == "left"
    rows = sign_map.COORDINATE_SIGN_MAP
    # The same axis and the same push toward +y: toward the dorsum of the left
    # hand (extension) and toward the palm of the right hand (flexion).
    assert rows["LWInputY"].meaning.startswith("extension")
    assert rows["RWInputY"].meaning.startswith("flexion")


def test_wrist_ranges_use_the_measured_senses() -> None:
    lo, hi = HUMAN_RANGES_DEG["LWInputX"]
    assert (lo, hi) == (-40.0, 25.0)  # ulnar .. radial, positive radial


# --- forearm pronation, hips, knees ------------------------------------------


def test_forearm_axis_is_proximal_and_pronation_sign_flips_by_side(
    compiled: tuple[Any, Any], measured: dict[str, dict[str, Any]]
) -> None:
    model, data = compiled
    pelvis = _pelvis_rotation(model, data)
    for side, pronation_sign in (("L", -1.0), ("R", +1.0)):
        forearm_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, _long_name(model, f"{side}LowerForearm")
        )
        hand_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_BODY,
            _long_name(model, "Clubface Vector" if side == "L" else "RHandStandoff"),
        )
        distal = pelvis.T @ (data.xpos[hand_id] - data.xpos[forearm_id])
        axis = measured[f"{side}FInput"]["axis"]
        # Pronation is positive about the proximal direction of a right
        # forearm and about the distal direction of a left forearm.
        about_proximal = -float(np.dot(axis, distal / np.linalg.norm(distal)))
        assert about_proximal > 0.99
        meaning = sign_map.COORDINATE_SIGN_MAP[f"{side}FInput"].meaning
        assert meaning.startswith("pronation" if pronation_sign > 0 else "supination")


def _long_name(model: Any, suffix: str) -> str:
    for index in range(model.nbody):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, index)
        if name and _short(name) == suffix:
            return name
    raise AssertionError(suffix)


@pytest.mark.parametrize("side", ["r", "l"])
def test_hips_and_knees_follow_the_opensim_signs(
    side: str, measured: dict[str, dict[str, Any]]
) -> None:
    rows = sign_map.COORDINATE_SIGN_MAP
    assert measured[f"hip_flexion_{side}"]["motion"] == "up"  # forward probe rises
    assert rows[f"hip_flexion_{side}"].meaning.startswith("flexion")
    assert rows[f"hip_adduction_{side}"].meaning.startswith("adduction")
    assert rows[f"hip_rotation_{side}"].meaning.startswith("internal rotation")
    # Knee flexion is negative: the large side of the range is negative and +
    # swings the shank forward (extension).
    lo, hi = HUMAN_RANGES_DEG[f"knee_angle_{side}"]
    assert abs(lo) > abs(hi)
    assert rows[f"knee_angle_{side}"].meaning.startswith("extension")


def test_hip_rotation_axes_are_opposite_by_side(
    measured: dict[str, dict[str, Any]],
) -> None:
    right, left = measured["hip_rotation_r"]["axis"], measured["hip_rotation_l"]["axis"]
    assert float(np.dot(right, left)) < -0.99


# --- trunk order --------------------------------------------------------------


def _world_axis(model: Any, data: Any, name: str, q: np.ndarray) -> np.ndarray:
    _forward(model, data, q)
    joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    return np.asarray(data.xaxis[joint]).copy()


def _qpos(model: Any, **degrees: float) -> np.ndarray:
    q = np.zeros(model.nq)
    for name, value in degrees.items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        q[model.jnt_qposadr[joint]] = np.radians(value)
    return q


def test_trunk_twist_is_applied_before_tilt(compiled: tuple[Any, Any]) -> None:
    model, data = compiled
    z = np.array([0.0, 0.0, 1.0])
    # Twist is about the pelvis z axis whatever the tilt that follows it.
    for tilt in (0.0, 20.0):
        axis = _world_axis(model, data, "TorsoInput", _qpos(model, SpineInputY=tilt))
        np.testing.assert_allclose(axis, z, atol=1e-12)
    # The tilt axes ride on the twist: twisting by 30 degrees yaws them.
    yaw = np.radians(30.0)
    expected = {
        "SpineInputX": [np.cos(yaw), np.sin(yaw), 0.0],
        "SpineInputY": [-np.sin(yaw), np.cos(yaw), 0.0],
    }
    for name, vector in expected.items():
        axis = _world_axis(model, data, name, _qpos(model, TorsoInput=30.0))
        np.testing.assert_allclose(axis, vector, atol=1e-12)
    rows = sign_map.COORDINATE_SIGN_MAP
    assert rows["TorsoInput"].parent_body == "LowerTorso"
    assert rows["SpineInputX"].parent_body == rows["TorsoInput"].body
    assert tuple(sign_map.TRUNK_CHAIN_ORDER) == (
        "TorsoInput",
        "SpineInputX",
        "SpineInputY",
    )


def test_trunk_twist_is_left_rotation_about_pelvis_z(
    measured: dict[str, dict[str, Any]],
) -> None:
    row = sign_map.COORDINATE_SIGN_MAP["TorsoInput"]
    assert row.axis_model == (0.0, 0.0, 1.0)
    assert measured["TorsoInput"]["motion"] == "left"
    assert row.meaning.startswith("axial rotation to the left")


# --- spec family: was WRIST_ROLL applied? -------------------------------------


def test_wrist_family_probe_separates_rolled_from_native_specs() -> None:
    native = json.loads(NATIVE_PATH.read_text(encoding="utf-8"))
    assert sign_map.wrist_roll_applied(_spec(), native) is True
    assert (
        sign_map.wrist_roll_applied(
            _spec(MODELS / "full_body_spec_anthro_driver.json"), native
        )
        is True
    )
    assert (
        sign_map.wrist_roll_applied(_spec(MODELS / "full_body_spec_v1.json"), native)
        is False
    )
    assert sign_map.wrist_roll_applied(native, native) is False


def test_wrist_family_probe_matches_the_builder_constant() -> None:
    native = json.loads(NATIVE_PATH.read_text(encoding="utf-8"))
    spec = _spec()
    cocks = [
        j
        for j in spec["joints"]
        if "LWInputX" in [p["coordinate"] for p in j["primitives"]]
    ]
    nats = [
        j
        for j in native["joints"]
        if "LWInputX" in [p["coordinate"] for p in j["primitives"]]
    ]
    rolled = np.asarray(cocks[0]["parent_to_base"])[:3, :3]
    base = np.asarray(nats[0]["parent_to_base"])[:3, :3]
    np.testing.assert_allclose(rolled, WRIST_ROLL @ base, atol=1e-9)


def test_wrist_family_probe_rejects_a_foreign_basis() -> None:
    native = json.loads(NATIVE_PATH.read_text(encoding="utf-8"))
    spec = json.loads(json.dumps(native))
    for joint in spec["joints"]:
        if "LWInputX" in [p["coordinate"] for p in joint["primitives"]]:
            joint["parent_to_base"] = np.diag([1.0, -1.0, -1.0, 1.0]).tolist()
    with pytest.raises(ValueError, match="neither"):
        sign_map.wrist_roll_applied(spec, native)


def test_compiled_geometry_separates_the_families() -> None:
    def second_axis_vs_forearm(path: Path) -> float:
        model, data = _compile(path)
        _forward(model, data, np.zeros(model.nq))
        axes = {
            n: np.asarray(
                data.xaxis[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)]
            )
            for n in ("LWInputY", "LFInput")
        }
        return abs(float(np.dot(axes["LWInputY"], axes["LFInput"])))

    # Native: the second wrist primitive spins about the forearm (redundant with
    # pronation). Rolled: it is the flexion/extension axis, perpendicular to it.
    assert second_axis_vs_forearm(MODELS / "full_body_spec_v1.json") > 1 - 1e-9
    assert second_axis_vs_forearm(SPEC_PATH) < 1e-9


# --- ISB basis ----------------------------------------------------------------


def test_isb_basis_map() -> None:
    basis = sign_map.MODEL_TO_ISB
    np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-15)
    assert np.linalg.det(basis) == pytest.approx(1.0)
    np.testing.assert_allclose(sign_map.model_to_isb([1, 0, 0]), [1, 0, 0])  # anterior
    np.testing.assert_allclose(sign_map.model_to_isb([0, 0, 1]), [0, 1, 0])  # up
    np.testing.assert_allclose(sign_map.model_to_isb([0, 1, 0]), [0, 0, -1])  # left
    np.testing.assert_allclose(sign_map.isb_to_model([0, 0, 1]), [0, -1, 0])  # right
    vector = np.array([0.3, -1.2, 2.5])
    np.testing.assert_allclose(
        sign_map.isb_to_model(sign_map.model_to_isb(vector)), vector
    )


def test_isb_rotation_keeps_angle_and_handedness() -> None:
    angle = 0.4
    c, s = np.cos(angle), np.sin(angle)
    about_model_z = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    about_isb_y = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    np.testing.assert_allclose(
        sign_map.model_rotation_to_isb(about_model_z), about_isb_y, atol=1e-15
    )
    axis, got = sign_map.axis_angle_from_rotation(
        sign_map.model_rotation_to_isb(about_model_z)
    )
    np.testing.assert_allclose(axis, [0, 1, 0], atol=1e-12)
    assert got == pytest.approx(angle)
    # A left twist (positive about model z) is positive about ISB Y.
    assert sign_map.isb_axis_label([0, 1, 0]) == "Y (up)"


# --- design by contract -------------------------------------------------------


def _row(**changes: Any) -> sign_map.CoordinateSign:
    base: dict[str, Any] = {
        "coordinate": "X",
        "parent_body": "P",
        "body": "B",
        "axis_model": (0.0, 0.0, 1.0),
        "probe": "forward",
        "probe_motion": "left",
        "meaning": "axial rotation to the left",
    }
    return sign_map.CoordinateSign(**{**base, **changes})


def test_contracts() -> None:
    assert _row().coordinate == "X"
    with pytest.raises(ValueError, match="unit"):
        _row(axis_model=(0.0, 0.0, 2.0))
    with pytest.raises(ValueError, match="probe"):
        _row(probe="sideways")
    with pytest.raises(ValueError, match="probe_motion"):
        _row(probe_motion="sideways")
    with pytest.raises(ValueError, match="meaning"):
        _row(meaning=" ")
    with pytest.raises(ValueError, match="finite"):
        _row(axis_model=(0.0, float("nan"), 1.0))
    with pytest.raises(ValueError, match="3-vector"):
        sign_map.model_to_isb([1.0, 2.0])
    with pytest.raises(ValueError, match="zero"):
        sign_map.dominant_direction([0.0, 0.0, 0.0])


# --- other engines ------------------------------------------------------------


def _hinge_rows() -> list[sign_map.CoordinateSign]:
    return [r for r in sign_map.COORDINATE_SIGN_MAP.values() if r.probe != "origin"]


def _child_of(spec: dict[str, Any]) -> dict[str, str]:
    return {
        p["coordinate"]: j["child"] for j in spec["joints"] for p in j["primitives"]
    }


def _axes_from_rotation_provider(
    spec: dict[str, Any], rotation_of: Any
) -> dict[str, np.ndarray]:
    """Axis of each hinge from ``rotation_of(coordinate_values, child) -> 3x3``."""
    order = list(spec["coordinate_order"])
    child = _child_of(spec)
    zero = dict.fromkeys(order, 0.0)
    root = next(j["child"] for j in spec["joints"] if j["parent"] == "world")
    pelvis = rotation_of(zero, root)
    axes = {}
    for row in _hinge_rows():
        step = dict(zero)
        step[row.coordinate] = np.radians(STEP_DEG)
        body0 = rotation_of(zero, child[row.coordinate])
        body1 = rotation_of(step, child[row.coordinate])
        axis, angle = sign_map.axis_angle_from_rotation(
            pelvis.T @ body1 @ body0.T @ pelvis
        )
        assert angle == pytest.approx(np.radians(STEP_DEG), abs=1e-9), row.coordinate
        axes[row.coordinate] = axis
    return axes


def _assert_axes_match_table(axes: dict[str, np.ndarray]) -> None:
    for row in _hinge_rows():
        got = _angle_deg(axes[row.coordinate], np.asarray(row.axis_model))
        assert got < sign_map.AXIS_TOLERANCE_DEG, (row.coordinate, got)


@pytest.mark.requires_pinocchio
def test_pinocchio_forward_kinematics_reproduces_the_table() -> None:
    pin = pytest.importorskip("pinocchio")
    if not hasattr(pin, "SE3"):
        pytest.skip("real pinocchio required")
    from src.engines.physics_engines.pinocchio.python.native_model import (
        FullBodyPinocchioModel,
    )

    spec = _spec()
    model = FullBodyPinocchioModel(spec)

    def rotation_of(values: dict[str, float], body: str) -> np.ndarray:
        q = model.configuration(values)
        pin.forwardKinematics(model.model, model.data, q)
        joint, pose = model.body_placement(body)
        return np.asarray(model.data.oMi[joint].rotation @ pose.rotation)

    _assert_axes_match_table(_axes_from_rotation_provider(spec, rotation_of))


@pytest.mark.requires_drake
def test_drake_forward_kinematics_reproduces_the_table() -> None:
    pytest.importorskip("pydrake")
    from src.engines.physics_engines.drake.python.full_body_model import (
        FullBodyDrakeModel,
    )

    spec = _spec()
    model = FullBodyDrakeModel(spec)
    links = model.metadata["body_links"]

    def rotation_of(values: dict[str, float], body: str) -> np.ndarray:
        model.set_configuration(np.array([values[n] for n in model.names]))
        link = model.plant.GetBodyByName(links[body], model._instance)
        return np.asarray(
            model.plant.EvalBodyPoseInWorld(model.context, link).rotation().matrix()
        )

    _assert_axes_match_table(_axes_from_rotation_provider(spec, rotation_of))
