"""The default golfer has a neck and a trapezius slope below the jaw (#15292)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import document_from_dict, library, neck
from src.shared.python.model_appearance.body_meshes import body_meshes
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton
from src.shared.python.model_appearance.head import build_neck_parts
from src.shared.python.model_appearance.defaults import default_document_dict
from src.shared.python.model_appearance.schema import HeadSettings

pytestmark = pytest.mark.unit

MODELS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
SPEC_BYTES = (MODELS / "full_body_spec_anthro_driver.json").read_bytes()
HEAD_M = 0.2429
ACCEPT = neck.NECK_ACCEPTANCE
SPEC = json.loads(SPEC_BYTES)
DOC = document_from_dict({"schema_version": "appearance-v1"})


ADDRESS = json.loads(
    (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "model_appearance"
        / "address_pose.json"
    ).read_text()
)


def _scene(appearance: dict, pose: dict | None = None):
    mujoco = pytest.importorskip("mujoco")
    from src.tools.native_viewer_export.backends.mujoco_worker import build_scene_xml

    xml, _ = build_scene_xml(SPEC_BYTES, appearance)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    for name, value in (pose or {}).items():
        joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        data.qpos[model.jnt_qposadr[joint]] = value
    mujoco.mj_forward(model, data)
    return mujoco, model, data


def _vertices(mujoco, model, data, geom: int) -> np.ndarray:
    mesh = model.geom_dataid[geom]
    first, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
    rot = data.geom_xmat[geom].reshape(3, 3)
    return model.mesh_vert[first : first + count] @ rot.T + data.geom_xpos[geom]


def _heights() -> dict[str, float]:
    """Highest trunk solid, yoke, and the chin, up from the neck point (metres)."""
    mujoco, model, data = _scene(default_document_dict())
    head = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "GolfSwing3D_Kinetic/Head"
    )
    frame, origin = data.xmat[head].reshape(3, 3), data.xpos[head]
    out: dict[str, float] = {
        "trunk": -9.0,
        "front": -9.0,
        "yoke": -9.0,
        "cap": -9.0,
        "chin": 9.0,
    }
    zone = ACCEPT.neck_zone_half_width * HEAD_M
    for geom in range(model.ngeom):
        name = model.geom(geom).name
        if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_MESH or not name:
            continue
        body = model.body(model.geom_bodyid[geom]).name
        local = (_vertices(mujoco, model, data, geom) - origin) @ frame
        if name == "visual_head_skin_mesh":
            face = local[local[:, 0] > 0.25 * HEAD_M]
            out["chin"] = float(face[:, 2].min())
        elif name.startswith("visual_vmesh") and ("skin" in name or "garment" in name):
            key = "yoke" if "Hubto" in body else "trunk" if "COMRod" in body else ""
            if "UpperArm" in body:
                out["cap"] = max(out["cap"], float(local[:, 2].max()))
            if key:  # the yoke counts where it meets the neck, not out at the cap
                near = local[np.abs(local[:, 1]) <= zone] if key == "yoke" else local
                if len(near):
                    out[key] = max(out[key], float(near[:, 2].max()))
                if key == "trunk":  # the collar's front, beside the sternum
                    ahead = local[
                        (local[:, 0] > 0.25 * HEAD_M) & (np.abs(local[:, 1]) < zone)
                    ]
                    out["front"] = max(out["front"], float(ahead[:, 2].max()))
    return out


def test_the_neck_is_visible_at_the_front_of_the_collar() -> None:
    h = _heights()
    visible = h["chin"] - h["front"]
    assert visible >= ACCEPT.min_front_visible_neck * HEAD_M, h


def test_the_collar_is_highest_at_the_back_and_slants_down_to_the_front() -> None:
    h = _heights()
    assert h["trunk"] <= ACCEPT.max_back_top_above_neck_point * HEAD_M, h
    assert h["trunk"] - h["front"] >= ACCEPT.min_collar_slant * HEAD_M, h


def test_the_upper_arm_does_not_rise_behind_the_shoulder_joint() -> None:
    """The cap sits on the joint: nothing pokes up the neck side of it.

    A T-pose height test would measure the arm lying sideways, so this is the
    pose-independent form of "the cap top is below the neck point": with the
    arm hanging, its overshoot behind the joint is exactly how far the cap
    rises above the joint (itself 4 cm below the neck point).
    """
    caps = {
        c.body: c
        for c in derive_visual_skeleton(SPEC).capsules
        if library.classify_body(c.body) == "upper_arm"
    }
    meshes = [
        m
        for m in body_meshes(SPEC, DOC)
        if m.body in caps and m.label.startswith(("skin", "garment"))
    ]
    assert len(meshes) >= 4
    for mesh in meshes:
        cap = caps[mesh.body]
        axis = (np.asarray(cap.end_m) - np.asarray(cap.start_m)) / cap.length_m()
        behind = -((mesh.mesh_in_body_frame.vertices - np.asarray(cap.start_m)) @ axis)
        allowed = ACCEPT.max_cap_behind_joint * library.PART_RADIUS_M["upper_arm"]
        assert behind.max() <= allowed, (mesh.body, mesh.label, behind.max())


def _address_cap_tops() -> dict[str, float]:
    """Highest upper-arm and outboard yoke vertex, head frame, at address."""
    mujoco, model, data = _scene(default_document_dict(), ADDRESS)
    head = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "GolfSwing3D_Kinetic/Head"
    )
    frame, origin = data.xmat[head].reshape(3, 3), data.xpos[head]
    zone = ACCEPT.neck_zone_half_width * HEAD_M
    tops: dict[str, float] = {}
    for geom in range(model.ngeom):
        name = model.geom(geom).name
        body = model.body(model.geom_bodyid[geom]).name
        if not name.startswith("visual_vmesh") or not body.endswith(
            ("LUpperArm", "RUpperArm", "HubtoLS", "HubtoRS")
        ):
            continue
        local = (_vertices(mujoco, model, data, geom) - origin) @ frame
        if "Hubto" in body:
            local = local[np.abs(local[:, 1]) > zone]
        if len(local):
            tops[body[-9:] + name] = float(local[:, 2].max())
    return tops


def test_the_shoulder_caps_stay_below_the_neck_point_at_address() -> None:
    tops = _address_cap_tops()
    assert len(tops) >= 8, tops
    limit = -ACCEPT.min_cap_below_neck_point * HEAD_M
    assert max(tops.values()) <= limit, tops


def _collar_ring(local: np.ndarray, sides: int = 28) -> np.ndarray:
    return local[-2 - sides : -2]  # the top ring, just before the two poles


def test_the_collar_opening_hugs_the_neck() -> None:
    mujoco, model, data = _scene(default_document_dict())
    head = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "GolfSwing3D_Kinetic/Head"
    )
    frame, origin = data.xmat[head].reshape(3, 3), data.xpos[head]
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "visual_vmesh_2_garment0")
    ring = _collar_ring((_vertices(mujoco, model, data, geom) - origin) @ frame)
    x, y = ring[:, 0], ring[:, 1]
    area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
    width, depth = _section_extent(_neck_part(), -0.05)
    neck_area = math.pi / 4.0 * width * depth * HEAD_M**2
    assert np.ptp(y) <= ACCEPT.max_opening_over_neck_width * width * HEAD_M
    assert area <= ACCEPT.max_collar_area_over_neck * neck_area, (area, neck_area)


def test_the_shirt_rises_from_the_shoulder_line_to_the_opening_without_a_lid() -> None:
    mujoco, model, data = _scene(default_document_dict())
    head = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "GolfSwing3D_Kinetic/Head"
    )
    frame, origin = data.xmat[head].reshape(3, 3), data.xpos[head]
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "visual_vmesh_2_garment0")
    local = (_vertices(mujoco, model, data, geom) - origin) @ frame
    rings = local[:-2].reshape(-1, 28, 3)[-ACCEPT.collar_rings :]
    ring_x = rings[-1, :, 0]
    front = ring_x > ring_x.mean()  # the front half of the opening
    rise = np.diff(rings[:, :, 2], axis=0)[:, front]
    run = np.linalg.norm(np.diff(rings[:, :, :2], axis=0), axis=2)[:, front]
    assert (rise / np.maximum(run, 1e-9)).min() >= ACCEPT.min_collar_wall_slope


def _garment_rings() -> np.ndarray:
    mujoco, model, data = _scene(default_document_dict())
    head = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "GolfSwing3D_Kinetic/Head"
    )
    frame, origin = data.xmat[head].reshape(3, 3), data.xpos[head]
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "visual_vmesh_2_garment0")
    local = (_vertices(mujoco, model, data, geom) - origin) @ frame
    return local[:-2].reshape(-1, 28, 3)  # 28 sides, rings bottom to top


def test_the_shirt_top_is_a_dome_not_a_shelf() -> None:
    rings = _garment_rings()
    height = rings[:, :, 2].mean(axis=1)
    above = rings[height >= ACCEPT.shoulder_line_height * HEAD_M]
    side = np.abs(above[:, 0, 1])  # the side meridian, outward distance
    rise = np.diff(above[:, 0, 2])
    angles = np.degrees(np.arctan2(rise, np.maximum(-np.diff(side), 1e-9)))
    assert angles.min() >= ACCEPT.min_collar_side_angle_deg, angles
    assert np.abs(np.diff(angles)).max() <= ACCEPT.max_collar_kink_deg, angles


def test_the_collar_outline_has_no_sharp_corner() -> None:
    for ring in _garment_rings()[-ACCEPT.rounded_collar_rings :]:
        x, y = ring[:, 0], ring[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
        fill = area / (np.ptp(x) * np.ptp(y))  # 0.785 is an ellipse, 1 a box
        assert fill <= ACCEPT.max_collar_fill, fill


def test_no_trunk_or_yoke_rises_to_the_chin() -> None:
    h = _heights()
    margin = ACCEPT.min_cap_below_chin * HEAD_M
    assert h["yoke"] <= h["chin"] - margin, h
    assert h["trunk"] <= h["chin"] - margin, h


def _neck_part():
    parts = {p.name: p for p in build_neck_parts(HEAD_M, HeadSettings())}
    return parts["neck"].mesh


def test_the_neck_solid_is_closed_with_positive_volume() -> None:
    mesh = _neck_part()
    edges = np.sort(
        np.vstack([mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]])
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert np.all(counts == 2)
    a, b, c = (mesh.vertices[mesh.faces[:, i]] for i in range(3))
    assert float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum()) / 6.0 > 0.0


def test_the_neck_solid_mirrors_left_to_right_and_leans_forward() -> None:
    mesh = _neck_part()
    mirrored = mesh.vertices * np.array([1.0, -1.0, 1.0])
    for point in mirrored[::5]:
        assert np.linalg.norm(mesh.vertices - point, axis=1).min() < 1e-9
    axis = mesh.vertices[-1] - mesh.vertices[-2]  # the poles are last: base, top
    lean = math.degrees(math.atan2(axis[0], axis[2]))
    assert 5.0 <= lean <= 35.0, lean  # the top is forward of the base
    assert mesh.vertices[:, 2].min() < 0.0  # it reaches down below the neck point


def _section_extent(mesh, height: float) -> tuple[float, float]:
    """Width and depth (head lengths) of the neck near ``height`` head lengths."""
    band = np.abs(mesh.vertices[:-2, 2] / HEAD_M - height) < 0.03
    ring = mesh.vertices[:-2][band] / HEAD_M
    return float(np.ptp(ring[:, 1])), float(np.ptp(ring[:, 0]))


def test_the_neck_is_a_slim_column_without_a_base_bulge() -> None:
    mesh = _neck_part()
    width, depth = _section_extent(mesh, -0.05)
    assert ACCEPT.neck_width[0] <= width <= ACCEPT.neck_width[1], width
    assert ACCEPT.neck_depth[0] <= depth <= ACCEPT.neck_depth[1], depth
    base_width, _ = _section_extent(mesh, -0.25)
    assert base_width <= ACCEPT.max_base_bulge * width, (base_width, width)


def test_the_neck_scales_with_the_head() -> None:
    small = {p.name: p for p in build_neck_parts(0.8 * HEAD_M, HeadSettings())}
    assert np.allclose(small["neck"].mesh.vertices, 0.8 * _neck_part().vertices)


def test_the_trapezius_slopes_inside_its_band() -> None:
    mujoco, model, data = _scene(default_document_dict())
    head = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "GolfSwing3D_Kinetic/Head"
    )
    frame, origin = data.xmat[head].reshape(3, 3), data.xpos[head]
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "visual_vmesh_2_skin0")
    local = (_vertices(mujoco, model, data, geom) - origin) @ frame
    rings = np.unique(np.round(local[:, 2], 4))
    reach = np.array(
        [np.abs(local[np.isclose(local[:, 2], z, atol=6e-5), 1]).max() for z in rings]
    )
    shoulder = rings[np.where(reach >= 0.97 * reach.max())[0].max()]
    rise = rings.max() - shoulder
    run = reach[rings == shoulder][0] - reach[rings == rings.max()][0]
    angle = math.degrees(math.atan2(rise, run))
    low, high = ACCEPT.slope_deg_band
    assert low <= angle <= high, angle


def test_mass_properties_do_not_depend_on_the_visual_neck() -> None:
    _, with_head, _ = _scene(default_document_dict())
    bare = {**default_document_dict(), "head": {"enabled": False}}
    _, without, _ = _scene(bare)
    for field in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "body_pos"):
        assert np.array_equal(getattr(with_head, field), getattr(without, field)), field
    assert np.array_equal(with_head.jnt_pos, without.jnt_pos)
    assert with_head.nbody == without.nbody


def test_the_parametric_head_keeps_its_own_neck() -> None:
    assert build_neck_parts(HEAD_M, HeadSettings(model="parametric")) == []
    with pytest.raises(ValueError, match="positive"):
        build_neck_parts(0.0)


def _neck_samples() -> np.ndarray:
    """Dense points on the neck surface (every face, 8 barycentric points)."""
    mesh = _neck_part()
    tri = mesh.vertices[mesh.faces]
    w = np.array([(i, j) for i in range(9) for j in range(9 - i)]) / 8.0
    return np.concatenate(
        [np.einsum("fvc,v->fc", tri, [a, b, 1 - a - b]) for a, b in w]
    )


def _top_ring(pose: dict | None) -> np.ndarray:
    mujoco, model, data = _scene(default_document_dict(), pose)
    head = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "GolfSwing3D_Kinetic/Head"
    )
    frame, origin = data.xmat[head].reshape(3, 3), data.xpos[head]
    geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "visual_vmesh_2_garment0")
    local = (_vertices(mujoco, model, data, geom) - origin) @ frame
    return local[:-2].reshape(-1, 28, 3)[-1]  # metres, head frame


def test_the_shirt_opening_is_centred_on_the_neck_axis() -> None:
    cfg = neck.NeckSettings()
    base = HEAD_M * np.array([cfg.base_forward, 0.0, cfg.base_height])
    top = HEAD_M * np.array([cfg.top_forward, 0.0, cfg.top_height])
    axis = (top - base) / np.linalg.norm(top - base)
    offset = _top_ring(None).mean(axis=0) - base
    off_axis = np.linalg.norm(offset - (offset @ axis) * axis)
    assert off_axis <= ACCEPT.max_opening_off_axis_m, off_axis


def test_the_shirt_opening_is_an_even_gap_round_the_neck() -> None:
    samples = _neck_samples()
    gap = np.array([np.linalg.norm(samples - v, axis=1).min() for v in _top_ring(None)])
    low, high = ACCEPT.opening_gap_m
    assert low <= gap.min() and gap.max() <= high, (gap.min(), gap.max())


def test_at_address_the_neck_does_not_stick_out_of_the_opening_front() -> None:
    ring = _top_ring(ADDRESS)
    front = ring[np.argmin(ring[:, 2])]
    samples = _neck_samples()
    near = samples[np.abs(samples[:, 2] - front[2]) < 0.01]
    ahead = float(near[:, 0].max() - front[0])
    assert ahead <= ACCEPT.max_neck_ahead_of_opening_m, ahead
