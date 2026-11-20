"""The skinned CC0 body in the MuJoCo export: default off, appearance only (#15299)."""

from __future__ import annotations

import copy
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from src.engines.physics_engines.mujoco.python import body_skin_layer as layer
from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter
from src.shared.python.model_appearance import body_skin_fit as fit
from src.shared.python.model_appearance import body_skin_mesh as mesh
from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance import skinned_wrist_sections as sws
from src.shared.python.model_appearance import body_skin_export as export
from src.shared.python.model_appearance import document_from_dict, library
from src.shared.python.motion_matching.execution.skinned_hand_wrists import (
    apply_skinned_hand_wrists,
)

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = json.loads(
    (
        ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    ).read_text(encoding="utf-8")
)
ADDRESS = json.loads(
    (ROOT / "tests/fixtures/model_appearance/address_pose.json").read_text()
)
SWING_FILE = json.loads(
    (ROOT / "tests/fixtures/model_appearance/swing_poses.json").read_text()
)
SWING, SWING_WRISTS = (
    SWING_FILE["changes_from_address"],
    SWING_FILE["skinned_hand_wrists"],
)
SWING_LIMITS = json.loads((fit.ASSET_DIR / "export_settings.json").read_text())[
    "swing_limits"
]
WRIST_LIMITS = json.loads((fit.ASSET_DIR / "export_settings.json").read_text())[
    "wrist_limits"
]
HEAD_BASE = json.loads((fit.ASSET_DIR / "export_settings.json").read_text())[
    "head_base_limits"
]
HEAD_JOIN = json.loads((fit.ASSET_DIR / "export_settings.json").read_text())[
    "head_join_limits"
]
SETTINGS = export.load_export_settings()
DOCUMENT = document_from_dict({"schema_version": "appearance-v1"})
LAYERS = ("skin", "shirt", "trousers")


def _export(backend: str | None) -> tuple[str, dict[str, Any]]:
    spec = copy.deepcopy(SPEC)
    if backend is not None:
        spec["subject"]["body_visual_backend"] = backend
    return exporter.export_full_body_mjcf(
        json.dumps(spec).encode(), appearance=DOCUMENT
    )


@pytest.fixture(scope="module")
def loft() -> Any:
    return mujoco.MjModel.from_xml_string(_export(None)[0])


@pytest.fixture(scope="module")
def skinned() -> tuple[Any, dict[str, Any]]:
    xml, summary = _export(export.SKINNED_BACKEND)
    return mujoco.MjModel.from_xml_string(xml), summary["visual_layer"]["body_skin"]


def test_the_loft_body_is_the_default_and_has_no_skin() -> None:
    xml, summary = _export(None)
    assert not export.body_skin_enabled(SPEC)
    assert "<deformable" not in xml and "<skin" not in xml
    assert "body_skin" not in summary["visual_layer"]
    assert _export(export.LOFT_BACKEND)[0] == xml
    with pytest.raises(ValueError, match="body_visual_backend"):
        _export("marble")


def test_the_skin_changes_no_physics(loft: Any, skinned: tuple[Any, Any]) -> None:
    model = skinned[0]
    sizes = ("nq", "nv", "nbody", "njnt", "nu", "neq")
    assert [getattr(model, n) for n in sizes] == [getattr(loft, n) for n in sizes]
    for name in ("body_mass", "body_inertia", "body_pos", "body_quat", "body_ipos"):
        assert np.array_equal(getattr(model, name), getattr(loft, name)), name


def test_three_layers_follow_the_bodies(skinned: tuple[Any, Any]) -> None:
    model, summary = skinned
    assert model.nskin == len(LAYERS)
    assert [model.skin(i).name for i in range(3)] == [f"body_skin_{n}" for n in LAYERS]
    assert summary["bones"] == {"skin": 12, "shirt": 11, "trousers": 6}
    assert model.nskinbone == sum(summary["bones"].values())
    assert model.nskinvert == sum(summary["vertices"].values())
    roles = summary["bodies"]
    assert library.classify_body(roles["thorax"]) == "torso"
    assert roles["upper_arm_l"].endswith("LUpperArm")
    assert roles["shank_r"] == "tibia_r"


def test_the_skin_replaces_the_loft_and_the_neck(
    loft: Any, skinned: tuple[Any, Any]
) -> None:
    def visuals(model: Any) -> dict[str, list[str]]:
        out: dict[str, list[str]] = {}
        for index in range(model.ngeom):
            geom = model.geom(index)
            part = library.classify_body(model.body(geom.bodyid[0]).name or "world")
            if geom.name.startswith(("visual_vmesh", "visual_head")):
                # the number in a mesh geom's name counts the drawn bodies
                out.setdefault(part, []).append(re.sub(r"vmesh_\d+_", "", geom.name))
        return out

    before, after = visuals(loft), visuals(skinned[0])
    for part in SETTINGS.covered_parts:
        assert before[part] and part not in after, part
    for part in ("foot", "club"):
        assert after[part] == before[part]
    necks = [n for n in before["head"] if n not in after["head"]]
    assert necks and all("neck" in name for name in necks)
    assert len(after["head"]) == len(before["head"]) - len(necks)


def test_the_skinned_model_steps_at_swing_poses(skinned: tuple[Any, Any]) -> None:
    model = skinned[0]
    data = mujoco.MjData(model)
    for pose in ({}, SETTINGS.bind_pose, ADDRESS):
        mujoco.mj_resetData(model, data)
        for name, value in pose.items():
            data.qpos[model.joint(name).qposadr[0]] = value
        mujoco.mj_forward(model, data)
        mujoco.mj_step(model, data)
        assert np.isfinite(data.qpos).all() and np.isfinite(data.xpos).all()
        assert not data.warning.number.any()


def _surface(model: Any, pose: dict[str, float]) -> dict[str, tuple[Any, Any]]:
    data = mujoco.MjData(model)
    for name, value in pose.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    return layer.posed_skin_layers(model, data)


@pytest.mark.parametrize("name", sorted(SWING))
def test_the_surface_survives_swing_like_poses(
    name: str, loft: Any, skinned: tuple[Any, Any]
) -> None:
    model = skinned[0]
    pose = dict(ADDRESS)
    for joint, change in SWING[name].items():
        pose[joint] = pose.get(joint, 0.0) + change
    rest = _surface(model, dict(SETTINGS.bind_pose))
    posed = _surface(model, pose)
    assert sorted(posed) == sorted(f"body_skin_{n}" for n in LAYERS)
    band = SWING_LIMITS["area_ratio"]
    for layer_name, (vertices, faces) in posed.items():
        assert np.isfinite(vertices).all() and len(faces) > 1000
        ratio = mesh.area_ratio(rest[layer_name][0], vertices, faces)
        stretched = float(((ratio > band) | (ratio < 1.0 / band)).mean())
        folded = float(mesh.folded_faces(vertices, faces).mean())
        assert stretched <= SWING_LIMITS["stretched_share"], (layer_name, stretched)
        assert folded <= SWING_LIMITS["folded_share"], (layer_name, folded)
    moved = posed["body_skin_shirt"][0] - rest["body_skin_shirt"][0]
    assert np.linalg.norm(moved, axis=1).max() > 0.1
    with pytest.raises(ValueError, match="no skin"):
        layer.posed_skin_layers(loft, mujoco.MjData(loft))


def _head_shell(model: Any, pose: dict[str, float]) -> tuple[Any, Any, Any, Any]:
    """The head shell at ``pose``: its vertices in the world and in the head
    frame, its faces, and the posed data."""
    geom = model.geom("visual_head_skin_mesh")
    shell = model.mesh(geom.dataid[0])
    local = model.mesh_vert[shell.vertadr[0] : shell.vertadr[0] + shell.vertnum[0]]
    faces = model.mesh_face[shell.faceadr[0] : shell.faceadr[0] + shell.facenum[0]]
    data = mujoco.MjData(model)
    for name, value in pose.items():
        data.qpos[model.joint(name).qposadr[0]] = value
    mujoco.mj_forward(model, data)
    world = data.geom_xpos[geom.id] + local @ data.geom_xmat[geom.id].reshape(3, 3).T
    head = geom.bodyid[0]
    in_head = (world - data.xpos[head]) @ data.xmat[head].reshape(3, 3)
    return world, in_head, faces, data


def _head_base(model: Any, pose: dict[str, float]) -> tuple[Any, Any, Any]:
    """The head shell's base at ``pose``: its vertices in the world and in the
    head frame, and the posed data."""
    world, in_head, _, data = _head_shell(model, pose)
    low = in_head[:, 2] < HEAD_BASE["below_m"]
    return world[low], in_head[low], data


def _rim_cover(model: Any, pose: dict[str, float]) -> Any:
    """Cover angle (degrees) of each front vertex of the skin's neck rim.

    In the head frame (z up the neck, x forward) a sight line passes over a rim
    vertex and dips towards the neck axis. It meets the head shell if it dips
    less than the steepest line from the vertex to a shell point nearer the axis
    in the same direction (within ``window_deg``); a steeper one passes under the
    shell's base into the open neck. That steepest dip is the vertex's cover.
    """
    _, shell, faces, data = _head_shell(model, pose)
    skin, skin_faces = layer.posed_skin_layers(model, data)["body_skin_skin"]
    head = model.geom("visual_head_skin_mesh").bodyid[0]
    skin = (skin - data.xpos[head]) @ data.xmat[head].reshape(3, 3)
    edges = mesh.boundary_edges(skin_faces)
    graph = coo_matrix((np.ones(len(edges)), edges.T), shape=(len(skin),) * 2)
    part = connected_components(graph, directed=False)[1]
    loops = [np.flatnonzero(part == k) for k in np.unique(part[edges.ravel()])]
    rim = skin[max(loops, key=lambda ids: skin[ids, 2].mean())]  # the top opening
    sides = [shell[faces[:, [i, i - 1]]].mean(axis=1) for i in range(3)]
    wall = np.vstack([shell, *sides])
    wall_turn = np.degrees(np.arctan2(wall[:, 1], wall[:, 0]))
    wall_reach = np.hypot(wall[:, 0], wall[:, 1])
    turn = np.degrees(np.arctan2(rim[:, 1], rim[:, 0]))
    cover = []
    for point, at in zip(rim, turn, strict=True):
        if abs(at) > HEAD_JOIN["sector_deg"]:
            continue
        reach = np.hypot(point[0], point[1])
        near = np.abs((wall_turn - at + 180.0) % 360.0 - 180.0)
        inner = (near <= HEAD_JOIN["window_deg"]) & (wall_reach < reach - 1e-4)
        dip = (point[2] - wall[inner, 2]) / (reach - wall_reach[inner])
        cover.append(np.degrees(np.arctan(dip.max())))
    return np.array(cover)


@pytest.mark.parametrize("name", ["top", "finish"])
def test_the_head_base_stays_inside_the_neck_at_a_swing_head_turn(
    name: str, loft: Any, skinned: tuple[Any, Any]
) -> None:
    """A fitted driver swing turns the head 1.4 rad against the chest at the
    top; the head model's base must not come out of the skin's neck there."""
    model = skinned[0]
    pose = dict(ADDRESS)
    for joint, change in SWING[name].items():
        pose[joint] = pose.get(joint, 0.0) + change
    for turn in (HEAD_BASE["turn_rad"], -HEAD_BASE["turn_rad"]):
        base, held, data = _head_base(model, pose | {"NeckInputZ": turn})
        skin, faces = layer.posed_skin_layers(model, data)["body_skin_skin"]
        near = cKDTree(skin).query(base)[1]
        normals = mesh.vertex_normals(skin, faces)[near]
        outside = np.einsum("ij,ij->i", base - skin[near], normals).max()
        assert len(base) > 50 and outside <= HEAD_BASE["outside_m"], (turn, outside)
        front, back = SETTINGS.head_base_front_m, SETTINGS.head_base_back_m
        depth = np.where(held[:, 0] > 0.0, front, back)
        inside = np.hypot(held[:, 0] / depth, held[:, 1] / SETTINGS.head_base_side_m)
        assert inside.max() <= 1.0 + 1e-4
    assert SETTINGS.head_base_height_m == HEAD_BASE["below_m"]
    # the loft body keeps the head model's own, wider base under its neck column
    wide = np.abs(_head_base(loft, pose)[1][:, 1]).max()
    assert wide > 1.5 * SETTINGS.head_base_side_m


@pytest.mark.parametrize("name", sorted(SWING))
def test_the_head_base_covers_the_skins_neck_rim_at_the_front(
    name: str, skinned: tuple[Any, Any]
) -> None:
    """Under the jaw the skin's neck ends in an open rim round the head model's
    neck stub. A base held too far in, or ending level with the rim, lets a view
    from the front pass between the two and out through the back of the neck."""
    pose = dict(ADDRESS)
    for joint, change in SWING[name].items():
        pose[joint] = pose.get(joint, 0.0) + change
    cover = _rim_cover(skinned[0], pose)
    assert len(cover) >= 10
    assert cover.mean() >= HEAD_JOIN["mean_cover_deg"], cover.mean()
    assert cover.min() >= HEAD_JOIN["least_cover_deg"], cover.min()


def test_the_forearms_end_under_the_skinned_hands_wrists() -> None:
    spec = copy.deepcopy(SPEC)
    spec["subject"]["hand_visual_backend"] = "skinned_human"
    club = ca.assembly_from_spec(spec)
    by_body = layer.reference_frames(spec, SETTINGS.bind_pose)
    roles = export.body_roles(spec, {b: f.translation for b, f in by_body.items()})
    frames = {role: by_body[body] for role, body in roles.items()}
    settings = fit.load_fit_settings()
    skin = fit.load_body_skin()
    fitted = fit.fit_body_skin(skin, export.subject_lengths(frames), settings)
    binds = export.bind_poses(frames, fitted)
    turns = layer.spec_turns(spec, roles, frames, SETTINGS.bind_pose)
    assert layer.wrist_rings(SPEC, club, roles, binds, turns) == {}  # other hands
    assert layer.hand_binds(SPEC, club, roles, binds, turns, fitted.joints) == {}
    rings = layer.wrist_rings(spec, club, roles, binds, turns)
    sections = sws.forearm_wrist_sections(spec, club)
    assert sorted(rings) == ["l", "r"] and len(sections) == 2
    matched = fit.match_wrists(skin, fitted, rings, settings)
    for side, ring in rings.items():
        section = sections[roles[f"forearm_{side}"]]
        assert ring.half_width_m == section.half_width_m > ring.half_depth_m
        elbow, wrist = (fitted.joints[f"{j}_{side}"] for j in ("elbow", "wrist"))
        bone = (wrist - elbow) / np.linalg.norm(wrist - elbow)
        assert abs(float(ring.width_axis @ bone)) < 0.2  # across the forearm
        columns = [g == f"forearm_distal_{side}" for g in skin.groups]
        end = skin.weights[:, columns].sum(axis=1) > 0.5
        end &= (matched.vertices - wrist) @ bone > -settings.wrist_match_full_m
        off = matched.vertices[end] - wrist - ring.centre_m
        radius = np.linalg.norm(off - np.outer(off @ bone, bone), axis=1)
        # The ring lies inside the hand's section, by the named 2 mm at most
        # off its two half axes.
        assert radius.max() <= ring.half_width_m - settings.wrist_under_m + 3e-4
        assert radius.min() >= ring.half_depth_m - settings.wrist_under_m - 3e-4


@pytest.fixture(scope="module")
def on_club() -> tuple[Any, dict[str, Any], Any]:
    """The skinned body with the skinned hands rigid on the club, the wrist
    joints on the hands' (and the same model with the loft body)."""
    spec = copy.deepcopy(SPEC)
    spec["subject"]["hand_visual_backend"] = "skinned_human"
    spec = apply_skinned_hand_wrists(spec)
    models = []
    for backend in (export.LOFT_BACKEND, export.SKINNED_BACKEND):
        spec["subject"]["body_visual_backend"] = backend
        xml, summary = exporter.export_full_body_mjcf(
            json.dumps(spec).encode(), appearance=DOCUMENT
        )
        models.append(mujoco.MjModel.from_xml_string(xml))
    return models[1], summary["visual_layer"]["body_skin"], models[0]


def _hand_vertices(model: Any, data: Any, label: str) -> Any:
    """World vertices of a drawn hand proper: its largest connected part."""
    geom = next(g for g in range(model.ngeom) if model.geom(g).name.endswith(label))
    mesh = model.mesh(model.geom_dataid[geom])
    count = int(mesh.vertnum[0])
    faces = model.mesh_face[mesh.faceadr[0] : mesh.faceadr[0] + mesh.facenum[0]]
    ends = faces[:, [0, 1]].ravel(), faces[:, [1, 2]].ravel()
    graph = coo_matrix((np.ones(len(ends[0])), ends), shape=(count, count))
    part = connected_components(graph, directed=False)[1]
    used = np.unique(faces)
    used = used[part[used] == np.bincount(part[used]).argmax()]
    local = model.mesh_vert[mesh.vertadr[0] : mesh.vertadr[0] + count][used]
    return data.geom_xpos[geom] + local @ data.geom_xmat[geom].reshape(3, 3).T


def test_the_forearm_ends_stay_under_the_hands_on_the_club(
    skinned: tuple[Any, Any], on_club: tuple[Any, Any, Any]
) -> None:
    model, summary, loft = on_club
    assert summary["wrists_blended"] == ["l", "r"] and not skinned[1]["wrists_blended"]
    assert summary["bones"]["skin"] == skinned[1]["bones"]["skin"] + 2
    for name in ("nq", "nv", "nbody", "njnt", "nu", "neq"):
        assert getattr(model, name) == getattr(loft, name), name
    for name in ("body_mass", "body_inertia", "body_pos", "body_quat", "body_ipos"):
        assert np.array_equal(getattr(model, name), getattr(loft, name)), name
    rest = _surface(model, dict(SETTINGS.bind_pose))["body_skin_skin"][0]
    bend, band = np.radians(WRIST_LIMITS["bend_deg"]), SWING_LIMITS["area_ratio"]
    last = model.skin_boneadr[0] + summary["bones"]["skin"]
    for side, prefix, label in (("l", "L", "lead_hand"), ("r", "R", "trail_hand")):
        hand = model.body(summary["bodies"][f"hand_{side}"]).id
        bone = next(b for b in range(last) if model.skin_bonebodyid[b] == hand)
        first, count = model.skin_bonevertadr[bone], model.skin_bonevertnum[bone]
        whole = model.skin_bonevertweight[first : first + count] == 1.0
        ring = model.skin_bonevertid[first : first + count][whole]
        assert len(ring) >= 40  # the end ring
        seen = []
        for x, y in ((0, 0), (bend, 0), (-bend, 0), (0, bend), (0, -bend)):
            data = mujoco.MjData(model)
            for joint, value in ADDRESS.items():
                data.qpos[model.joint(joint).qposadr[0]] = value
            data.qpos[model.joint(f"{prefix}WInputX").qposadr[0]] += x
            data.qpos[model.joint(f"{prefix}WInputY").qposadr[0]] += y
            mujoco.mj_forward(model, data)
            vertices, faces = layer.posed_skin_layers(model, data)["body_skin_skin"]
            turn = data.xmat[hand].reshape(3, 3)
            seen.append((vertices[ring] - data.xpos[hand]) @ turn)  # in the hand
            ratio = mesh.area_ratio(rest, vertices, faces)
            stretched = float(((ratio > band) | (ratio < 1.0 / band)).mean())
            folded = float(mesh.folded_faces(vertices, faces).mean())
            assert stretched <= SWING_LIMITS["stretched_share"], (side, x, y)
            assert folded <= SWING_LIMITS["folded_share"], (side, x, y)
        shift = max(np.linalg.norm(one - seen[0], axis=1).max() for one in seen[1:])
        assert shift <= WRIST_LIMITS["end_shift_m"], (side, shift)
        points = _hand_vertices(model, data, label)
        if side == "r":  # drawn on the club: carried to where the grip weld holds
            a, b = (data.site(f"native_closure_{k}") for k in "ab")
            turn = b.xmat.reshape(3, 3) @ a.xmat.reshape(3, 3).T
            points = a.xpos + (points - b.xpos) @ turn
        gap = cKDTree(points).query(vertices[ring])[0].max()
        assert gap <= WRIST_LIMITS["end_to_hand_m"], (side, gap)


@pytest.mark.parametrize("name", sorted(SWING_WRISTS))
def test_the_forearms_keep_their_depth_at_a_swing_wrist(
    name: str, on_club: tuple[Any, Any, Any]
) -> None:
    """The hands are turned 60 to 100 degrees against the forearms in a
    swing; the forearm must not be wrung thin behind the wrist there."""
    model, summary, _ = on_club
    pose = dict(ADDRESS) | SWING_WRISTS[name]
    for joint, change in SWING[name].items():
        pose[joint] = pose.get(joint, 0.0) + change
    posed, faces = _surface(model, pose)["body_skin_skin"]
    rest = np.array(model.skin_vert[: len(posed)], dtype=float)  # straight wrists
    near, far = WRIST_LIMITS["depth_band_m"]
    assert near < SETTINGS.wrist_blend_length_m < far  # the band spans the blend
    thin = {}
    for side in ("l", "r"):
        arm, joint = np.zeros(len(rest), dtype=bool), {}
        for bone in range(model.skin_bonenum[0]):  # the first skin's bones
            for role in ("elbow", "forearm", "hand"):
                body = model.body(summary["bodies"][f"{role}_{side}"]).id
                if model.skin_bonebodyid[bone] == body:
                    first, count = (
                        model.skin_bonevertadr[bone],
                        model.skin_bonevertnum[bone],
                    )
                    arm[model.skin_bonevertid[first : first + count]] = True
                    joint[role] = model.skin_bonebindpos[bone]
        along = joint["hand"] - joint["elbow"]
        before = (joint["hand"] - rest) @ along / np.linalg.norm(along)
        part = faces[arm[faces].all(axis=1)]
        ids = np.flatnonzero(arm & (before > near) & (before < far))
        depth = mesh.inward_thickness(rest, part, ids)
        ratio = mesh.inward_thickness(posed, part, ids) / depth
        assert len(ids) > 40 and np.isfinite(depth).all(), side
        thin[side] = float(np.quantile(ratio, WRIST_LIMITS["thin_share"]))
    assert min(thin.values()) >= WRIST_LIMITS["depth_ratio"], thin
