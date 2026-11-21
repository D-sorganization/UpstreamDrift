"""One skin piece per segment, same drawn surface, straight borders (#15150)."""

from __future__ import annotations

import dataclasses
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
from defusedxml import ElementTree as ET
from scipy.spatial import cKDTree

from src.engines.physics_engines.mujoco.python import body_skin_layer as layer
from src.engines.physics_engines.mujoco.python import body_skin_pieces as pieces
from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter
from src.shared.python.model_appearance import document_from_dict
from src.shared.python.model_appearance.body_skin_fit import ASSET_DIR

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.unit

SETTINGS = pieces.load_piece_settings()
MAJORITY = dataclasses.replace(SETTINGS, border_method="neighbour_majority")
COLUMNS, ROWS, BEND_RAD = 6, 2, 0.7
SPEC = Path(__file__).resolve().parents[4] / (
    "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
# The drawn surface is compared in MuJoCo's float32 scene: 1e-6 of a metre (and
# of the strip's unit) is above its rounding and far below anything visible.
SURFACE_TOL, AREA_RTOL, NEAREST = 1e-6, 1e-6, 24
# Measured 2026-11-20 on the skinned golfer (12 closed borders over 3 skins):
# neighbour_majority turns 487.2 rad over 8.59 m of border, iso_line 152.7 rad
# over 5.68 m; twelve flat closed loops cannot turn less than 75.4 rad.
TURNING_RATIO, LENGTH_RATIO = 0.4, 0.75


def _strip(seam_at: int | None = None):
    """A flat strip of quads along x, two triangles each; an unwelded seam on demand."""
    grid = [
        (float(i), float(j), 0.0) for i in range(COLUMNS + 1) for j in range(ROWS + 1)
    ]
    vertices, faces = np.array(grid), []
    for i in range(COLUMNS):
        for j in range(ROWS):
            a, b = i * (ROWS + 1) + j, (i + 1) * (ROWS + 1) + j
            faces += [(a, b, a + 1), (b, b + 1, a + 1)]
    faces = np.array(faces)
    if seam_at is not None:  # the faces right of the seam get their own copies
        column = np.flatnonzero(vertices[:, 0] == seam_at)
        right = (vertices[faces].mean(axis=1)[:, 0] > seam_at)[:, None]
        copies = {int(v): len(vertices) + n for n, v in enumerate(column)}
        moved = np.vectorize(lambda v: copies.get(int(v), int(v)))(faces)
        faces = np.where(right, moved, faces)
        vertices = np.vstack([vertices, vertices[column]])
    return vertices, faces


def _xml(texture: bool = False, shift: float = 0.0) -> str:
    vertices, faces = _strip()
    far = np.clip(vertices[:, 0] / COLUMNS + shift, 0.0, 1.0)
    ids = " ".join(str(i) for i in range(len(vertices)))
    bones = "".join(
        f'<bone body="{body}" bindpos="{x} 0 0" bindquat="1 0 0 0" vertid="{ids}"'
        f' vertweight="{" ".join(format(w, ".6g") for w in share)}"/>'
        for body, x, share in (("near", 0, 1.0 - far), ("far", COLUMNS / 2, far))
    )
    asset = '<material name="cloth" rgba="0.2 0.4 0.6 1"'
    if texture:
        asset = (
            '<texture name="weave" type="2d" builtin="flat" rgb1="1 1 1" width="4"'
            f' height="4"/>{asset} texture="weave" texrepeat="2 2"'
        )
    return (
        f"<mujoco><asset>{asset}/></asset><worldbody>"
        '<body name="near"><geom size="0.1"/>'
        f'<body name="far" pos="{COLUMNS / 2} 0 0"><joint name="bend" axis="0 1 0"/>'
        '<geom size="0.1"/></body></body></worldbody><deformable>'
        f'<skin name="sleeve" material="cloth" group="2"'
        f' vertex="{" ".join(format(v, "g") for v in vertices.ravel())}"'
        f' face="{" ".join(str(i) for i in faces.ravel())}">{bones}</skin>'
        "</deformable></mujoco>"
    )


def _drawn(xml: str, bend: float = BEND_RAD) -> np.ndarray:
    """Every drawn triangle's corners ``(F, 3, 3)`` with each hinge turned ``bend``."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    hinges = model.jnt_type == mujoco.mjtJoint.mjJNT_HINGE
    data.qpos[model.jnt_qposadr[hinges]] += bend
    mujoco.mj_forward(model, data)
    drawn = [v[f] for v, f in layer.posed_skin_layers(model, data).values()]
    return np.concatenate(drawn)


def _triangles(xml: str) -> np.ndarray:
    """:func:`_drawn` as rows in a fixed order."""
    flat = _drawn(xml).reshape(-1, 9)
    return flat[np.lexsort(np.round(flat, 6).T)]


def _areas(triangles: np.ndarray) -> np.ndarray:
    a, b, c = triangles[:, 0], triangles[:, 1], triangles[:, 2]
    return np.linalg.norm(np.cross(b - a, c - a), axis=1) / 2


def _split_only(cut: np.ndarray, whole: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The ``cut`` triangles not drawn as in ``whole``, and those they replace."""
    gap, twin = cKDTree(whole.reshape(-1, 9)).query(cut.reshape(-1, 9))
    same = gap <= SURFACE_TOL
    return cut[~same], np.delete(whole, twin[same], axis=0)


def _off_surface(cut: np.ndarray, whole: np.ndarray) -> float:
    """Farthest any corner or centre of the ``cut`` triangles is off ``whole``.

    With equal total areas, zero means the two draw the same surface. A point
    is on a triangle when its foot in the triangle's plane is inside it; the
    ``NEAREST`` triangles by centre are tried.
    """
    points = np.concatenate([cut.reshape(-1, 3), cut.mean(axis=1)])
    _, near = cKDTree(whole.mean(axis=1)).query(points, k=min(NEAREST, len(whole)))
    a, b, c = (whole[near][:, :, k] for k in range(3))
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    height = ((points[:, None] - a) * normal).sum(axis=-1)
    foot = points[:, None] - height[..., None] * normal
    inside = np.ones(height.shape, bool)
    for start, end in ((a, b), (b, c), (c, a)):
        side = (np.cross(end - start, foot - start) * normal).sum(axis=-1)
        inside &= side >= -SURFACE_TOL * np.linalg.norm(end - start, axis=-1)
    return float(np.where(inside, np.abs(height), np.inf).min(axis=1).max())


def _borders(xml: str, settings) -> tuple[float, float, int]:
    """Turning angle (rad), length and corner count of the borders of a cut.

    A border edge joins two pieces of one skin; vertices are joined by place.
    The turning angle is summed over the corners where two border edges meet,
    so a saw-tooth scores its every tooth and a straight line nothing.
    """
    cut, _ = pieces.split_mjcf_skins(xml, settings)
    owners, places = defaultdict(set), {}
    for piece in ET.fromstring(cut).iter("skin"):
        vertices = np.array(piece.get("vertex").split(), float).reshape(-1, 3)
        faces = np.array(piece.get("face").split(), int).reshape(-1, 3)
        skin = piece.get("name").split("__seg")[0]
        keys = [(skin, *np.round(v, SETTINGS.weld_decimals)) for v in vertices]
        places.update(zip(keys, vertices, strict=True))
        for face in faces:
            for i, j in ((0, 1), (1, 2), (2, 0)):
                owners[frozenset((keys[face[i]], keys[face[j]]))].add(piece.get("name"))
    meets, length = defaultdict(list), 0.0
    for edge in (e for e, names in owners.items() if len(names) == 2):
        a, b = tuple(edge)
        meets[a].append(b)
        meets[b].append(a)
        length += float(np.linalg.norm(places[a] - places[b]))
    turning = 0.0
    for corner, (a, b) in ((c, m) for c, m in meets.items() if len(m) == 2):
        before, after = places[corner] - places[a], places[b] - places[corner]
        turning += float(
            np.arctan2(np.linalg.norm(np.cross(before, after)), before @ after)
        )
    return turning, length, len(meets)


@pytest.fixture(scope="module")
def golfer() -> str:
    """The repo's skinned golfer as MJCF (three skins: skin, shirt, trousers)."""
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    spec["subject"]["body_visual_backend"] = "skinned_cc0"
    document = document_from_dict({"schema_version": "appearance-v1"})
    payload = json.dumps(spec).encode()
    xml, _ = exporter.export_full_body_mjcf(payload, appearance=document)
    return xml


def test_settings_come_from_the_json_file_and_are_checked():
    raw = json.loads((ASSET_DIR / "piece_settings.json").read_text())
    assert SETTINGS.border_rounds == raw["border_rounds"]
    assert SETTINGS.border_votes == raw["border_votes"]
    assert set(raw["sources"]) == set(raw) - {"sources"}
    with pytest.raises(ValueError, match="majority"):
        dataclasses.replace(SETTINGS, border_votes=1)
    with pytest.raises(ValueError, match=">= 0"):
        dataclasses.replace(SETTINGS, border_rounds=-1)
    with pytest.raises(TypeError, match="integer"):
        dataclasses.replace(SETTINGS, weld_decimals=1.5)
    assert SETTINGS.border_method == raw["border_method"] == "iso_line"
    with pytest.raises(ValueError, match="border_method"):
        dataclasses.replace(SETTINGS, border_method="by_eye")
    for fraction in (0.0, 0.5):
        with pytest.raises(ValueError, match="snap fraction"):
            dataclasses.replace(SETTINGS, cut_snap_fraction=fraction)
    with pytest.raises(TypeError, match="integer"):
        dataclasses.replace(SETTINGS, cut_smooth_rounds=1.0)


def test_a_face_goes_to_the_body_holding_most_of_its_weight():
    faces = np.array([[0, 1, 2], [1, 2, 3]])
    weights = np.array([[1.0, 0.0], [0.6, 0.4], [0.6, 0.4], [0.0, 1.0]])
    assert pieces.face_segments(faces, weights).tolist() == [0, 1]
    with pytest.raises(ValueError, match="vertex"):
        pieces.face_segments(np.array([[0, 1, 9]]), weights)


def test_a_stray_face_takes_its_neighbours_label_and_a_straight_border_stays():
    vertices, faces = _strip()
    side = (vertices[faces].mean(axis=1)[:, 0] > COLUMNS / 2).astype(np.int64)
    assert pieces.straighten_borders(vertices, faces, side, SETTINGS).tolist() == (
        side.tolist()
    )
    stray = side.copy()
    inner = int(np.flatnonzero(side == 0)[ROWS + 1])  # a face with three neighbours
    stray[inner] = 1
    assert pieces.straighten_borders(vertices, faces, stray, SETTINGS).tolist() == (
        side.tolist()
    )
    none = dataclasses.replace(SETTINGS, border_rounds=0)
    assert pieces.straighten_borders(vertices, faces, stray, none).tolist() == (
        stray.tolist()
    )
    with pytest.raises(ValueError, match="labels"):
        pieces.straighten_borders(vertices, faces, side[:-1], SETTINGS)


def test_vertices_doubled_along_a_seam_do_not_make_a_border():
    whole, _ = _strip()
    vertices, faces = _strip(seam_at=1)
    assert len(vertices) == len(whole) + ROWS + 1
    # the top face on the seam: one neighbour across it, one beside it, no third
    centre = vertices[faces].mean(axis=1)
    stray = np.flatnonzero(
        (centre[:, 0] > 0.5) & (centre[:, 0] < 1) & (centre[:, 1] > ROWS - 1)
    )
    assert len(stray) == 1
    labels = np.zeros(len(faces), np.int64)
    labels[stray] = 1
    assert not pieces.straighten_borders(vertices, faces, labels, SETTINGS).any()
    apart = vertices.copy()
    apart[len(whole) :, 2] += 1.0  # the copies moved off: the seam is a real border
    assert pieces.straighten_borders(apart, faces, labels, SETTINGS)[stray] == 1


def test_the_cut_keeps_the_drawn_surface_and_gives_each_piece_a_material():
    xml = _xml(texture=True)
    cut, table = pieces.split_mjcf_skins(xml, MAJORITY)
    assert table == {"sleeve__seg0": "far", "sleeve__seg1": "near"}
    assert np.allclose(_triangles(cut), _triangles(xml), atol=1e-6)
    root = ET.fromstring(cut)
    skins = {s.get("name"): s for s in root.iter("skin")}
    assert set(skins) == set(table)
    assert {s.get("group") for s in skins.values()} == {"2"}
    materials = {m.get("name"): m for m in root.iter("material")}
    owned = [s.get("material") for s in skins.values()]
    assert owned == ["cloth__seg__sleeve__0", "cloth__seg__sleeve__1"]
    for name in owned:
        assert materials[name].get("rgba") == "0.2 0.4 0.6 1"
        assert not set(SETTINGS.dropped_material_keys) & set(materials[name].keys())
    assert materials["cloth"].get("texture") == "weave"  # the source material stays
    model = mujoco.MjModel.from_xml_string(cut)
    assert model.nskin == len(table) and len(set(model.skin_matid)) == len(table)


def test_a_skin_that_cannot_be_cut_is_rejected():
    with pytest.raises(ValueError, match="no material"):
        pieces.split_mjcf_skins(_xml().replace(' material="cloth"', ""))
    with pytest.raises(ValueError, match="off its vertices"):
        pieces.split_mjcf_skins(_xml().replace('vertid="0 ', 'vertid="999 '))
    with pytest.raises(ValueError, match="lacks 'face'"):
        pieces.split_mjcf_skins(_xml().replace(" face=", " faces="))


@pytest.mark.parametrize("bend", [0.0, BEND_RAD])
def test_the_iso_line_cut_is_one_straight_line_on_the_same_drawn_surface(bend):
    xml = _xml(shift=0.07)  # the weights are equal at x = 2.58, inside the faces
    plain = dataclasses.replace(SETTINGS, cut_smooth_rounds=0)
    cut, table = pieces.split_mjcf_skins(xml, plain)
    assert table == {"sleeve__seg0": "far", "sleeve__seg1": "near"}
    rest = [
        np.array(s.get("vertex").split(), float).reshape(-1, 3)
        for s in ET.fromstring(cut).iter("skin")
    ]
    shared = [v for v in rest[0] if (np.abs(rest[1] - v).sum(axis=1) < 1e-9).any()]
    # one crossing on each of the 2 * ROWS + 1 edges the line meets, all on it
    assert len(shared) == 2 * ROWS + 1
    assert np.allclose(np.array(shared)[:, 0], COLUMNS * (0.5 - 0.07), atol=1e-9)
    parts, split = _split_only(_drawn(cut, bend), _drawn(xml, bend))
    assert len(split) == 2 * ROWS and len(parts) == 3 * len(split)
    assert _off_surface(parts, split) <= SURFACE_TOL
    assert _areas(parts).sum() == pytest.approx(_areas(split).sum(), rel=AREA_RTOL)
    assert _areas(parts).min() >= plain.cut_snap_fraction**2 * _areas(split).min()
    # the old method is still there and still cuts along the faces
    old, _ = pieces.split_mjcf_skins(xml, MAJORITY)
    assert np.allclose(_triangles(old), _triangles(xml), atol=1e-6)


def test_a_crossing_near_a_vertex_snaps_to_it_and_three_segments_meet_at_a_centre():
    faces = np.array([[0, 1, 2]])
    corners = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    snap = SETTINGS.cut_snap_fraction
    near = np.array([[0.51, 0.49], [0.0, 1.0], [0.0, 1.0]])  # crosses 2 % along
    labels, cross, added, share = pieces._crossings(faces, near, snap)
    assert cross.tolist() == [[0, -1, 0]] and len(added) == 0
    kept, parts, centred = pieces._cut_faces(corners, faces, labels, cross)
    assert kept.tolist() == [[1, 2, 0]] and parts.tolist() == [1] and not len(centred)
    labels, cross, added, share = pieces._crossings(faces, np.eye(3), snap)
    assert sorted(cross[0]) == [3, 4, 5] and np.allclose(share[:, :2], 0.5)
    mids = np.einsum("mk,mkx->mx", share, corners[added])
    kept, parts, centred = pieces._cut_faces(
        np.vstack([corners, mids]), faces, labels, cross
    )
    assert centred.tolist() == [[0, 1, 2]] and sorted(parts) == [0, 0, 1, 1, 2, 2]
    points = np.vstack([corners, mids, corners.mean(axis=0)])
    assert np.allclose(_areas(points[kept]), 1 / 12)  # six equal parts, none lost


def test_the_golfers_borders_are_much_less_ragged_along_the_iso_line(golfer):
    old_turn, old_length, old_corners = _borders(golfer, MAJORITY)
    turn, length, corners = _borders(golfer, SETTINGS)
    assert old_corners > 0 and corners > 0
    assert turn < TURNING_RATIO * old_turn, (turn, old_turn)
    assert length < LENGTH_RATIO * old_length, (length, old_length)


@pytest.mark.parametrize("bend", [0.0, BEND_RAD])
def test_the_iso_line_cut_keeps_the_golfers_drawn_surface(golfer, bend):
    cut, table = pieces.split_mjcf_skins(golfer)
    assert table == pieces.split_mjcf_skins(golfer, MAJORITY)[1]  # the same pieces
    # every face the cut left alone is drawn as it was; the rest are compared
    parts, split = _split_only(_drawn(cut, bend), _drawn(golfer, bend))
    assert len(parts) > len(split) > 0
    assert _off_surface(parts, split) <= SURFACE_TOL
    assert _areas(parts).sum() == pytest.approx(_areas(split).sum(), rel=AREA_RTOL)
    floor = SETTINGS.cut_snap_fraction**2 * _areas(split).min()
    assert _areas(parts).min() >= floor > 0.0
