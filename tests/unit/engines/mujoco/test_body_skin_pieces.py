"""One skin piece per segment, same drawn surface, straightened borders (#15150)."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest
from defusedxml import ElementTree as ET

from src.engines.physics_engines.mujoco.python import body_skin_layer as layer
from src.engines.physics_engines.mujoco.python import body_skin_pieces as pieces
from src.shared.python.model_appearance.body_skin_fit import ASSET_DIR

mujoco = pytest.importorskip("mujoco")
pytestmark = pytest.mark.unit

SETTINGS = pieces.load_piece_settings()
COLUMNS, ROWS, BEND_RAD = 6, 2, 0.7


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


def _xml(texture: bool = False) -> str:
    vertices, faces = _strip()
    far = np.clip(vertices[:, 0] / COLUMNS, 0.0, 1.0)
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


def _triangles(xml: str) -> np.ndarray:
    """Every drawn triangle's corner positions at the bent pose, in a fixed order."""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    data.qpos[0] = BEND_RAD
    mujoco.mj_forward(model, data)
    drawn = [v[f] for v, f in layer.posed_skin_layers(model, data).values()]
    flat = np.concatenate(drawn).reshape(-1, 9)
    return flat[np.lexsort(np.round(flat, 6).T)]


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
    cut, table = pieces.split_mjcf_skins(xml)
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
