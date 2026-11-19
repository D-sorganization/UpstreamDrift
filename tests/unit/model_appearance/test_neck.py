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


def _head_skin() -> np.ndarray:
    from src.shared.python.model_appearance.head import build_head_parts

    parts = {p.name: p for p in build_head_parts(HEAD_M, HeadSettings())}
    return parts["skin_mesh"].mesh.vertices


def test_the_neck_section_is_round_not_boxy() -> None:
    rings = _neck_part().vertices[:-2].reshape(-1, 36, 3)
    for ring in rings:
        x, y = ring[:, 0], ring[:, 1]
        area = 0.5 * abs(float(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))
        assert area / (np.ptp(x) * np.ptp(y)) <= ACCEPT.max_neck_fill


def test_the_neck_is_narrower_than_the_head_at_mid_height() -> None:
    width, _ = _section_extent(_neck_part(), ACCEPT.neck_mid_height)
    head = np.ptp(_head_skin()[:, 1]) / HEAD_M
    low, high = ACCEPT.neck_over_head_width
    assert low <= width / head <= high, (width, head)


def test_the_neck_enters_the_head_without_a_step_at_the_jaw() -> None:
    skin, neck_v = _head_skin(), _neck_part().vertices[:-2]
    for z in np.unique(np.round(neck_v[:, 2], 3)):
        if z < ACCEPT.neck_head_overlap_from * HEAD_M:
            continue
        reach = np.abs(skin[np.abs(skin[:, 2] - z) < 0.006 * HEAD_M * 4, 1]).max()
        past = np.abs(neck_v[np.abs(neck_v[:, 2] - z) < 1e-3, 1]).max() - reach
        assert past <= ACCEPT.max_neck_past_head_m, (z, past)


def test_the_neck_is_one_skin_part() -> None:
    from src.shared.python.model_appearance.head import build_head_parts

    names = [p.name for p in build_head_parts(HEAD_M, HeadSettings())]
    names += [p.name for p in build_neck_parts(HEAD_M, HeadSettings())]
    assert names.count("neck") == 1
