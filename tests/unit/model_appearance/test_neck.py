"""The default golfer has a neck and a trapezius slope below the jaw (#15292)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import neck
from src.shared.python.model_appearance.head import build_neck_parts
from src.shared.python.model_appearance.defaults import default_document_dict
from src.shared.python.model_appearance.schema import HeadSettings

pytestmark = pytest.mark.unit

MODELS = Path(__file__).resolve().parents[3] / "docs/development/full_body_models"
SPEC_BYTES = (MODELS / "full_body_spec_anthro_driver.json").read_bytes()
HEAD_M = 0.2429
ACCEPT = neck.NECK_ACCEPTANCE


def _scene(appearance: dict):
    mujoco = pytest.importorskip("mujoco")
    from src.tools.native_viewer_export.backends.mujoco_worker import build_scene_xml

    xml, _ = build_scene_xml(SPEC_BYTES, appearance)
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
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
    out: dict[str, float] = {"trunk": -9.0, "yoke": -9.0, "chin": 9.0}
    zone = ACCEPT.neck_zone_half_width_head_fraction * HEAD_M
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
            if key:  # the yoke counts where it meets the neck, not out at the cap
                near = local[np.abs(local[:, 1]) <= zone] if key == "yoke" else local
                if len(near):
                    out[key] = max(out[key], float(near[:, 2].max()))
    return out


def test_the_neck_is_visible_between_the_trunk_and_the_chin() -> None:
    h = _heights()
    visible = h["chin"] - h["trunk"]
    assert visible >= ACCEPT.min_visible_neck_head_fraction * HEAD_M, h


def test_no_trunk_yoke_or_cap_rises_to_the_chin() -> None:
    h = _heights()
    margin = ACCEPT.min_cap_below_chin_head_fraction * HEAD_M
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
