"""Articulated pelvis, abdomen and thorax visuals for the v2 trunk (HRG-D3, #13174).

full-body-v2 splits the v1 trunk into ``pelvis -> trunk_lumbar -> thorax``.
Each segment carries its own de Leva solid (pelvis, middle_trunk,
upper_trunk) and its own ``visual_hints`` ellipsoid, so the visual layer
draws three rigid trunk visuals that follow the lumbar and thoracic joints.

Model-level checks (the stills need a GL host, see
``tests/unit/scripts/test_render_v2_trunk_stills.py``):

* every trunk body gets its own visual geom on the right MJCF body;
* the visuals leave the compiled physics unchanged;
* the seams neither open a gap nor let the pelvis and thorax interpenetrate
  over sampled soft-ROM poses.
"""

from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.model_appearance.library import classify_body
from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching import full_body_v2 as v2
from src.shared.python.motion_matching.full_body_v2_geometry import (
    EXPORT_STATUS,
    derive_landmark_v2_spec,
)
from src.shared.python.motion_matching.visual_skeleton import (
    Shape,
    derive_visual_skeleton,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SPECS = (
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json",
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_iron7.json",
)
#: de Leva solid drawn by each trunk role (solid-name suffix).
ROLE_SOLID = {"pelvis": "pelvis", "lumbar": "middle_trunk", "thorax": "upper_trunk"}
SEED = 13174
N_RANDOM_POSES = 64


@pytest.fixture(scope="module", params=SPECS, ids=lambda p: p.stem)
def v2_doc(request: pytest.FixtureRequest) -> dict[str, Any]:
    v1 = json.loads(request.param.read_text(encoding="utf-8"))
    return derive_landmark_v2_spec(v1, provenance="HRG-D3 trunk visual fixture")


@pytest.fixture(scope="module")
def v2_bytes(v2_doc: dict[str, Any]) -> bytes:
    return json.dumps(v2_doc, sort_keys=True).encode("utf-8")


def _trunk_shapes(doc: dict[str, Any]) -> dict[str, Shape]:
    """The one hint ellipsoid of each trunk role, keyed by role."""
    roles = v2.trunk_bodies(doc)
    shapes = derive_visual_skeleton(doc).shapes
    out = {}
    for role, body in roles.items():
        own = [s for s in shapes if s.body == body]
        assert len(own) == 1, f"{role} ({body}) must carry exactly one trunk shape"
        out[role] = own[0]
    return out


def test_each_trunk_segment_has_its_own_shape_from_its_own_solid(v2_doc) -> None:
    shapes = _trunk_shapes(v2_doc)
    assert set(shapes) == set(ROLE_SOLID)
    for role, shape in shapes.items():
        assert shape.kind == "ellipsoid"
        assert shape.label.endswith("/" + ROLE_SOLID[role])


def test_abdomen_ellipsoid_sits_on_its_solid_com(v2_doc) -> None:
    """The reparented middle_trunk hint is in the trunk_lumbar frame."""
    body = next(b for b in v2_doc["bodies"] if b["name"] == v2.LUMBAR_BODY)
    (solid,) = body["solids"]
    placement = np.asarray(solid["placement"], dtype=float)
    com = placement[:3, :3] @ np.asarray(solid["com_m"]) + placement[:3, 3]
    shape = _trunk_shapes(v2_doc)["lumbar"]
    # The hint centre is a design offset from the COM, inside the ellipsoid.
    offset = (com - np.asarray(shape.center_m)) / np.asarray(shape.half_size_m)
    assert float(np.sum(offset**2)) < 0.25


def test_abdomen_dresses_as_torso() -> None:
    """trunk_lumbar wears the shirt like the thorax, not an unclothed 'other'."""
    assert classify_body(v2.LUMBAR_BODY) == "torso"


def _rom_samples(doc: dict[str, Any]) -> list[dict[str, float]]:
    """Every corner of the v2 trunk soft-ROM box plus seeded interior poses."""
    ranges = v2.v2_trunk_ranges_deg()
    names = list(ranges)
    rng = np.random.default_rng(SEED)
    poses = [
        dict(zip(names, c, strict=True))
        for c in itertools.product(*(ranges[n] for n in names))
    ]
    poses += [
        {n: float(rng.uniform(*ranges[n])) for n in names}
        for _ in range(N_RANDOM_POSES)
    ]
    out = []
    for pose in poses:
        q = dict.fromkeys(doc["coordinate_order"], 0.0)
        q.update({n: float(np.radians(v)) for n, v in pose.items()})
        out.append(q)
    return out


def _ellipsoid_level(points: np.ndarray, shape: Shape, pose: np.ndarray) -> np.ndarray:
    """Normalised level of world ``points`` in a body ellipsoid (< 1 inside)."""
    local = (points - pose[:3, 3]) @ pose[:3, :3]
    scaled = (local - np.asarray(shape.center_m)) / np.asarray(shape.half_size_m)
    return np.sum(scaled**2, axis=-1)


def _surface(shape: Shape, pose: np.ndarray, n: int = 2000) -> np.ndarray:
    rng = np.random.default_rng(SEED)
    unit = rng.normal(size=(n, 3))
    unit /= np.linalg.norm(unit, axis=1)[:, None]
    local = np.asarray(shape.center_m) + unit * np.asarray(shape.half_size_m)
    return local @ pose[:3, :3].T + pose[:3, 3]


def _seam_joints(doc: dict[str, Any]) -> list[dict[str, Any]]:
    roles = v2.trunk_bodies(doc)
    trunk = {roles["pelvis"], roles["lumbar"], roles["thorax"]}
    joints = [j for j in doc["joints"] if j["parent"] in trunk and j["child"] in trunk]
    assert len(joints) == 2
    return joints


def test_seams_have_no_gap_over_sampled_rom(v2_doc) -> None:
    """Each regional joint centre lies inside both adjacent segment visuals."""
    by_body = {
        body: _trunk_shapes(v2_doc)[role]
        for role, body in v2.trunk_bodies(v2_doc).items()
    }
    joints = _seam_joints(v2_doc)
    for q in _rom_samples(v2_doc):
        poses = spec_body_poses(v2_doc, q)
        for joint in joints:
            parent, child = joint["parent"], joint["child"]
            centre = (
                poses[parent]
                @ np.append(np.asarray(joint["parent_to_base"])[:3, 3], 1.0)
            )[:3]
            from_child = (
                poses[child]
                @ np.append(np.asarray(joint["child_to_follower"])[:3, 3], 1.0)
            )[:3]
            np.testing.assert_allclose(from_child, centre, atol=1e-9)
            for body in (parent, child):
                level = _ellipsoid_level(centre[None], by_body[body], poses[body])
                assert level[0] < 1.0, f"gap at {joint['name']} on {body}"


def test_segments_do_not_telescope_or_interpenetrate_over_sampled_rom(
    v2_doc,
) -> None:
    """Adjacent visuals overlap only at the seam; pelvis and thorax never touch."""
    shapes = _trunk_shapes(v2_doc)
    roles = v2.trunk_bodies(v2_doc)
    for q in _rom_samples(v2_doc):
        poses = spec_body_poses(v2_doc, q)
        world = {r: poses[b] for r, b in roles.items()}
        for parent, child in (("pelvis", "lumbar"), ("lumbar", "thorax")):
            centre = world[child] @ np.append(shapes[child].center_m, 1.0)
            level = _ellipsoid_level(centre[None, :3], shapes[parent], world[parent])
            assert level[0] > 1.0, f"{child} visual telescopes into {parent}"
        thorax_surface = _surface(shapes["thorax"], world["thorax"])
        level = _ellipsoid_level(thorax_surface, shapes["pelvis"], world["pelvis"])
        assert level.min() > 1.0, "thorax visual interpenetrates the pelvis"


def test_mujoco_export_status_names_the_visual_layer() -> None:
    assert "HRG-D3" not in EXPORT_STATUS["mujoco"]["reason"]
    assert "visual" in EXPORT_STATUS["mujoco"]["reason"]


# --- MuJoCo export ---------------------------------------------------------


def _geoms_on(model: Any, mujoco: Any, body: str) -> list[tuple[str, int]]:
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    assert bid >= 0, body
    return [
        (
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "",
            int(model.geom_type[g]),
        )
        for g in range(model.ngeom)
        if model.geom_bodyid[g] == bid
    ]


def test_mujoco_visual_attaches_one_trunk_ellipsoid_per_segment(
    v2_doc, v2_bytes
) -> None:
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    xml, meta = export_full_body_mjcf(v2_bytes, visual=True)
    assert "visual_layer" in meta
    model = mujoco.MjModel.from_xml_string(xml)
    for role, body in v2.trunk_bodies(v2_doc).items():
        geoms = _geoms_on(model, mujoco, body)
        ellipsoids = [n for n, t in geoms if t == mujoco.mjtGeom.mjGEOM_ELLIPSOID]
        assert ellipsoids == [n for n in ellipsoids if n.endswith("_" + body)]
        assert len(ellipsoids) == 1, f"{role} needs exactly one trunk ellipsoid"
        assert any(t == mujoco.mjtGeom.mjGEOM_CAPSULE for _, t in geoms), role


def test_mujoco_appearance_skins_and_dresses_each_trunk_segment(
    v2_doc, v2_bytes
) -> None:
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.model_appearance.schema import AppearanceDocument

    xml, _ = export_full_body_mjcf(v2_bytes, appearance=AppearanceDocument())
    model = mujoco.MjModel.from_xml_string(xml)
    for role, body in v2.trunk_bodies(v2_doc).items():
        meshes = [
            n
            for n, t in _geoms_on(model, mujoco, body)
            if t == mujoco.mjtGeom.mjGEOM_MESH
        ]
        assert any("skin" in n for n in meshes), f"{role} has no skin mesh"
        assert any("garment" in n for n in meshes), f"{role} is undressed"


_PHYSICS_FIELDS = (
    "body_parentid",
    "body_pos",
    "body_quat",
    "body_ipos",
    "body_iquat",
    "body_mass",
    "body_inertia",
    "jnt_type",
    "jnt_bodyid",
    "jnt_pos",
    "jnt_axis",
    "jnt_range",
    "dof_damping",
    "dof_armature",
    "eq_type",
    "eq_data",
)


def _physics_signature(model: Any) -> dict[str, np.ndarray]:
    out = {name: np.array(getattr(model, name), copy=True) for name in _PHYSICS_FIELDS}
    out["gravity"] = np.array(model.opt.gravity, copy=True)
    return out


@pytest.mark.parametrize("mode", ["visual", "appearance"])
def test_trunk_visuals_leave_compiled_physics_unchanged(v2_doc, v2_bytes, mode) -> None:
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.model_appearance.schema import AppearanceDocument

    plain_xml, _ = export_full_body_mjcf(v2_bytes)
    kwargs: dict[str, Any] = (
        {"visual": True} if mode == "visual" else {"appearance": AppearanceDocument()}
    )
    visual_xml, _ = export_full_body_mjcf(v2_bytes, **kwargs)
    plain = mujoco.MjModel.from_xml_string(plain_xml)
    visual = mujoco.MjModel.from_xml_string(visual_xml)
    assert (plain.nq, plain.nv, plain.nbody, plain.njnt) == (
        visual.nq,
        visual.nv,
        visual.nbody,
        visual.njnt,
    )
    a, b = _physics_signature(plain), _physics_signature(visual)
    for key in a:
        np.testing.assert_array_equal(a[key], b[key], err_msg=key)

    rng = np.random.default_rng(SEED)
    datas = [mujoco.MjData(plain), mujoco.MjData(visual)]
    qpos = rng.uniform(-0.3, 0.3, size=plain.nq)
    qvel = rng.uniform(-1.0, 1.0, size=plain.nv)
    for model, data in zip((plain, visual), datas, strict=True):
        data.qpos[:] = qpos
        data.qvel[:] = qvel
        mujoco.mj_forward(model, data)
    np.testing.assert_array_equal(datas[0].xpos, datas[1].xpos)
    np.testing.assert_array_equal(datas[0].qacc, datas[1].qacc)


# --- Still render CLI (GL call mocked; the stills themselves need a GL host) ----


def test_still_render_cli_accepts_v2_spec(v2_bytes, tmp_path: Path) -> None:
    """The showcase CLI takes a full-body-v2 spec for the address/top/impact stills."""
    from scripts import render_golfer_realism_showcase as showcase

    spec = tmp_path / "full_body_v2.json"
    spec.write_bytes(v2_bytes)
    argv = ["--swing", "driver=bundle.npz", "--v2-spec", str(spec)]
    args = showcase.build_parser().parse_args(argv)
    assert args.v2_spec == spec
    assert showcase.build_parser().parse_args(["--swing", "d=b.npz"]).v2_spec is None
