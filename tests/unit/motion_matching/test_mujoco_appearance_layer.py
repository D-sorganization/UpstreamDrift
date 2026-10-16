"""MuJoCo appearance layer: purely visual, physics bit-identical."""

import json
from pathlib import Path

import numpy as np
import pytest
from defusedxml import ElementTree as ET

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter
from src.shared.python.model_appearance import (
    document_from_dict,
    physics_spec_sha256,
)

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
FULL_BODY = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
DOC = {"schema_version": "appearance-v1"}


def _models(mujoco):
    raw = FULL_BODY.read_bytes()
    appearance = document_from_dict(DOC)
    exports = {
        "plain": exporter.export_full_body_mjcf(raw),
        "capsules": exporter.export_full_body_mjcf(raw, visual=True),
        "appearance": exporter.export_full_body_mjcf(raw, appearance=appearance),
    }
    return {
        k: (mujoco.MjModel.from_xml_string(x), meta) for k, (x, meta) in exports.items()
    }


def _rollout(mujoco, model, steps: int = 12):
    data = mujoco.MjData(model)
    rng = np.random.default_rng(7)
    data.qpos[:] = rng.normal(scale=0.02, size=model.nq)
    data.qvel[:] = rng.normal(scale=0.02, size=model.nv)
    trace = []
    for _ in range(steps):
        mujoco.mj_step(model, data)
        trace.append((data.qpos.copy(), data.qvel.copy(), data.qacc.copy()))
    assert all(np.isfinite(a).all() for step in trace for a in step)
    return trace


def test_physics_is_bit_identical_with_and_without_appearance() -> None:
    mujoco = pytest.importorskip("mujoco")
    models = _models(mujoco)
    plain = models["plain"][0]
    for key in ("capsules", "appearance"):
        other = models[key][0]
        assert other.nbody == plain.nbody and other.nq == plain.nq
        for attr in (
            "body_mass",
            "body_inertia",
            "body_ipos",
            "body_iquat",
            "body_pos",
            "body_quat",
            "dof_armature",
            "dof_damping",
            "jnt_axis",
            "jnt_pos",
            "opt_gravity" if hasattr(plain, "opt_gravity") else "body_subtreemass",
        ):
            np.testing.assert_array_equal(
                getattr(other, attr), getattr(plain, attr), err_msg=attr
            )
        for a, b in zip(_rollout(mujoco, plain), _rollout(mujoco, other), strict=True):
            for left, right in zip(a, b, strict=True):
                assert np.array_equal(left, right)  # bit-identical, no tolerance


def test_appearance_geoms_are_visual_only() -> None:
    mujoco = pytest.importorskip("mujoco")
    models = _models(mujoco)
    plain, _ = models["plain"]
    model, meta = models["appearance"]
    layer = meta["visual_layer"]
    assert layer["meshes"] > plain.nbody and layer["garments"] > 0
    assert layer["offscreen"] == [960, 720] and layer["lights"] == 3
    new = [i for i in range(model.ngeom) if model.geom_group[i] == 1]
    assert len(new) >= layer["meshes"]
    for i in new:
        assert model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0
    # Collision geometry (everything outside the visual groups) is unchanged.
    keep = lambda m: [  # noqa: E731
        (m.geom_type[i], tuple(m.geom_size[i]), tuple(m.geom_pos[i]))
        for i in range(m.ngeom)
        if m.geom_group[i] not in (1, 3)
    ]
    assert keep(model) == keep(plain)
    assert model.nmesh == layer["meshes"] and model.nmat >= 5 and model.ntex >= 3
    assert "visual_layer" not in models["plain"][1]


def test_appearance_does_not_change_the_spec_hash() -> None:
    raw = FULL_BODY.read_bytes()
    spec = json.loads(raw)
    _, meta = exporter.export_full_body_mjcf(raw, appearance=document_from_dict(DOC))
    _, plain_meta = exporter.export_full_body_mjcf(raw)
    assert meta["model_sha256"] == plain_meta["model_sha256"]
    assert physics_spec_sha256(spec) == physics_spec_sha256(raw)


def test_file_textures_are_rejected_clearly() -> None:
    doc = document_from_dict(
        {
            **DOC,
            "materials": {
                "scan": {
                    "base_color": [1, 1, 1],
                    "texture": {"kind": "file", "path": "scan.png"},
                }
            },
            "segments": [{"match": "*", "material": "scan"}],
        }
    )
    with pytest.raises(ValueError, match="File textures"):
        exporter.export_full_body_mjcf(FULL_BODY.read_bytes(), appearance=doc)


@pytest.mark.requires_gl
def test_appearance_renders_960x720_offscreen() -> None:
    mujoco = pytest.importorskip("mujoco")
    model = _models(mujoco)["appearance"][0]
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    try:
        renderer = mujoco.Renderer(model, 720, 960)
    except (RuntimeError, ValueError, OSError) as error:
        pytest.skip(f"no offscreen GL: {error}")
    try:
        renderer.update_scene(data)
        image = renderer.render()
    finally:
        renderer.close()
    assert image.shape == (720, 960, 3)
    assert image.mean() > 20.0 and image.std() > 10.0


# --- R1 (#12680): trunk and shoe cross-sections in the exported MJCF -----

ANTHRO = (
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json",
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_iron7.json",
)


def _skin_meshes_in_spec_frames(spec_path: Path) -> tuple[dict, dict]:
    """Skin mesh vertices per body, mapped back from MJCF into spec body frames.

    The MJCF body frame is the joint follower frame, so a spec-body point p
    sits at ``inv(child_to_follower) @ p`` in MJCF; invert that here.
    """
    raw = spec_path.read_bytes()
    spec = json.loads(raw)
    xml, _ = exporter.export_full_body_mjcf(raw, appearance=document_from_dict(DOC))
    root = ET.fromstring(xml)
    verts = {
        m.get("name"): np.array(m.get("vertex", "").split(), float).reshape(-1, 3)
        for m in root.iter("mesh")
    }
    follower = {j["child"]: np.asarray(j["child_to_follower"]) for j in spec["joints"]}
    out: dict[str, list[np.ndarray]] = {}
    for body in root.iter("body"):
        name = body.get("name", "")
        for geom in body.findall("geom"):
            mesh = geom.get("mesh") or ""
            if "_skin" not in mesh:
                continue
            offset = np.linalg.inv(follower[name])
            local = (verts[mesh] - offset[:3, 3]) @ offset[:3, :3]
            out.setdefault(name, []).append(local)
    return spec, out


def _mid_ring_extents(
    vertices: np.ndarray, start, end, width: np.ndarray
) -> tuple[float, float]:
    """(width-axis, depth-axis) extent of the ring nearest the segment middle."""
    a = np.subtract(end, start) / np.linalg.norm(np.subtract(end, start))
    s = (vertices - np.asarray(start)) @ a
    mid = 0.5 * float(np.linalg.norm(np.subtract(end, start)))
    ring = vertices[np.isclose(s, s[np.argmin(np.abs(s - mid))], atol=1e-6)]
    depth = np.cross(a, width)
    return float(np.ptp(ring @ width)), float(np.ptp(ring @ depth))


@pytest.mark.parametrize("spec_path", ANTHRO, ids=lambda p: p.stem)
def test_trunk_and_shoe_cross_sections_are_wider_than_deep(spec_path) -> None:
    from src.shared.python.model_appearance.cross_section import body_sections
    from src.shared.python.model_appearance.library import PART_ASPECT, classify_body
    from src.shared.python.motion_matching.visual_skeleton import (
        derive_visual_skeleton,
    )

    spec, meshes = _skin_meshes_in_spec_frames(spec_path)
    sections = body_sections(spec)
    by_body: dict[str, list] = {}
    for cap in derive_visual_skeleton(spec).capsules:
        by_body.setdefault(cap.body, []).append(cap)
    seen: set[str] = set()
    for body, caps in by_body.items():
        part = classify_body(body)
        if part not in ("torso", "pelvis", "foot") or body not in meshes:
            continue
        ml = sections[body].ml_axis
        for cap, verts in zip(caps, meshes[body], strict=True):
            axis = np.subtract(cap.end_m, cap.start_m)
            if cap.length_m() < 0.05 or abs(np.dot(axis, ml)) > 0.1 * np.linalg.norm(
                axis
            ):
                continue  # short stub, or a lobe that runs along ML (to a hip)
            width_ext, depth_ext = _mid_ring_extents(verts, cap.start_m, cap.end_m, ml)
            assert width_ext > depth_ext, (body, width_ext, depth_ext)
            assert depth_ext / width_ext == pytest.approx(
                PART_ASPECT[part], abs=0.02
            ), body
            seen.add(part if part != "foot" else body)
    assert {"calcn_r", "calcn_l"} <= seen and ({"torso", "pelvis"} & seen)
    pelvis = next(b for b in by_body if b.endswith("LowerTorso"))
    pts = np.vstack(meshes[pelvis])
    depth = sections[pelvis].depth_axis
    assert np.ptp(pts @ sections[pelvis].ml_axis) > np.ptp(pts @ depth)
