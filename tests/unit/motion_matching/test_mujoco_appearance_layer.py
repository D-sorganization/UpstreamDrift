"""MuJoCo appearance layer: purely visual, physics bit-identical."""

import json
from pathlib import Path

import numpy as np
import pytest

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
