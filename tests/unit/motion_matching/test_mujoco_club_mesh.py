"""MuJoCo club head is the shared mesh: visual only, massless, non-colliding."""

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter
from src.shared.python.model_appearance import document_from_dict

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SPECS = {
    "driver": ROOT
    / "docs/development/full_body_models/full_body_spec_anthro_driver.json",
    "iron7": ROOT
    / "docs/development/full_body_models/full_body_spec_anthro_iron7.json",
}
DOC = document_from_dict({"schema_version": "appearance-v1"})


def _pair(mujoco, key):
    raw = SPECS[key].read_bytes()
    plain = mujoco.MjModel.from_xml_string(exporter.export_full_body_mjcf(raw)[0])
    xml, meta = exporter.export_full_body_mjcf(raw, appearance=DOC)
    return plain, mujoco.MjModel.from_xml_string(xml), meta["visual_layer"]


@pytest.mark.parametrize("key", sorted(SPECS))
def test_club_meshes_replace_the_ellipsoid_and_keep_physics(key) -> None:
    mujoco = pytest.importorskip("mujoco")
    plain, model, layer = _pair(mujoco, key)
    assert layer["club_head"] == key
    names = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i)
        for i in range(model.nmesh)
    }
    club = {n for n in names if n.endswith(("_head", "_shaft", "_grip"))}
    assert len(club) == 3
    head_id = next(
        i
        for i in range(model.nmesh)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i) or "").endswith(
            "_head"
        )
    )
    assert (
        model.mesh_vertnum[head_id] > 500
    )  # a real head, not a 12x20 ellipsoid (242 vertices)
    for i in range(model.ngeom):
        if model.geom_type[i] == mujoco.mjtGeom.mjGEOM_MESH:
            assert model.geom_contype[i] == 0 and model.geom_conaffinity[i] == 0
            assert model.geom_group[i] == 1
    np.testing.assert_array_equal(model.body_mass, plain.body_mass)
    np.testing.assert_array_equal(model.body_inertia, plain.body_inertia)
    d_plain, d_vis = mujoco.MjData(plain), mujoco.MjData(model)
    q = np.random.default_rng(3).normal(scale=0.05, size=plain.nq)
    d_plain.qpos[:] = q
    d_vis.qpos[:] = q
    mujoco.mj_forward(plain, d_plain)
    mujoco.mj_forward(model, d_vis)
    np.testing.assert_array_equal(d_vis.qacc, d_plain.qacc)
    np.testing.assert_array_equal(d_vis.xpos, d_plain.xpos)


def test_head_vertex_extent_matches_the_club() -> None:
    mujoco = pytest.importorskip("mujoco")
    _, model, _ = _pair(mujoco, "driver")
    head_id = next(
        i
        for i in range(model.nmesh)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i) or "").endswith(
            "_head"
        )
    )
    adr, num = model.mesh_vertadr[head_id], model.mesh_vertnum[head_id]
    extent = np.ptp(model.mesh_vert[adr : adr + num], axis=0)
    assert 0.11 < float(extent.max()) < 0.20  # head plus hosel, metres


@pytest.mark.parametrize("key", sorted(SPECS))
def test_plain_visual_layer_shows_the_mesh_head_not_the_ellipsoid(key) -> None:
    """MyoSuite renders the plain ``visual=True`` export, so it needs the mesh."""
    mujoco = pytest.importorskip("mujoco")
    xml, _ = exporter.export_full_body_mjcf(SPECS[key].read_bytes(), visual=True)
    model = mujoco.MjModel.from_xml_string(xml)
    meshes = {
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_MESH, i)
        for i in range(model.nmesh)
    }
    assert {"vmesh_club_head", "vmesh_club_shaft", "vmesh_club_grip"} <= meshes
    for i in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
        if "Clubface" in name:
            assert model.geom_type[i] != mujoco.mjtGeom.mjGEOM_ELLIPSOID
