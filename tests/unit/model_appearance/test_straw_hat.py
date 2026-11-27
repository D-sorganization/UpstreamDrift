"""Straw-hat style is visual-only, closed and clears the cropped hair."""

import numpy as np
import pytest

from src.shared.python.model_appearance import (
    build_head_parts,
    document_from_dict,
    document_to_dict,
)
from src.shared.python.model_appearance import head_cap, head_hair, head_mesh
from src.shared.python.model_appearance import head as head_model
from src.shared.python.model_appearance.head_morph import HeadMorph, warp_vertices

pytestmark = pytest.mark.unit


def test_straw_style_roundtrip_and_geometry():
    """Keep crown, full brim and dark band closed and material-bound."""
    doc = document_from_dict(
        {
            "schema_version": "appearance-v1",
            "head": {"cap_style": "straw_hat", "cap_material": "hat_straw"},
        }
    )
    assert document_from_dict(document_to_dict(doc)) == doc
    parts = {p.name: p for p in build_head_parts(1.0, doc.head)}
    for name in ("straw_crown", "straw_brim", "straw_band"):
        mesh = parts[name].mesh
        edges = np.concatenate(
            (mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]])
        )
        assert sorted(map(tuple, edges)) == sorted(map(tuple, edges[:, ::-1]))
        assert mesh.volume() > 0
    assert head_model.part_material_name(parts["straw_crown"], doc) == "hat_straw"
    assert head_model.part_material_name(parts["straw_band"], doc) == "hat_band_dark"
    brim, crown = parts["straw_brim"].mesh.vertices, parts["straw_crown"].mesh.vertices
    for axis in (0, 1):
        assert np.ptp(brim[:, axis]) > np.ptp(crown[:, axis]) + 0.3


def test_straw_cap_clearance_and_common_morph():
    """Crown clears cropped hair and the existing head warp moves all hat parts."""
    skin = head_mesh._default_asset().skin
    crop = head_cap.straw_hat_crop()
    crown = dict(head_cap.build_straw_hat(skin))["straw_crown"]
    hair = head_hair.build_short_crop(skin, "left", crop.hair_under)
    assert crop.offset - crop.wall > crop.hair_under.top_thickness
    assert crown.vertices[:, 2].max() > hair.vertices[:, 2].max()
    # Test rays through cropped hair against the actual crown shell.
    centre = np.array([crop.centre_x, 0.0, 0.55])
    vertices = hair.vertices[hair.vertices[:, 2] > 0.80][::15]
    triangles = crown.vertices[crown.faces]
    e1, e2 = triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]
    for point in vertices:
        ray = point - centre
        ray /= np.linalg.norm(ray)
        h = np.cross(ray, e2)
        det = np.einsum("ij,ij->i", e1, h)
        valid = np.abs(det) > 1e-12
        inverse = 1.0 / np.where(valid, det, 1.0)
        relative = point - triangles[:, 0]
        u = inverse * np.einsum("ij,ij->i", relative, h)
        cross = np.cross(relative, e1)
        v = inverse * (cross @ ray)
        t = inverse * np.einsum("ij,ij->i", e2, cross)
        assert np.any(
            valid & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 0)
        )
    base = document_from_dict(
        {
            "schema_version": "appearance-v1",
            "head": {"cap_style": "straw_hat", "cap_material": "hat_straw"},
        }
    )
    morph = HeadMorph(width=0.1, jaw=-0.1)
    morphed = document_from_dict(
        {
            "schema_version": "appearance-v1",
            "head": {
                "cap_style": "straw_hat",
                "cap_material": "hat_straw",
                "morph": {"width": 0.1, "jaw": -0.1},
            },
        }
    )
    a = {p.name: p.mesh for p in build_head_parts(1.0, base.head)}
    b = {p.name: p.mesh for p in build_head_parts(1.0, morphed.head)}
    for name in ("straw_crown", "straw_brim", "straw_band"):
        assert np.array_equal(a[name].faces, b[name].faces)
        np.testing.assert_allclose(
            b[name].vertices, warp_vertices(a[name].vertices, morph)
        )


@pytest.mark.parametrize("style", ["baseball_cap", "flat_cap"])
def test_existing_cap_meshes_remain_exact(style):
    """Preserve existing cap mesh bytes when the straw style is added."""
    settings = document_from_dict(
        {"schema_version": "appearance-v1", "head": {"cap_style": style}}
    ).head
    parts = {p.name: p.mesh for p in build_head_parts(1.0, settings)}
    crop = head_cap.flat_cap_crop() if style == "flat_cap" else head_cap.CapCrop()
    reference = dict(
        head_cap.build_cap(head_mesh._default_asset().skin, settings.cap_brim, crop)
    )
    for name in ("cap_crown", "cap_brim"):
        np.testing.assert_array_equal(parts[name].vertices, reference[name].vertices)
        np.testing.assert_array_equal(parts[name].faces, reference[name].faces)


@pytest.mark.timeout(240)
def test_native_straw_hat_does_not_change_physics():
    """Compare current native physics with and without the new hat."""
    import json

    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.motion_matching.execution.golfer_reference_specs import (
        reference_spec,
    )

    raw = json.dumps(reference_spec("driver")).encode()
    docs = [
        document_from_dict({"schema_version": "appearance-v1", "head": head})
        for head in ({}, {"cap_style": "straw_hat", "cap_material": "hat_straw"})
    ]
    models = [
        mujoco.MjModel.from_xml_string(export_full_body_mjcf(raw, appearance=doc)[0])
        for doc in docs
    ]
    a, b = models
    for key in (
        "body_mass",
        "body_inertia",
        "body_pos",
        "body_quat",
        "jnt_type",
        "jnt_qposadr",
        "jnt_dofadr",
    ):
        np.testing.assert_array_equal(getattr(a, key), getattr(b, key))
    states = []
    for model in models:
        data = mujoco.MjData(model)
        data.qpos[:] = np.linspace(-0.02, 0.02, model.nq)
        data.qvel[:] = np.linspace(0.01, -0.01, model.nv)
        mujoco.mj_forward(model, data)
        states.append(data)
    for key in ("qM", "qfrc_bias", "qacc"):
        np.testing.assert_array_equal(getattr(states[0], key), getattr(states[1], key))
