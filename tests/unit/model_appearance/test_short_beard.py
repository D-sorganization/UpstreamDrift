"""Optional conforming short facial hair on the CC0 head."""

import json
from dataclasses import replace

import numpy as np
import pytest

from src.shared.python.model_appearance.head_mesh import _closed, build_mesh_parts
from src.shared.python.model_appearance.head_morph import HeadMorph
from src.shared.python.model_appearance.schema import (
    AppearanceDocument,
    HeadSettings,
    document_from_dict,
    document_to_dict,
    validate_document,
)

pytestmark = pytest.mark.unit


def test_short_beard_is_opt_in_and_roundtrips():
    """Preserve default parts and encode the explicit short beard choice."""
    doc = AppearanceDocument()
    assert "facial_hair" not in document_to_dict(doc)["head"]
    data = document_to_dict(replace(doc, head=HeadSettings(facial_hair="short_beard")))
    assert document_from_dict(data).head.facial_hair == "short_beard"
    with pytest.raises(ValueError):
        HeadSettings(model="parametric", facial_hair="short_beard")
    base = build_mesh_parts(0.24, "hair", HeadSettings())
    explicit = build_mesh_parts(
        0.24, "hair", document_from_dict(document_to_dict(doc)).head
    )
    for a, b in zip(base, explicit, strict=True):
        assert a[0] == b[0] and a[2] == b[2]
        np.testing.assert_array_equal(a[1].vertices, b[1].vertices)
        np.testing.assert_array_equal(a[1].faces, b[1].faces)


def test_beard_is_closed_and_follows_morph_and_caps():
    """Check beard, moustache, mouth opening and cap compatibility."""
    for facial_hair in ("short_beard", "dense_short_beard"):
        for cap in ("none", "baseball_cap", "flat_cap"):
            settings = HeadSettings(
                facial_hair=facial_hair,
                cap_style=cap,
                morph=HeadMorph(jaw=0.18, width=-0.1, nose=0.1),
            )
            parts = build_mesh_parts(0.24, "hair", settings)
            beard = [
                (n, mesh)
                for n, mesh, _ in parts
                if n.startswith(("beard", "moustache"))
            ]
            assert len(beard) >= 3 and any(n.startswith("moustache") for n, _ in beard)
            for _, mesh in beard:
                assert np.isfinite(mesh.vertices).all() and _closed(mesh.faces)
                assert mesh.volume() > 0
            assert any(n.startswith("mouth") for n, _, _ in parts)


@pytest.mark.parametrize("facial_hair", ("short_beard", "dense_short_beard"))
def test_beard_leaves_the_lips_open_and_uses_shared_warp(facial_hair):
    """Keep the lip opening empty and warp the beard with the head skin."""
    from src.shared.python.model_appearance.head_features import front_depth
    from src.shared.python.model_appearance.head_morph import warp_vertices

    morph = HeadMorph(jaw=0.18, width=-0.1, nose=0.1)
    plain = build_mesh_parts(0.24, "none", HeadSettings(facial_hair=facial_hair))
    styled = build_mesh_parts(
        0.24, "none", HeadSettings(facial_hair=facial_hair, morph=morph)
    )
    before = {name: mesh for name, mesh, _ in plain}["beard_shell"]
    after = {name: mesh for name, mesh, _ in styled}["beard_shell"]
    vertices = before.vertices / 0.24
    assert not np.any(
        np.hypot(vertices[:, 1] / 0.08, (vertices[:, 2] - 0.27) / 0.035) < 1
    )
    from src.shared.python.model_appearance.geometry import Mesh

    assert np.isnan(
        front_depth(
            Mesh(vertices, before.faces),
            np.array([[y, 0.27] for y in (-0.06, -0.03, 0, 0.03, 0.06)]),
        )
    ).all()
    np.testing.assert_allclose(
        after.vertices, warp_vertices(vertices, morph) * 0.24, rtol=0, atol=1e-15
    )
    np.testing.assert_array_equal(after.faces, before.faces)


def test_beard_material_tracks_skin_and_hair():
    """Use distinct short-stubble colors for the wearer's skin and hair."""
    from src.shared.python.model_appearance import head, library
    from src.shared.python.model_appearance.head_mesh import load_head_asset

    part = head.HeadPart("beard", load_head_asset().skin, "beard")
    doc = AppearanceDocument(head=HeadSettings(headwear_material="hair_dark_brown"))
    first = head.part_material_name(part, doc)
    second = head.part_material_name(part, replace(doc, skin_tone="skin_6"))
    assert first != second
    assert library.MATERIALS[first].rgba != library.MATERIALS[second].rgba


def test_dense_short_beard_is_a_distinct_opt_in_schema_style():
    """Expose the denser style without changing the default or existing stubble."""
    doc = AppearanceDocument()
    assert document_to_dict(doc)["head"].get("facial_hair", "none") == "none"
    data = document_to_dict(
        replace(doc, head=HeadSettings(facial_hair="dense_short_beard"))
    )
    validate_document(data)
    assert document_from_dict(data).head.facial_hair == "dense_short_beard"
    invalid = {**data, "head": {**data["head"], "facial_hair": "full_beard"}}
    with pytest.raises(ValueError, match="facial_hair"):
        validate_document(invalid)
    with pytest.raises(ValueError):
        HeadSettings(model="parametric", facial_hair="dense_short_beard")


def test_dense_short_beard_keeps_default_and_existing_stubble_geometry():
    """Add only opt-in beard meshes; leave the other variants byte-identical."""
    base_settings = HeadSettings(
        cap_style="baseball_cap",
        morph=HeadMorph(jaw=0.12, width=-0.06, nose=0.08),
    )
    base = build_mesh_parts(0.24, "hair", base_settings)
    stubble = build_mesh_parts(
        0.24, "hair", replace(base_settings, facial_hair="short_beard")
    )
    dense = build_mesh_parts(
        0.24, "hair", replace(base_settings, facial_hair="dense_short_beard")
    )
    for variant in (stubble, dense):
        by_name = {name: (mesh, role) for name, mesh, role in variant}
        for name, mesh, role in base:
            assert name in by_name
            variant_mesh, variant_role = by_name[name]
            assert role == variant_role
            np.testing.assert_array_equal(mesh.vertices, variant_mesh.vertices)
            np.testing.assert_array_equal(mesh.faces, variant_mesh.faces)
    stubble_shell = {name: mesh for name, mesh, _ in stubble}["beard_shell"]
    dense_shell = {name: mesh for name, mesh, _ in dense}["beard_shell"]
    assert dense_shell.vertices[:, 2].max() > stubble_shell.vertices[:, 2].max()
    dense_meshes = [
        (name, mesh, role)
        for name, mesh, role in dense
        if "beard" in name or "moustache" in name
    ]
    assert len(dense_meshes) >= 3
    for _, mesh, _ in dense_meshes:
        assert _closed(mesh.faces) and mesh.volume() > 0
        assert np.isfinite(mesh.vertices).all()


def test_dense_beard_material_is_darker_than_stubble_for_dark_hair():
    """Use a separate beard palette role so the old stubble remains unchanged."""
    from src.shared.python.model_appearance import head, library
    from src.shared.python.model_appearance.head_mesh import load_head_asset

    doc = AppearanceDocument(head=HeadSettings(headwear_material="hair_dark_brown"))
    skin = load_head_asset().skin
    stubble = head.part_material_name(head.HeadPart("beard", skin, "beard"), doc)
    dense = head.part_material_name(
        head.HeadPart("beard_shell", skin, "dense_beard"), doc
    )

    def luminance(rgb):
        return sum(c * w for c, w in zip(rgb, (0.2126, 0.7152, 0.0722), strict=True))

    assert dense != stubble
    assert luminance(library.MATERIALS[dense].rgba[:3]) < luminance(
        library.MATERIALS[stubble].rgba[:3]
    )


@pytest.mark.timeout(300)
def test_beard_keeps_native_physics():
    """Verify all physical arrays and qM, bias, qacc across representative states."""
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.motion_matching.execution.golfer_reference_specs import (
        reference_spec,
    )

    raw = json.dumps(reference_spec("driver")).encode()
    docs = [
        AppearanceDocument(head=HeadSettings(facial_hair=kind))
        for kind in ("none", "short_beard", "dense_short_beard")
    ]
    models = [
        mujoco.MjModel.from_xml_string(export_full_body_mjcf(raw, appearance=d)[0])
        for d in docs
    ]
    for model in models[1:]:
        for name in (
            "body_mass",
            "body_inertia",
            "body_pos",
            "body_quat",
            "jnt_type",
            "jnt_qposadr",
            "jnt_dofadr",
            "jnt_range",
            "qpos0",
        ):
            np.testing.assert_array_equal(
                getattr(models[0], name), getattr(model, name)
            )
    for q in (0.0, 0.03, -0.04):
        states = []
        for model in models:
            data = mujoco.MjData(model)
            data.qpos[:] += q
            data.qvel[:] = q
            mujoco.mj_forward(model, data)
            states.append(data)
        for state in states[1:]:
            for name in ("qM", "qfrc_bias", "qacc"):
                np.testing.assert_array_equal(
                    getattr(states[0], name), getattr(state, name)
                )
