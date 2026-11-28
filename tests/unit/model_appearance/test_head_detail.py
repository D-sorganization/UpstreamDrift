"""CC0 local face detail is bounded, closed and visual only."""

import json

import numpy as np
import pytest

from src.shared.python.model_appearance.head_detail import HeadDetail, apply_detail
from src.shared.python.model_appearance.head_mesh import (
    build_mesh_parts,
    load_head_asset,
)
from src.shared.python.model_appearance.schema import (
    AppearanceDocument,
    HeadSettings,
    document_from_dict,
    document_to_dict,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(240)]


def test_detail_schema_and_default_identity():
    """Keep legacy output identical and require bounded finite controls."""
    skin = load_head_asset().skin
    assert apply_detail(skin, HeadDetail()) is skin
    assert "detail" not in document_to_dict(AppearanceDocument())["head"]
    doc = AppearanceDocument(
        head=HeadSettings(detail=HeadDetail(nose_curve=0.7, cheek_bones=-0.5))
    )
    assert document_from_dict(document_to_dict(doc)) == doc
    for bad in (float("nan"), float("inf"), 1.01, -1.01, True):
        with pytest.raises(ValueError):
            HeadDetail(nose_curve=bad)
    with pytest.raises(ValueError):
        HeadSettings(model="parametric", detail=HeadDetail(nose_curve=0.5))


@pytest.fixture(scope="module")
def default_head_parts():
    """Return legacy hair and facial accessories for exact attachment checks."""
    return {
        name: mesh for name, mesh, _ in build_mesh_parts(1.0, "hair", HeadSettings())
    }


@pytest.mark.parametrize("nose,cheek", [(a, b) for a in (-1, 0, 1) for b in (-1, 0, 1)])
def test_detail_keeps_closed_mesh_and_facial_accessories(
    nose, cheek, default_head_parts
):
    """Retain topology, neck/top, hair and facial-feature attachment across bounds."""
    from src.shared.python.model_appearance import head_mesh

    source = load_head_asset().skin
    detail = HeadDetail(nose_curve=nose, cheek_bones=cheek)
    changed = apply_detail(source, detail)
    np.testing.assert_array_equal(changed.faces, source.faces)
    original_tri = source.vertices[source.faces]
    changed_tri = changed.vertices[changed.faces]
    old_normal = np.cross(
        original_tri[:, 1] - original_tri[:, 0], original_tri[:, 2] - original_tri[:, 0]
    )
    new_normal = np.cross(
        changed_tri[:, 1] - changed_tri[:, 0], changed_tri[:, 2] - changed_tri[:, 0]
    )
    assert (np.einsum("ij,ij->i", old_normal, new_normal) > 0).all()
    assert changed.vertices[:, 2].max() == source.vertices[:, 2].max()
    np.testing.assert_array_equal(changed.vertices[-1], source.vertices[-1])
    head_mesh._check_mesh("detail", changed)
    parts = build_mesh_parts(1.0, "hair", HeadSettings(detail=detail))
    named = {name: mesh for name, mesh, role in parts}
    for name in ("brow_l", "brow_r", "mouth", "eye_l", "eye_r", "hair_mesh"):
        np.testing.assert_allclose(
            named[name].vertices, default_head_parts[name].vertices, atol=1e-12, rtol=0
        )
    for _, mesh, _ in parts:
        assert np.isfinite(mesh.vertices).all()
        assert head_mesh._closed(mesh.faces)
        assert mesh.volume() > 0


def test_detail_does_not_change_native_dynamics():
    """Compare native inertia, joints and dynamics with and without face detail."""
    mujoco = pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.motion_matching.execution.golfer_reference_specs import (
        reference_spec,
    )

    raw = json.dumps(reference_spec("driver")).encode()
    appearances = [
        AppearanceDocument(),
        AppearanceDocument(
            head=HeadSettings(detail=HeadDetail(nose_curve=0.7, cheek_bones=-0.7))
        ),
    ]
    models = [
        mujoco.MjModel.from_xml_string(export_full_body_mjcf(raw, appearance=d)[0])
        for d in appearances
    ]
    for attr in (
        "body_mass",
        "body_inertia",
        "body_ipos",
        "body_pos",
        "jnt_qposadr",
        "jnt_range",
    ):
        np.testing.assert_array_equal(
            getattr(models[0], attr), getattr(models[1], attr)
        )
    for phase in (0, 0.03, -0.04):
        states = []
        for model in models:
            data = mujoco.MjData(model)
            data.qpos[:] = model.qpos0 + phase
            data.qvel[:] = phase
            mujoco.mj_forward(model, data)
            states.append(data)
        for attr in ("qM", "qfrc_bias", "qacc"):
            np.testing.assert_array_equal(
                getattr(states[0], attr), getattr(states[1], attr)
            )


@pytest.mark.parametrize(
    "style", ["baseball_cap", "flat_cap", "straw_hat", "newsboy_cap"]
)
def test_detail_keeps_headwear_and_face_parts_aligned(style):
    """Keep the existing cap, underlying hair, eyes and mouth at their baseline positions."""
    base = HeadSettings(cap_style=style)
    altered = HeadSettings(
        cap_style=style, detail=HeadDetail(nose_curve=1, cheek_bones=-1)
    )
    left = {name: mesh for name, mesh, _ in build_mesh_parts(1.0, "hair", base)}
    right = {name: mesh for name, mesh, _ in build_mesh_parts(1.0, "hair", altered)}
    assert set(left) == set(right)
    for name in left.keys() - {"skin_mesh"}:
        np.testing.assert_allclose(
            left[name].vertices, right[name].vertices, atol=1e-12, rtol=0
        )
        np.testing.assert_array_equal(left[name].faces, right[name].faces)


def test_detail_composes_with_head_morph():
    """Keep all parts closed when local and global face controls are combined."""
    from src.shared.python.model_appearance import head_mesh
    from src.shared.python.model_appearance.head_morph import HeadMorph

    settings = HeadSettings(morph=HeadMorph(width=0.1), detail=HeadDetail(0.7, -0.7))
    assert all(
        head_mesh._closed(mesh.faces) and mesh.volume() > 0
        for _, mesh, _ in build_mesh_parts(1, "hair", settings)
    )
