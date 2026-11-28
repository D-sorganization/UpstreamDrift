"""Period hat shapes and crown-to-peak fastenings stay visual-only."""

import numpy as np
import pytest

from src.shared.python.model_appearance import (
    build_head_parts,
    document_from_dict,
    document_to_dict,
)
from src.shared.python.model_appearance import head_cap, head_mesh

pytestmark = pytest.mark.unit


def _parts(style):
    """Build the chosen headwear in unit-head coordinates."""
    doc = document_from_dict(
        {"schema_version": "appearance-v1", "head": {"cap_style": style}}
    )
    assert document_from_dict(document_to_dict(doc)) == doc
    return {p.name: p.mesh for p in build_head_parts(1.0, doc.head)}


def test_period_caps_have_closed_snapped_peaks_and_distinct_crowns():
    """Keep each peak button and larger newsboy crown in the cap material."""
    flat, news = _parts("flat_cap"), _parts("newsboy_cap")
    for parts in (flat, news):
        for name in ("cap_crown", "cap_brim", "cap_snap"):
            m = parts[name]
            edges = np.concatenate(
                (m.faces[:, [0, 1]], m.faces[:, [1, 2]], m.faces[:, [2, 0]])
            )
            assert sorted(map(tuple, edges)) == sorted(map(tuple, edges[:, ::-1]))
            assert np.isfinite(m.vertices).all() and m.volume() > 0
        crown, brim = parts["cap_crown"].vertices, parts["cap_brim"].vertices
        surface = parts["cap_snap"].vertices
        for mesh in (crown, brim):
            assert (
                np.linalg.norm(mesh[:, None] - surface[None, :], axis=2).min() < 0.003
            )
    assert (
        news["cap_crown"].vertices[:, 2].max()
        > flat["cap_crown"].vertices[:, 2].max() + 0.015
    )


def test_straw_crown_flattens_without_losing_hair_clearance():
    """A flat-topped straw crown covers hair while keeping its closed shell."""
    skin = head_mesh._default_asset().skin
    cfg = head_cap.straw_hat_crop()
    straw = dict(head_cap.build_straw_hat(skin))["straw_crown"]
    rounded = dict(head_cap.build_cap(skin, 0.3, cfg))["cap_crown"]
    top = straw.vertices[straw.vertices[:, 2] > straw.vertices[:, 2].max() - 0.02]
    assert np.ptp(top[:, 0]) > 0.40 and np.ptp(top[:, 1]) > 0.40
    assert straw.vertices[:, 2].max() <= rounded.vertices[:, 2].max() + 1e-9
    assert straw.vertices[:, 2].max() > 1.02
    assert straw.volume() > 0
    band = dict(head_cap.build_straw_hat(skin))["straw_band"]
    assert 0.075 < np.ptp(band.vertices[:, 2]) < 0.105


def test_baseball_geometry_is_unchanged():
    """Keep the existing baseball assembly exactly on its former path."""
    skin = head_mesh._default_asset().skin
    old = dict(head_cap.build_cap(skin, 0.3))
    current = _parts("baseball_cap")
    assert "cap_snap" not in current
    for name in old:
        np.testing.assert_array_equal(old[name].vertices, current[name].vertices)
        np.testing.assert_array_equal(old[name].faces, current[name].faces)


@pytest.mark.timeout(600)
def test_period_hats_do_not_change_native_physics():
    """Compare native mass, joints and dynamics against the same bare head."""
    import json
    import mujoco
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.motion_matching.execution.golfer_reference_specs import (
        reference_spec,
    )

    raw = json.dumps(reference_spec("driver")).encode()
    docs = [
        document_from_dict(
            {"schema_version": "appearance-v1", "head": {"cap_style": style}}
        )
        for style in (
            "none",
            "baseball_cap_structured",
            "flat_cap",
            "newsboy_cap",
            "straw_hat",
        )
    ]
    models = [
        mujoco.MjModel.from_xml_string(export_full_body_mjcf(raw, appearance=doc)[0])
        for doc in docs
    ]
    base = models[0]
    for model in models[1:]:
        for key in (
            "body_mass",
            "body_inertia",
            "body_pos",
            "body_quat",
            "jnt_axis",
            "jnt_pos",
            "jnt_range",
            "qpos0",
        ):
            np.testing.assert_array_equal(getattr(base, key), getattr(model, key))
    rng = np.random.default_rng(53)
    for _ in range(3):
        q, v = base.qpos0 + rng.normal(0, 0.03, base.nq), rng.normal(0, 0.05, base.nv)
        states = []
        for model in models:
            state = mujoco.MjData(model)
            state.qpos[:], state.qvel[:] = q, v
            mujoco.mj_forward(model, state)
            states.append(state)
        for state in states[1:]:
            for key in ("qM", "qfrc_bias", "qacc"):
                np.testing.assert_array_equal(
                    getattr(states[0], key), getattr(state, key)
                )


def test_short_peak_snap_is_at_the_front():
    """On short period peaks, pull the crown down to a front snap."""
    skin = head_mesh._default_asset().skin
    for crop in (head_cap.flat_cap_crop(), head_cap.newsboy_cap_crop()):
        parts = dict(head_cap.build_period_cap(skin, 0.12, crop))
        brim, snap = parts["cap_brim"].vertices, parts["cap_snap"].vertices
        assert snap[:, 0].mean() > brim[:, 0].max() - 0.02


def test_crown_pull_preserves_triangle_winding():
    """Keep the reshaped crown locally oriented throughout the brim range."""
    skin = head_mesh._default_asset().skin
    for crop in (head_cap.flat_cap_crop(), head_cap.newsboy_cap_crop()):
        for length in (0.1, 0.12, 0.3, 0.8):
            original = dict(head_cap.build_cap(skin, length, crop))["cap_crown"]
            changed = dict(head_cap.build_period_cap(skin, length, crop))["cap_crown"]
            a, b = original.vertices[original.faces], changed.vertices[changed.faces]
            na = np.cross(a[:, 1] - a[:, 0], a[:, 2] - a[:, 0])
            nb = np.cross(b[:, 1] - b[:, 0], b[:, 2] - b[:, 0])
            assert (np.einsum("ij,ij->i", na, nb) > 0).all()


def test_owner_requested_brim_and_cap_fit():
    """Use a narrower straw brim and a fuller, forward-leaning period cap."""
    skin = head_mesh._default_asset().skin
    brim = dict(head_cap.build_straw_hat(skin))["straw_brim"].vertices
    assert np.ptp(brim[:, 1]) < 1.20
    for crop in (head_cap.flat_cap_crop(), head_cap.newsboy_cap_crop()):
        assert crop.offset >= 0.044
        parts = dict(head_cap.build_period_cap(skin, 0.12, crop))
        old = dict(head_cap.build_cap(skin, 0.12, crop))
        assert (
            parts["cap_brim"].vertices[:, 2].mean()
            < old["cap_brim"].vertices[:, 2].mean() - 0.005
        )
