"""Named hairstyles preserve the legacy crop and closed visual geometry."""

import numpy as np
import pytest

from src.shared.python.model_appearance import (
    build_head_parts,
    document_from_dict,
    document_to_dict,
)
from src.shared.python.model_appearance import head_hair, head_mesh

pytestmark = pytest.mark.unit
STYLES = ("side_swept", "swept_back", "slick_part", "receding_crop", "tousled_wave")


def _parts(style, cap="none"):
    """Build named hair on a shared unit head."""
    doc = document_from_dict(
        {
            "schema_version": "appearance-v1",
            "head": {"hair_style": style, "cap_style": cap},
        }
    )
    assert document_from_dict(document_to_dict(doc)) == doc
    return {p.name: p.mesh for p in build_head_parts(1.0, doc.head)}


def test_styles_are_closed_outward_and_distinct():
    """Each named style is a finite, closed shell with a different silhouette."""
    meshes = [_parts(style)["hair_mesh"] for style in STYLES]
    for mesh in meshes:
        edges = np.concatenate(
            (mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]])
        )
        assert sorted(map(tuple, edges)) == sorted(map(tuple, edges[:, ::-1]))
        assert np.isfinite(mesh.vertices).all() and mesh.volume() > 0
    signatures = [
        tuple(np.round(np.r_[m.vertices.min(axis=0), m.vertices.max(axis=0)], 4))
        for m in meshes
    ]
    assert len(set(signatures)) == len(STYLES)


def test_legacy_and_capped_hair_are_unchanged():
    """Preserve old crop bytes and the qualified cap-specific clearance geometry."""
    skin = head_mesh._default_asset().skin
    old = head_hair.build_short_crop(skin, "left")
    legacy = _parts("short_crop")["hair_mesh"]
    assert np.array_equal(old.vertices, legacy.vertices)
    assert np.array_equal(old.faces, legacy.faces)
    for cap in ("baseball_cap", "flat_cap", "straw_hat"):
        expected = _parts("short_crop", cap)["hair_mesh"]
        for style in STYLES:
            actual = _parts(style, cap)["hair_mesh"]
            assert np.array_equal(actual.vertices, expected.vertices)
            assert np.array_equal(actual.faces, expected.faces)


@pytest.mark.timeout(600)
def test_named_style_keeps_native_physics():
    """Changing visible hair leaves native mass, joints and dynamics identical."""
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
            {"schema_version": "appearance-v1", "head": {"hair_style": s}}
        )
        for s in ("short_crop", *STYLES)
    ]
    models = [
        mujoco.MjModel.from_xml_string(export_full_body_mjcf(raw, appearance=d)[0])
        for d in docs
    ]
    baseline = models[0]
    assert all(not np.array_equal(baseline.mesh_vert, m.mesh_vert) for m in models[1:])
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
            np.testing.assert_array_equal(getattr(baseline, key), getattr(model, key))
    rng = np.random.default_rng(48)
    for _ in range(3):
        q = baseline.qpos0 + rng.normal(0, 0.03, baseline.nq)
        v = rng.normal(0, 0.05, baseline.nv)
        states = []
        for model in models:
            data = mujoco.MjData(model)
            data.qpos[:], data.qvel[:] = q, v
            mujoco.mj_forward(model, data)
            states.append(data)
        for data in states[1:]:
            for key in ("qM", "qfrc_bias", "qacc"):
                np.testing.assert_array_equal(
                    getattr(states[0], key), getattr(data, key)
                )


def test_side_part_changes_the_forehead_hairline():
    """A side-swept part makes a visible asymmetric forehead boundary."""
    from src.shared.python.model_appearance import head_hair_styles

    skin = head_mesh._default_asset().skin
    left = head_hair_styles.build_hair_style(skin, "left", "side_swept")
    right = head_hair_styles.build_hair_style(skin, "right", "side_swept")
    # Opposite parts must move the front rim relative to each other.
    front_l = left.vertices[(left.vertices[:, 0] > 0.35) & (left.vertices[:, 2] < 0.84)]
    front_r = right.vertices[
        (right.vertices[:, 0] > 0.35) & (right.vertices[:, 2] < 0.84)
    ]
    assert abs(front_l[:, 1].mean() - front_r[:, 1].mean()) > 0.003


def test_asymmetry_bounds_and_default_line():
    """Reject invalid sweeps and retain the exact symmetric default hairline."""
    from dataclasses import replace

    from scipy.interpolate import PchipInterpolator

    crop = head_hair.load_hair_crop()
    angles = np.linspace(-np.pi, np.pi, 81)
    points = np.asarray(crop.hairline)
    legacy = PchipInterpolator(points[:, 0], points[:, 1])(
        np.clip(np.abs(angles), points[0, 0], points[-1, 0])
    )
    np.testing.assert_array_equal(head_hair.hairline_height(crop, angles), legacy)
    skin = head_mesh._default_asset().skin
    for invalid in (-0.151, 0.151, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="asymmetry"):
            head_hair.build_short_crop(
                skin, crop=replace(crop, hairline_asymmetry=invalid)
            )


def test_full_styles_keep_crown_below_pompadour_height():
    """Limit crown height while retaining the longer rear hairline."""
    from src.shared.python.model_appearance import head_hair_styles

    skin = head_mesh._default_asset().skin
    for style in ("side_swept", "swept_back", "tousled_wave"):
        vertices = head_hair_styles.build_hair_style(skin, "left", style).vertices
        assert vertices[:, 2].max() < 1.09
        assert vertices[:, 2].min() < 0.30


def test_tousled_wave_is_schema_validated_and_roundtrips():
    """Expose the wavy style through the public appearance schema."""
    doc = document_from_dict(
        {"schema_version": "appearance-v1", "head": {"hair_style": "tousled_wave"}}
    )
    restored = document_from_dict(document_to_dict(doc))
    assert restored.head.hair_style == "tousled_wave"


def test_tousled_wave_bends_ridges_on_a_closed_scalp():
    """Add a wavy phase to the CC0 scalp while keeping its shell watertight."""
    from dataclasses import replace

    skin = head_mesh._default_asset().skin
    base = replace(
        head_hair.load_hair_crop(), comb_amplitude=0.015, comb_wavelength=0.10
    )
    straight = head_hair.build_short_crop(
        skin, "none", replace(base, wave_bend=0.0, wave_wavelength=0.24)
    )
    wave = head_hair.build_short_crop(
        skin, "none", replace(base, wave_bend=0.45, wave_wavelength=0.24)
    )
    assert not np.array_equal(straight.vertices, wave.vertices)
    edges = np.concatenate(
        (wave.faces[:, [0, 1]], wave.faces[:, [1, 2]], wave.faces[:, [2, 0]])
    )
    assert sorted(map(tuple, edges)) == sorted(map(tuple, edges[:, ::-1]))
    assert np.isfinite(wave.vertices).all() and wave.volume() > 0


def test_tousled_wave_softens_the_front_hairline():
    """The front boundary gets a small undulation, within head-length limits."""
    from dataclasses import replace

    angles = np.linspace(-0.8, 0.8, 81)
    base = head_hair.load_hair_crop()
    straight = head_hair.hairline_height(base, angles)
    wave = head_hair.hairline_height(replace(base, hairline_waviness=0.006), angles)
    delta = wave - straight
    assert np.max(np.abs(delta)) <= 0.006
    assert np.ptp(delta) > 0.005


def test_wave_parameters_are_bounded():
    """Reject non-finite and excessive ridge warps before creating hair geometry."""
    from dataclasses import replace

    skin = head_mesh._default_asset().skin
    crop = head_hair.load_hair_crop()
    for changes in (
        {"wave_bend": -0.01},
        {"wave_bend": 0.76},
        {"wave_bend": float("nan")},
        {"wave_wavelength": 0.079},
        {"wave_wavelength": 1.01},
        {"hairline_waviness": 0.016},
        {"hairline_waviness": float("nan")},
    ):
        with pytest.raises(ValueError, match="hair wave|hairline waviness"):
            head_hair.build_short_crop(skin, crop=replace(crop, **changes))
