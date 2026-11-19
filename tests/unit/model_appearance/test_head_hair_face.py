"""Dirty-blonde short-crop hair and a featured face on the CC0 head (#15276)."""

import numpy as np
import pytest
from scipy.spatial import cKDTree

from src.shared.python.model_appearance import (
    build_head_parts,
    defaults,
    document_from_dict,
    document_to_dict,
    library,
)
from src.shared.python.model_appearance import head as head_model
from src.shared.python.model_appearance.schema import HeadSettings

LENGTH = 0.2429
BARE = {"schema_version": "appearance-v1"}


def _doc(**head):
    return document_from_dict({**BARE, "head": head})


def _parts(**head):
    doc = _doc(**head)
    return doc, {p.name: p for p in build_head_parts(LENGTH, doc.head)}


def _closed(faces) -> bool:
    """Every directed edge is used once and has its opposite edge."""
    edges = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]])
    n = int(edges.max()) + 1
    fwd = np.sort(edges[:, 0] * n + edges[:, 1])
    rev = np.sort(edges[:, 1] * n + edges[:, 0])
    return len(np.unique(fwd)) == len(fwd) and np.array_equal(fwd, rev)


def _luma(rgb) -> float:
    return 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]


@pytest.mark.unit
def test_dirty_blonde_is_in_the_palette_and_is_the_default_hair() -> None:
    ids = [h["id"] for h in defaults.hair_palette()]
    assert "hair_dirty_blonde" in ids
    doc = defaults.default_appearance()
    assert library.headwear_material_name(doc) == "hair_dirty_blonde"
    r, g, b = library.MATERIALS["hair_dirty_blonde"].base_color[:3]
    assert r > g > b, "dirty blonde is warm"
    assert 0.45 <= _luma((r, g, b)) <= 0.62
    assert (r - b) < 0.40, "lighter and less red than the brown it replaces"


@pytest.mark.unit
def test_hair_style_and_eye_colour_are_validated_settings() -> None:
    cfg = HeadSettings()
    assert (cfg.hair_style, cfg.eye_color) == ("short_crop", "blue_grey")
    for bad in ({"hair_style": "mohawk"}, {"eye_color": "violet"}):
        with pytest.raises(ValueError):
            _doc(**bad)
    doc = _doc(hair_style="dome", eye_color="hazel", hair_part="right")
    again = document_from_dict(document_to_dict(doc))
    assert (again.head.hair_style, again.head.eye_color, again.head.hair_part) == (
        "dome",
        "hazel",
        "right",
    )


@pytest.mark.unit
def test_short_crop_hair_is_a_closed_positive_volume_shell() -> None:
    _, parts = _parts()
    hair = parts["hair_mesh"].mesh
    assert _closed(hair.faces)
    assert hair.volume() > 0.0
    dome = _parts(hair_style="dome")[1]["hair_mesh"].mesh
    assert hair.volume() > 1.3 * dome.volume(), "more volume than the thin shell"


@pytest.mark.unit
def test_symmetric_part_gives_a_left_right_mirrored_haircut() -> None:
    _, parts = _parts(hair_part="none")
    v = parts["hair_mesh"].mesh.vertices
    gaps, _ = cKDTree(v).query(v * np.array([1.0, -1.0, 1.0]))
    assert gaps.max() < 0.02 * LENGTH
    _, side = _parts(hair_part="left")
    sv = side["hair_mesh"].mesh.vertices
    assert not np.allclose(np.sort(sv[:, 1]), np.sort(-sv[:, 1]), atol=1e-6)


def _skin_crossings(skin, origin, targets) -> list[np.ndarray]:
    """Ray parameters where the origin-to-target rays cross the skin, as a
    multiple of the target distance (so ``> 1`` lies beyond the target)."""
    tri = skin.vertices[skin.faces]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    out = []
    for target in targets:
        d = target - origin
        p = np.cross(d, e2)
        det = np.einsum("ij,ij->i", e1, p)
        ok = np.abs(det) > 1e-14
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = origin - tri[:, 0]
        u = np.einsum("ij,ij->i", s, p) * inv
        q = np.cross(s, e1)
        v = (q @ d) * inv
        t = np.einsum("ij,ij->i", e2, q) * inv
        out.append(t[ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 0)])
    return out


@pytest.mark.unit
def test_hair_hugs_outside_the_skull() -> None:
    _, parts = _parts()
    skin, hair = parts["skin_mesh"].mesh, parts["hair_mesh"].mesh
    centre = np.array([0.10, 0.0, 0.62]) * LENGTH
    crossings = _skin_crossings(skin, centre, hair.vertices)
    beyond = np.array([(c > 1.0 + 1e-9).sum() for c in crossings])
    assert (beyond % 2 == 0).all(), "a hair vertex sits inside the skull"
    gap = np.array([1.0 - c[c <= 1.0 + 1e-9].max() for c in crossings])
    dist = np.linalg.norm(hair.vertices - centre, axis=1)
    assert (gap * dist).max() < 0.2 * LENGTH, "hair is a helmet, not a haircut"


@pytest.mark.unit
def test_hair_sits_above_the_brow_and_leaves_the_face_bare() -> None:
    _, parts = _parts()
    hair = parts["hair_mesh"].mesh.vertices / LENGTH
    brow_top = max(parts[f"brow_{s}"].mesh.vertices[:, 2].max() for s in "lr") / LENGTH
    forehead = hair[hair[:, 0] > 0.40]
    assert len(forehead) > 0
    assert forehead[:, 2].min() > brow_top + 0.05, "hairline clear of the brow"
    assert hair[:, 2].max() > 1.0, "volume on top"
    eye_top = max(parts[f"eye_{s}"].mesh.vertices[:, 2].max() for s in "lr") / LENGTH
    assert hair[:, 2].min() < eye_top, "sides and nape are cut lower"
    nape = hair[hair[:, 0] < -0.05]
    assert nape[:, 2].min() < 0.40, "hair reaches a normal nape hairline"


@pytest.mark.unit
def test_hairline_depth_is_a_setting_and_the_default_is_a_normal_haircut() -> None:
    from dataclasses import replace

    from src.shared.python.model_appearance import head_hair, head_mesh

    skin = head_mesh._default_asset().skin
    crop = head_hair.HairCrop()
    az = np.array([np.pi, 2.4, 1.8])
    low = head_hair.hairline_height(crop, az)
    assert low[0] < 0.35, "nape hairline sits near the top of the neck"
    assert low[1] < 0.45, "side hairline drops toward the ear bottom"
    high = replace(crop, hairline=((0.0, 0.8), (np.pi, 0.8)))
    assert head_hair.hairline_height(high, az)[0] == pytest.approx(0.8)
    mesh = head_hair.build_short_crop(skin, "left", high)
    assert mesh.vertices[:, 2].min() > 0.7, "a raised hairline leaves the nape bare"


@pytest.mark.unit
def test_eyes_have_white_iris_pupil_lid_and_matching_darker_brows() -> None:
    doc, parts = _parts()
    for side in "lr":
        for name in ("eye", "iris", "pupil", "lid", "brow"):
            assert f"{name}_{side}" in parts
    names = {
        n: head_model.part_material_name(parts[f"{n}_l"], doc)
        for n in ("eye", "iris", "pupil", "lid", "brow")
    }
    assert len(set(names.values())) == 5, names
    brow = library.MATERIALS[names["brow"]].base_color
    hair = library.MATERIALS[library.headwear_material_name(doc) or ""].base_color
    assert _luma(brow) < _luma(hair) and brow[0] > brow[2], "darker, same hue family"
    hazel = _doc(eye_color="hazel")
    assert head_model.part_material_name(
        parts["iris_l"], hazel
    ) != head_model.part_material_name(parts["iris_l"], doc)
    eye, iris = parts["eye_l"].mesh.vertices, parts["iris_l"].mesh.vertices
    assert iris[:, 0].max() > eye[:, 0].max() - 0.005 * LENGTH, "iris faces forward"
    assert eye[:, 1].mean() > 0 > parts["eye_r"].mesh.vertices[:, 1].mean()
    assert abs(eye[:, 1].mean() + parts["eye_r"].mesh.vertices[:, 1].mean()) < 1e-4


@pytest.mark.unit
def test_features_are_closed_meshes_and_the_mouth_is_tinted() -> None:
    doc, parts = _parts()
    for name in ("eye_l", "iris_l", "pupil_l", "lid_l", "brow_l", "mouth"):
        assert _closed(parts[name].mesh.faces), name
        assert parts[name].mesh.volume() > 0.0, name
    assert head_model.part_material_name(parts["mouth"], doc) != doc.skin_tone
