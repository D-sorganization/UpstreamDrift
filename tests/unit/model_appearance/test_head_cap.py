"""Baseball cap option for the CC0 head (#15276): off by default, no poke-through."""

import numpy as np
import pytest

from src.shared.python.model_appearance import (
    build_head_parts,
    document_from_dict,
    head_cap,
    head_hair,
    head_mesh,
)
from src.shared.python.model_appearance import head as head_model
from src.shared.python.model_appearance import library

LENGTH = 0.2429
CAP = {"cap_style": "baseball_cap"}
FLAT_CAP = {"cap_style": "flat_cap"}


def _doc(**head):
    return document_from_dict({"schema_version": "appearance-v1", "head": head})


def _names(**head) -> set[str]:
    return {p.name for p in build_head_parts(LENGTH, _doc(**head).head)}


def _volume_and_closed(mesh) -> tuple[float, bool]:
    f = mesh.faces
    e = np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]])
    n = int(e.max()) + 1
    fwd, rev = np.sort(e[:, 0] * n + e[:, 1]), np.sort(e[:, 1] * n + e[:, 0])
    tri = mesh.vertices[f]
    vol = np.einsum("ij,ij->i", tri[:, 0], np.cross(tri[:, 1], tri[:, 2])).sum() / 6
    return float(vol), bool(
        len(np.unique(fwd)) == len(fwd) and np.array_equal(fwd, rev)
    )


def _assert_hair_under_crown(crown, hair, centre: np.ndarray) -> None:
    p = hair.vertices[hair.vertices[:, 2] > 0.80]
    p = p[:: max(1, len(p) // 300)]
    d = (p - centre) / np.linalg.norm(p - centre, axis=1, keepdims=True)
    tri = crown.vertices[crown.faces]
    e1, e2 = tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]
    for pt, ray in zip(p, d, strict=True):
        h = np.cross(ray, e2)
        det = np.einsum("ij,ij->i", e1, h)
        ok = np.abs(det) > 1e-12
        inv = np.where(ok, 1.0 / np.where(ok, det, 1.0), 0.0)
        s = pt - tri[:, 0]
        u = inv * np.einsum("ij,ij->i", s, h)
        q = np.cross(s, e1)
        v = inv * (q @ ray)
        t = inv * np.einsum("ij,ij->i", e2, q)
        hits = ok & (u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9) & (t > 0)
        assert hits.sum() >= 1, "hair vertex outside the cap crown"


@pytest.mark.unit
def test_cap_is_off_by_default_and_adds_geometry_when_enabled() -> None:
    assert _doc().head.cap_style == "none"
    assert not {"cap_crown", "cap_brim"} & _names()
    assert {"cap_crown", "cap_brim", "hair_mesh"} <= _names(**CAP)


@pytest.mark.unit
def test_cap_parts_are_closed_and_use_the_cap_colour_setting() -> None:
    doc = _doc(**CAP, cap_material="cap_red")
    parts = {p.name: p for p in build_head_parts(LENGTH, doc.head)}
    for name in ("cap_crown", "cap_brim"):
        volume, closed = _volume_and_closed(parts[name].mesh)
        assert closed and volume > 0.0, name
        assert head_model.part_material_name(parts[name], doc) == "cap_red"
    assert _doc(**CAP).head.cap_material == "cap_navy"
    assert "cap_red" in library.MATERIALS


@pytest.mark.unit
def test_flat_cap_is_a_distinct_closed_low_profile_style() -> None:
    doc = _doc(**FLAT_CAP)
    assert doc.head.cap_style == "flat_cap"
    parts = {p.name: p for p in build_head_parts(LENGTH, doc.head)}
    baseball = {p.name: p for p in build_head_parts(LENGTH, _doc(**CAP).head)}
    for name in ("cap_crown", "cap_brim"):
        assert _volume_and_closed(parts[name].mesh)[1]
        assert _volume_and_closed(parts[name].mesh)[0] > 0.0
    assert parts["cap_crown"].mesh.vertices[:, 2].max() < (
        baseball["cap_crown"].mesh.vertices[:, 2].max()
    )
    assert parts["cap_crown"].mesh.vertices[:, 0].max() > (
        baseball["cap_crown"].mesh.vertices[:, 0].max() + 0.02
    )
    assert parts["cap_brim"].mesh.vertices[:, 0].max() < (
        baseball["cap_brim"].mesh.vertices[:, 0].max()
    )


@pytest.mark.unit
def test_flat_cap_hair_is_cropped_below_the_crown() -> None:
    skin = head_mesh._default_asset().skin
    settings = _doc(**FLAT_CAP).head
    parts = {p.name: p for p in build_head_parts(LENGTH, settings)}
    assert "hair_mesh" in parts
    crop = head_cap.flat_cap_crop()
    crown = head_cap.build_cap(skin, settings.cap_brim, crop)[0][1]
    hair = head_hair.build_short_crop(skin, settings.hair_part, crop.hair_under)
    _assert_hair_under_crown(crown, hair, np.array([crop.centre_x, 0.0, 0.55]))
    assert hair.vertices[:, 2].max() < crown.vertices[:, 2].max()


@pytest.mark.unit
def test_brim_points_forward_and_length_is_a_setting() -> None:
    skin = head_mesh._default_asset().skin
    reach = {
        b: dict(head_cap.build_cap(skin, b))["cap_brim"].vertices[:, 0].max()
        for b in (0.2, 0.6)
    }
    assert reach[0.6] > reach[0.2] + 0.2
    assert reach[0.2] > skin.vertices[:, 0].max() - 0.1, "brim sits at the forehead"


@pytest.mark.unit
def test_hair_never_pokes_through_the_crown_and_shows_below_it() -> None:
    skin = head_mesh._default_asset().skin
    crop = head_cap.CapCrop()
    crown = dict(head_cap.build_cap(skin, 0.3))["cap_crown"]
    hair = head_hair.build_short_crop(skin, "left", crop.hair_under)
    _assert_hair_under_crown(crown, hair, np.array([crop.centre_x, 0.0, 0.55]))
    line_min = head_cap.cap_line_height(crop, np.array([np.pi]))[0]
    assert hair.vertices[:, 2].min() < line_min - 0.1, "hair shows below the cap"


@pytest.mark.unit
@pytest.mark.parametrize("brim", [0.0, 0.05, 2.0, float("nan")])
def test_bad_cap_brim_is_refused(brim: float) -> None:
    with pytest.raises(ValueError, match="cap_brim"):
        _doc(**CAP, cap_brim=brim)
    with pytest.raises(ValueError, match="brim"):
        head_cap.build_cap(head_mesh._default_asset().skin, brim)


@pytest.mark.unit
def test_bad_cap_style_material_and_model_are_refused() -> None:
    with pytest.raises(ValueError, match="cap_style"):
        _doc(cap_style="top_hat")
    with pytest.raises(ValueError, match="cc0_mesh"):
        _doc(**CAP, model="parametric")
    with pytest.raises(ValueError, match="cap_material"):
        library.check_references(_doc(**CAP, cap_material="no_such_material"))
