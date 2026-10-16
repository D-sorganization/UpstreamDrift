"""CC0 head mesh slot of ``model_appearance.head`` (R7, #12687, epic #12678)."""

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import (
    build_head_parts,
    document_from_dict,
    document_to_dict,
    head_mesh,
    resolve_head_anchor,
)
from src.shared.python.model_appearance.body_meshes import resolve_appearance
from src.shared.python.model_appearance.schema import HeadSettings

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
ASSET_DIR = ROOT / "assets/head_cc0"
SPECS = ROOT / "docs/development/full_body_models"
LENGTH = 0.2429
MESH = HeadSettings(model="cc0_mesh")
MAX_ASSET_BYTES = 150_000  # measured 63 kB; keep the repo light
MAX_TOTAL_TRIANGLES = 12_000  # measured ~9.4k; 60 fps renders keep their cost


def _by_name(parts):
    return {p.name: p for p in parts}


def test_asset_matches_its_provenance_record() -> None:
    record = json.loads((ASSET_DIR / "PROVENANCE.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256((ASSET_DIR / record["asset"]).read_bytes()).hexdigest()
    assert digest == record["asset_sha256"]
    assert record["licence"] == "CC0-1.0"
    for key in (
        "source_url",
        "author",
        "licence_text",
        "download_date",
        "source_sha256",
    ):
        assert str(record[key]).strip(), key
    assert (ASSET_DIR / record["licence_file"]).is_file()


def test_tampered_asset_fails_closed(tmp_path: Path) -> None:
    bad = tmp_path / "head_cc0.npz"
    bad.write_bytes((ASSET_DIR / "head_cc0.npz").read_bytes() + b"x")
    with pytest.raises(ValueError, match="sha256"):
        head_mesh.load_head_asset(bad, ASSET_DIR / "PROVENANCE.json")


def test_missing_provenance_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="provenance"):
        head_mesh.load_head_asset(ASSET_DIR / "head_cc0.npz", tmp_path / "none.json")


def test_asset_size_budget() -> None:
    assert (ASSET_DIR / "head_cc0.npz").stat().st_size <= MAX_ASSET_BYTES
    parts = build_head_parts(LENGTH, MESH)
    assert sum(len(p.mesh.faces) for p in parts) <= MAX_TOTAL_TRIANGLES


def test_head_slot_loads_the_mesh_with_skin_and_short_hair() -> None:
    parts = _by_name(build_head_parts(LENGTH, MESH))
    assert set(parts) == {"skin_mesh", "hair_mesh"}
    assert parts["skin_mesh"].material == "skin"
    assert parts["hair_mesh"].material == "headwear"
    assert len(parts["skin_mesh"].mesh.faces) > 4000


def test_bare_head_has_no_hair_and_cap_is_refused() -> None:
    bare = build_head_parts(LENGTH, HeadSettings(model="cc0_mesh", headwear="none"))
    assert [p.name for p in bare] == ["skin_mesh"]
    with pytest.raises(ValueError, match="cap"):
        build_head_parts(LENGTH, HeadSettings(model="cc0_mesh", headwear="cap"))


def test_every_part_is_closed_outward_with_positive_volume() -> None:
    for part in build_head_parts(LENGTH, MESH):
        assert part.mesh.volume() > 0, part.name
        edges: dict[tuple[int, int], int] = {}
        for a, b, c in part.mesh.faces:
            for u, v in ((a, b), (b, c), (c, a)):
                edges[(u, v)] = edges.get((u, v), 0) + 1
        assert set(edges.values()) == {1}, f"{part.name}: not a manifold"
        assert all((v, u) in edges for u, v in edges), f"{part.name}: not closed"


@pytest.mark.parametrize("scale", [1.0, 1.25])
def test_scale_follows_the_spec_head_length(scale: float) -> None:
    spec = json.loads((SPECS / "full_body_spec_anthro_driver.json").read_text())
    anchor = resolve_head_anchor(spec)
    settings = HeadSettings(model="cc0_mesh", scale=scale)
    skin = _by_name(build_head_parts(anchor.length_m, settings))["skin_mesh"]
    top = skin.mesh.vertices[:, 2].max()
    assert top == pytest.approx(scale * anchor.length_m, abs=1e-4)
    assert skin.mesh.vertices[:, 2].min() == pytest.approx(0.0, abs=1e-4)


def test_face_points_forward_in_the_canonical_frame() -> None:
    v = _by_name(build_head_parts(LENGTH, MESH))["skin_mesh"].mesh.vertices
    assert v[:, 0].max() > -v[:, 0].min()  # nose and brow reach further than the back
    assert abs(v[:, 1].mean()) < 1e-3  # symmetric about the mid-sagittal plane
    nose_height = v[np.argmax(v[:, 0]), 2]
    assert 0.3 * LENGTH < nose_height < 0.7 * LENGTH


def test_hair_sits_on_the_crown_above_the_brow_and_outside_the_skull() -> None:
    parts = _by_name(build_head_parts(LENGTH, MESH))
    hair, skin = parts["hair_mesh"].mesh.vertices, parts["skin_mesh"].mesh.vertices
    assert hair[:, 2].max() > skin[:, 2].max()  # covers the crown
    forehead = skin[(skin[:, 0] > 0.30 * LENGTH) & (np.abs(skin[:, 1]) < 0.02)]
    assert hair[:, 2].min() < forehead[:, 2].max()  # hairline reaches the forehead
    assert hair[:, 2].min() > 0.4 * LENGTH  # short: it never reaches the neck
    # Eyes sit at ~0.53 of the head length; no hair in front of the brow plane
    # may reach down to them (the face region stays bare).
    front = hair[:, 0] > 0.30 * LENGTH
    assert front.any() and hair[front][:, 2].min() > 0.62 * LENGTH


def test_left_right_symmetry_within_a_millimetre() -> None:
    v = _by_name(build_head_parts(LENGTH, MESH))["skin_mesh"].mesh.vertices
    mirrored = v * np.array([1.0, -1.0, 1.0])
    # Every vertex has a mirror partner within 1 mm (nearest in a coarse grid).
    keys = {tuple(np.rint(p / 1e-3).astype(int)) for p in v}
    hits = sum(
        any(
            (k[0] + dx, k[1] + dy, k[2] + dz) in keys
            for dx in (-1, 0, 1)
            for dy in (-1, 0, 1)
            for dz in (-1, 0, 1)
        )
        for k in (tuple(np.rint(p / 1e-3).astype(int)) for p in mirrored)
    )
    assert hits / len(v) > 0.98


def test_deterministic() -> None:
    a = build_head_parts(LENGTH, MESH)
    b = build_head_parts(LENGTH, MESH)
    for p, q in zip(a, b, strict=True):
        np.testing.assert_array_equal(p.mesh.vertices, q.mesh.vertices)


def test_parametric_head_stays_selectable_and_unchanged() -> None:
    cfg = HeadSettings(model="parametric")
    names = [p.name for p in build_head_parts(LENGTH, cfg)]
    assert names[:2] == ["neck", "skull"] and "nose" in names and "hair" in names
    assert not any(n.endswith("_mesh") for n in names)


def test_schema_round_trip_and_invalid_model() -> None:
    doc = document_from_dict(
        {"schema_version": "appearance-v1", "head": {"model": "cc0_mesh"}}
    )
    assert doc.head.model == "cc0_mesh"
    assert document_from_dict(document_to_dict(doc)) == doc
    # The default (cc0_mesh, #12949) stays out of the document.
    plain = document_to_dict(document_from_dict({"schema_version": "appearance-v1"}))
    assert "model" not in plain["head"]
    with pytest.raises(ValueError):
        document_from_dict({"schema_version": "appearance-v1", "head": {"model": "x"}})


def test_resolved_appearance_carries_the_head_mesh_reference() -> None:
    spec = json.loads((SPECS / "full_body_spec_anthro_driver.json").read_text())
    on = resolve_appearance(
        spec,
        document_from_dict(
            {"schema_version": "appearance-v1", "head": {"model": "cc0_mesh"}}
        ),
    )
    assert on.head_mesh is not None
    assert on.head_mesh.licence == "CC0-1.0"
    assert on.head_mesh.path.endswith("head_cc0.npz")
    off = resolve_appearance(
        spec,
        document_from_dict(
            {"schema_version": "appearance-v1", "head": {"model": "parametric"}}
        ),
    )
    assert off.head_mesh is None
    default = resolve_appearance(
        spec, document_from_dict({"schema_version": "appearance-v1"})
    )
    assert default.head_mesh == on.head_mesh


def test_head_disabled_keeps_the_old_primitive_head_exactly() -> None:
    from src.shared.python.model_appearance.body_meshes import body_meshes

    spec = json.loads((SPECS / "full_body_spec_anthro_driver.json").read_text())
    plain = document_from_dict(
        {"schema_version": "appearance-v1", "head": {"enabled": False}}
    )
    meshed = document_from_dict(
        {
            "schema_version": "appearance-v1",
            "head": {"enabled": False, "model": "cc0_mesh"},
        }
    )
    a, b = body_meshes(spec, plain), body_meshes(spec, meshed)
    assert [(m.body, m.label, m.material_name) for m in a] == [
        (m.body, m.label, m.material_name) for m in b
    ]
    for m, n in zip(a, b, strict=True):
        np.testing.assert_array_equal(
            m.mesh_in_body_frame.vertices, n.mesh_in_body_frame.vertices
        )
