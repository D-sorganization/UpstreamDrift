"""The owner-ruled CC0 short-hair head is the default head (#12949, epic #12678)."""

import pytest

from src.shared.python.model_appearance import (
    build_head_parts,
    document_from_dict,
    document_to_dict,
)
from src.shared.python.model_appearance.schema import HeadSettings

BARE = {"schema_version": "appearance-v1"}
CC0_PARTS = {"skin_mesh", "hair_mesh"}


def _names(doc) -> set[str]:
    return {p.name for p in build_head_parts(0.2429, doc.head)}


@pytest.mark.unit
def test_default_head_model_is_cc0_mesh() -> None:
    assert HeadSettings().model == "cc0_mesh"
    assert HeadSettings().headwear == "hair"


@pytest.mark.unit
def test_bare_document_resolves_cc0_head_with_hair() -> None:
    assert _names(document_from_dict(BARE)) == CC0_PARTS


@pytest.mark.unit
def test_parametric_stays_selectable_and_round_trips() -> None:
    doc = document_from_dict({**BARE, "head": {"model": "parametric"}})
    assert doc.head.model == "parametric"
    assert "skull" in _names(doc)
    assert document_to_dict(doc)["head"]["model"] == "parametric"
    again = document_from_dict(document_to_dict(doc))
    assert again.head.model == "parametric"


@pytest.mark.unit
def test_default_document_round_trips_as_cc0() -> None:
    doc = document_from_dict(document_to_dict(document_from_dict(BARE)))
    assert doc.head.model == "cc0_mesh"


@pytest.mark.unit
def test_cap_without_explicit_model_keeps_the_parametric_cap() -> None:
    doc = document_from_dict({**BARE, "head": {"headwear": "cap"}})
    assert doc.head.model == "parametric"
    assert "cap" in _names(doc)


@pytest.mark.unit
def test_cap_with_explicit_cc0_model_fails_closed() -> None:
    doc = document_from_dict({**BARE, "head": {"headwear": "cap", "model": "cc0_mesh"}})
    with pytest.raises(ValueError, match="cap"):
        build_head_parts(0.2429, doc.head)


@pytest.mark.unit
def test_missing_cc0_asset_fails_closed(monkeypatch, tmp_path) -> None:
    from src.shared.python.model_appearance import head_mesh

    monkeypatch.setattr(head_mesh, "ASSET_PATH", tmp_path / "gone.npz")
    monkeypatch.setattr(head_mesh, "PROVENANCE_PATH", tmp_path / "gone.json")
    head_mesh._default_asset.cache_clear()
    try:
        with pytest.raises(ValueError, match="provenance"):
            build_head_parts(0.2429, HeadSettings())
    finally:
        head_mesh._default_asset.cache_clear()
