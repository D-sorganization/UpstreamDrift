"""Variant roots and registry (#9793)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.motion_capture.variants import (
    ensure_variant,
    get_variant,
    list_variants,
    register_variant,
    variant_dir,
)

pytestmark = pytest.mark.unit


def test_variant_dir_resolves_default_and_named(tmp_path: Path) -> None:
    assert variant_dir(tmp_path, "") == tmp_path
    assert variant_dir(tmp_path, "all3") == tmp_path / "variants" / "all3"
    assert ensure_variant(tmp_path, "pair_fd").is_dir()
    for bad in ("a b", "x/y", "", "n" * 41):
        if bad == "":
            continue
        with pytest.raises(Exception, match="invalid variant name"):
            variant_dir(tmp_path, bad)


def test_register_and_list_variants(tmp_path: Path) -> None:
    first = register_variant(
        tmp_path, "pair_fd", views=("face_on", "down_line"), anchors=["shank=0.42"]
    )
    assert first.source == {"kind": "triangulate"} and first.created_utc
    register_variant(
        tmp_path,
        "cam_face",
        views=("face_on",),
        observation_set="observations_openpose",
        source={"kind": "image_space", "cameras_from": "pair_fd"},
    )
    register_variant(tmp_path, "", views=("face_on", "down_line", "overhead"))
    names = [v.name for v in list_variants(tmp_path)]
    assert names == ["", "cam_face", "pair_fd"]
    cam = get_variant(tmp_path, "cam_face")
    assert cam is not None and cam.source["cameras_from"] == "pair_fd"
    assert cam.observation_set == "observations_openpose"
    assert get_variant(tmp_path, "pair_fd").extra == {"anchors": ["shank=0.42"]}
    index = json.loads((tmp_path / "variants" / "index.json").read_text("utf-8"))
    assert index["schema_version"] == "variants-index/1.0.0"
    assert index["provenance"]["parameters"] == {"count": 3}
    # Re-registering replaces.
    register_variant(tmp_path, "pair_fd", views=("face_on", "overhead"))
    assert get_variant(tmp_path, "pair_fd").views == ("face_on", "overhead")
    assert get_variant(tmp_path, "missing") is None


def test_register_preconditions(tmp_path: Path) -> None:
    with pytest.raises(Exception, match="at least one view"):
        register_variant(tmp_path, "x", views=())
    with pytest.raises(Exception, match="source kind"):
        register_variant(tmp_path, "x", views=("a",), source={"kind": "magic"})
