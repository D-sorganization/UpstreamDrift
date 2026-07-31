"""Unit tests for motion_pipeline.scaling.marker_maps."""

from __future__ import annotations

import pytest

from src.shared.python.motion_pipeline.scaling.marker_maps import (
    IOR,
    MARKER_SETS,
    PLUG_IN_GAIT,
    THEIA,
    VICON_FULL_BODY,
    MarkerSet,
    get_marker_set,
)


@pytest.mark.parametrize(
    "marker_set,expected_name",
    [
        (PLUG_IN_GAIT, "Plug-in-Gait"),
        (IOR, "IOR"),
        (THEIA, "Theia"),
        (VICON_FULL_BODY, "Vicon-Full-Body"),
    ],
)
def test_marker_set_names(marker_set: MarkerSet, expected_name: str) -> None:
    assert marker_set.name == expected_name


@pytest.mark.parametrize(
    "marker_set",
    [PLUG_IN_GAIT, IOR, THEIA, VICON_FULL_BODY],
)
def test_marker_sets_are_non_empty(marker_set: MarkerSet) -> None:
    assert len(marker_set.markers) > 0
    assert len(marker_set.marker_to_segment) > 0
    assert len(marker_set.segment_pairs) > 0


@pytest.mark.parametrize(
    "marker_set",
    [PLUG_IN_GAIT, IOR, THEIA, VICON_FULL_BODY],
)
def test_marker_to_segment_keys_subset_of_markers(marker_set: MarkerSet) -> None:
    """Every key in marker_to_segment should be a real marker name."""
    for marker_name in marker_set.marker_to_segment:
        assert (
            marker_name in marker_set.markers
        ), f"{marker_name} maps to segment but is not in markers list"


@pytest.mark.parametrize(
    "marker_set",
    [PLUG_IN_GAIT, IOR, THEIA, VICON_FULL_BODY],
)
def test_segment_pairs_reference_valid_markers(marker_set: MarkerSet) -> None:
    for prox, dist in marker_set.segment_pairs:
        assert prox in marker_set.markers
        assert dist in marker_set.markers


@pytest.mark.parametrize(
    "alias",
    ["plug-in-gait", "plugingait", "pig", "ior", "theia", "vicon", "vicon-full-body"],
)
def test_get_marker_set_known_aliases(alias: str) -> None:
    assert isinstance(get_marker_set(alias), MarkerSet)


def test_get_marker_set_case_insensitive() -> None:
    a = get_marker_set("plug-in-gait")
    b = get_marker_set("PLUG-IN-GAIT")
    assert a is b


def test_get_marker_set_unknown_raises() -> None:
    with pytest.raises(ValueError, match="Unknown marker set"):
        get_marker_set("not-a-real-set")


def test_marker_sets_registry_contains_all_named() -> None:
    expected_keys = {
        "plug-in-gait",
        "plugingait",
        "pig",
        "ior",
        "theia",
        "vicon",
        "vicon-full-body",
    }
    assert expected_keys.issubset(set(MARKER_SETS.keys()))


def test_plug_in_gait_pelvis_markers_documented() -> None:
    pelvis_markers = {
        m for m, seg in PLUG_IN_GAIT.marker_to_segment.items() if seg == "pelvis"
    }
    assert pelvis_markers == {"RASI", "LASI", "RPSI", "LPSI"}


def test_theia_has_left_right_symmetry_for_legs() -> None:
    """Theia segments map symmetrically across left and right."""
    seg_set = set(THEIA.marker_to_segment.values())
    assert "left_thigh" in seg_set
    assert "right_thigh" in seg_set
    assert "left_shank" in seg_set
    assert "right_shank" in seg_set
