"""
Comprehensive unit tests for humanoid_character_builder.core.anthropometry.

Tests cover:
- de Leva data loading and completeness
- Gender interpolation (male=1.0, female=0.0, neutral=0.5)
- Segment mass estimation and normalization
- Segment dimension estimation
- Inertia estimation from gyration radii
- COM location computation
- Extreme height/mass values
- Segment name mapping
"""

from __future__ import annotations

import pytest
from humanoid_character_builder.core.anthropometry import (
    _SEGMENT_NAME_MAP,
    DE_LEVA_DATA,
    estimate_segment_dimensions,
    estimate_segment_inertia_from_gyration,
    estimate_segment_masses,
    get_anthropometry_key,
    get_com_location,
    get_segment_length_ratio,
    get_segment_mass_ratio,
)

# ── de Leva Data Loading ────────────────────────────────────────────────────


class TestDeLeva:
    """Test that the de Leva anthropometric data is loaded and complete."""

    EXPECTED_SEGMENTS = {
        "head",
        "neck",
        "thorax",
        "lumbar",
        "pelvis",
        "upper_arm",
        "forearm",
        "hand",
        "thigh",
        "shin",
        "foot",
        "shoulder",
        "hip",
    }

    def test_male_segments_present(self) -> None:
        for seg in self.EXPECTED_SEGMENTS:
            assert seg in DE_LEVA_DATA.male, f"Missing male segment: {seg}"

    def test_female_segments_present(self) -> None:
        for seg in self.EXPECTED_SEGMENTS:
            assert seg in DE_LEVA_DATA.female, f"Missing female segment: {seg}"

    def test_male_mass_ratios_positive(self) -> None:
        for name, data in DE_LEVA_DATA.male.items():
            assert data.mass_ratio > 0, f"Male {name} mass_ratio <= 0"

    def test_female_mass_ratios_positive(self) -> None:
        for name, data in DE_LEVA_DATA.female.items():
            assert data.mass_ratio > 0, f"Female {name} mass_ratio <= 0"

    def test_length_ratios_positive(self) -> None:
        for name, data in DE_LEVA_DATA.male.items():
            assert data.length_ratio > 0, f"Male {name} length_ratio <= 0"

    def test_gyration_radii_positive(self) -> None:
        for name, data in DE_LEVA_DATA.male.items():
            assert data.gyration_sagittal > 0, f"Male {name} sagittal <= 0"
            assert data.gyration_transverse > 0, f"Male {name} transverse <= 0"
            assert data.gyration_longitudinal > 0, f"Male {name} longitudinal <= 0"

    def test_com_proximal_ratio_in_range(self) -> None:
        for name, data in DE_LEVA_DATA.male.items():
            ratio = data.com_proximal_ratio
            msg = f"Male {name} com_proximal_ratio out of [0,1]"
            assert 0.0 < ratio < 1.0, msg


# ── Gender Interpolation ────────────────────────────────────────────────────


class TestGenderInterpolation:
    """Test gender interpolation between male and female data."""

    def test_male_factor_returns_male_data(self) -> None:
        data = DE_LEVA_DATA.get_segment_data("head", gender_factor=1.0)
        male = DE_LEVA_DATA.male["head"]
        assert data.mass_ratio == pytest.approx(male.mass_ratio)

    def test_female_factor_returns_female_data(self) -> None:
        data = DE_LEVA_DATA.get_segment_data("head", gender_factor=0.0)
        female = DE_LEVA_DATA.female["head"]
        assert data.mass_ratio == pytest.approx(female.mass_ratio)

    def test_neutral_is_average(self) -> None:
        data = DE_LEVA_DATA.get_segment_data("head", gender_factor=0.5)
        male = DE_LEVA_DATA.male["head"]
        female = DE_LEVA_DATA.female["head"]
        expected = (male.mass_ratio + female.mass_ratio) / 2.0
        assert data.mass_ratio == pytest.approx(expected)

    def test_all_fields_interpolated(self) -> None:
        data = DE_LEVA_DATA.get_segment_data("thigh", gender_factor=0.3)
        male = DE_LEVA_DATA.male["thigh"]
        female = DE_LEVA_DATA.female["thigh"]
        expected_length = female.length_ratio + 0.3 * (male.length_ratio - female.length_ratio)
        assert data.length_ratio == pytest.approx(expected_length)

    def test_unknown_segment_returns_default(self) -> None:
        data = DE_LEVA_DATA.get_segment_data("nonexistent_segment", gender_factor=0.5)
        assert data is not None
        assert data.mass_ratio > 0


# ── Segment Mass Estimation ─────────────────────────────────────────────────


class TestSegmentMasses:
    """Test estimate_segment_masses function."""

    def test_total_mass_preserved(self) -> None:
        total = 75.0
        masses = estimate_segment_masses(total)
        assert abs(sum(masses.values()) - total) < 1e-6

    def test_all_segments_present(self) -> None:
        masses = estimate_segment_masses(75.0)
        for seg_name in _SEGMENT_NAME_MAP:
            assert seg_name in masses, f"Missing segment: {seg_name}"

    def test_all_masses_positive(self) -> None:
        masses = estimate_segment_masses(75.0)
        for name, mass in masses.items():
            assert mass > 0, f"Segment {name} has non-positive mass: {mass}"

    def test_male_vs_female_differ(self) -> None:
        male_masses = estimate_segment_masses(75.0, gender_factor=1.0)
        female_masses = estimate_segment_masses(75.0, gender_factor=0.0)
        # At least some segments should differ
        diffs = [abs(male_masses[k] - female_masses[k]) for k in male_masses]
        assert max(diffs) > 0.01  # non-trivial difference

    def test_heavy_person(self) -> None:
        masses = estimate_segment_masses(150.0)
        assert abs(sum(masses.values()) - 150.0) < 1e-6

    def test_light_person(self) -> None:
        masses = estimate_segment_masses(30.0)
        assert abs(sum(masses.values()) - 30.0) < 1e-6
        assert all(m > 0 for m in masses.values())


# ── Segment Dimension Estimation ─────────────────────────────────────────────


class TestSegmentDimensions:
    """Test estimate_segment_dimensions function."""

    def test_all_segments_have_dimensions(self) -> None:
        dims = estimate_segment_dimensions(1.75)
        for seg_name in _SEGMENT_NAME_MAP:
            assert seg_name in dims, f"Missing segment: {seg_name}"
            assert "length" in dims[seg_name]
            assert "width" in dims[seg_name]
            assert "depth" in dims[seg_name]

    def test_all_dimensions_positive(self) -> None:
        dims = estimate_segment_dimensions(1.75)
        for name, d in dims.items():
            assert d["length"] > 0, f"{name} length not positive"
            assert d["width"] > 0, f"{name} width not positive"
            assert d["depth"] > 0, f"{name} depth not positive"

    def test_taller_person_has_longer_segments(self) -> None:
        short = estimate_segment_dimensions(1.50)
        tall = estimate_segment_dimensions(2.00)
        for seg_name in _SEGMENT_NAME_MAP:
            t_len = tall[seg_name]["length"]
            s_len = short[seg_name]["length"]
            assert t_len > s_len, f"Taller should have longer {seg_name}"

    def test_dimensions_scale_linearly_with_height(self) -> None:
        dims_1 = estimate_segment_dimensions(1.0)
        dims_2 = estimate_segment_dimensions(2.0)
        for seg_name in _SEGMENT_NAME_MAP:
            ratio = dims_2[seg_name]["length"] / dims_1[seg_name]["length"]
            assert ratio == pytest.approx(2.0, rel=1e-6)


# ── Inertia from Gyration ───────────────────────────────────────────────────


class TestInertiaFromGyration:
    """Test estimate_segment_inertia_from_gyration."""

    def test_returns_all_components(self) -> None:
        result = estimate_segment_inertia_from_gyration("thigh", 10.0, 0.4)
        assert "ixx" in result
        assert "iyy" in result
        assert "izz" in result
        assert "ixy" in result
        assert result["ixy"] == 0.0

    def test_inertia_positive(self) -> None:
        result = estimate_segment_inertia_from_gyration("head", 5.0, 0.2)
        assert result["ixx"] > 0
        assert result["iyy"] > 0
        assert result["izz"] > 0

    def test_inertia_scales_with_mass(self) -> None:
        light = estimate_segment_inertia_from_gyration("forearm", 1.0, 0.3)
        heavy = estimate_segment_inertia_from_gyration("forearm", 2.0, 0.3)
        assert heavy["ixx"] == pytest.approx(2.0 * light["ixx"])

    def test_inertia_scales_with_length_squared(self) -> None:
        short = estimate_segment_inertia_from_gyration("shin", 5.0, 0.2)
        long = estimate_segment_inertia_from_gyration("shin", 5.0, 0.4)
        assert long["ixx"] == pytest.approx(4.0 * short["ixx"])


# ── COM Location ─────────────────────────────────────────────────────────────


class TestCOMLocation:
    """Test get_com_location."""

    def test_com_along_z_axis(self) -> None:
        com = get_com_location("thigh", 0.4)
        assert com[0] == 0.0
        assert com[1] == 0.0
        assert 0.0 < com[2] < 0.4

    def test_com_scales_with_length(self) -> None:
        com_short = get_com_location("forearm", 0.2)
        com_long = get_com_location("forearm", 0.4)
        assert com_long[2] == pytest.approx(2.0 * com_short[2])

    def test_com_differs_by_gender(self) -> None:
        com_male = get_com_location("thigh", 0.4, gender_factor=1.0)
        com_female = get_com_location("thigh", 0.4, gender_factor=0.0)
        # Should differ due to different com_proximal_ratio
        assert com_male[2] != pytest.approx(com_female[2])


# ── Segment Name Mapping ────────────────────────────────────────────────────


class TestSegmentNameMapping:
    """Test get_anthropometry_key mapping."""

    def test_left_right_map_to_same(self) -> None:
        assert get_anthropometry_key("left_upper_arm") == get_anthropometry_key("right_upper_arm")
        assert get_anthropometry_key("left_foot") == get_anthropometry_key("right_foot")

    def test_direct_name_passthrough(self) -> None:
        assert get_anthropometry_key("head") == "head"
        assert get_anthropometry_key("pelvis") == "pelvis"

    def test_unknown_name_passthrough(self) -> None:
        assert get_anthropometry_key("unknown_segment") == "unknown_segment"

    def test_mass_ratio_consistent(self) -> None:
        left = get_segment_mass_ratio("left_thigh")
        right = get_segment_mass_ratio("right_thigh")
        assert left == pytest.approx(right)

    def test_length_ratio_consistent(self) -> None:
        left = get_segment_length_ratio("left_forearm")
        right = get_segment_length_ratio("right_forearm")
        assert left == pytest.approx(right)


# ── Extreme Values ───────────────────────────────────────────────────────────


class TestExtremeValues:
    """Test behavior with extreme but valid inputs."""

    def test_very_tall_person(self) -> None:
        dims = estimate_segment_dimensions(3.0, gender_factor=0.5)
        assert all(d["length"] > 0 for d in dims.values())

    def test_very_short_person(self) -> None:
        dims = estimate_segment_dimensions(0.5, gender_factor=0.5)
        assert all(d["length"] > 0 for d in dims.values())

    def test_very_heavy_person(self) -> None:
        masses = estimate_segment_masses(300.0)
        assert abs(sum(masses.values()) - 300.0) < 1e-4
        assert all(m > 0 for m in masses.values())

    def test_very_light_person(self) -> None:
        masses = estimate_segment_masses(1.0)
        assert abs(sum(masses.values()) - 1.0) < 1e-6
        assert all(m > 0 for m in masses.values())
