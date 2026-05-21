# ruff: noqa: E501
"""Tests for pipe_database.py — get_roughness, get_pipe_spec, etc.

Targets: 29% → 100% coverage.
"""

from __future__ import annotations

import pytest
from upstream_drift_tools.process_calculators.pressure_drop_calculator.utils.pipe_database import (
    MATERIAL_ROUGHNESS,
    STEEL_PIPE_DIMENSIONS,
    create_custom_pipe,
    get_pipe_spec,
    get_roughness,
    list_available_sizes,
    list_schedules_for_size,
)

# ---------------------------------------------------------------------------
# get_roughness
# ---------------------------------------------------------------------------


class TestGetRoughness:
    def test_commercial_steel_in_meters(self):
        """Lines 75-76: unit='m' returns mm/1000."""
        r = get_roughness("Commercial Steel", unit="m")
        assert abs(r - 0.045 / 1000.0) < 1e-10

    def test_commercial_steel_in_mm(self):
        """Lines 77-78: unit='mm' returns mm value."""
        r = get_roughness("Commercial Steel", unit="mm")
        assert r == 0.045

    def test_commercial_steel_in_ft(self):
        """Lines 79-80: unit='ft' returns ft value."""
        r = get_roughness("Commercial Steel", unit="ft")
        assert r == 0.00015

    def test_unknown_material_raises(self):
        """Lines 70-71: unknown material → ValueError."""
        with pytest.raises(ValueError, match="not found in database"):
            get_roughness("Magic Material", unit="m")

    def test_unknown_unit_raises(self):
        """Lines 81-82: unknown unit → ValueError."""
        with pytest.raises(ValueError, match="not recognized"):
            get_roughness("Commercial Steel", unit="cm")

    def test_all_materials_resolve_in_meters(self):
        """Every entry in MATERIAL_ROUGHNESS can be fetched in meters."""
        for material in MATERIAL_ROUGHNESS:
            result = get_roughness(material, "m")
            assert result > 0


# ---------------------------------------------------------------------------
# get_pipe_spec
# ---------------------------------------------------------------------------


class TestGetPipeSpec:
    def test_4_inch_schedule_40(self):
        """Lines 309-324: known NPS+schedule → PipeSpecification."""
        spec = get_pipe_spec("4", "40")
        assert abs(spec.inner_diameter - 102.26) < 0.01
        assert spec.nominal_size == "4"
        assert spec.schedule == "40"
        assert spec.material == "Commercial Steel"

    def test_custom_material_stored(self):
        """Line 323: material propagated to PipeSpecification."""
        spec = get_pipe_spec("2", "STD", material="Stainless Steel 316")
        assert spec.material == "Stainless Steel 316"

    def test_unknown_size_raises(self):
        """Lines 310-313: unknown key → ValueError."""
        with pytest.raises(ValueError):
            get_pipe_spec("99", "40")

    def test_known_schedule_xxthin(self):
        """NPS 1/2 Schedule 5S."""
        spec = get_pipe_spec("1/2", "5S")
        assert abs(spec.outer_diameter - 21.3) < 0.01

    def test_dimensions_are_positive(self):
        """All pipe specs have positive OD, wall, ID."""
        for (nps, sch), (od, wall, id_val) in STEEL_PIPE_DIMENSIONS.items():
            assert od > 0, f"OD should be positive for ({nps}, {sch})"
            assert wall > 0, f"Wall should be positive for ({nps}, {sch})"
            assert id_val > 0, f"ID should be positive for ({nps}, {sch})"


# ---------------------------------------------------------------------------
# list_available_sizes
# ---------------------------------------------------------------------------


class TestListAvailableSizes:
    def test_returns_list_of_strings(self):
        """Lines 328-333: returns sorted list."""
        sizes = list_available_sizes()
        assert isinstance(sizes, list)
        assert len(sizes) > 0

    def test_contains_expected_sizes(self):
        sizes = list_available_sizes()
        assert "2" in sizes
        assert "4" in sizes
        assert "6" in sizes
        assert "8" in sizes

    def test_smallest_size_is_half_inch(self):
        """Fractional size '1/2' should be in the list."""
        sizes = list_available_sizes()
        assert "1/2" in sizes
        # Verify sizes are unique strings
        assert all(isinstance(s, str) for s in sizes)

    def test_no_duplicates(self):
        sizes = list_available_sizes()
        assert len(sizes) == len(set(sizes))


# ---------------------------------------------------------------------------
# list_schedules_for_size
# ---------------------------------------------------------------------------


class TestListSchedulesForSize:
    def test_4_inch_has_standard_schedules(self):
        """Lines 336-341: NPS 4 has '40', '80', 'STD', 'XS'."""
        schedules = list_schedules_for_size("4")
        assert "40" in schedules
        assert "80" in schedules
        assert "STD" in schedules
        assert "XS" in schedules

    def test_unknown_size_returns_empty(self):
        """If size doesn't exist, returns empty list."""
        schedules = list_schedules_for_size("99")
        assert schedules == []

    def test_returns_all_available_schedules(self):
        """All schedules for NPS 6 are present."""
        schedules = list_schedules_for_size("6")
        assert len(schedules) > 0
        assert "40" in schedules
        assert "80" in schedules


# ---------------------------------------------------------------------------
# create_custom_pipe
# ---------------------------------------------------------------------------


class TestCreateCustomPipe:
    def test_defaults_to_commercial_steel(self):
        """Lines 355-363: material defaults to Commercial Steel."""
        spec = create_custom_pipe(100.0)
        assert spec.nominal_size == "Custom"
        assert spec.schedule == "Custom"
        assert spec.inner_diameter == 100.0
        assert spec.outer_diameter == 100.0
        assert spec.wall_thickness == 0.0
        assert spec.material == "Commercial Steel"

    def test_custom_material(self):
        """Line 362: custom material stored."""
        spec = create_custom_pipe(50.0, material="PVC")
        assert spec.material == "PVC"

    def test_custom_diameter_stored(self):
        spec = create_custom_pipe(200.0)
        assert spec.inner_diameter == 200.0
