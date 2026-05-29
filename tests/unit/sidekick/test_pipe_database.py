"""Tests for pressure_drop_calculator pipe_database utilities (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from sidekick.process_calculators.pressure_drop_calculator.utils.pipe_database import (
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
    def test_commercial_steel_meters(self) -> None:
        r = get_roughness("Commercial Steel", "m")
        assert r > 0.0

    def test_commercial_steel_mm_larger_than_m(self) -> None:
        r_m = get_roughness("Commercial Steel", "m")
        r_mm = get_roughness("Commercial Steel", "mm")
        assert r_mm > r_m

    def test_smooth_lower_roughness_than_concrete(self) -> None:
        r_smooth = get_roughness("Glass", "m")
        r_concrete = get_roughness("Concrete", "m")
        assert r_smooth < r_concrete

    def test_unknown_material_raises(self) -> None:
        with pytest.raises(ValueError):
            get_roughness("UNKNOWN_MATERIAL_XYZ")

    def test_unknown_unit_raises(self) -> None:
        with pytest.raises(ValueError):
            get_roughness("Commercial Steel", "light_years")


# ---------------------------------------------------------------------------
# get_pipe_spec
# ---------------------------------------------------------------------------


class TestGetPipeSpec:
    def test_known_spec_returns_object(self) -> None:
        spec = get_pipe_spec("4", "40")
        assert spec is not None

    def test_inner_diameter_positive(self) -> None:
        spec = get_pipe_spec("4", "40")
        assert spec.inner_diameter > 0.0

    def test_inner_diam_less_than_outer(self) -> None:
        spec = get_pipe_spec("4", "40")
        assert spec.inner_diameter < spec.outer_diameter

    def test_unknown_size_raises(self) -> None:
        with pytest.raises(ValueError):
            get_pipe_spec("999", "40")

    def test_unknown_schedule_raises(self) -> None:
        with pytest.raises(ValueError):
            get_pipe_spec("4", "SCHEDULE_NEVER")

    def test_heavier_schedule_thicker_wall(self) -> None:
        sch40 = get_pipe_spec("4", "40")
        sch80 = get_pipe_spec("4", "80")
        assert sch80.wall_thickness > sch40.wall_thickness


# ---------------------------------------------------------------------------
# list_available_sizes / list_schedules_for_size
# ---------------------------------------------------------------------------


class TestListHelpers:
    def test_list_available_sizes_nonempty(self) -> None:
        sizes = list_available_sizes()
        assert len(sizes) > 0

    def test_list_available_sizes_contains_4(self) -> None:
        sizes = list_available_sizes()
        assert "4" in sizes

    def test_list_schedules_for_size_nonempty(self) -> None:
        schedules = list_schedules_for_size("4")
        assert len(schedules) > 0

    def test_list_schedules_for_size_contains_40(self) -> None:
        schedules = list_schedules_for_size("4")
        assert "40" in schedules


# ---------------------------------------------------------------------------
# create_custom_pipe
# ---------------------------------------------------------------------------


class TestCreateCustomPipe:
    def test_returns_spec(self) -> None:
        spec = create_custom_pipe(50.0)
        assert spec is not None

    def test_inner_diameter_matches(self) -> None:
        spec = create_custom_pipe(75.0)
        assert spec.inner_diameter == 75.0

    def test_custom_nominal_size(self) -> None:
        spec = create_custom_pipe(50.0)
        assert spec.nominal_size == "Custom"
