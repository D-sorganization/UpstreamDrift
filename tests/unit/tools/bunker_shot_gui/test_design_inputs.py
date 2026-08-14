"""The designer input layer: vocabulary in, value objects out (issue #8618).

These tests never import Qt. They check that the W2/W3/swing controls a
designer types map onto the ADR-0032 value objects without mixing the two
bounce conventions and without letting an impossible sole through.
"""

from __future__ import annotations

import math

import pytest

from bunkershot3d.geometry import WedgeGeometry, get_preset
from bunkershot3d.sand import PlayingCondition
from src.tools.bunker_shot_gui.design import (
    DEFAULT_GRIND_PRESET,
    FIRMNESS_RANGE_KG_PER_CM2,
    SandCondition,
    SolverSetup,
    SwingSetup,
    WedgeDesign,
    WorkbenchInputError,
    grind_preset_names,
    playing_condition_names,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


class TestWedgeDesign:
    def test_default_design_resolves_to_its_preset(self) -> None:
        geometry = WedgeDesign(name="A").geometry()
        preset = get_preset(DEFAULT_GRIND_PRESET).geometry
        assert isinstance(geometry, WedgeGeometry)
        assert geometry.loft_deg == pytest.approx(preset.loft_deg)
        assert geometry.sole_width_m == pytest.approx(preset.sole_width_m)

    def test_marketed_bounce_is_the_design_variable(self) -> None:
        """The number a designer types is the number the geometry reports."""
        geometry = WedgeDesign(name="A", marketed_bounce_deg=11.0).geometry()
        assert geometry.marketed_bounce.angle_deg == pytest.approx(11.0)

    def test_geometric_bounce_is_rederived_not_carried(self) -> None:
        """Widening the sole must not silently move the marketed bounce.

        The patent convention and the marketed convention are not
        interchangeable (ADR-0032 structural decision 2); the design layer
        pins the marketed one and recomputes the geometric one.
        """
        narrow = WedgeDesign(name="A", marketed_bounce_deg=10.0, sole_width_mm=18.0)
        wide = WedgeDesign(name="A", marketed_bounce_deg=10.0, sole_width_mm=24.0)
        first, second = narrow.geometry(), wide.geometry()
        assert first.marketed_bounce.angle_deg == pytest.approx(10.0)
        assert second.marketed_bounce.angle_deg == pytest.approx(10.0)
        assert first.geometric_bounce.angle_deg != pytest.approx(
            second.geometric_bounce.angle_deg
        )

    def test_blank_name_is_rejected(self) -> None:
        with pytest.raises(WorkbenchInputError, match="non-empty name"):
            WedgeDesign(name="   ")

    def test_unknown_preset_is_rejected_by_name(self) -> None:
        with pytest.raises(WorkbenchInputError, match="unknown grind preset"):
            WedgeDesign(name="A", grind_preset="no-such-grind")

    def test_impossible_sole_is_an_input_error_not_a_crash(self) -> None:
        design = WedgeDesign(name="A", heel_relief_fraction=0.85)
        with pytest.raises(WorkbenchInputError, match="constructible sole"):
            design.geometry()

    def test_every_preset_resolves(self) -> None:
        for name in grind_preset_names():
            assert isinstance(
                WedgeDesign(name=name, grind_preset=name).geometry(), WedgeGeometry
            )


class TestSandCondition:
    def test_preset_names_cover_the_four_playing_conditions(self) -> None:
        assert set(playing_condition_names()) == {
            condition.value for condition in PlayingCondition
        }

    def test_default_condition_is_firm(self) -> None:
        assert SandCondition().preset is PlayingCondition.FIRM

    def test_firmness_override_reaches_the_sand_state(self) -> None:
        state = SandCondition().with_firmness(2.0).sand_state()
        assert state.firmness_kg_per_cm2 == pytest.approx(2.0)

    def test_firmness_override_holds_the_preset_gradation(self) -> None:
        """The published sweep isolates firmness; nothing else may move."""
        base = SandCondition().sand_state()
        swept = SandCondition().with_firmness(1.8).sand_state()
        assert swept.d50_m == pytest.approx(base.d50_m)
        assert swept.angularity is base.angularity

    def test_string_preset_is_accepted(self) -> None:
        assert SandCondition(preset="wet").preset is PlayingCondition.WET

    def test_unknown_preset_rejected(self) -> None:
        with pytest.raises(WorkbenchInputError, match="unknown playing condition"):
            SandCondition(preset="soggy")

    @pytest.mark.parametrize("value", [0.0, -1.0, math.nan])
    def test_non_physical_firmness_rejected(self, value: float) -> None:
        with pytest.raises(WorkbenchInputError, match="positive penetrometer"):
            SandCondition(firmness_kg_per_cm2=value)

    def test_published_sweep_bounds_are_the_documented_band(self) -> None:
        assert FIRMNESS_RANGE_KG_PER_CM2 == (1.6, 2.8)


class TestSwingSetup:
    def test_default_delivery_is_inertially_dominated(self) -> None:
        """25 m/s is well past the 6.8 m/s depth/inertia crossover."""
        assert SwingSetup().swing_condition().is_inertially_dominated

    def test_ascending_blow_is_refused(self) -> None:
        with pytest.raises(WorkbenchInputError, match="must be negative"):
            SwingSetup(attack_angle_deg=2.0)

    def test_level_blow_is_refused(self) -> None:
        with pytest.raises(WorkbenchInputError, match="must be negative"):
            SwingSetup(attack_angle_deg=0.0)

    def test_entry_at_or_past_the_ball_is_refused(self) -> None:
        with pytest.raises(WorkbenchInputError, match="entry_distance_behind_ball_m"):
            SwingSetup(entry_distance_behind_ball_m=0.0)

    def test_implausible_speed_is_refused(self) -> None:
        with pytest.raises(WorkbenchInputError, match="unusable swing condition"):
            SwingSetup(clubhead_speed_mps=120.0)

    def test_delivery_carries_the_three_angles(self) -> None:
        delivery = SwingSetup(
            attack_angle_deg=-9.0, face_open_deg=12.0, shaft_lean_deg=5.0
        ).delivery()
        assert delivery.attack_angle_deg == pytest.approx(-9.0)
        assert delivery.face_open_deg == pytest.approx(12.0)
        assert delivery.shaft_lean_deg == pytest.approx(5.0)

    def test_with_attack_angle_is_non_mutating(self) -> None:
        original = SwingSetup()
        swept = original.with_attack_angle(-11.0)
        assert original.attack_angle_deg == pytest.approx(-8.0)
        assert swept.attack_angle_deg == pytest.approx(-11.0)


class TestSolverSetup:
    def test_defaults_are_constructible(self) -> None:
        assert SolverSetup().playability_points >= 2

    @pytest.mark.parametrize("stations", [0, 4])
    def test_too_few_stations_rejected(self, stations: int) -> None:
        with pytest.raises(WorkbenchInputError, match="n_stations"):
            SolverSetup(n_stations=stations)

    def test_too_few_profile_points_rejected(self) -> None:
        with pytest.raises(WorkbenchInputError, match="n_profile_points"):
            SolverSetup(n_profile_points=11)

    def test_step_larger_than_horizon_rejected(self) -> None:
        with pytest.raises(WorkbenchInputError, match="exceeds max_time_s"):
            SolverSetup(time_step_s=1.0, max_time_s=0.01)

    @pytest.mark.parametrize("tolerance", [0.0, 1.5])
    def test_tolerance_outside_unit_interval_rejected(self, tolerance: float) -> None:
        with pytest.raises(WorkbenchInputError, match="carry_tolerance_fraction"):
            SolverSetup(carry_tolerance_fraction=tolerance)

    def test_single_point_grid_rejected(self) -> None:
        with pytest.raises(WorkbenchInputError, match="at least 2 stations"):
            SolverSetup(playability_points=1)
