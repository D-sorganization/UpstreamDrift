"""Preset and provenance-honesty tests (issue #8610, guarding against #7999).

No published bulk density, internal friction angle or angle of repose specific
to *golf bunker* sand was found in the literature. The DRFT constants used here
(phi = 34 deg, packing fraction 0.60, rho_grain = 2600 kg/m^3) come from
Quikrete medium sand, 0.3-0.8 mm, as an analogue. Every preset must say so.
Issue #7999 was exactly this failure mode -- a fabricated constant presented as
a calibration -- so these tests are a regression guard, not decoration.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest
from bunkershot3d.sand.firmness import (
    FIRMNESS_SWEEP_KG_PER_CM2,
    FirmnessRating,
)
from bunkershot3d.sand.moisture import MoistureRegime
from bunkershot3d.sand.presets import (
    USGA_GSR_2020_WINDY_PSD,
    USGA_LAB_MIDBAND_PSD,
    PlayingCondition,
    all_presets,
    firmness_sweep,
    playing_condition,
)
from bunkershot3d.sand.provenance import ProvenanceBasis
from bunkershot3d.sand.specification import (
    USGA_LAB_SPECIFICATION,
    evaluate_compliance,
)
from bunkershot3d.sand.state import SandState

pytestmark = pytest.mark.unit


class TestPresetSieveTables:
    def test_midband_is_usga_compliant(self) -> None:
        assert evaluate_compliance(USGA_LAB_MIDBAND_PSD, USGA_LAB_SPECIFICATION).passed

    def test_midband_uniformity_coefficient_is_in_band(self) -> None:
        assert 2.0 <= USGA_LAB_MIDBAND_PSD.uniformity_coefficient <= 5.0

    def test_windy_preset_targets_the_erosion_resistant_gradation(self) -> None:
        psd = USGA_GSR_2020_WINDY_PSD
        assert psd.fraction_between(2.5e-4, 1e-3) > 0.80
        assert 0.10 <= psd.fraction_between(1e-3, 2e-3) <= 0.20


class TestPlayingConditions:
    @pytest.mark.parametrize("condition", list(PlayingCondition))
    def test_every_condition_builds_a_valid_state(
        self, condition: PlayingCondition
    ) -> None:
        state = playing_condition(condition)
        assert isinstance(state, SandState)
        assert state.name
        assert state.dry_bulk_density_kg_m3 > 1200.0
        assert state.bulk_density_kg_m3 >= state.dry_bulk_density_kg_m3

    def test_regimes_are_explicit_and_distinct(self) -> None:
        regimes = {
            condition: playing_condition(condition).moisture.regime
            for condition in PlayingCondition
        }
        assert regimes[PlayingCondition.FIRM] is MoistureRegime.DAMP_CAPILLARY
        assert regimes[PlayingCondition.FLUFFY] is MoistureRegime.DRY
        assert regimes[PlayingCondition.WET] is MoistureRegime.SATURATED
        assert regimes[PlayingCondition.PLUGGED] is MoistureRegime.DRY

    def test_firm_is_denser_than_fluffy_and_plugged(self) -> None:
        firm = playing_condition(PlayingCondition.FIRM)
        fluffy = playing_condition(PlayingCondition.FLUFFY)
        plugged = playing_condition(PlayingCondition.PLUGGED)
        assert firm.relative_density > fluffy.relative_density
        assert fluffy.relative_density > plugged.relative_density
        assert firm.dry_bulk_density_kg_m3 > plugged.dry_bulk_density_kg_m3

    def test_firm_dry_bulk_density_matches_the_drft_reference(self) -> None:
        """DRFT alpha_z was measured at rho ~ 1600 kg/m^3 (ADR-0032)."""
        firm = playing_condition(PlayingCondition.FIRM)
        assert firm.dry_bulk_density_kg_m3 == pytest.approx(1600.0, rel=0.05)

    def test_plugged_lie_is_rated_undesirable_by_the_penetrometer_scale(self) -> None:
        plugged = playing_condition(PlayingCondition.PLUGGED)
        assert plugged.firmness_rating is FirmnessRating.UNDESIRABLE

    def test_damp_apparent_cohesion_is_in_the_published_band(self) -> None:
        firm = playing_condition(PlayingCondition.FIRM)
        cohesion_pa = firm.cohesive_strength_pa()
        assert 1.0e3 <= cohesion_pa <= 1.0e4

    def test_wet_requires_an_explicit_dilation_suction_and_is_capped(self) -> None:
        wet = playing_condition(PlayingCondition.WET)
        gain = wet.cohesive_strength_pa(dilation_suction_pa=5.0e6)
        assert gain == pytest.approx(6.7e4, rel=0.05)

    def test_unknown_condition_raises(self) -> None:
        with pytest.raises(ValueError, match="condition"):
            playing_condition("mud")  # type: ignore[arg-type]


class TestFirmnessSweep:
    def test_sweep_covers_the_four_published_points(self) -> None:
        states = firmness_sweep()
        assert len(states) == len(FIRMNESS_SWEEP_KG_PER_CM2)
        values = [round(state.firmness_kg_per_cm2, 6) for state in states]
        assert values == list(FIRMNESS_SWEEP_KG_PER_CM2)

    def test_sweep_is_monotone_in_density(self) -> None:
        densities = [state.dry_bulk_density_kg_m3 for state in firmness_sweep()]
        assert densities == sorted(densities)
        assert densities[-1] > densities[0]

    def test_sweep_shares_one_particle_size_distribution(self) -> None:
        states = firmness_sweep()
        assert len({state.psd for state in states}) == 1

    def test_sweep_ratings_span_the_scale(self) -> None:
        ratings = [state.firmness_rating for state in firmness_sweep()]
        assert ratings[0] is FirmnessRating.UNDESIRABLE
        assert ratings[-1] is FirmnessRating.DESIRABLE


class TestProvenanceHonesty:
    @pytest.mark.parametrize("name", sorted(all_presets()))
    def test_every_preset_declares_provenance_for_the_borrowed_constants(
        self, name: str
    ) -> None:
        state = all_presets()[name]
        for key in ("friction_angle_deg", "particle_density_kg_m3", "packing"):
            entry = state.provenance.entry(key)
            assert entry.basis is ProvenanceBasis.BORROWED_ANALOGUE
            assert "Quikrete" in entry.source

    @pytest.mark.parametrize("name", sorted(all_presets()))
    def test_no_preset_claims_a_measured_bunker_sand_friction_angle(
        self, name: str
    ) -> None:
        state = all_presets()[name]
        assert (
            state.provenance.entry("friction_angle_deg").basis
            is not ProvenanceBasis.MEASURED
        )
        assert "friction_angle_deg" in state.provenance.borrowed_properties()

    @pytest.mark.parametrize("name", sorted(all_presets()))
    def test_particle_size_distribution_is_a_published_specification(
        self, name: str
    ) -> None:
        entry = all_presets()[name].provenance.entry("particle_size_distribution")
        assert entry.basis is ProvenanceBasis.SPECIFICATION

    def test_friction_angle_is_the_quikrete_value(self) -> None:
        for state in all_presets().values():
            assert state.friction_angle_deg == pytest.approx(34.0)
            assert state.friction_angle_rad == pytest.approx(math.radians(34.0))

    def test_particle_density_is_the_quikrete_value(self) -> None:
        for state in all_presets().values():
            assert state.packing.particle_density_kg_m3 == pytest.approx(2600.0)

    def test_the_measurement_gap_is_stated_in_the_note(self) -> None:
        note = (
            all_presets()["usga-firm"]
            .provenance.entry("friction_angle_deg")
            .note.lower()
        )
        assert "bunker" in note
        assert "no published" in note or "not measured" in note

    def test_presets_module_never_declares_a_measured_basis(self) -> None:
        """A source-level guard, mirroring the #7999 regression test."""
        from bunkershot3d.sand import presets

        source = inspect.getsource(presets)
        assert "ProvenanceBasis.MEASURED" not in source
        assert "Quikrete" in source

    def test_provenance_summary_is_human_readable(self) -> None:
        summary = all_presets()["usga-firm"].provenance.summary()
        assert "borrowed_analogue" in summary
        assert "friction_angle_deg" in summary

    def test_the_honesty_statement_ships_with_the_package(self) -> None:
        from bunkershot3d.sand import provenance as provenance_module

        source = Path(provenance_module.__file__).read_text(encoding="utf-8")
        assert "golf bunker" in source
        assert "7999" in source
