"""SandState aggregate tests (issue #8610)."""

from __future__ import annotations

import math

import pytest
from bunkershot3d.sand import (
    Angularity,
    BunkerBedGeometry,
    InfeasibleBedError,
    MoistureRegime,
    PlayingCondition,
    SandState,
    playing_condition,
)
from bunkershot3d.sand.bed import BedZone
from bunkershot3d.sand.exceptions import ProvenanceError
from bunkershot3d.sand.provenance import (
    REQUIRED_PROVENANCE_KEYS,
    PropertyProvenance,
    ProvenanceBasis,
    SandProvenance,
)

pytestmark = pytest.mark.unit


class TestDerivedQuantities:
    def test_state_exposes_the_designer_facing_summary(self) -> None:
        state = playing_condition(PlayingCondition.FIRM)
        assert state.solid_fraction == pytest.approx(1.0 / (1.0 + state.void_ratio))
        assert state.d50_m == state.psd.d50_m
        assert state.uniformity_coefficient == state.psd.uniformity_coefficient
        assert state.regime is MoistureRegime.DAMP_CAPILLARY
        assert state.friction_angle_rad == pytest.approx(math.radians(34.0))

    def test_moist_bulk_density_includes_the_water(self) -> None:
        state = playing_condition(PlayingCondition.WET)
        expected = state.dry_bulk_density_kg_m3 * (
            1.0 + state.moisture.gravimetric_water_content
        )
        assert state.bulk_density_kg_m3 == pytest.approx(expected)

    def test_state_is_frozen(self) -> None:
        state = playing_condition(PlayingCondition.FIRM)
        with pytest.raises((AttributeError, TypeError)):
            state.friction_angle_deg = 40.0  # type: ignore[misc]

    def test_with_bed_returns_a_new_state(self) -> None:
        state = playing_condition(PlayingCondition.FIRM)
        face = BunkerBedGeometry(
            depth_m=0.0625,
            plan_length_m=0.4,
            plan_width_m=0.3,
            zone=BedZone.FACE,
        )
        moved = state.with_bed(face)
        assert moved is not state
        assert moved.bed.zone is BedZone.FACE
        assert state.bed.zone is BedZone.FLOOR
        assert moved.psd is state.psd


class TestFeasibilityIntegration:
    def test_state_reports_the_required_grain_count(self) -> None:
        state = playing_condition(PlayingCondition.FIRM)
        count = state.required_grain_count(grain_diameter_m=4.0e-4)
        assert count > 1.0e8

    def test_state_refuses_the_b29_configuration(self) -> None:
        state = playing_condition(PlayingCondition.FIRM)
        with pytest.raises(InfeasibleBedError, match="solid fraction"):
            state.require_feasible_bed(grain_count=50_000, grain_diameter_m=4.0e-4)

    def test_state_accepts_a_consistent_configuration(self) -> None:
        state = playing_condition(PlayingCondition.FIRM)
        diameter_m = 4.0e-3
        count = state.required_grain_count(grain_diameter_m=diameter_m)
        state.require_feasible_bed(grain_count=count, grain_diameter_m=diameter_m)
        report = state.bed_feasibility(grain_count=count, grain_diameter_m=diameter_m)
        assert report.is_feasible

    def test_the_required_count_uses_the_state_solid_fraction(self) -> None:
        loose = playing_condition(PlayingCondition.PLUGGED)
        dense = playing_condition(PlayingCondition.FIRM)
        assert dense.required_grain_count(4.0e-4) > loose.required_grain_count(4.0e-4)

    def test_the_psd_derived_count_is_the_honest_one(self) -> None:
        """Silt dominates the grain count; the default must not hide that."""
        state = playing_condition(PlayingCondition.FIRM)
        from_psd = state.required_grain_count()
        from_d50 = state.required_grain_count(state.d50_m)
        assert from_psd > 100.0 * from_d50
        assert from_psd > 1.0e11


class TestProvenanceEnforcement:
    def _minimal_provenance(self, keys: tuple[str, ...]) -> SandProvenance:
        entry = PropertyProvenance(
            basis=ProvenanceBasis.BORROWED_ANALOGUE,
            source="test",
            note="test",
        )
        return SandProvenance(entries=dict.fromkeys(keys, entry))

    def test_required_keys_are_the_honesty_critical_ones(self) -> None:
        assert "friction_angle_deg" in REQUIRED_PROVENANCE_KEYS
        assert "particle_density_kg_m3" in REQUIRED_PROVENANCE_KEYS
        assert "packing" in REQUIRED_PROVENANCE_KEYS

    def test_missing_required_provenance_is_refused(self) -> None:
        provenance = self._minimal_provenance(("particle_size_distribution",))
        with pytest.raises(ProvenanceError, match="friction_angle_deg"):
            provenance.require_keys(REQUIRED_PROVENANCE_KEYS)

    def test_state_requires_a_provenance_record(self) -> None:
        base = playing_condition(PlayingCondition.FIRM)
        with pytest.raises(ProvenanceError):
            SandState(
                name="no-provenance",
                psd=base.psd,
                packing=base.packing,
                moisture=base.moisture,
                bed=base.bed,
                angularity=Angularity.ANGULAR,
                friction_angle_deg=34.0,
                penetrometer_firmness_pa=base.penetrometer_firmness_pa,
                provenance=self._minimal_provenance(("packing",)),
            )

    def test_borrowed_and_measured_properties_are_separable(self) -> None:
        provenance = playing_condition(PlayingCondition.FIRM).provenance
        assert "friction_angle_deg" in provenance.borrowed_properties()
        assert provenance.measured_properties() == ()


class TestValidation:
    @pytest.mark.parametrize("friction_angle_deg", [0.0, -5.0, 90.0])
    def test_unphysical_friction_angle_is_refused(
        self, friction_angle_deg: float
    ) -> None:
        base = playing_condition(PlayingCondition.FIRM)
        with pytest.raises(ValueError, match="friction angle"):
            SandState(
                name="bad-friction",
                psd=base.psd,
                packing=base.packing,
                moisture=base.moisture,
                bed=base.bed,
                angularity=base.angularity,
                friction_angle_deg=friction_angle_deg,
                penetrometer_firmness_pa=base.penetrometer_firmness_pa,
                provenance=base.provenance,
            )

    def test_non_positive_firmness_is_refused(self) -> None:
        base = playing_condition(PlayingCondition.FIRM)
        with pytest.raises(ValueError, match="firmness"):
            SandState(
                name="bad-firmness",
                psd=base.psd,
                packing=base.packing,
                moisture=base.moisture,
                bed=base.bed,
                angularity=base.angularity,
                friction_angle_deg=34.0,
                penetrometer_firmness_pa=0.0,
                provenance=base.provenance,
            )


class TestPublicApi:
    def test_package_exports_the_domain_objects(self) -> None:
        import bunkershot3d.sand as sand

        for name in (
            "SandState",
            "ParticleSizeDistribution",
            "PackingState",
            "MoistureState",
            "MoistureRegime",
            "BunkerBedGeometry",
            "InfeasibleBedError",
            "playing_condition",
            "firmness_sweep",
            "CAVITATION_PORE_PRESSURE_PA",
        ):
            assert name in sand.__all__
            assert hasattr(sand, name)
