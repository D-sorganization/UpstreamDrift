"""The per-element sole load field, over time (issue #8705, epic #8699).

The workbench already binned the strike onto a static 12x12 sole map. That map
answers "where did the sole carry load, summed over the whole shot"; it cannot
answer "when", and it cannot say **which of the two DRFT terms** carried it.
Both questions are answerable from data the F0 solver already produces per
surface element, and this module pins that they are.

Everything here is headless: no Qt, no matplotlib, no display. The drawing is
tested in ``test_sole_load_render`` and ``tests/tools/bunker_shot_gui``.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.metrics import SoleLoadTrace
from bunkershot3d.solvers import EnvelopeStatus, FidelityTier
from src.tools.bunker_shot_gui.design import SandCondition, SwingSetup, WedgeDesign
from src.tools.bunker_shot_gui.field import (
    ContactPatch,
    LoadComponent,
    LoadScale,
    SoleLoadField,
    contact_patch,
)
from src.tools.bunker_shot_gui.model import WorkbenchModel

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


@pytest.fixture(scope="session")
def nominal_field(nominal_shot) -> SoleLoadField:  # type: ignore[no-untyped-def]
    """The per-element load field of the nominal shot."""
    field = nominal_shot.sole_field
    assert field is not None, nominal_shot.unavailable
    return field


class TestTheFieldIsPerElementAndPerSample:
    """The whole point of #8705: neither axis is collapsed."""

    def test_the_field_has_one_row_per_solver_sample(
        self, nominal_field: SoleLoadField
    ) -> None:
        assert nominal_field.n_frames == nominal_field.time_s.size
        assert nominal_field.n_frames > 2

    def test_the_field_has_one_column_per_sole_element(
        self, nominal_field: SoleLoadField
    ) -> None:
        assert nominal_field.n_elements == nominal_field.element_area_m2.size
        assert nominal_field.n_elements > 12 * 12 // 4

    def test_the_field_is_not_the_binned_map(
        self, nominal_field: SoleLoadField, nominal_shot
    ) -> None:  # type: ignore[no-untyped-def]
        """A 12x12 grid is a summary of this, not a substitute for it."""
        assert nominal_shot.sole_load is not None
        assert nominal_shot.sole_load.density_pa_s.shape == (12, 12)
        assert nominal_field.depth_normal_force_N.shape == (
            nominal_field.n_frames,
            nominal_field.n_elements,
        )

    def test_time_is_reported_in_seconds_and_strictly_increases(
        self, nominal_field: SoleLoadField
    ) -> None:
        assert np.all(np.diff(nominal_field.time_s) > 0.0)
        assert nominal_field.time_s[-1] < 1.0

    def test_pressure_is_force_over_area(self, nominal_field: SoleLoadField) -> None:
        force = nominal_field.component_force_N(LoadComponent.TOTAL)
        pressure = nominal_field.component_pressure_pa(LoadComponent.TOTAL)
        np.testing.assert_allclose(
            pressure, force / nominal_field.element_area_m2, rtol=1e-12
        )


class TestTheTwoTermsAreSeparated:
    """#8705: a designer needs to know which term a feature is fighting."""

    def test_both_terms_are_carried_separately(
        self, nominal_field: SoleLoadField
    ) -> None:
        depth = nominal_field.component_force_N(LoadComponent.DEPTH)
        inertial = nominal_field.component_force_N(LoadComponent.INERTIAL)
        assert depth.shape == inertial.shape
        assert not np.allclose(depth, inertial)

    def test_the_terms_recombine_into_the_load_the_map_is_built_from(
        self, nominal_field: SoleLoadField
    ) -> None:
        """The split must not change the number the existing metric consumes."""
        combined = np.maximum(
            nominal_field.depth_normal_force_N + nominal_field.inertial_normal_force_N,
            0.0,
        )
        np.testing.assert_allclose(
            nominal_field.component_force_N(LoadComponent.TOTAL), combined, rtol=1e-12
        )

    def test_the_field_yields_the_trace_the_bounce_map_already_consumes(
        self, nominal_field: SoleLoadField
    ) -> None:
        trace = nominal_field.load_trace()
        assert isinstance(trace, SoleLoadTrace)
        assert trace.n_elements == nominal_field.n_elements
        np.testing.assert_allclose(trace.time_s, nominal_field.time_s)

    def test_each_term_reports_when_it_peaked(
        self, nominal_field: SoleLoadField
    ) -> None:
        for component in (LoadComponent.DEPTH, LoadComponent.INERTIAL):
            moment = nominal_field.peak_time_s(component)
            assert nominal_field.time_s[0] <= moment <= nominal_field.time_s[-1]

    def test_the_inertial_term_dominates_at_greenside_speed(
        self, nominal_field: SoleLoadField
    ) -> None:
        """ADR-0032 predicts a ~0.9 inertial share at 25 m/s. Check it holds."""
        depth = nominal_field.peak_resultant_force_N(LoadComponent.DEPTH)
        inertial = nominal_field.peak_resultant_force_N(LoadComponent.INERTIAL)
        assert inertial > depth
        assert nominal_field.peak_inertial_share > 0.9

    def test_an_unknown_component_is_refused(
        self, nominal_field: SoleLoadField
    ) -> None:
        with pytest.raises(ValueError, match="component"):
            nominal_field.component_force_N("sideways")  # type: ignore[arg-type]


class TestTheFieldCarriesItsOwnHonesty:
    """The status has to reach the frame, so it travels with the data."""

    def test_the_field_carries_the_verdict_of_the_shot_it_came_from(
        self, nominal_field: SoleLoadField, nominal_shot
    ) -> None:  # type: ignore[no-untyped-def]
        assert nominal_field.verdict is nominal_shot.verdict
        assert nominal_field.status is EnvelopeStatus.BEYOND_VALIDATION

    def test_the_field_carries_the_fidelity_tier(
        self, nominal_field: SoleLoadField
    ) -> None:
        assert nominal_field.fidelity_tier is FidelityTier.F0

    def test_a_refused_shot_carries_no_field_at_all(
        self,
        model: WorkbenchModel,
        nominal_design: WedgeDesign,
        firm_sand: SandCondition,
        quasi_static_swing: SwingSetup,
    ) -> None:
        refused = model.run_shot(
            nominal_design.geometry(), firm_sand.sand_state(), quasi_static_swing
        )
        assert refused.refused
        assert refused.sole_field is None
        assert refused.contact_patch is None


class TestTheFieldRefusesMalformedInput:
    """``raise``, not ``assert``: ``python -O`` must not switch these off."""

    def _valid(self) -> dict[str, object]:
        return {
            "time_s": np.array([0.0, 1.0]),
            "element_centroid_body_m": np.zeros((2, 3)),
            "element_area_m2": np.array([1.0, 2.0]),
            "depth_normal_force_N": np.zeros((2, 2)),
            "inertial_normal_force_N": np.ones((2, 2)),
            "verdict": None,
            "fidelity_tier": FidelityTier.F0,
        }

    def test_a_mismatched_force_block_is_refused(self, nominal_shot) -> None:  # type: ignore[no-untyped-def]
        fields = self._valid()
        fields["verdict"] = nominal_shot.verdict
        fields["inertial_normal_force_N"] = np.ones((3, 2))
        with pytest.raises(ValueError, match="inertial_normal_force_N"):
            SoleLoadField(**fields)  # type: ignore[arg-type]

    def test_a_single_sample_is_refused(self, nominal_shot) -> None:  # type: ignore[no-untyped-def]
        fields = self._valid()
        fields["verdict"] = nominal_shot.verdict
        fields["time_s"] = np.array([0.0])
        fields["depth_normal_force_N"] = np.zeros((1, 2))
        fields["inertial_normal_force_N"] = np.zeros((1, 2))
        with pytest.raises(ValueError, match="at least 2 samples"):
            SoleLoadField(**fields)  # type: ignore[arg-type]

    def test_a_non_finite_load_is_refused(self, nominal_shot) -> None:  # type: ignore[no-untyped-def]
        fields = self._valid()
        fields["verdict"] = nominal_shot.verdict
        fields["depth_normal_force_N"] = np.full((2, 2), np.nan)
        with pytest.raises(ValueError, match="finite"):
            SoleLoadField(**fields)  # type: ignore[arg-type]

    def test_a_zero_area_element_is_refused(self, nominal_shot) -> None:  # type: ignore[no-untyped-def]
        fields = self._valid()
        fields["verdict"] = nominal_shot.verdict
        fields["element_area_m2"] = np.array([1.0, 0.0])
        with pytest.raises(ValueError, match="positive"):
            SoleLoadField(**fields)  # type: ignore[arg-type]


class TestTheColourScaleIsFixed:
    """Auto-scaling each frame to its own max makes two grinds incomparable."""

    def test_a_scale_covers_every_frame_not_one(
        self, nominal_field: SoleLoadField
    ) -> None:
        scale = nominal_field.scale(LoadComponent.TOTAL)
        pressure = nominal_field.component_pressure_pa(LoadComponent.TOTAL)
        assert scale.peak_pa >= float(pressure.max())
        for frame in range(nominal_field.n_frames):
            assert scale.peak_pa >= float(pressure[frame].max())

    def test_zero_is_always_inside_the_scale(
        self, nominal_field: SoleLoadField
    ) -> None:
        scale = nominal_field.scale(LoadComponent.TOTAL)
        low, high = scale.limits_pa
        assert low <= 0.0 <= high

    def test_normalising_maps_the_peak_to_one_and_zero_into_range(
        self, nominal_field: SoleLoadField
    ) -> None:
        scale = nominal_field.scale(LoadComponent.INERTIAL)
        normalised = scale.normalise(
            np.array([scale.limits_pa[0], 0.0, scale.limits_pa[1]])
        )
        assert normalised[-1] == pytest.approx(1.0)
        assert np.all((normalised >= 0.0) & (normalised <= 1.0))

    def test_two_designs_can_be_put_on_one_scale(
        self,
        model: WorkbenchModel,
        firm_sand: SandCondition,
        tour_swing: SwingSetup,
        nominal_field: SoleLoadField,
    ) -> None:
        other = model.run_shot(
            WedgeDesign(name="wide", sole_width_mm=22.0).geometry(),
            firm_sand.sand_state(),
            tour_swing,
        )
        assert other.sole_field is not None
        merged = LoadScale.covering(
            LoadComponent.TOTAL, (nominal_field, other.sole_field)
        )
        for field in (nominal_field, other.sole_field):
            assert merged.peak_pa >= field.scale(LoadComponent.TOTAL).peak_pa

    def test_scales_of_different_components_do_not_merge(
        self, nominal_field: SoleLoadField
    ) -> None:
        depth = nominal_field.scale(LoadComponent.DEPTH)
        inertial = nominal_field.scale(LoadComponent.INERTIAL)
        with pytest.raises(ValueError, match="same component"):
            depth.merged(inertial)

    def test_each_component_keeps_its_own_scale(
        self, nominal_field: SoleLoadField
    ) -> None:
        """The depth term is orders smaller; one shared ramp would erase it."""
        depth = nominal_field.scale(LoadComponent.DEPTH)
        inertial = nominal_field.scale(LoadComponent.INERTIAL)
        assert depth.component is LoadComponent.DEPTH
        assert inertial.peak_pa > depth.peak_pa

    def test_a_scale_states_its_unit(self, nominal_field: SoleLoadField) -> None:
        assert nominal_field.scale(LoadComponent.TOTAL).unit == "Pa"


class TestTheContactPatch:
    """#8707 rides on the same field: the engaged element set over time."""

    @pytest.fixture(scope="class")
    def patch(self, nominal_field: SoleLoadField) -> ContactPatch:
        return contact_patch(nominal_field)

    def test_the_patch_is_reported_for_every_frame(
        self, patch: ContactPatch, nominal_field: SoleLoadField
    ) -> None:
        assert patch.area_m2.shape == (nominal_field.n_frames,)
        assert patch.engaged.shape == nominal_field.engaged_mask.shape

    def test_the_patch_area_never_exceeds_the_sole(
        self, patch: ContactPatch, nominal_field: SoleLoadField
    ) -> None:
        assert float(patch.area_m2.max()) <= nominal_field.total_area_m2 + 1e-15

    def test_the_shot_engages_and_then_disengages(self, patch: ContactPatch) -> None:
        assert patch.initial_frame >= 0
        assert patch.initial_area_m2 > 0.0
        assert patch.peak_area_m2 >= patch.initial_area_m2

    def test_the_patch_is_located_against_the_leading_edge(
        self, patch: ContactPatch
    ) -> None:
        assert patch.trailing_edge_m > patch.leading_edge_m
        reach = patch.reach_m[patch.engaged.any(axis=1)]
        assert np.all(reach >= 0.0)
        assert patch.closest_approach_m == pytest.approx(float(reach.min()))

    def test_nothing_is_claimed_for_a_frame_with_no_contact(
        self, patch: ContactPatch
    ) -> None:
        idle = ~patch.engaged.any(axis=1)
        if not idle.any():
            pytest.skip("this shot engaged from the first sample")
        assert np.all(np.isnan(patch.reach_m[idle]))
        assert np.all(patch.area_m2[idle] == 0.0)


class TestTheOutcomeExposesTheField:
    """The workbench model owns the computation; the view owns the drawing."""

    def test_the_outcome_carries_both_the_field_and_the_patch(
        self, nominal_shot
    ) -> None:  # type: ignore[no-untyped-def]
        assert isinstance(nominal_shot.sole_field, SoleLoadField)
        assert isinstance(nominal_shot.contact_patch, ContactPatch)

    def test_the_patch_on_the_outcome_matches_its_field(self, nominal_shot) -> None:  # type: ignore[no-untyped-def]
        np.testing.assert_allclose(
            nominal_shot.contact_patch.time_s, nominal_shot.sole_field.time_s
        )

    def test_the_binned_map_is_still_produced(self, nominal_shot) -> None:  # type: ignore[no-untyped-def]
        """#8705 raises the load view; it does not remove the existing one."""
        assert nominal_shot.sole_load is not None
        assert nominal_shot.sole_load.utilisation.utilisation_fraction > 0.0
