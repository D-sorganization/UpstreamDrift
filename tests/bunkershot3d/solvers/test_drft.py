"""Behaviour of the F0 solver itself (issue #8611).

Covers the acceptance criteria that are about the physics rather than
about invariance: monotonicity, the depth/inertia crossover, the
end-to-end force magnitude, the leading-edge and depth masks, and the
rule that every result carries its tier and its verdict.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from bunkershot3d.geometry import build_wedge_mesh, get_preset, preset_names
from bunkershot3d.sand import PlayingCondition, playing_condition
from bunkershot3d.solvers import (
    DEFAULT_FEATURE_SCALES_M,
    Caveat,
    DRFTSolver,
    EnvelopeStatus,
    FidelityTier,
    GranularSolver,
    IntrusionState,
    MaterialResponse,
    OutOfEnvelopeError,
    RefusalPolicy,
    SolverInputError,
    SolverResult,
    SurfaceElements,
    Wrench,
    ZeroDepression,
)

from .conftest import box_mesh

pytestmark = pytest.mark.unit

# A descending blow at 25 m/s with a 6 degree attack angle. The head
# frame has +x *rearward* (geometry.lofting), so travel toward the target
# is along -x and the club face leads.
_ATTACK_ANGLE_DEG = 6.0
_DELIVERY_SPEED_M_S = 25.0


def _delivery_velocity(speed_m_s: float = _DELIVERY_SPEED_M_S) -> np.ndarray:
    """Velocity of a descending blow at ``speed_m_s``."""
    angle = math.radians(_ATTACK_ANGLE_DEG)
    return speed_m_s * np.array([-math.cos(angle), 0.0, -math.sin(angle)])


def _plunge(
    solver: DRFTSolver,
    elements: SurfaceElements,
    speed_m_s: float,
) -> SolverResult:
    """Drive the body straight down at ``speed_m_s``."""
    return solver.solve(
        IntrusionState(elements, (0.0, 0.0, -speed_m_s), free_surface_height_m=0.0)
    )


class TestProtocolConformance:
    """ADR-0032: every tier implements one protocol."""

    def test_the_drft_solver_is_a_granular_solver(self, solver: DRFTSolver) -> None:
        assert isinstance(solver, GranularSolver)

    def test_reports_the_f0_tier(self, solver: DRFTSolver) -> None:
        assert solver.fidelity_tier is FidelityTier.F0

    def test_every_result_carries_its_tier_and_verdict(
        self, solver: DRFTSolver, sole_elements: SurfaceElements
    ) -> None:
        result = _plunge(solver, sole_elements, 25.0)
        assert result.fidelity_tier is FidelityTier.F0
        assert result.verdict.status is EnvelopeStatus.BEYOND_VALIDATION
        assert "BEYOND_VALIDATION" in result.summary()

    def test_a_result_cannot_be_built_without_a_verdict(self) -> None:
        with pytest.raises(SolverInputError, match="must carry a ValidityVerdict"):
            SolverResult(
                wrench=Wrench.zero(),
                fidelity_tier=FidelityTier.F0,
                verdict="looks fine to me",  # type: ignore[arg-type]
                depth_force_n=np.zeros(3),
                inertial_force_n=np.zeros(3),
                n_active_elements=0,
                active_area_m2=0.0,
                max_depth_m=0.0,
            )

    def test_the_envelope_can_be_judged_without_solving(
        self, solver: DRFTSolver, sole_elements: SurfaceElements
    ) -> None:
        verdict = solver.envelope(
            IntrusionState(sole_elements, (0.0, 0.0, -25.0), free_surface_height_m=0.0)
        )
        assert verdict.status is EnvelopeStatus.BEYOND_VALIDATION

    def test_rejects_a_state_that_is_not_an_intrusion_state(
        self, solver: DRFTSolver
    ) -> None:
        with pytest.raises(SolverInputError):
            solver.solve("a wedge, going quite fast")  # type: ignore[arg-type]


class TestForceSmokeTest:
    """The end-to-end magnitude anchor from the research addendum."""

    def test_a_sole_at_delivery_speed_gives_order_1550_newtons(
        self, solver: DRFTSolver, sole_elements: SurfaceElements
    ) -> None:
        # The addendum's anchor: a 20 x 80 mm sole at 25 m/s gives about
        # 1550 N, which is 527 g on a 0.30 kg head and stops it in ~5 ms
        # of submerged travel.
        #
        # The assertion is a band, not a point, because the anchor is
        # itself an order-of-magnitude estimate: lambda alone is only
        # known to within a factor of 2.8 across motion types, RFT is
        # documented to over-predict natural sand by ~35%, and delta_h is
        # uncalibrated. A tighter assertion would be pinning the model to
        # its own arithmetic and calling that agreement.
        result = _plunge(solver, sole_elements, _DELIVERY_SPEED_M_S)
        assert 500.0 < result.force_magnitude_n < 5000.0

    def test_that_force_would_stop_a_wedge_head_in_a_few_milliseconds(
        self, solver: DRFTSolver, sole_elements: SurfaceElements
    ) -> None:
        result = _plunge(solver, sole_elements, _DELIVERY_SPEED_M_S)
        head_mass_kg = 0.30
        stopping_time_s = head_mass_kg * _DELIVERY_SPEED_M_S / result.force_magnitude_n
        assert 0.001 < stopping_time_s < 0.020

    def test_the_force_opposes_the_motion(
        self, solver: DRFTSolver, sole_elements: SurfaceElements
    ) -> None:
        velocity = _delivery_velocity()
        result = solver.solve(
            IntrusionState(sole_elements, velocity, free_surface_height_m=0.0)
        )
        assert float(result.wrench.force_n @ velocity) < 0.0


class TestMonotonicity:
    """Deeper and faster must both mean harder."""

    @pytest.mark.parametrize("speed_m_s", [1.0, 5.0, 25.0])
    def test_force_increases_with_depth(
        self, solver: DRFTSolver, speed_m_s: float
    ) -> None:
        previous = 0.0
        for depth_m in (0.005, 0.010, 0.020, 0.040, 0.080):
            elements = SurfaceElements.from_mesh(
                box_mesh(0.020, 0.080, 0.004, centre_m=(0.0, 0.0, -depth_m))
            )
            magnitude = _plunge(solver, elements, speed_m_s).force_magnitude_n
            assert magnitude > previous
            previous = magnitude

    def test_force_increases_with_speed(
        self, solver: DRFTSolver, sole_elements: SurfaceElements
    ) -> None:
        previous = 0.0
        for speed_m_s in (0.5, 1.0, 3.0, 7.0, 15.0, 25.0, 40.0):
            magnitude = _plunge(solver, sole_elements, speed_m_s).force_magnitude_n
            assert magnitude > previous
            previous = magnitude

    @settings(
        deadline=None,
        max_examples=40,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(
        depth_m=st.floats(min_value=0.002, max_value=0.10),
        speed_m_s=st.floats(min_value=0.2, max_value=40.0),
        depth_gain_m=st.floats(min_value=1e-3, max_value=0.05),
        speed_gain_m_s=st.floats(min_value=0.5, max_value=20.0),
    )
    def test_monotone_in_both_arguments_over_the_whole_design_range(
        self,
        solver: DRFTSolver,
        depth_m: float,
        speed_m_s: float,
        depth_gain_m: float,
        speed_gain_m_s: float,
    ) -> None:
        def magnitude(depth: float, speed: float) -> float:
            elements = SurfaceElements.from_mesh(
                box_mesh(0.020, 0.080, 0.004, centre_m=(0.0, 0.0, -depth))
            )
            return _plunge(solver, elements, speed).force_magnitude_n

        base = magnitude(depth_m, speed_m_s)
        assert magnitude(depth_m + depth_gain_m, speed_m_s) >= base
        assert magnitude(depth_m, speed_m_s + speed_gain_m_s) >= base


class TestDepthInertiaCrossover:
    """The whole reason F0 is *dynamic* RFT and not RFT."""

    def test_the_depth_term_dominates_well_below_the_crossover(
        self, solver: DRFTSolver, sole_elements: SurfaceElements
    ) -> None:
        result = _plunge(solver, sole_elements, 1.0)
        assert result.depth_force_magnitude_n > result.inertial_force_magnitude_n
        assert result.inertial_fraction < 0.1

    def test_the_inertial_term_dominates_at_and_above_the_published_crossover(
        self, solver: DRFTSolver, sole_elements: SurfaceElements
    ) -> None:
        # Research digest: the two terms cross at 6.8 m/s.
        for speed_m_s in (6.8, 10.0, 25.0):
            result = _plunge(solver, sole_elements, speed_m_s)
            assert result.inertial_force_magnitude_n > result.depth_force_magnitude_n

    def test_at_delivery_speed_the_inertial_term_carries_most_of_the_load(
        self, solver: DRFTSolver, sole_elements: SurfaceElements
    ) -> None:
        result = _plunge(solver, sole_elements, _DELIVERY_SPEED_M_S)
        assert result.inertial_fraction > 0.9

    def test_without_the_structural_correction_the_crossover_matches_the_formula(
        self, material: MaterialResponse, sole_elements: SurfaceElements
    ) -> None:
        # delta_h attenuates the depth term, so the crossover it produces
        # sits below the analytic one. With the plate limit delta_h = 0
        # the solver must reproduce MaterialResponse.crossover_speed_m_s.
        quasi = DRFTSolver(
            material=material,
            structural_correction=ZeroDepression(),
            refusal_policy=RefusalPolicy.REPORT,
        )
        predicted = material.crossover_speed_m_s(0.040)
        below = _plunge(quasi, sole_elements, predicted * 0.8)
        above = _plunge(quasi, sole_elements, predicted * 1.25)
        assert below.depth_force_magnitude_n > below.inertial_force_magnitude_n
        assert above.inertial_force_magnitude_n > above.depth_force_magnitude_n

    def test_the_structural_correction_only_ever_softens_the_depth_term(
        self, material: MaterialResponse, sole_elements: SurfaceElements
    ) -> None:
        default_solver = DRFTSolver(
            material=material, refusal_policy=RefusalPolicy.REPORT
        )
        quasi = DRFTSolver(
            material=material,
            structural_correction=ZeroDepression(),
            refusal_policy=RefusalPolicy.REPORT,
        )
        for speed_m_s in (1.0, 7.0, 25.0):
            corrected = _plunge(default_solver, sole_elements, speed_m_s)
            uncorrected = _plunge(quasi, sole_elements, speed_m_s)
            assert (
                corrected.depth_force_magnitude_n
                <= uncorrected.depth_force_magnitude_n + 1e-9
            )
            # ...and never inverts it, which is the failure the source
            # paper reported for the inertial term on its own.
            assert corrected.depth_force_magnitude_n >= 0.0


class TestMasks:
    """Only leading-edge, submerged elements may contribute."""

    def test_a_body_entirely_above_the_surface_returns_a_null_wrench(
        self, solver: DRFTSolver
    ) -> None:
        elements = SurfaceElements.from_mesh(
            box_mesh(0.020, 0.080, 0.004, centre_m=(0.0, 0.0, 0.050))
        )
        result = _plunge(solver, elements, 25.0)
        assert result.n_active_elements == 0
        assert result.force_magnitude_n == 0.0
        assert result.active_area_m2 == 0.0
        # A null answer is still an answer, and still carries its verdict.
        assert result.fidelity_tier is FidelityTier.F0
        assert result.verdict.status is EnvelopeStatus.BEYOND_VALIDATION

    def test_a_stationary_body_feels_nothing(
        self, solver: DRFTSolver, sole_elements: SurfaceElements
    ) -> None:
        result = solver.solve(
            IntrusionState(sole_elements, (0.0, 0.0, 0.0), free_surface_height_m=0.0)
        )
        assert result.n_active_elements == 0
        assert result.force_magnitude_n == 0.0

    def test_only_the_leading_faces_of_a_box_engage(
        self, solver: DRFTSolver, sole_elements: SurfaceElements
    ) -> None:
        # Twelve triangles: two on the bottom (leading), two on the top
        # (trailing, excluded), eight on the sides (v . n = 0, included
        # by the >= 0 criterion).
        response = solver.element_response(
            IntrusionState(sole_elements, (0.0, 0.0, -25.0), free_surface_height_m=0.0)
        )
        assert len(sole_elements) == 12
        assert int(response.index.size) == 10

    def test_the_active_area_never_exceeds_the_body_area(
        self, solver: DRFTSolver, sole_elements: SurfaceElements
    ) -> None:
        result = _plunge(solver, sole_elements, 25.0)
        assert 0.0 < result.active_area_m2 <= sole_elements.total_area_m2

    def test_a_partly_submerged_body_engages_only_its_submerged_part(
        self, solver: DRFTSolver
    ) -> None:
        elements = SurfaceElements.from_mesh(
            box_mesh(0.020, 0.080, 0.040, centre_m=(0.0, 0.0, 0.0))
        )
        result = _plunge(solver, elements, 25.0)
        assert 0 < result.n_active_elements < len(elements)
        assert result.max_depth_m <= 0.020 + 1e-12


class TestFittedDomainClamping:
    """Upward-facing leading edges are outside the fit, and are declared."""

    def test_a_lofted_wedge_face_is_clamped_and_the_verdict_says_so(
        self, solver: DRFTSolver
    ) -> None:
        preset = get_preset(preset_names()[0])
        mesh = build_wedge_mesh(preset.geometry, n_profile_points=20, n_stations=9)
        elements = SurfaceElements.from_mesh(mesh).translated((0.0, 0.0, -0.030))
        result = solver.solve(
            IntrusionState(elements, _delivery_velocity(), free_surface_height_m=0.0)
        )
        # A lofted face points up and forward, so it is simultaneously a
        # leading edge and outside the polynomial's fitted domain.
        assert result.verdict.clamped_area_fraction > 0.0
        assert Caveat.UPWARD_FACING_LEADING_EDGE in result.verdict.caveats

    def test_a_flat_sole_needs_no_clamping(
        self, solver: DRFTSolver, sole_elements: SurfaceElements
    ) -> None:
        result = _plunge(solver, sole_elements, 25.0)
        assert result.verdict.clamped_area_fraction == 0.0
        assert Caveat.UPWARD_FACING_LEADING_EDGE not in result.verdict.caveats

    def test_the_fitted_angles_stay_inside_the_published_domain(
        self, solver: DRFTSolver
    ) -> None:
        preset = get_preset(preset_names()[0])
        mesh = build_wedge_mesh(preset.geometry, n_profile_points=20, n_stations=9)
        elements = SurfaceElements.from_mesh(mesh).translated((0.0, 0.0, -0.030))
        response = solver.element_response(
            IntrusionState(elements, _delivery_velocity(), free_surface_height_m=0.0)
        )
        half_pi = math.pi / 2.0 + 1e-12
        for name, values in (
            ("beta", response.beta_rad),
            ("gamma", response.gamma_rad),
            ("psi", response.psi_rad),
        ):
            assert np.all(np.abs(values) <= half_pi), f"{name} left the fitted domain"


class TestRefusal:
    """Out-of-envelope queries refuse rather than answer."""

    def test_quasi_static_rft_at_delivery_speed_is_refused(
        self, material: MaterialResponse, sole_elements: SurfaceElements
    ) -> None:
        quasi_static = DRFTSolver(
            material=material,
            dynamic_terms_active=False,
            refusal_policy=RefusalPolicy.STRICT,
        )
        with pytest.raises(OutOfEnvelopeError) as excinfo:
            _plunge(quasi_static, sole_elements, 25.0)
        assert "dynamic terms switched off" in str(excinfo.value)

    def test_the_same_query_returns_a_flagged_result_under_a_reporting_policy(
        self, material: MaterialResponse, sole_elements: SurfaceElements
    ) -> None:
        reporting = DRFTSolver(
            material=material,
            dynamic_terms_active=False,
            refusal_policy=RefusalPolicy.REPORT,
        )
        result = _plunge(reporting, sole_elements, 25.0)
        assert result.verdict.status is EnvelopeStatus.REFUSED
        assert result.verdict.is_refusal

    def test_a_slow_query_is_never_refused(
        self, material: MaterialResponse, sole_elements: SurfaceElements
    ) -> None:
        strict = DRFTSolver(material=material, refusal_policy=RefusalPolicy.STRICT)
        result = _plunge(strict, sole_elements, 0.3)
        assert not result.verdict.is_refusal

    def test_a_feature_scale_of_a_few_grains_refuses(
        self, material: MaterialResponse, sole_elements: SurfaceElements
    ) -> None:
        strict = DRFTSolver(
            material=material,
            refusal_policy=RefusalPolicy.STRICT,
            feature_scales_m={"hair-thin edge": 1.0e-3},
        )
        with pytest.raises(OutOfEnvelopeError, match="no continuum"):
            _plunge(strict, sole_elements, 0.3)


class TestConstruction:
    """Preconditions on the solver itself."""

    def test_rejects_a_material_that_is_not_a_material_response(self) -> None:
        with pytest.raises(SolverInputError):
            DRFTSolver(material="firm sand, probably")  # type: ignore[arg-type]

    def test_rejects_an_empty_feature_scale_map(
        self, material: MaterialResponse
    ) -> None:
        with pytest.raises(SolverInputError):
            DRFTSolver(material=material, feature_scales_m={})

    def test_rejects_a_structural_correction_that_is_not_one(
        self, material: MaterialResponse
    ) -> None:
        with pytest.raises(SolverInputError):
            DRFTSolver(material=material, structural_correction=0.0)  # type: ignore[arg-type]

    def test_default_feature_scales_are_the_addendum_s_three(self) -> None:
        assert set(DEFAULT_FEATURE_SCALES_M) == {
            "clubhead",
            "sole width",
            "leading edge",
        }


class TestMaterialSensitivity:
    """Changing the sand must change the answer, through the cubic."""

    def test_firmer_sand_resists_more(self, sole_elements: SurfaceElements) -> None:
        firm = MaterialResponse.from_sand_state(
            playing_condition(PlayingCondition.FIRM)
        )
        fluffy = MaterialResponse.from_sand_state(
            playing_condition(PlayingCondition.FLUFFY)
        )
        assert firm.normal_stress_scale_pa_per_m > fluffy.normal_stress_scale_pa_per_m
        firm_force = _plunge(
            DRFTSolver(material=firm, refusal_policy=RefusalPolicy.REPORT),
            sole_elements,
            3.0,
        ).force_magnitude_n
        fluffy_force = _plunge(
            DRFTSolver(material=fluffy, refusal_policy=RefusalPolicy.REPORT),
            sole_elements,
            3.0,
        ).force_magnitude_n
        assert firm_force > fluffy_force

    def test_a_larger_lambda_raises_the_dynamic_term_proportionally(
        self, firm_sand: object, sole_elements: SurfaceElements
    ) -> None:
        light = MaterialResponse.from_sand_state(firm_sand, inertial_lambda=1.0)  # type: ignore[arg-type]
        heavy = MaterialResponse.from_sand_state(firm_sand, inertial_lambda=2.8)  # type: ignore[arg-type]
        light_result = _plunge(
            DRFTSolver(material=light, refusal_policy=RefusalPolicy.REPORT),
            sole_elements,
            25.0,
        )
        heavy_result = _plunge(
            DRFTSolver(material=heavy, refusal_policy=RefusalPolicy.REPORT),
            sole_elements,
            25.0,
        )
        ratio = (
            heavy_result.inertial_force_magnitude_n
            / light_result.inertial_force_magnitude_n
        )
        assert ratio == pytest.approx(2.8, rel=1e-9)
