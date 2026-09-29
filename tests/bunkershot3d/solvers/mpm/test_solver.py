"""The F1 solver as a ``GranularSolver``: contract, CFL, envelope, refusals.

ADR-0032's whole point is that every tier implements one protocol, so the
first thing tested here is that F0 and F1 are genuinely interchangeable --
not that they agree, which they do not and need not, but that a caller
written against the protocol can hold either one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.sand import playing_condition
from bunkershot3d.sand.presets import PlayingCondition
from bunkershot3d.solvers import (
    DRFTSolver,
    EnvelopeStatus,
    FidelityTier,
    GranularSolver,
    IntrusionState,
    MaterialResponse,
    OutOfEnvelopeError,
    RefusalPolicy,
    SolverResult,
    SurfaceElements,
)
from bunkershot3d.solvers.envelope import Caveat
from bunkershot3d.solvers.exceptions import SolverInputError
from bunkershot3d.solvers.mpm.constitutive import SandContinuum
from bunkershot3d.solvers.mpm.envelope import (
    RefusedQuantity,
    evaluate_f1_envelope,
    require_quotable,
)
from bunkershot3d.solvers.mpm.solver import (
    DEFAULT_CFL_NUMBER,
    PlaneStrainMPMSolver,
    cfl_time_step_s,
)

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def material() -> SandContinuum:
    return SandContinuum.from_sand_state(playing_condition(PlayingCondition.FIRM))


def sole_state(
    speed_m_s: float = 12.0,
    attack_deg: float = 20.0,
    free_surface_height_m: float = 0.0,
) -> IntrusionState:
    """A 40 x 16 mm sole section entering at a stated attack angle."""
    corners = np.array(
        [
            [-0.020, 0.0, -0.008],
            [0.020, 0.0, -0.008],
            [0.020, 0.0, 0.008],
            [-0.020, 0.0, 0.008],
            [0.024, 0.0, -0.004],
        ]
    )
    normals = np.tile([0.0, 0.0, -1.0], (corners.shape[0], 1))
    areas = np.full(corners.shape[0], 4.0e-4)
    angle = math.radians(attack_deg)
    return IntrusionState(
        SurfaceElements(corners, normals, areas),
        (speed_m_s * math.cos(angle), 0.0, -speed_m_s * math.sin(angle)),
        free_surface_height_m=free_surface_height_m,
    )


@pytest.fixture(scope="module")
def solver(material: SandContinuum) -> PlaneStrainMPMSolver:
    return PlaneStrainMPMSolver(
        material=material,
        cell_size_m=0.004,
        effective_width_m=0.030,
        bed_depth_m=0.06,
        refusal_policy=RefusalPolicy.REPORT,
        max_steps=4000,
    )


class TestProtocolConformance:
    """F1 is swappable with F0 because they implement the same protocol."""

    def test_it_is_a_granular_solver(self, solver: PlaneStrainMPMSolver) -> None:
        assert isinstance(solver, GranularSolver)

    def test_its_tier_is_f1(self, solver: PlaneStrainMPMSolver) -> None:
        assert solver.fidelity_tier is FidelityTier.F1

    def test_a_caller_can_hold_either_tier(
        self, solver: PlaneStrainMPMSolver, material: SandContinuum
    ) -> None:
        sand = playing_condition(PlayingCondition.FIRM)
        f0 = DRFTSolver(
            material=MaterialResponse.from_sand_state(sand),
            refusal_policy=RefusalPolicy.REPORT,
        )
        tiers: list[GranularSolver] = [f0, solver]
        assert {tier.fidelity_tier for tier in tiers} == {
            FidelityTier.F0,
            FidelityTier.F1,
        }
        for tier in tiers:
            assert isinstance(tier.envelope(sole_state()).status, EnvelopeStatus)

    @pytest.mark.slow
    def test_solve_returns_a_verdict_carrying_result(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        result = solver.solve(sole_state())
        assert isinstance(result, SolverResult)
        assert result.fidelity_tier is FidelityTier.F1
        assert result.verdict.status is EnvelopeStatus.BEYOND_VALIDATION

    @pytest.mark.slow
    def test_the_two_force_parts_add_to_the_resultant(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        """The split is an exact partition of the contact impulse."""
        result = solver.solve(sole_state())
        np.testing.assert_allclose(
            result.depth_force_n + result.inertial_force_n,
            result.wrench.force_n,
            rtol=1e-12,
            atol=1e-12,
        )

    @pytest.mark.slow
    def test_the_force_opposes_the_motion(self, solver: PlaneStrainMPMSolver) -> None:
        state = sole_state()
        result = solver.solve(state)
        assert float(result.wrench.force_n @ state.velocity_m_s) < 0.0

    @pytest.mark.slow
    def test_plane_strain_leaves_no_out_of_plane_force(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        result = solver.solve(sole_state())
        assert float(result.wrench.force_n[1]) == 0.0
        assert float(result.wrench.torque_n_m[0]) == 0.0
        assert float(result.wrench.torque_n_m[2]) == 0.0


class TestConstruction:
    """The one argument that has no default, and why."""

    def test_effective_width_is_required(self, material: SandContinuum) -> None:
        with pytest.raises(TypeError):
            PlaneStrainMPMSolver(material=material, cell_size_m=0.004)  # type: ignore[call-arg]

    @pytest.mark.parametrize("width", [0.0, -0.01, float("nan")])
    def test_an_unusable_width_is_refused(
        self, material: SandContinuum, width: float
    ) -> None:
        with pytest.raises(SolverInputError, match="effective_width_m"):
            PlaneStrainMPMSolver(
                material=material, cell_size_m=0.004, effective_width_m=width
            )

    def test_a_non_continuum_material_is_refused(self) -> None:
        with pytest.raises(SolverInputError, match="SandContinuum"):
            PlaneStrainMPMSolver(
                material=object(),  # type: ignore[arg-type]
                cell_size_m=0.004,
                effective_width_m=0.03,
            )


class TestTimeStep:
    """Computed from the elastic wave speed, and checked with a raise."""

    def test_it_uses_the_material_wave_speed(
        self, material: SandContinuum, solver: PlaneStrainMPMSolver
    ) -> None:
        state = sole_state()
        expected = (
            DEFAULT_CFL_NUMBER
            * solver.cell_size_m
            / (material.elastic_wave_speed_m_s + state.speed_m_s)
        )
        assert solver.time_step_s(state) == pytest.approx(expected, rel=1e-12)

    def test_a_stiffer_sand_shortens_the_step_by_itself(
        self, material: SandContinuum
    ) -> None:
        soft = cfl_time_step_s(
            cell_size_m=0.004,
            elastic_wave_speed_m_s=100.0,
            max_material_speed_m_s=0.0,
        )
        stiff = cfl_time_step_s(
            cell_size_m=0.004,
            elastic_wave_speed_m_s=400.0,
            max_material_speed_m_s=0.0,
        )
        assert stiff == pytest.approx(soft / 4.0, rel=1e-12)

    def test_the_body_speed_enters_the_condition(self) -> None:
        """This is what stops the club crossing a cell in one step."""
        still = cfl_time_step_s(
            cell_size_m=0.004,
            elastic_wave_speed_m_s=138.0,
            max_material_speed_m_s=0.0,
        )
        fast = cfl_time_step_s(
            cell_size_m=0.004,
            elastic_wave_speed_m_s=138.0,
            max_material_speed_m_s=25.0,
        )
        assert fast < still

    @pytest.mark.parametrize(
        ("kwargs", "match"),
        [
            ({"cell_size_m": 0.0}, "cell_size_m"),
            ({"elastic_wave_speed_m_s": 0.0}, "wave speed"),
            ({"max_material_speed_m_s": -1.0}, "max material speed"),
            ({"cfl_number": 0.0}, "cfl_number"),
            ({"cfl_number": 1.5}, "cfl_number"),
        ],
    )
    def test_unusable_inputs_raise(self, kwargs: dict, match: str) -> None:
        base = {
            "cell_size_m": 0.004,
            "elastic_wave_speed_m_s": 138.0,
            "max_material_speed_m_s": 10.0,
        }
        with pytest.raises(SolverInputError, match=match):
            cfl_time_step_s(**{**base, **kwargs})

    def test_a_tunnelling_step_is_refused_at_runtime(
        self, material: SandContinuum
    ) -> None:
        """A raise, not an assert, so ``python -O`` cannot remove it."""
        from bunkershot3d.solvers.mpm.body import RigidSection

        solver = PlaneStrainMPMSolver(
            material=material, cell_size_m=0.001, effective_width_m=0.03
        )
        section = RigidSection(
            np.array([[0.0, 0.0], [0.01, 0.0], [0.01, 0.01]]),
            velocity_m_s=(25.0, 0.0),
        )
        with pytest.raises(SolverInputError, match="in one step"):
            solver._require_courant((section,), 1.0e-3)


class TestApproachHistory:
    """A continuum has no instantaneous answer, and says so."""

    def test_a_static_query_is_refused(self, solver: PlaneStrainMPMSolver) -> None:
        state = IntrusionState(
            sole_state().elements, (0.0, 0.0, 0.0), free_surface_height_m=0.0
        )
        with pytest.raises(SolverInputError, match="no approach direction"):
            solver.run(state)

    def test_a_bodiless_query_is_refused(self, solver: PlaneStrainMPMSolver) -> None:
        empty = SurfaceElements(np.zeros((0, 3)), np.zeros((0, 3)), np.zeros(0))
        state = IntrusionState(empty, (10.0, 0.0, -4.0))
        with pytest.raises(SolverInputError, match="no surface elements"):
            solver.envelope(state)

    def test_the_section_is_the_in_plane_hull_of_the_body(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        section = solver.section_from_state(sole_state())
        assert section.area_m2 > 0.0
        np.testing.assert_allclose(
            section.velocity_m_s[0], 12.0 * math.cos(math.radians(20.0))
        )


class TestEnvelope:
    """F1's own limits, judged in F1's own vocabulary."""

    def test_no_query_beats_beyond_validation(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        """Even a slow, well-resolved one: there is nothing to validate against."""
        slow = solver.envelope(sole_state(speed_m_s=0.5))
        fast = solver.envelope(sole_state(speed_m_s=25.0))
        assert slow.status is EnvelopeStatus.BEYOND_VALIDATION
        assert fast.status is EnvelopeStatus.BEYOND_VALIDATION

    def test_the_structural_caveats_are_on_every_verdict(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        verdict = solver.envelope(sole_state())
        for caveat in (
            Caveat.PLANE_STRAIN_NO_OUT_OF_PLANE,
            Caveat.RATE_INDEPENDENT_PLASTICITY,
            Caveat.DECLARED_EFFECTIVE_WIDTH,
            Caveat.NO_MEASURED_COMPARISON,
        ):
            assert caveat in verdict.caveats
        assert verdict.summary()

    def test_no_rft_specific_caveat_leaks_onto_an_f1_verdict(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        """Describing a continuum's flaws in RFT's vocabulary hides its own."""
        verdict = solver.envelope(sole_state())
        for caveat in (
            Caveat.SHADOWING,
            Caveat.SHARP_CORNERS,
            Caveat.TRANSIENT_RESPONSE,
            Caveat.ELEMENT_SIZE_EFFECTS,
        ):
            assert caveat not in verdict.caveats

    def test_the_under_resolved_edge_is_declared(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        verdict = solver.envelope(sole_state())
        assert Caveat.UNDER_RESOLVED_LEADING_EDGE in verdict.caveats

    def test_the_declared_width_travels_on_the_verdict(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        verdict = solver.envelope(sole_state())
        assert verdict.details["effective_width_m"] == pytest.approx(0.030)

    def test_too_fine_a_grid_is_refused(self, material: SandContinuum) -> None:
        """The mirror image of F0's trap: refining below the grain scale."""
        verdict = evaluate_f1_envelope(
            speed_m_s=10.0,
            feature_lengths_m={"clubhead": 0.1},
            grain_diameter_m=material.grain_diameter_m,
            cell_size_m=material.grain_diameter_m,
            effective_width_m=0.03,
        )
        assert verdict.status is EnvelopeStatus.REFUSED
        assert "grain diameters" in " ".join(verdict.reasons)

    def test_a_feature_of_a_few_grains_is_refused(
        self, material: SandContinuum
    ) -> None:
        verdict = evaluate_f1_envelope(
            speed_m_s=10.0,
            feature_lengths_m={"speck": 3.0 * material.grain_diameter_m},
            grain_diameter_m=material.grain_diameter_m,
            cell_size_m=0.002,
            effective_width_m=0.03,
        )
        assert verdict.status is EnvelopeStatus.REFUSED

    def test_no_feature_scale_is_refused(self, material: SandContinuum) -> None:
        with pytest.raises(SolverInputError, match="at least one feature scale"):
            evaluate_f1_envelope(
                speed_m_s=10.0,
                feature_lengths_m={},
                grain_diameter_m=material.grain_diameter_m,
                cell_size_m=0.002,
                effective_width_m=0.03,
            )

    def test_a_strict_policy_raises_on_a_refusal(self, material: SandContinuum) -> None:
        strict = PlaneStrainMPMSolver(
            material=material,
            cell_size_m=material.grain_diameter_m,
            effective_width_m=0.03,
            refusal_policy=RefusalPolicy.STRICT,
        )
        with pytest.raises(OutOfEnvelopeError):
            strict.solve(sole_state())


class TestRefusedQuantities:
    """ADR-0033's "Refused" means the API raises, not that docs discourage."""

    @pytest.mark.parametrize("quantity", list(RefusedQuantity))
    def test_every_refused_quantity_raises(self, quantity: RefusedQuantity) -> None:
        with pytest.raises(OutOfEnvelopeError, match=quantity.value):
            require_quotable(quantity)

    def test_club_force_names_f0_as_its_owner(self) -> None:
        with pytest.raises(OutOfEnvelopeError, match="F0"):
            require_quotable(RefusedQuantity.CLUB_FORCE)

    def test_out_of_plane_is_refused_rather_than_approximated(self) -> None:
        with pytest.raises(OutOfEnvelopeError, match="no approximate answer"):
            require_quotable(RefusedQuantity.OUT_OF_PLANE)

    def test_an_unrecognised_quantity_raises_rather_than_passing(self) -> None:
        """A silent no-op the first time somebody misspells one is the bug."""
        with pytest.raises(SolverInputError, match="not a refused F1 quantity"):
            require_quotable("club force")


@pytest.mark.slow
class TestRunTrace:
    """What the march exposes for verification and for the later field work."""

    @pytest.fixture(scope="class")
    def run(self, material: SandContinuum):
        solver = PlaneStrainMPMSolver(
            material=material,
            cell_size_m=0.004,
            effective_width_m=0.030,
            bed_depth_m=0.06,
            refusal_policy=RefusalPolicy.REPORT,
            max_steps=4000,
        )
        return solver.run(sole_state())

    def test_mass_is_invariant_across_the_whole_march(self, run) -> None:
        masses = {step.total_mass_kg_per_m for step in run.steps}
        assert len(masses) == 1

    def test_it_digs_a_divot(self, run) -> None:
        assert run.divot_depth_m() > 0.0

    def test_the_peak_load_has_a_time(self, run) -> None:
        assert 0.0 < run.peak_force_time_s() <= run.duration_s

    def test_the_backstop_stays_a_backstop(self, run) -> None:
        """Constant pushouts would mean the swept contact is not working."""
        assert run.max_pushed_out() < 0.05 * run.particles.n_particles

    def test_an_averaging_window_is_required(self, run) -> None:
        with pytest.raises(SolverInputError, match="window_s"):
            run.averaged_force_n_per_m(0.0)

    def test_a_zero_step_march_is_refused(self, material: SandContinuum, run) -> None:
        solver = PlaneStrainMPMSolver(
            material=material, cell_size_m=0.004, effective_width_m=0.03
        )
        with pytest.raises(SolverInputError, match="n_steps must be positive"):
            solver.march(
                run.particles,
                None,
                run.grid,
                n_steps=0,
                time_step_s=1e-5,
                free_surface_height_m=0.0,
                bed_x_bounds_m=(-0.1, 0.1),
            )
