"""Code verification: is the maths right? (issue #8616).

**No experimental data appears in this file.** Every expected value is
either a closed form derived on paper or an exact identity of the scheme.

Three groups:

* analytic limit cases -- the quasi-static flat plate and the zero-speed
  limit, both with closed forms;
* the order of accuracy of the surface quadrature, against the cylinder's
  exact inertial integral;
* the exact identities the discretisation must satisfy at any refinement.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.solvers import (
    VERTICAL_PLATE_ALPHA_Z,
    DRFTSolver,
    IntrusionState,
    MaterialResponse,
)
from bunkershot3d.vandv import (
    RefinementLevel,
    VerificationError,
    cylinder_case,
    cylinder_inertial_force_n,
    cylinder_side_elements,
    flat_plate_elements,
    inertial_power_is_dissipative,
    observed_order_from_errors,
    quasi_static_plate_case,
    refinement_errors,
)

pytestmark = [pytest.mark.unit, pytest.mark.scientific]

#: Facet counts for the surface refinement study. Multiples of four, each
#: double the last, so the refinement ratio in the facet chord is exactly 2.
REFINEMENT_FACET_COUNTS = (64, 128, 256, 512)


class TestQuasiStaticFlatPlateLimit:
    """The analytic limit RFT's one-shot calibration is defined against."""

    def test_depth_force_matches_the_closed_form_to_round_off(
        self, exact_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        """``F_z = xi_n alpha_z(0, pi/2, 0) |z| A``, exactly."""
        case = quasi_static_plate_case(material)
        result = exact_solver.solve(case.state())
        assert result.depth_force_n[2] == pytest.approx(
            case.exact_depth_force_n, rel=1e-12
        )

    def test_the_depth_force_is_purely_vertical(
        self, exact_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        """A horizontal plate driven straight down has no lateral response."""
        case = quasi_static_plate_case(material)
        result = exact_solver.solve(case.state())
        lateral = float(np.abs(result.depth_force_n[:2]).max())
        assert lateral <= 1e-12 * abs(case.exact_depth_force_n)

    def test_the_force_is_exactly_linear_in_depth(
        self, exact_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        """Doubling the depth doubles the depth force, with ``delta_h = 0``."""
        shallow = quasi_static_plate_case(material, depth_m=0.020)
        deep = quasi_static_plate_case(material, depth_m=0.040)
        shallow_force = exact_solver.solve(shallow.state()).depth_force_n[2]
        deep_force = exact_solver.solve(deep.state()).depth_force_n[2]
        assert deep_force == pytest.approx(2.0 * shallow_force, rel=1e-12)

    def test_the_force_is_exactly_linear_in_area(
        self, exact_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        """The surface integral of a uniform traction is exact at any area."""
        small = quasi_static_plate_case(material, area_m2=1e-3)
        large = quasi_static_plate_case(material, area_m2=3e-3)
        small_force = exact_solver.solve(small.state()).depth_force_n[2]
        large_force = exact_solver.solve(large.state()).depth_force_n[2]
        assert large_force == pytest.approx(3.0 * small_force, rel=1e-12)

    def test_the_closed_form_inverts_the_one_shot_calibration(
        self, exact_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        """Solve forward, calibrate back, and recover the same ``xi_n``.

        ``MaterialResponse.from_vertical_plate_intrusion`` divides by the
        same ``alpha_z`` this case multiplies by, so a sign or factor
        error in either shows up as a mismatch here and nowhere else.
        """
        case = quasi_static_plate_case(material)
        measured = exact_solver.solve(case.state()).depth_force_n[2]
        recovered = MaterialResponse.from_vertical_plate_intrusion(
            force_n=float(measured),
            area_m2=case.area_m2,
            depth_m=case.depth_m,
            bulk_density_kg_m3=material.bulk_density_kg_m3,
            friction_angle_deg=material.friction_angle_deg,
            grain_diameter_m=material.grain_diameter_m,
        )
        assert recovered.normal_stress_scale_pa_per_m == pytest.approx(
            material.normal_stress_scale_pa_per_m, rel=1e-12
        )

    def test_the_calibration_anchor_is_the_polynomial_value(self) -> None:
        """Guard the literal the whole limit case rests on."""
        assert pytest.approx(0.87574, abs=5e-6) == VERTICAL_PLATE_ALPHA_Z


class TestZeroSpeedLimit:
    """The inertial term must vanish, quadratically, as the speed goes to zero."""

    def test_the_inertial_force_scales_as_the_square_of_speed(
        self, exact_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        """Halving the speed quarters the inertial force, exactly."""
        fast = quasi_static_plate_case(material, speed_m_s=2.0)
        slow = quasi_static_plate_case(material, speed_m_s=1.0)
        fast_force = exact_solver.solve(fast.state()).inertial_force_n[2]
        slow_force = exact_solver.solve(slow.state()).inertial_force_n[2]
        assert fast_force == pytest.approx(4.0 * slow_force, rel=1e-12)

    def test_the_inertial_term_is_negligible_far_below_the_crossover(
        self, exact_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        """At a thousandth of the crossover speed it is a millionth of the load."""
        crossover = material.crossover_speed_m_s(0.040)
        case = quasi_static_plate_case(
            material, depth_m=0.040, speed_m_s=crossover / 1000.0
        )
        result = exact_solver.solve(case.state())
        assert result.inertial_fraction < 2e-6

    def test_a_stationary_body_feels_nothing(
        self, exact_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        """At exactly zero speed no element is active and the wrench is zero."""
        elements = flat_plate_elements(area_m2=1.6e-3, depth_m=0.040)
        result = exact_solver.solve(IntrusionState(elements, (0.0, 0.0, 0.0)))
        assert result.n_active_elements == 0
        assert result.force_magnitude_n == 0.0

    def test_the_two_terms_cross_exactly_at_the_published_crossover(
        self, exact_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        """``crossover_speed_m_s`` is an identity, not a comment.

        With ``delta_h = 0`` the depth force is ``xi_n alpha_z |z| A`` and
        the inertial force is ``lambda rho v^2 A``. They are equal exactly
        when ``v = sqrt(xi_n alpha_z |z| / (lambda rho))``, which is what
        :meth:`MaterialResponse.crossover_speed_m_s` returns.
        """
        depth = 0.040
        case = quasi_static_plate_case(
            material, depth_m=depth, speed_m_s=material.crossover_speed_m_s(depth)
        )
        result = exact_solver.solve(case.state())
        assert result.inertial_force_n[2] == pytest.approx(
            result.depth_force_n[2], rel=1e-12
        )

    def test_the_crossover_is_near_the_seven_metres_per_second_the_digest_quotes(
        self, material: MaterialResponse
    ) -> None:
        """A cross-check on the material scaling, not a tolerance to tune."""
        assert material.crossover_speed_m_s(0.040) == pytest.approx(7.0, abs=1.0)

    def test_the_inertial_term_never_adds_energy(
        self, exact_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        """``-lambda rho v_n^3 <= 0`` on every active element, provably."""
        case = cylinder_case(material, n_facets=128)
        power, dissipative = inertial_power_is_dissipative(exact_solver, case.state())
        assert dissipative
        assert power < 0.0


class TestSurfaceQuadratureOrderOfAccuracy:
    """Refinement of the DRFT surface discretisation, against a closed form."""

    def test_the_exact_cylinder_integral_has_the_expected_form(
        self, material: MaterialResponse
    ) -> None:
        """``-(4/3) lambda rho v^2 R L``, derived in ``cases``."""
        expected = -(4.0 / 3.0) * (
            material.inertial_stress_scale_pa_s2_per_m2 * 25.0**2 * 0.020 * 0.080
        )
        assert cylinder_inertial_force_n(material) == pytest.approx(expected, rel=1e-15)

    def test_the_quadrature_converges_at_second_order(
        self, exact_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        """The dominant error is the chord area, so ``p`` must be about 2.

        The composite midpoint rule lands on ``cos^3``, whose derivative
        vanishes at both integration limits, so its own error is
        ``O(dtheta^4)``. What survives is the chord-versus-arc area,
        ``2 R sin(dtheta/2)`` against ``R dtheta``, a relative error of
        ``-dtheta^2/24``.
        """
        levels = []
        exact = 0.0
        for count in REFINEMENT_FACET_COUNTS:
            case = cylinder_case(material, n_facets=count)
            force = exact_solver.solve(case.state()).inertial_force_n[0]
            levels.append(RefinementLevel(case.cell_size_m, float(force), f"N={count}"))
            exact = case.exact_inertial_force_x_n
        sizes, errors = refinement_errors(levels, exact_value=exact)
        observed = observed_order_from_errors(sizes, errors)
        assert observed.order == pytest.approx(2.0, abs=0.05)
        assert observed.monotone
        assert observed.spread < 0.1

    def test_the_out_of_plane_force_vanishes_by_symmetry_at_every_level(
        self, exact_solver: DRFTSolver, material: MaterialResponse
    ) -> None:
        """Facets pair across ``theta -> -theta``, so ``F_z`` cancels exactly."""
        for count in REFINEMENT_FACET_COUNTS:
            case = cylinder_case(material, n_facets=count)
            force = exact_solver.solve(case.state()).inertial_force_n
            assert abs(float(force[2])) <= 1e-12 * abs(case.exact_inertial_force_x_n), (
                f"N={count}"
            )

    def test_exactly_half_the_facets_are_leading_edges(
        self, material: MaterialResponse
    ) -> None:
        """The multiple-of-four rule is what makes the split unambiguous."""
        for count in REFINEMENT_FACET_COUNTS:
            case = cylinder_case(material, n_facets=count)
            leading = (case.elements.normals @ np.array([1.0, 0.0, 0.0])) >= 0.0
            assert int(leading.sum()) == count // 2

    def test_a_facet_count_that_is_not_a_multiple_of_four_is_refused(self) -> None:
        with pytest.raises(VerificationError, match="multiple of 4"):
            cylinder_side_elements(n_facets=30)

    def test_a_cylinder_that_would_break_the_surface_is_refused(self) -> None:
        with pytest.raises(VerificationError, match="break the free surface"):
            cylinder_side_elements(n_facets=32, radius_m=0.2, centre_depth_m=0.1)


class TestRefinementStudyGuards:
    """The order machinery must refuse a study that cannot mean anything."""

    def test_a_single_level_is_refused(self) -> None:
        with pytest.raises(VerificationError, match="at least two levels"):
            refinement_errors([RefinementLevel(1e-3, 1.0)], exact_value=0.0)

    def test_a_level_landing_exactly_on_the_reference_is_refused(self) -> None:
        """Usually a sign the reference came from the same computation."""
        levels = [RefinementLevel(1e-3, 2.0), RefinementLevel(2e-3, 2.5)]
        with pytest.raises(VerificationError, match="last bit"):
            refinement_errors(levels, exact_value=2.0)

    def test_mismatched_series_lengths_are_refused(self) -> None:
        with pytest.raises(VerificationError, match="one error per size"):
            observed_order_from_errors([1e-3, 2e-3], [1.0])

    def test_a_known_power_law_is_recovered_exactly(self) -> None:
        """Sanity: a synthetic ``C h^3`` series must fit ``p = 3``."""
        sizes = [1.0, 0.5, 0.25, 0.125]
        errors = [7.0 * size**3 for size in sizes]
        assert observed_order_from_errors(sizes, errors).order == pytest.approx(
            3.0, abs=1e-9
        )

    def test_the_refinement_ratio_in_the_facet_chord_is_exactly_two(
        self, material: MaterialResponse
    ) -> None:
        """Doubling the facet count halves the chord in the fine limit."""
        coarse = cylinder_case(material, n_facets=256)
        fine = cylinder_case(material, n_facets=512)
        assert coarse.cell_size_m / fine.cell_size_m == pytest.approx(2.0, abs=1e-4)

    def test_the_facet_chord_is_used_rather_than_the_root_mean_area(
        self, material: MaterialResponse
    ) -> None:
        """The two differ, and using the wrong one reports ``p = 4``."""
        case = cylinder_case(material, n_facets=128)
        assert case.cell_size_m == pytest.approx(
            2.0 * 0.020 * math.sin(math.pi / 128), rel=1e-12
        )
        assert case.cell_size_m != pytest.approx(
            case.elements.characteristic_length_m, rel=1e-3
        )
