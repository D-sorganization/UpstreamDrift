"""The 3D-RFT response surface and its material scaling (issue #8611).

These tests are the reproduction check the issue asks for: the published
anchors -- the vertical flat-plate intrusion coefficient, the material
scaling table and the depth/inertia crossover -- have to come out of the
implementation, not out of a comment.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.sand import PlayingCondition, playing_condition
from bunkershot3d.solvers import (
    LAMBDA_BY_MOTION,
    PLATE_DRAG_LAMBDA,
    RFT_COEFFICIENT_PROVENANCE,
    RFT_POLYNOMIAL_COEFFICIENTS,
    VERTICAL_PLATE_ALPHA_Z,
    CalibrationError,
    MaterialResponse,
    generic_alpha,
    internal_friction_mu,
    material_scaling_pa_per_m,
    polynomial_terms,
    scaling_shape_function,
)

pytestmark = pytest.mark.unit


def _alpha(beta: float, gamma: float, psi: float) -> tuple[float, float, float]:
    """Scalar convenience wrapper over the vectorised response."""
    arrays = tuple(np.array([value], dtype=np.float64) for value in (beta, gamma, psi))
    radial, tangential, vertical = generic_alpha(*arrays)
    return float(radial[0]), float(tangential[0]), float(vertical[0])


class TestPolynomialTable:
    """The 20-term table itself."""

    def test_has_twenty_terms_and_three_coefficient_columns(self) -> None:
        assert RFT_POLYNOMIAL_COEFFICIENTS.shape == (20, 3)

    def test_table_is_immutable(self) -> None:
        with pytest.raises(ValueError):
            RFT_POLYNOMIAL_COEFFICIENTS[0, 0] = 0.0

    def test_design_matrix_matches_the_published_term_order(self) -> None:
        x1, x2, x3 = (np.array([2.0]), np.array([3.0]), np.array([5.0]))
        terms = polynomial_terms(x1, x2, x3)[0]
        expected = [
            1.0,
            2.0,
            3.0,
            5.0,
            4.0,
            9.0,
            25.0,
            6.0,
            15.0,
            10.0,
            8.0,
            27.0,
            125.0,
            18.0,
            12.0,
            75.0,
            45.0,
            20.0,
            50.0,
            30.0,
        ]
        assert terms.tolist() == expected


class TestPublishedAnchors:
    """Values that exist in the literature, recomputed from the table."""

    def test_vertical_flat_plate_intrusion_reproduces_the_calibration_anchor(
        self,
    ) -> None:
        # The addendum's one-shot calibration case: a horizontal plate
        # driven straight down. beta = 0, gamma = pi/2, psi = 0.
        radial, tangential, vertical = _alpha(0.0, math.pi / 2.0, 0.0)
        assert vertical == pytest.approx(VERTICAL_PLATE_ALPHA_Z, abs=1e-12)
        assert vertical == pytest.approx(0.87574, abs=1e-5)
        # A plate driven straight down has no horizontal response at all.
        assert radial == pytest.approx(0.0, abs=1e-15)
        assert tangential == pytest.approx(0.0, abs=1e-15)

    def test_vertical_plate_response_resists_the_motion(self) -> None:
        # Moving down (gamma = +pi/2), the medium pushes up.
        assert _alpha(0.0, math.pi / 2.0, 0.0)[2] > 0.0

    def test_horizontal_drag_of_a_vertical_plate_opposes_the_motion(self) -> None:
        radial, _, vertical = _alpha(math.pi / 2.0, 0.0, 0.0)
        assert radial < 0.0
        # Passive earth pressure is roughly an order of magnitude below
        # bearing capacity, and the ratio here is about 0.22.
        assert 0.05 < abs(radial) / VERTICAL_PLATE_ALPHA_Z < 0.5
        assert vertical > 0.0


class TestMaterialScalingCubic:
    """``xi_n = rho_c g f_hat(mu)`` against the addendum's table."""

    @pytest.mark.parametrize(
        ("density", "mu", "expected"),
        [
            (1450.0, 0.6, 1.53e6),
            (1450.0, 0.7, 2.56e6),
            (1450.0, 0.84, 4.73e6),
            (1550.0, 0.6, 1.64e6),
            (1550.0, 0.7, 2.73e6),
            (1550.0, 0.84, 5.05e6),
            (1700.0, 0.6, 1.79e6),
            (1700.0, 0.7, 3.00e6),
            (1700.0, 0.84, 5.54e6),
        ],
    )
    def test_reproduces_the_published_table(
        self, density: float, mu: float, expected: float
    ) -> None:
        scale = density * 9.81 * scaling_shape_function(mu)
        assert scale == pytest.approx(expected, rel=3e-3)

    @pytest.mark.parametrize(
        ("angle_deg", "expected_mu"), [(31.0, 0.60), (35.0, 0.70), (40.0, 0.84)]
    )
    def test_friction_coefficient_is_the_tangent_of_the_friction_angle(
        self, angle_deg: float, expected_mu: float
    ) -> None:
        assert internal_friction_mu(angle_deg) == pytest.approx(expected_mu, abs=5e-3)

    def test_refuses_to_extrapolate_the_cubic_past_its_fit(self) -> None:
        with pytest.raises(CalibrationError, match="0.3-0.9"):
            scaling_shape_function(1.5)
        with pytest.raises(CalibrationError, match="0.3-0.9"):
            scaling_shape_function(0.1)

    def test_rejects_a_nonsensical_friction_angle(self) -> None:
        with pytest.raises(CalibrationError):
            internal_friction_mu(0.0)
        with pytest.raises(CalibrationError):
            internal_friction_mu(90.0)

    def test_rejects_a_nonsensical_density(self) -> None:
        with pytest.raises(CalibrationError):
            material_scaling_pa_per_m(bulk_density_kg_m3=0.0, friction_angle_deg=34.0)


class TestCrossCheckAgainstThePlateMeasurement:
    """The scaling cubic against the independently measured 2.02 N/cm^3."""

    def test_predicted_plate_coefficient_is_within_ten_percent_of_the_measurement(
        self,
    ) -> None:
        # Quikrete medium sand: Phi = 34 deg, and the addendum's 1550
        # kg/m^3 bunker-sand bulk density. The measured alpha_z(0, pi/2)
        # is 2.02 N/cm^3. The scaling cubic knows nothing about that
        # measurement, so agreement here is a real cross-check.
        material = MaterialResponse(
            normal_stress_scale_pa_per_m=material_scaling_pa_per_m(
                bulk_density_kg_m3=1550.0, friction_angle_deg=34.0
            ),
            bulk_density_kg_m3=1550.0,
            inertial_lambda=PLATE_DRAG_LAMBDA,
            surface_friction_mu=0.45,
            grain_diameter_m=0.33e-3,
            friction_angle_deg=34.0,
            provenance=_provenance(),
        )
        predicted = material.vertical_plate_alpha_z_n_per_cm3
        assert predicted == pytest.approx(2.02, rel=0.10)

    def test_depth_inertia_crossover_is_near_the_published_seven_metres_per_second(
        self,
    ) -> None:
        material = MaterialResponse(
            normal_stress_scale_pa_per_m=2.02e6 / VERTICAL_PLATE_ALPHA_Z,
            bulk_density_kg_m3=1600.0,
            inertial_lambda=1.1,
            surface_friction_mu=0.45,
            grain_diameter_m=0.33e-3,
            friction_angle_deg=34.0,
            provenance=_provenance(),
        )
        # Research digest: 6.8 m/s for a 40 mm divot.
        assert material.crossover_speed_m_s(0.040) == pytest.approx(6.8, rel=0.05)

    def test_crossover_rejects_a_non_positive_depth(self) -> None:
        material = MaterialResponse(
            normal_stress_scale_pa_per_m=2.4e6,
            bulk_density_kg_m3=1600.0,
            inertial_lambda=1.1,
            surface_friction_mu=0.45,
            grain_diameter_m=0.33e-3,
            friction_angle_deg=34.0,
            provenance=_provenance(),
        )
        with pytest.raises(CalibrationError):
            material.crossover_speed_m_s(0.0)


class TestInertialLambda:
    """The primary calibration target, and its honest spread."""

    def test_default_is_the_oblique_plate_value(self) -> None:
        assert pytest.approx(1.1) == PLATE_DRAG_LAMBDA
        assert LAMBDA_BY_MOTION["oblique_horizontal_plate"] == pytest.approx(1.1)

    def test_published_spread_covers_a_factor_of_nearly_three(self) -> None:
        values = list(LAMBDA_BY_MOTION.values())
        assert min(values) == pytest.approx(1.0)
        assert max(values) == pytest.approx(2.8)


class TestProvenanceHonesty:
    """Issue #7999's rule, applied to the solver's own constants."""

    def test_every_fitted_coefficient_is_borrowed(self) -> None:
        for name, record in RFT_COEFFICIENT_PROVENANCE.items():
            assert not record.is_measured, (
                f"'{name}' claims to be measured; nothing in the F0 solver was "
                "measured on golf bunker sand"
            )

    def test_a_sand_derived_material_reports_no_measured_constants(self) -> None:
        material = MaterialResponse.from_sand_state(
            playing_condition(PlayingCondition.FIRM)
        )
        assert material.measured_constants() == ()
        assert "rft_polynomial" in material.borrowed_constants()
        assert "inertial_lambda" in material.borrowed_constants()

    def test_one_shot_plate_calibration_is_the_only_route_to_a_measured_scale(
        self,
    ) -> None:
        material = MaterialResponse.from_vertical_plate_intrusion(
            force_n=120.0,
            area_m2=0.0025,
            depth_m=0.030,
            bulk_density_kg_m3=1550.0,
            friction_angle_deg=34.0,
            grain_diameter_m=0.33e-3,
        )
        assert material.measured_constants() == ("normal_stress_scale_pa_per_m",)
        # xi_n = F / (alpha_z_gen * ds * |z|), inverted exactly.
        recovered = (
            material.normal_stress_scale_pa_per_m
            * VERTICAL_PLATE_ALPHA_Z
            * 0.0025
            * 0.030
        )
        assert recovered == pytest.approx(120.0, rel=1e-12)

    def test_plate_calibration_rejects_a_non_positive_measurement(self) -> None:
        with pytest.raises(CalibrationError):
            MaterialResponse.from_vertical_plate_intrusion(
                force_n=0.0,
                area_m2=0.0025,
                depth_m=0.030,
                bulk_density_kg_m3=1550.0,
                friction_angle_deg=34.0,
                grain_diameter_m=0.33e-3,
            )


class TestReflectionSymmetryOfTheResponse:
    """Only ``alpha_theta`` is odd in the twist angle.

    This is what makes the metamorphic reflection test an identity
    rather than an approximation, so it is asserted directly on the
    response surface as well as end to end.
    """

    @pytest.mark.parametrize("beta", [-1.2, -0.4, 0.0, 0.5, 1.4])
    @pytest.mark.parametrize("gamma", [-1.0, 0.0, 0.7, 1.5])
    @pytest.mark.parametrize("psi", [0.2, 0.9, 1.5])
    def test_radial_and_vertical_are_even_and_tangential_is_odd(
        self, beta: float, gamma: float, psi: float
    ) -> None:
        forward = _alpha(beta, gamma, psi)
        mirrored = _alpha(beta, gamma, -psi)
        assert mirrored[0] == pytest.approx(forward[0], rel=1e-15, abs=1e-18)
        assert mirrored[1] == pytest.approx(-forward[1], rel=1e-15, abs=1e-18)
        assert mirrored[2] == pytest.approx(forward[2], rel=1e-15, abs=1e-18)


class TestVectorisation:
    """The response must stay array-granular."""

    def test_evaluates_a_whole_array_in_one_call(self) -> None:
        count = 257
        rng = np.random.default_rng(20260814)
        beta = rng.uniform(-math.pi / 2, math.pi / 2, count)
        gamma = rng.uniform(-math.pi / 2, math.pi / 2, count)
        psi = rng.uniform(-math.pi / 2, math.pi / 2, count)
        radial, tangential, vertical = generic_alpha(beta, gamma, psi)
        assert radial.shape == tangential.shape == vertical.shape == (count,)
        assert np.all(np.isfinite(radial))
        # Element-by-element agreement with the same call on scalars.
        for index in (0, 13, count - 1):
            single = _alpha(beta[index], gamma[index], psi[index])
            assert single[0] == pytest.approx(radial[index], rel=1e-14)
            assert single[2] == pytest.approx(vertical[index], rel=1e-14)


def _provenance() -> object:
    """A minimal provenance record for hand-built material responses."""
    from bunkershot3d.sand.provenance import SandProvenance

    return SandProvenance(entries=dict(RFT_COEFFICIENT_PROVENANCE))
