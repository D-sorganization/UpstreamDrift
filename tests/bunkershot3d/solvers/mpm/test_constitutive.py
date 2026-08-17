"""The F1 constitutive model: capped Drucker-Prager on Hencky strains.

The sharpest available check on a return mapping is the yield function
itself: after projection **every** state must be admissible, and every
state that was actually projected must sit exactly on the surface.  That
is an identity of the scheme, so it is tested to round-off rather than to
a tolerance chosen to make it pass.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.sand import Angularity, playing_condition, usga_reference_sand
from bunkershot3d.sand.exceptions import MoistureRegimeError
from bunkershot3d.sand.presets import PlayingCondition
from bunkershot3d.solvers.exceptions import CalibrationError
from bunkershot3d.solvers.mpm.constitutive import (
    PLANE_STRAIN_DIMENSION,
    SandContinuum,
    drucker_prager_alpha,
    hencky_kirchhoff_principal,
    principal_stretches,
    project_to_yield_surface,
    reconstruct,
    yield_function,
)

pytestmark = pytest.mark.unit

_MU_PA = 9.0e6
_LAMBDA_PA = 13.5e6
_ALPHA = 0.374
_CAP = -0.02


def _project(strain: np.ndarray, *, tip: float = 0.0, cap: float = _CAP):
    return project_to_yield_surface(
        strain,
        shear_modulus_pa=_MU_PA,
        lame_lambda_pa=_LAMBDA_PA,
        alpha=_ALPHA,
        tip_volumetric_strain=tip,
        cap_volumetric_strain=cap,
    )


def _yield(strain: np.ndarray, *, tip: float = 0.0) -> np.ndarray:
    return yield_function(
        strain,
        shear_modulus_pa=_MU_PA,
        lame_lambda_pa=_LAMBDA_PA,
        alpha=_ALPHA,
        tip_volumetric_strain=tip,
    )


class TestConeSlope:
    """``alpha`` is the inner-cone Mohr-Coulomb match."""

    def test_reproduces_klar_formula(self) -> None:
        sin_phi = math.sin(math.radians(34.0))
        expected = math.sqrt(2.0 / 3.0) * 2.0 * sin_phi / (3.0 - sin_phi)
        assert drucker_prager_alpha(34.0) == pytest.approx(expected, rel=1e-14)

    def test_grows_with_friction_angle(self) -> None:
        assert drucker_prager_alpha(25.0) < drucker_prager_alpha(40.0)

    @pytest.mark.parametrize("angle", [0.0, 90.0, -5.0, float("nan")])
    def test_refuses_an_unusable_angle(self, angle: float) -> None:
        with pytest.raises(CalibrationError):
            drucker_prager_alpha(angle)


class TestReturnMapping:
    """The projected state is admissible, and the projection is exact."""

    def test_admissible_states_are_untouched(self) -> None:
        # Isotropic compression well inside the cone and inside the cap.
        strain = np.array([[-0.001, -0.001], [-0.002, -0.0019]])
        projected, yielded, capped = _project(strain)
        assert not yielded.any()
        assert not capped.any()
        np.testing.assert_array_equal(projected, strain)

    def test_every_projected_state_is_admissible(self) -> None:
        rng = np.random.default_rng(20260816)
        strain = rng.normal(scale=0.05, size=(4000, PLANE_STRAIN_DIMENSION))
        projected, _, _ = _project(strain)
        # Scale the round-off budget by the stress scale the yield function
        # is measured in, since y has units of pascals.
        stress_scale = (2.0 * _MU_PA + PLANE_STRAIN_DIMENSION * _LAMBDA_PA) * float(
            np.abs(strain).max()
        )
        assert float(_yield(projected).max()) <= 1e-9 * stress_scale

    def test_yielded_states_land_exactly_on_the_surface(self) -> None:
        rng = np.random.default_rng(7)
        # Pure shear well outside the cone: guaranteed case-III projections.
        deviator = rng.normal(scale=0.05, size=(500,))
        strain = np.stack([deviator - 0.002, -deviator - 0.002], axis=1)
        projected, yielded, _ = _project(strain)
        assert yielded.any()
        cone = yielded & (projected.sum(axis=1) < 0.0)
        residual = np.abs(_yield(projected)[cone])
        stress_scale = 2.0 * _MU_PA * float(np.abs(deviator).max())
        assert float(residual.max()) <= 1e-12 * stress_scale

    def test_cone_projection_preserves_volume(self) -> None:
        """Non-associated flow: case III changes the deviator only."""
        strain = np.array([[0.03, -0.034], [0.05, -0.06]])
        projected, yielded, capped = _project(strain)
        assert yielded.all()
        assert not capped.any()
        np.testing.assert_allclose(
            projected.sum(axis=1), strain.sum(axis=1), rtol=0.0, atol=1e-15
        )

    def test_tension_is_released_to_the_tip(self) -> None:
        """Sand cannot be stretched: a tensile trial state goes to the tip."""
        strain = np.array([[0.02, 0.01]])
        projected, yielded, _ = _project(strain)
        assert yielded.all()
        np.testing.assert_allclose(projected, np.zeros_like(strain), atol=1e-16)

    def test_cohesive_tip_carries_isotropic_tension(self) -> None:
        bulk = 2.0 * _MU_PA + PLANE_STRAIN_DIMENSION * _LAMBDA_PA
        tensile_pa = 2000.0
        tip = PLANE_STRAIN_DIMENSION * tensile_pa / bulk
        strain = np.array([[0.25 * tip, 0.25 * tip]])
        projected, yielded, _ = _project(strain, tip=tip)
        # Half-way to the tip is admissible for a cohesive sand and is not
        # for a cohesionless one.
        assert not yielded.any()
        assert _project(strain, tip=0.0)[1].all()
        np.testing.assert_allclose(projected, strain)

    def test_compressive_cap_clamps_only_the_volumetric_part(self) -> None:
        strain = np.array([[-0.05, -0.05]])  # tr = -0.10, cap = -0.02
        projected, _, capped = _project(strain)
        assert capped.all()
        assert projected.sum(axis=1) == pytest.approx(_CAP, rel=1e-14)

    def test_cap_leaves_the_deviator_alone(self) -> None:
        strain = np.array([[-0.05 + 0.001, -0.05 - 0.001]])
        projected, _, capped = _project(strain)
        assert capped.all()
        trial_dev = strain - strain.mean(axis=1, keepdims=True)
        projected_dev = projected - projected.mean(axis=1, keepdims=True)
        np.testing.assert_allclose(projected_dev, trial_dev, atol=1e-16)

    def test_a_non_compressive_cap_is_refused(self) -> None:
        with pytest.raises(CalibrationError, match="must be negative"):
            _project(np.zeros((1, 2)), cap=0.0)

    def test_a_malformed_strain_array_is_refused(self) -> None:
        with pytest.raises(CalibrationError, match=r"shape \(n, d\)"):
            _project(np.zeros(4))


class TestHenckyStress:
    """The elastic law and its Cauchy conversion."""

    def test_zero_strain_is_stress_free(self) -> None:
        stress = hencky_kirchhoff_principal(
            np.zeros((3, 2)), shear_modulus_pa=_MU_PA, lame_lambda_pa=_LAMBDA_PA
        )
        np.testing.assert_array_equal(stress, np.zeros((3, 2)))

    def test_uniaxial_strain_gives_the_constrained_modulus(self) -> None:
        strain = np.array([[0.0, -1e-4]])
        stress = hencky_kirchhoff_principal(
            strain, shear_modulus_pa=_MU_PA, lame_lambda_pa=_LAMBDA_PA
        )
        assert stress[0, 1] == pytest.approx(-1e-4 * (_LAMBDA_PA + 2.0 * _MU_PA))
        assert stress[0, 0] == pytest.approx(-1e-4 * _LAMBDA_PA)


class TestSvdRoundTrip:
    """The closed-form 2x2 decomposition the return mapping runs inside.

    Only three properties are relied on downstream, so those are what get
    tested -- not whether the factors happen to agree with LAPACK's, which
    they need not and do not.
    """

    @staticmethod
    def _sample(seed: int, scale: float = 0.05) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return np.eye(2) + rng.normal(scale=scale, size=(512, 2, 2))

    def test_reconstruction_is_exact(self) -> None:
        gradient = self._sample(11)
        left, stretches, right_transposed = principal_stretches(gradient)
        rebuilt = reconstruct(left, np.log(stretches), right_transposed)
        np.testing.assert_allclose(rebuilt, gradient, rtol=0.0, atol=1e-13)

    def test_reconstruction_survives_an_inverted_particle(self) -> None:
        """det(F) < 0 puts the second singular value below zero if unhandled."""
        gradient = self._sample(13, scale=1.5)
        assert float(np.linalg.det(gradient).min()) < 0.0
        left, stretches, right_transposed = principal_stretches(gradient)
        assert float(stretches.min()) >= 0.0
        rebuilt = reconstruct(left, np.log(stretches), right_transposed)
        np.testing.assert_allclose(rebuilt, gradient, rtol=0.0, atol=1e-12)

    def test_singular_values_match_lapack(self) -> None:
        gradient = self._sample(17, scale=0.8)
        _, stretches, _ = principal_stretches(gradient)
        expected = np.linalg.svd(gradient, compute_uv=False)
        np.testing.assert_allclose(stretches, expected, rtol=1e-12, atol=1e-14)

    def test_left_factor_is_orthogonal(self) -> None:
        """The Kirchhoff stress is coaxial with it, so it must be a rotation."""
        gradient = self._sample(19, scale=0.6)
        left, _, right_transposed = principal_stretches(gradient)
        identity = np.broadcast_to(np.eye(2), left.shape)
        np.testing.assert_allclose(
            left @ np.transpose(left, (0, 2, 1)), identity, atol=1e-13
        )
        np.testing.assert_allclose(
            right_transposed @ np.transpose(right_transposed, (0, 2, 1)),
            identity,
            atol=1e-13,
        )

    def test_a_degenerate_gradient_still_has_a_logarithm(self) -> None:
        gradient = np.zeros((1, 2, 2))
        _, stretches, _ = principal_stretches(gradient)
        assert np.all(np.isfinite(np.log(stretches)))

    def test_a_malformed_gradient_is_refused(self) -> None:
        with pytest.raises(CalibrationError, match=r"shape \(n, 2, 2\)"):
            principal_stretches(np.zeros((4, 3, 3)))


class TestSandContinuum:
    """Material constants come from the sand package, never from a second set."""

    def test_derives_from_a_preset(self) -> None:
        sand = playing_condition(PlayingCondition.FIRM)
        material = SandContinuum.from_sand_state(sand)
        assert material.density_kg_m3 == pytest.approx(sand.bulk_density_kg_m3)
        assert material.grain_diameter_m == pytest.approx(sand.d50_m)
        assert material.friction_angle_deg == pytest.approx(sand.friction_angle_deg)
        assert material.alpha == pytest.approx(
            drucker_prager_alpha(sand.friction_angle_deg)
        )

    def test_wave_speed_is_computed_not_pinned(self) -> None:
        sand = playing_condition(PlayingCondition.FIRM)
        soft = SandContinuum.from_sand_state(sand, shear_modulus_pa=1.0e6)
        stiff = SandContinuum.from_sand_state(sand, shear_modulus_pa=4.0e6)
        assert stiff.elastic_wave_speed_m_s == pytest.approx(
            2.0 * soft.elastic_wave_speed_m_s, rel=1e-12
        )
        assert soft.elastic_wave_speed_m_s == pytest.approx(
            math.sqrt(soft.p_wave_modulus_pa / soft.density_kg_m3)
        )

    def test_cap_tightens_as_the_bed_gets_denser(self) -> None:
        """Critical state, arriving from the packing machinery itself."""
        loose = SandContinuum.from_sand_state(
            usga_reference_sand("loose", 1.6, 0.05), shear_modulus_pa=9.0e6
        )
        dense = SandContinuum.from_sand_state(
            usga_reference_sand("dense", 2.8, 0.05), shear_modulus_pa=9.0e6
        )
        assert loose.cap_volumetric_strain < dense.cap_volumetric_strain < 0.0

    def test_a_damp_sand_carries_a_cohesive_tip(self) -> None:
        dry = SandContinuum.from_sand_state(
            usga_reference_sand("dry", 2.0, 0.0005), shear_modulus_pa=9.0e6
        )
        damp = SandContinuum.from_sand_state(
            usga_reference_sand("damp", 2.0, 0.050), shear_modulus_pa=9.0e6
        )
        assert damp.tip_volumetric_strain > dry.tip_volumetric_strain >= 0.0
        assert damp.tensile_strength_pa == pytest.approx(
            damp.cohesion_pa / math.tan(math.radians(damp.friction_angle_deg))
        )

    def test_a_saturated_bed_refuses_to_guess_its_suction(self) -> None:
        """The sand model has no default here, and F1 has no better one."""
        wet = playing_condition(PlayingCondition.WET)
        with pytest.raises(MoistureRegimeError, match="dilation_suction_pa"):
            SandContinuum.from_sand_state(wet, shear_modulus_pa=9.0e6)
        stated = SandContinuum.from_sand_state(
            wet, shear_modulus_pa=9.0e6, dilation_suction_pa=0.0
        )
        assert stated.tip_volumetric_strain >= 0.0

    def test_angularity_selects_the_modulus_branch(self) -> None:
        angular = SandContinuum.from_sand_state(
            usga_reference_sand("a", 2.0, 0.05, angularity=Angularity.ANGULAR)
        )
        rounded = SandContinuum.from_sand_state(
            usga_reference_sand("r", 2.0, 0.05, angularity=Angularity.ROUNDED)
        )
        assert angular.shear_modulus_pa != pytest.approx(rounded.shear_modulus_pa)

    def test_provenance_records_every_f1_specific_constant(self) -> None:
        material = SandContinuum.from_sand_state(
            playing_condition(PlayingCondition.FIRM)
        )
        for key in (
            "elastic_shear_modulus_pa",
            "poisson_ratio",
            "yield_surface",
            "compressive_cap",
        ):
            assert key in material.provenance.entries
        # The sand's own provenance survives the derivation.
        assert "friction_angle_deg" in material.provenance.entries

    def test_nothing_is_measured_on_bunker_sand(self) -> None:
        material = SandContinuum.from_sand_state(
            playing_condition(PlayingCondition.FIRM)
        )
        assert material.provenance.measured_properties() == ()

    def test_refuses_a_non_sand_state(self) -> None:
        with pytest.raises(CalibrationError, match="expected a SandState"):
            SandContinuum.from_sand_state(object())  # type: ignore[arg-type]

    @pytest.mark.parametrize("nu", [0.5, 0.6, -1.0])
    def test_refuses_an_unusable_poisson_ratio(self, nu: float) -> None:
        with pytest.raises(CalibrationError, match="poisson_ratio"):
            SandContinuum.from_sand_state(
                playing_condition(PlayingCondition.FIRM), poisson_ratio=nu
            )

    def test_cauchy_divides_by_the_jacobian(self) -> None:
        material = SandContinuum.from_sand_state(
            playing_condition(PlayingCondition.FIRM), shear_modulus_pa=9.0e6
        )
        strain = np.array([[-0.01, -0.005]])
        kirchhoff = hencky_kirchhoff_principal(
            strain,
            shear_modulus_pa=material.shear_modulus_pa,
            lame_lambda_pa=material.lame_lambda_pa,
        )
        cauchy = material.cauchy_from_hencky(strain)
        np.testing.assert_allclose(cauchy * math.exp(-0.015), kirchhoff, rtol=1e-13)
