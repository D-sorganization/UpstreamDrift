"""Regression tests for timestep stability and contact stiffness (#8612).

Covers baseline findings:

- **B13** — no timestep stability criterion existed anywhere. ``docs/comparison.md``
  claimed a 0.2 Rayleigh safety factor that no backend implemented.
- **B30** — Chrono used ``dt = 1 / output_rate_hz`` (5e-4 s) as the *integrator*
  step against a 0.2-Rayleigh limit of 4.2e-8 s: ~11 900x over.
- **canonical stiffness** — ``youngs_modulus: 1.0e7`` gives 47 % Hertzian grain
  interpenetration at 25 m/s.

The expected values are computed from the closed-form physics in
``_bunker_fixtures_8612`` rather than from the implementation.
"""

from __future__ import annotations

import math
from pathlib import Path

import pytest
import yaml
from _bunker_fixtures_8612 import (
    IMPACT_SPEED_MPS,
    QUARTZ_DENSITY,
    hertz_overlap_ratio,
    rayleigh_time,
)
from bunkershot3d.backends.stability import (
    ContactStiffnessError,
    StepBudgetExceededError,
    StepPlan,
    TimestepStabilityError,
    cfl_timestep,
)
from bunkershot3d.backends.stability import (
    hertz_overlap_ratio as impl_overlap_ratio,
)
from bunkershot3d.backends.stability import (
    plan_steps,
    rayleigh_timestep,
)
from bunkershot3d.backends.stability import (
    rayleigh_time as impl_rayleigh_time,
)
from bunkershot3d.backends.stability import (
    require_resolvable_contacts,
    require_stable_timestep,
    shear_modulus,
)
from bunkershot3d.config import BunkerShotConfig

pytestmark = pytest.mark.unit

_CANONICAL = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "bunkershot3d"
    / "calibration"
    / "configs"
    / "canonical.yaml"
)


class TestRayleighCriterion:
    """B13: the Rayleigh surface-wave transit time must be implemented."""

    def test_matches_closed_form(self) -> None:
        got = impl_rayleigh_time(
            radius=2.0e-4,
            density=QUARTZ_DENSITY,
            youngs_modulus=7.0e10,
            poisson_ratio=0.17,
        )
        want = rayleigh_time(2.0e-4, QUARTZ_DENSITY, 7.0e10, 0.17)
        assert got == pytest.approx(want, rel=1e-12)

    def test_reproduces_adr_0032_value(self) -> None:
        """ADR-0032 quotes t_R = 2.10e-7 s for a 0.2 mm-radius quartz grain."""
        t_r = impl_rayleigh_time(
            radius=2.0e-4,
            density=2650.0,
            youngs_modulus=7.0e10,
            poisson_ratio=0.17,
        )
        assert t_r == pytest.approx(2.10e-7, rel=0.02)
        assert 0.2 * t_r == pytest.approx(4.2e-8, rel=0.02)

    def test_shear_modulus_is_isotropic_relation(self) -> None:
        assert shear_modulus(7.0e10, 0.17) == pytest.approx(7.0e10 / (2.0 * 1.17))

    def test_rayleigh_timestep_applies_safety_factor(self) -> None:
        t_r = impl_rayleigh_time(
            radius=1.0e-3, density=2650.0, youngs_modulus=7.0e10, poisson_ratio=0.17
        )
        assert rayleigh_timestep(
            radius=1.0e-3,
            density=2650.0,
            youngs_modulus=7.0e10,
            poisson_ratio=0.17,
            safety_factor=0.2,
        ) == pytest.approx(0.2 * t_r)

    @pytest.mark.parametrize("bad", [0.0, -1.0])
    def test_non_positive_radius_rejected(self, bad: float) -> None:
        with pytest.raises(ValueError):
            impl_rayleigh_time(
                radius=bad,
                density=2650.0,
                youngs_modulus=7.0e10,
                poisson_ratio=0.17,
            )


class TestCflCriterion:
    """B13: a clubhead must not traverse a grain in a single step."""

    def test_cfl_is_courant_times_diameter_over_speed(self) -> None:
        assert cfl_timestep(diameter=4.0e-4, max_speed=25.0, courant=0.1) == (
            pytest.approx(0.1 * 4.0e-4 / 25.0)
        )

    def test_zero_speed_is_unconstrained(self) -> None:
        assert math.isinf(cfl_timestep(diameter=1.0e-3, max_speed=0.0))

    def test_a_millisecond_step_at_tour_speed_is_rejected(self) -> None:
        """dt = 1 ms moves the clubhead 25 mm — 62 grain diameters."""
        with pytest.raises(TimestepStabilityError, match="CFL|Courant|courant"):
            require_stable_timestep(
                1.0e-3,
                radius=2.0e-4,
                density=QUARTZ_DENSITY,
                youngs_modulus=7.0e10,
                poisson_ratio=0.17,
                max_speed=IMPACT_SPEED_MPS,
                enforce_rayleigh=False,
            )


class TestStabilityRefusal:
    """B30: running above the stability limit must raise, never proceed."""

    def test_output_rate_as_integration_step_is_refused(self) -> None:
        """The exact defect: dt = 1 / 2000 Hz used as the integrator step.

        ``max_speed=0`` isolates the Rayleigh criterion (the settle phase);
        the Courant limit is exercised separately above.
        """
        with pytest.raises(TimestepStabilityError) as excinfo:
            require_stable_timestep(
                1.0 / 2000.0,
                radius=2.0e-4,
                density=QUARTZ_DENSITY,
                youngs_modulus=7.0e10,
                poisson_ratio=0.17,
                max_speed=0.0,
            )
        message = str(excinfo.value)
        assert "Rayleigh" in message
        # The message must quantify how far over the limit the caller is.
        assert "5" in message  # 5.000e-04 appears in the reported dt

    def test_a_stable_step_is_accepted(self) -> None:
        require_stable_timestep(
            1.0e-9,
            radius=2.0e-4,
            density=QUARTZ_DENSITY,
            youngs_modulus=7.0e10,
            poisson_ratio=0.17,
            max_speed=IMPACT_SPEED_MPS,
        )

    def test_refusal_is_not_an_assert(self) -> None:
        """Safety checks must survive ``python -O`` (asserts are stripped)."""
        import bunkershot3d.backends.stability as module

        source = Path(module.__file__).read_text(encoding="utf-8")
        assert "\n    assert " not in source
        assert "\n        assert " not in source


class TestStepPlanSeparatesIntegrationFromOutput:
    """B30/B12: the sampling rate is not the integration timestep."""

    def test_output_every_derived_from_rate(self) -> None:
        plan = plan_steps(
            duration=1.0e-3,
            dt=1.0e-6,
            output_rate_hz=2000.0,
            max_steps=100_000,
        )
        assert isinstance(plan, StepPlan)
        assert plan.dt == pytest.approx(1.0e-6)
        assert plan.n_steps == 1000
        # 2 kHz sampling of a 1 MHz integration -> every 500 steps
        assert plan.output_every == 500

    def test_output_every_is_at_least_one(self) -> None:
        plan = plan_steps(
            duration=1.0e-3, dt=1.0e-3, output_rate_hz=2000.0, max_steps=10
        )
        assert plan.output_every == 1

    def test_budget_exceeded_raises_with_numbers(self) -> None:
        with pytest.raises(StepBudgetExceededError) as excinfo:
            plan_steps(
                duration=0.1,
                dt=4.2e-8,
                output_rate_hz=2000.0,
                max_steps=200_000,
            )
        message = str(excinfo.value)
        assert "200000" in message.replace(",", "").replace("_", "")


class TestHertzOverlapPrecondition:
    """The canonical config's 1e7 Pa gives 47 % interpenetration at 25 m/s."""

    @pytest.mark.parametrize(
        ("youngs_modulus", "expected"),
        [(1.0e7, 0.47), (1.0e8, 0.19), (1.0e10, 0.030), (7.0e10, 0.014)],
    )
    def test_reproduces_research_digest_table(
        self, youngs_modulus: float, expected: float
    ) -> None:
        ratio = impl_overlap_ratio(
            impact_speed=IMPACT_SPEED_MPS,
            density=QUARTZ_DENSITY,
            youngs_modulus=youngs_modulus,
            poisson_ratio=0.25,
        )
        assert ratio == pytest.approx(expected, rel=0.05)

    def test_ratio_is_independent_of_grain_size(self) -> None:
        """Coarse-graining cannot fix an over-soft stiffness."""
        reference = hertz_overlap_ratio(IMPACT_SPEED_MPS, QUARTZ_DENSITY, 1.0e7, 0.25)
        assert reference == pytest.approx(0.47, rel=0.02)
        assert impl_overlap_ratio(
            impact_speed=IMPACT_SPEED_MPS,
            density=QUARTZ_DENSITY,
            youngs_modulus=1.0e7,
            poisson_ratio=0.25,
        ) == pytest.approx(reference, rel=1e-9)

    def test_soft_stiffness_is_refused(self) -> None:
        with pytest.raises(ContactStiffnessError, match="overlap"):
            require_resolvable_contacts(
                impact_speed=IMPACT_SPEED_MPS,
                density=QUARTZ_DENSITY,
                youngs_modulus=1.0e7,
                poisson_ratio=0.25,
            )

    def test_quartz_stiffness_is_accepted(self) -> None:
        require_resolvable_contacts(
            impact_speed=IMPACT_SPEED_MPS,
            density=QUARTZ_DENSITY,
            youngs_modulus=7.0e10,
            poisson_ratio=0.17,
        )


class TestCanonicalConfigStiffness:
    """The shipped canonical config must satisfy the overlap precondition."""

    def test_canonical_overlap_is_below_two_percent(self) -> None:
        data = yaml.safe_load(_CANONICAL.read_text(encoding="utf-8"))
        contact = data["contact_model"]
        modulus = float(contact["youngs_modulus"])
        ratio = hertz_overlap_ratio(
            IMPACT_SPEED_MPS,
            float(data["grain_population"]["density"]),
            modulus,
            float(contact["poisson_ratio"]),
        )
        assert ratio <= 0.02, (
            f"canonical.yaml E={modulus:.3g} Pa gives {ratio:.1%} grain "
            f"interpenetration at {IMPACT_SPEED_MPS} m/s"
        )

    def test_numeric_fields_parse_as_numbers(self) -> None:
        """YAML 1.1 needs ``1.0e+7`` — ``1.0e7`` silently parses as a string."""
        data = yaml.safe_load(_CANONICAL.read_text(encoding="utf-8"))
        assert isinstance(data["contact_model"]["youngs_modulus"], float)
        assert isinstance(data["grain_population"]["diameter_mean"], float)

    def test_canonical_config_passes_the_runtime_precondition(self) -> None:
        config = BunkerShotConfig.from_yaml(_CANONICAL)
        require_resolvable_contacts(
            impact_speed=IMPACT_SPEED_MPS,
            density=config.grain_density,
            youngs_modulus=config.contact_params().youngs_modulus,
            poisson_ratio=config.contact_params().poisson_ratio,
        )
