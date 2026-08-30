"""A carry number may never be quoted as though it were measured (#8657).

Issue #8616 established that **no published value exists anywhere** for ball
speed, launch angle or spin out of a bunker: ``require_measurable`` refuses
them and ``ValidationComparison`` cannot be constructed against them. The ball
module produces exactly those quantities, which is what makes them dangerous.

So every launch result carries the two statements the rest of this package
carries, in the same shapes:

* a :class:`~bunkershot3d.solvers.envelope.ValidityVerdict`, combined with the
  solver's own so the carry can never read better than the shot behind it, and
  floored at ``BEYOND_VALIDATION`` because the sand-to-ball partition is past
  every published measurement;
* a :class:`~bunkershot3d.sand.provenance.SandProvenance` record naming the
  basis of every constant the partition uses.

The source-level scan mirrors
``tests/bunkershot3d/sand/test_presets.py::test_presets_module_never_declares_a_measured_basis``:
the module text is checked directly, because a constant that acquires a
``MEASURED`` basis in a later edit would otherwise pass every behavioural test.
"""

from __future__ import annotations

import inspect
import math
from pathlib import Path

import pytest

from bunkershot3d.ball import splash as splash_module
from bunkershot3d.ball.lie import BallLie, BallProperties
from bunkershot3d.ball.splash import (
    BALL_LAUNCH_MEASUREMENT_GAP,
    MomentumTransfer,
    SandDelivery,
    compute_ball_launch_from_splash,
)
from bunkershot3d.sand import ProvenanceBasis
from bunkershot3d.solvers import Caveat, EnvelopeStatus
from bunkershot3d.vandv import NoReferenceDataError, require_measurable

from .test_splash_transfer import GREENSIDE_LOFT_RAD, delivery, launch, solver_verdict

pytestmark = pytest.mark.unit

BALL_MODULE_DIR = Path(splash_module.__file__).resolve().parent


class TestNothingInTheBallModuleIsMeasured:
    """The honesty rule of #7999 and #8616, enforced on the source."""

    @pytest.mark.parametrize(
        "path", sorted(p.name for p in BALL_MODULE_DIR.glob("*.py"))
    )
    def test_no_module_declares_a_measured_basis(self, path: str) -> None:
        source = (BALL_MODULE_DIR / path).read_text(encoding="utf-8")
        assert "ProvenanceBasis.MEASURED" not in source, (
            f"{path} claims a measured constant. Per issue #8616 there is no "
            "published measurement of ball speed, launch angle or spin out of "
            "sand, so nothing in this module can be measured."
        )

    def test_the_module_glob_is_not_empty(self) -> None:
        """Guard the guard: an empty glob would make the scan vacuous."""
        assert len(list(BALL_MODULE_DIR.glob("*.py"))) >= 3

    def test_the_result_reports_no_measured_constants(self) -> None:
        assert launch(delivery()).measured_constants() == ()

    def test_the_provenance_names_no_measured_property(self) -> None:
        provenance = launch(delivery()).provenance
        assert provenance.measured_properties() == ()

    def test_the_transfer_efficiency_is_declared_uncalibrated(self) -> None:
        entry = launch(delivery()).provenance.entry("transfer_efficiency")
        assert entry.basis is not ProvenanceBasis.MEASURED
        assert entry.basis is ProvenanceBasis.ESTIMATED
        assert "uncalibrated" in entry.note.lower()

    @pytest.mark.parametrize(
        "name",
        ["transfer_efficiency", "intercepted_fraction", "launch_direction"],
    )
    def test_every_partition_parameter_carries_a_record(self, name: str) -> None:
        assert launch(delivery()).provenance.entry(name) is not None

    def test_the_measurement_gap_is_stated_in_words(self) -> None:
        assert "no published" in BALL_LAUNCH_MEASUREMENT_GAP.lower()
        assert "8616" in BALL_LAUNCH_MEASUREMENT_GAP

    def test_the_summary_is_human_readable(self) -> None:
        summary = launch(delivery()).provenance.summary()
        assert "transfer_efficiency" in summary
        assert "estimated" in summary


class TestTheVerdictTravelsWithTheCarry:
    """Same shape as the solver's, so the two can be combined."""

    def test_the_result_carries_a_validity_verdict(self) -> None:
        result = launch(delivery())
        assert result.verdict.groups
        assert result.verdict.summary().startswith("validity:")

    def test_a_perfect_solver_verdict_is_still_floored_at_beyond_validation(
        self,
    ) -> None:
        """The solver can be inside its envelope; the ball model never is."""
        within = solver_verdict(speed_m_s=0.1, feature_lengths_m={"clubhead": 0.1})
        assert within.status is EnvelopeStatus.WITHIN
        result = compute_ball_launch_from_splash(
            lie=BallLie(depth_m=0.005),
            ball=BallProperties(),
            # 0.02 N.s over 0.25 kg is 0.08 m/s of ejecta, admissible for a
            # 0.1 m/s head. The default 4 N.s would not be (issue #8659).
            delivery=delivery(speed_m_s=0.1, impulse_n_s=0.02, verdict=within),
            club_loft_rad=GREENSIDE_LOFT_RAD,
        )
        assert result.verdict.status is EnvelopeStatus.BEYOND_VALIDATION
        assert result.verdict.groups == within.groups

    def test_the_verdict_says_why(self) -> None:
        reasons = " ".join(launch(delivery()).verdict.reasons).lower()
        assert "ball" in reasons
        assert "uncalibrated" in reasons or "no published" in reasons

    def test_the_solver_s_caveats_are_carried_through(self) -> None:
        result = launch(delivery())
        assert Caveat.BORROWED_COEFFICIENTS in result.verdict.caveats
        assert Caveat.TRANSIENT_RESPONSE in result.verdict.caveats

    def test_the_mean_ejecta_speed_is_carried_in_the_details(self) -> None:
        result = launch(delivery(impulse_n_s=4.0, displaced_mass_kg=0.25))
        assert result.verdict.details["mean_ejecta_speed_m_s"] == pytest.approx(16.0)

    def test_an_inadmissible_strike_is_refused_before_it_becomes_a_launch(
        self,
    ) -> None:
        """Sand cannot leave faster than the head that threw it (issue #8659).

        #8657 reported this pair on the verdict and carried on. It is now
        refused where the two quantities first meet, because a delivery whose
        impulse and mass imply 80 m/s of ejecta out of a 25 m/s head is not a
        strike that happened and no launch is derivable from it.
        """
        with pytest.raises(ValueError, match="cannot leave faster"):
            delivery(impulse_n_s=4.0, displaced_mass_kg=0.05)

    def test_the_refusal_is_a_raise_and_not_an_assertion(self) -> None:
        """``python -O`` strips assertions; a momentum budget must survive it.

        Read off the module text rather than through ``inspect.getsource``,
        which is the pattern the source scans at the top of this file already
        use: the check has to survive an optimisation flag, so it is made
        against the shipped bytes.
        """
        source = Path(splash_module.__file__).read_text(encoding="utf-8")
        start = source.index("    def _require_admissible_ejecta")
        body = source[start : source.index("\n    @property", start)]

        assert "raise ValueError(" in body
        assert "assert " not in body
        assert SandDelivery._require_admissible_ejecta is not None

    def test_a_consistent_strike_is_built_and_carries_no_such_diagnostic(
        self,
    ) -> None:
        result = launch(delivery(impulse_n_s=4.0, displaced_mass_kg=0.50))
        assert not any("cannot leave faster" in r for r in result.verdict.reasons)
        assert result.ball_speed_m_s > 0.0

    def test_an_interval_reaching_below_the_momentum_floor_is_reported(self) -> None:
        """Half a band being excluded is information, not an error.

        The reported mass is admissible; its lower edge is not. That is a
        statement about the width of the interval and is reported rather than
        raised or clamped.
        """
        result = launch(
            delivery(
                impulse_n_s=4.0,
                displaced_mass_kg=0.50,
                displaced_mass_bounds_kg=(0.10, 1.00),
            )
        )
        assert any("momentum budget excludes" in r for r in result.verdict.reasons)

    def test_an_interval_clear_of_the_floor_raises_no_diagnostic(self) -> None:
        result = launch(
            delivery(
                impulse_n_s=4.0,
                displaced_mass_kg=0.50,
                displaced_mass_bounds_kg=(0.30, 1.00),
            )
        )
        assert not any("momentum budget excludes" in r for r in result.verdict.reasons)

    def test_the_refusal_does_not_stop_carry_responding_to_impulse(self) -> None:
        """The refusal is not the clamp #8657 rejected: nothing is capped.

        Both strikes below are admissible, and ball speed still tracks the
        delivered impulse exactly.
        """
        soft = launch(delivery(impulse_n_s=3.0, displaced_mass_kg=0.50))
        hard = launch(delivery(impulse_n_s=6.0, displaced_mass_kg=0.50))
        assert hard.ball_speed_m_s == pytest.approx(2.0 * soft.ball_speed_m_s)

    def test_a_worse_solver_verdict_wins(self) -> None:
        """A shot beyond validation cannot become merely extrapolated here."""
        greenside = launch(delivery(speed_m_s=25.0))
        assert greenside.verdict.status is EnvelopeStatus.BEYOND_VALIDATION
        assert any(
            "1.44" in reason or "published corpus" in reason
            for reason in greenside.verdict.reasons
        )


class TestTheQuantitiesCannotBeValidated:
    """The link back to #8616: the register refuses these by name."""

    @pytest.mark.parametrize(
        "quantity", ["ball_speed_m_s", "ball_launch_angle_rad", "ball_spin_rad_s"]
    )
    def test_the_register_refuses_the_launch_quantities(self, quantity: str) -> None:
        with pytest.raises(NoReferenceDataError):
            require_measurable(quantity)


class TestTheTransferParametersAreNamed:
    """Not buried as literals in the arithmetic."""

    def test_the_defaults_are_reachable_and_documented(self) -> None:
        transfer = MomentumTransfer()
        assert 0.0 < transfer.efficiency <= 1.0
        assert MomentumTransfer.__doc__
        assert "uncalibrated" in MomentumTransfer.__doc__.lower()

    def test_the_efficiency_scales_the_ball_momentum(self) -> None:
        strike = delivery()
        half = compute_ball_launch_from_splash(
            lie=BallLie(depth_m=0.005),
            ball=BallProperties(),
            delivery=strike,
            club_loft_rad=GREENSIDE_LOFT_RAD,
            transfer=MomentumTransfer(efficiency=0.25),
        )
        full = compute_ball_launch_from_splash(
            lie=BallLie(depth_m=0.005),
            ball=BallProperties(),
            delivery=strike,
            club_loft_rad=GREENSIDE_LOFT_RAD,
            transfer=MomentumTransfer(efficiency=0.50),
        )
        assert full.ball_speed_m_s == pytest.approx(2.0 * half.ball_speed_m_s)

    def test_the_source_names_the_efficiency_rather_than_inlining_it(self) -> None:
        source = inspect.getsource(splash_module)
        assert "BALL_MOMENTUM_TRANSFER_EFFICIENCY" in source
        assert "transfer.efficiency" in source

    def test_the_spin_lever_arm_is_a_declared_convention(self) -> None:
        entry = launch(delivery()).provenance.entry("spin_lever_arm")
        assert entry.basis is ProvenanceBasis.CONVENTION

    def test_the_launch_direction_is_a_declared_convention(self) -> None:
        entry = launch(delivery()).provenance.entry("launch_direction")
        assert entry.basis is ProvenanceBasis.CONVENTION
        assert "loft" in entry.note.lower()


def test_the_module_docstring_names_the_issue() -> None:
    """A reader arriving at the model must find the argument behind it."""
    assert splash_module.__doc__ is not None
    assert "8657" in splash_module.__doc__


def test_a_launch_angle_of_zero_loft_is_impossible() -> None:
    """Loft is a precondition, not something to clamp silently."""
    with pytest.raises(ValueError):
        compute_ball_launch_from_splash(
            lie=BallLie(depth_m=0.005),
            ball=BallProperties(),
            delivery=delivery(),
            club_loft_rad=0.0,
        )


def test_a_loft_beyond_a_right_angle_is_impossible() -> None:
    with pytest.raises(ValueError):
        compute_ball_launch_from_splash(
            lie=BallLie(depth_m=0.005),
            ball=BallProperties(),
            delivery=delivery(),
            club_loft_rad=math.pi,
        )
