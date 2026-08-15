"""Energy partition, against hand arithmetic (issue #8614, W7).

The decelerating trace is built so the sums are checkable on paper: a 0.300 kg
head slowing 25 -> 15 m/s in 10 ms loses ``0.5 * 0.3 * (625 - 225) = 60 J``, and
the constant ``-300 N`` holding it back does ``300 N * 0.200 m = 60 J`` of work
on the sand over the 0.200 m it travels. Residual zero, by construction.
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.metrics import (
    BallLaunch,
    HeadModel,
    StrikeTrace,
    energy_partition,
    head_kinetic_energy_J,
)

from .conftest import build_decelerating_trace, reference_head

pytestmark = pytest.mark.unit

#: 0.5 * 0.300 kg * (25^2 - 15^2) m^2/s^2.
EXPECTED_KE_LOSS_J = 60.0

#: A conforming golf ball at 20 m/s: 0.5 * 0.04593 kg * 400 m^2/s^2.
BALL_MASS_KG = 0.04593
BALL_SPEED_MPS = 20.0
EXPECTED_BALL_ENERGY_J = 0.5 * BALL_MASS_KG * BALL_SPEED_MPS**2


class TestHeadKineticEnergy:
    """The kinetic-energy series the partition is taken across."""

    def test_endpoints_match_the_prescribed_speeds(
        self, decelerating_trace, head
    ) -> None:
        """0.5 * 0.3 * 25^2 = 93.75 J at entry, 0.5 * 0.3 * 15^2 = 33.75 J at exit."""
        energy = head_kinetic_energy_J(decelerating_trace, head)

        assert energy[0] == pytest.approx(93.75, rel=1e-9)
        assert energy[-1] == pytest.approx(33.75, rel=1e-9)
        assert energy[0] - energy[-1] == pytest.approx(EXPECTED_KE_LOSS_J, rel=1e-9)

    def test_rotational_energy_is_included_when_the_inertia_is_known(
        self, head
    ) -> None:
        """A head spinning at 10 rad/s about +z carries 0.5 * 4e-4 * 100 = 0.02 J.

        The translational part is zero because the head is held still, so the
        whole of the reported energy is the rotational term.
        """
        rate_radps = 10.0
        time_s = np.linspace(0.0, 0.01, 51)
        angle = 0.5 * rate_radps * time_s
        quaternions = np.column_stack(
            [np.cos(angle), np.zeros_like(angle), np.zeros_like(angle), np.sin(angle)]
        )
        trace = StrikeTrace(
            time_s=time_s,
            head_position_m=np.zeros((time_s.size, 3)),
            head_orientation_quat=quaternions,
            sand_force_N=np.zeros((time_s.size, 3)),
            sand_moment_Nm=np.zeros((time_s.size, 3)),
        )

        energy = head_kinetic_energy_J(trace, head)

        assert energy[len(energy) // 2] == pytest.approx(
            0.5 * 4.0e-4 * rate_radps**2, rel=1e-6
        )


class TestEnergyPartition:
    """The three-way split, and the closure that makes it trustworthy."""

    def test_work_on_sand_equals_the_kinetic_energy_loss(
        self, decelerating_trace, head
    ) -> None:
        """Force and motion are consistent, so nothing is left over."""
        partition = energy_partition(decelerating_trace, head)

        assert partition.club_kinetic_energy_loss_J == pytest.approx(
            EXPECTED_KE_LOSS_J, rel=1e-9
        )
        assert partition.work_on_sand_J == pytest.approx(EXPECTED_KE_LOSS_J, rel=1e-9)
        assert partition.residual_J == pytest.approx(0.0, abs=1e-9)
        assert partition.sand_fraction == pytest.approx(1.0, rel=1e-9)

    def test_a_sand_driven_ball_takes_its_share_out_of_the_sand(
        self, decelerating_trace, head
    ) -> None:
        """9.186 J of 60 J is 15.31 %; the sand keeps 50.814 J."""
        ball = BallLaunch(mass_kg=BALL_MASS_KG, speed_mps=BALL_SPEED_MPS)

        partition = energy_partition(decelerating_trace, head, ball=ball)

        assert partition.ball_energy_J == pytest.approx(
            EXPECTED_BALL_ENERGY_J, rel=1e-12
        )
        assert partition.sand_retained_J == pytest.approx(
            EXPECTED_KE_LOSS_J - EXPECTED_BALL_ENERGY_J, rel=1e-9
        )
        assert partition.ball_fraction == pytest.approx(
            EXPECTED_BALL_ENERGY_J / EXPECTED_KE_LOSS_J, rel=1e-9
        )
        assert partition.ball_fraction == pytest.approx(0.1531, rel=1e-3)

    def test_a_directly_struck_ball_is_a_separate_branch(
        self, decelerating_trace, head
    ) -> None:
        """Then the sand keeps all 60 J and the residual carries the ball's share."""
        ball = BallLaunch(
            mass_kg=BALL_MASS_KG, speed_mps=BALL_SPEED_MPS, driven_by_sand=False
        )

        partition = energy_partition(decelerating_trace, head, ball=ball)

        assert partition.sand_retained_J == pytest.approx(EXPECTED_KE_LOSS_J, rel=1e-9)
        assert partition.residual_J == pytest.approx(-EXPECTED_BALL_ENERGY_J, rel=1e-9)
        assert partition.closes()

    @pytest.mark.parametrize("ball_speed_mps", [0.0, 10.0, 20.0, 30.0])
    def test_the_fractions_always_sum_to_one(
        self, decelerating_trace, head, ball_speed_mps: float
    ) -> None:
        """Closure is structural: the residual is defined as what is left."""
        ball = BallLaunch(mass_kg=BALL_MASS_KG, speed_mps=ball_speed_mps)

        partition = energy_partition(decelerating_trace, head, ball=ball)

        assert partition.fraction_sum == pytest.approx(1.0, abs=1e-12)
        assert partition.closes()

    def test_a_head_that_gains_energy_is_refused(self, head) -> None:
        """There is no partition of a negative loss, so it raises rather than signs it."""
        accelerating = build_decelerating_trace(
            entry_speed_mps=15.0, exit_speed_mps=25.0
        )

        with pytest.raises(ValueError, match="does not lose kinetic energy"):
            energy_partition(accelerating, head)

    def test_a_ball_richer_than_the_sand_work_is_refused(
        self, decelerating_trace, head
    ) -> None:
        """60 J of work cannot launch a 100 J ball; that is an input inconsistency."""
        ball = BallLaunch(mass_kg=BALL_MASS_KG, speed_mps=100.0)

        with pytest.raises(ValueError, match="more energy than the head delivered"):
            energy_partition(decelerating_trace, head, ball=ball)

    def test_rotation_reporting_is_honest_without_an_inertia_tensor(
        self, decelerating_trace
    ) -> None:
        """A head with no inertia tensor reports that rotation was left out."""
        reference = reference_head()
        head_without_inertia = HeadModel(
            mass_kg=reference.mass_kg,
            centre_of_mass_body_m=reference.centre_of_mass_body_m,
            sole_reference_body_m=reference.sole_reference_body_m,
            shaft_axis_body=reference.shaft_axis_body,
        )

        partition = energy_partition(decelerating_trace, head_without_inertia)

        assert partition.rotation_included is False


class TestBallLaunch:
    """Validation of the ball state, which the artifact cannot supply."""

    def test_spin_without_inertia_is_refused(self) -> None:
        """Reporting spin without inertia would silently drop the spin energy."""
        with pytest.raises(ValueError, match="spin_radps is non-zero"):
            BallLaunch(mass_kg=BALL_MASS_KG, speed_mps=20.0, spin_radps=900.0)

    def test_spin_energy_is_added_when_the_inertia_is_given(self) -> None:
        """A uniform-sphere ball at 900 rad/s: 0.5 * 3.72e-5 * 810000 = 15.066 J."""
        launch = BallLaunch(
            mass_kg=BALL_MASS_KG,
            speed_mps=0.0,
            spin_radps=900.0,
            inertia_kg_m2=3.72e-5,
        )

        assert launch.kinetic_energy_J == pytest.approx(0.5 * 3.72e-5 * 900.0**2)

    def test_negative_mass_is_refused(self) -> None:
        """A negative ball mass is not a modelling choice."""
        with pytest.raises(ValueError, match="mass_kg must be positive"):
            BallLaunch(mass_kg=-0.04, speed_mps=20.0)
