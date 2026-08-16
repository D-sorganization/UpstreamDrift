"""Conservation checks, split by class (issue #8616).

Code verification. **No experimental data appears in this file.**

The two classes get two different tests, and the suite enforces the
distinction rather than documenting it: a round-off residual that is
handed to an order test raises, and a truncation residual asked for a
fixed tolerance raises too.

The centrepiece is
:class:`TestAngularMomentumFindsWhatForceTestsCannot`, which monkeypatches
an axis swap into the solver's hand-written cross product and shows that
**the resultant force does not move at all** while both angular-momentum
residuals blow up by fourteen orders of magnitude.  That is the concrete
form of the digest's claim that angular momentum is the test which finds
this class of bug.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray

from bunkershot3d.solvers import (
    DRFTSolver,
    HeadKinematics,
    IntrusionState,
    ShotResult,
    ShotSettings,
    SurfaceElements,
    simulate_shot,
)
from bunkershot3d.vandv import (
    ROUND_OFF_TOLERANCE,
    ConservationClass,
    ConservationClassError,
    ConservationResidual,
    VerificationError,
    asymmetric_body_elements,
    element_moment_residual,
    energy_work_residual,
    flat_plate_elements,
    linear_impulse_residual,
    moment_transfer_residual,
    observed_order_from_residuals,
    spiral_body_elements,
)

pytestmark = [pytest.mark.unit, pytest.mark.scientific]

HEAD_MASS_KG = 0.300
"""A wedge head is 290-310 g."""

SHOT_WINDOW_S = 4.0e-4
"""Short enough that the body stays fully submerged at every step count.

Every refinement level must cover the *same* interval, or the energy
residuals are not comparable and the fitted order is meaningless."""

STEP_COUNTS = (16, 32, 64, 128)
"""Step counts for the timestep refinement, each double the last."""

FIRST_REFERENCE_M = (0.011, -0.007, -0.150)
SECOND_REFERENCE_M = (-0.019, 0.023, -0.131)
"""Two offset reference points, chosen so neither torque is axis-aligned."""


def _torque_state(elements: SurfaceElements) -> IntrusionState:
    """A query whose torque has three well-separated components."""
    return IntrusionState(
        elements, (18.0, 5.0, -3.0), reference_point_m=FIRST_REFERENCE_M
    )


def _run_shot(solver: DRFTSolver, *, n_steps: int) -> tuple[ShotResult, float]:
    """March the lopsided body through the bed for a fixed window.

    ``require_exit=False`` says what this shot is: a stated number of steps of
    a body that starts 100 mm down and is not meant to come out. The identity
    under test belongs to the update rule, not to a strike, so the march must
    not be required to reach an exit crossing.
    """
    step = SHOT_WINDOW_S / n_steps
    trace = simulate_shot(
        solver,
        spiral_body_elements(n_elements=64, radius_m=0.030),
        head_mass_kg=HEAD_MASS_KG,
        kinematics=HeadKinematics(
            velocity_m_s=(18.0, 0.0, -2.0), position_m=(0.0, 0.0, -0.100)
        ),
        settings=ShotSettings(
            time_step_s=step,
            max_time_s=SHOT_WINDOW_S,
            include_gravity=False,
            start_at_first_contact=False,
            require_exit=False,
        ),
    )
    return (trace, step)


class TestLinearMomentumRoundOffClass:
    """``m (v_k - v_0) = dt sum_{j<k} F_j``, an exact identity of the update."""

    def test_the_impulse_identity_holds_to_round_off(
        self, exact_solver: DRFTSolver
    ) -> None:
        trace, step = _run_shot(exact_solver, n_steps=64)
        residual = linear_impulse_residual(
            trace, head_mass_kg=HEAD_MASS_KG, time_step_s=step
        )
        assert residual.conservation_class is ConservationClass.ROUND_OFF
        assert residual.within_round_off
        assert residual.relative < ROUND_OFF_TOLERANCE

    def test_it_holds_at_every_step_count(self, exact_solver: DRFTSolver) -> None:
        """Round-off residuals do not improve with refinement, and need not."""
        for count in STEP_COUNTS:
            trace, step = _run_shot(exact_solver, n_steps=count)
            residual = linear_impulse_residual(
                trace, head_mass_kg=HEAD_MASS_KG, time_step_s=step
            )
            assert residual.within_round_off, f"n={count}"

    def test_a_wrong_mass_is_caught(self, exact_solver: DRFTSolver) -> None:
        """The identity is only an identity at the mass actually integrated."""
        trace, step = _run_shot(exact_solver, n_steps=32)
        residual = linear_impulse_residual(
            trace, head_mass_kg=HEAD_MASS_KG * 1.01, time_step_s=step
        )
        assert not residual.within_round_off

    def test_a_wrong_step_is_caught(self, exact_solver: DRFTSolver) -> None:
        trace, step = _run_shot(exact_solver, n_steps=32)
        residual = linear_impulse_residual(
            trace, head_mass_kg=HEAD_MASS_KG, time_step_s=step * 1.001
        )
        assert not residual.within_round_off

    def test_a_non_positive_mass_is_refused(self, exact_solver: DRFTSolver) -> None:
        trace, step = _run_shot(exact_solver, n_steps=16)
        with pytest.raises(VerificationError, match="head_mass_kg"):
            linear_impulse_residual(trace, head_mass_kg=0.0, time_step_s=step)


class TestAngularMomentumRoundOffClass:
    """The moment identities, which a linear-momentum test cannot reach."""

    def test_moment_transfer_holds_to_round_off(self, exact_solver: DRFTSolver) -> None:
        """``tau(p2) = tau(p1) + (p1 - p2) x F``, exactly."""
        state = _torque_state(asymmetric_body_elements())
        residual = moment_transfer_residual(
            exact_solver, state, reference_point_m=SECOND_REFERENCE_M
        )
        assert residual.conservation_class is ConservationClass.ROUND_OFF
        assert residual.within_round_off

    def test_the_element_moment_matches_the_naive_oracle(
        self, exact_solver: DRFTSolver
    ) -> None:
        """The vectorised cross product agrees with a per-element loop."""
        state = _torque_state(asymmetric_body_elements())
        residual = element_moment_residual(exact_solver, state)
        assert residual.conservation_class is ConservationClass.ROUND_OFF
        assert residual.within_round_off

    def test_a_rotating_body_is_refused_by_the_transfer_check(
        self, exact_solver: DRFTSolver
    ) -> None:
        """Moving the reference point of a rotating body changes the physics."""
        state = IntrusionState(
            asymmetric_body_elements(),
            (18.0, 5.0, -3.0),
            angular_velocity_rad_s=(0.0, 40.0, 0.0),
            reference_point_m=FIRST_REFERENCE_M,
        )
        with pytest.raises(VerificationError, match="non-rotating"):
            moment_transfer_residual(
                exact_solver, state, reference_point_m=SECOND_REFERENCE_M
            )

    def test_a_symmetric_configuration_is_refused_as_vacuous(
        self, exact_solver: DRFTSolver
    ) -> None:
        """A flat plate about its own centroid has no torque to check.

        This is the guard that stops the angular-momentum test passing for
        the wrong reason. A configuration with a vanishing torque
        component cannot detect an axis swap, so the suite refuses it
        rather than reporting a pass.
        """
        elements = flat_plate_elements(area_m2=1.6e-3, depth_m=0.040)
        state = IntrusionState(elements, (0.0, 0.0, -5.0))
        with pytest.raises(VerificationError, match="no net torque|component"):
            element_moment_residual(exact_solver, state)

    def test_a_query_engaging_nothing_is_refused(
        self, exact_solver: DRFTSolver
    ) -> None:
        elements = flat_plate_elements(area_m2=1.6e-3, depth_m=0.040)
        state = IntrusionState(elements, (0.0, 0.0, 5.0))
        with pytest.raises(VerificationError, match="no element"):
            element_moment_residual(exact_solver, state)


class TestAngularMomentumFindsWhatForceTestsCannot:
    """Swap two axes in the solver's cross product and see who notices.

    The solver replaces ``np.cross`` with a hand-written component form
    for speed. That is a sound optimisation, and it is also the exact
    place where an index or sign error survives review: **the resultant
    force never touches the cross product**, so no force test can move.
    """

    @staticmethod
    def _axis_swapped_cross_sum(
        lever: NDArray[np.float64], force: NDArray[np.float64]
    ) -> NDArray[np.float64]:
        """The correct cross product with its first two components swapped."""
        correct = np.array(
            [
                float((lever[:, 1] * force[:, 2] - lever[:, 2] * force[:, 1]).sum()),
                float((lever[:, 2] * force[:, 0] - lever[:, 0] * force[:, 2]).sum()),
                float((lever[:, 0] * force[:, 1] - lever[:, 1] * force[:, 0]).sum()),
            ],
            dtype=np.float64,
        )
        return correct[[1, 0, 2]]

    def test_the_resultant_force_does_not_move_at_all(
        self, exact_solver: DRFTSolver, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The premise: this defect is invisible to every force check."""
        state = _torque_state(asymmetric_body_elements())
        healthy = exact_solver.solve(state).wrench.force_n.copy()
        monkeypatch.setattr(
            "bunkershot3d.solvers.drft._cross_sum", self._axis_swapped_cross_sum
        )
        injured = exact_solver.solve(state).wrench.force_n
        assert np.array_equal(healthy, injured)

    def test_the_element_moment_oracle_catches_it(
        self, exact_solver: DRFTSolver, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = _torque_state(asymmetric_body_elements())
        monkeypatch.setattr(
            "bunkershot3d.solvers.drft._cross_sum", self._axis_swapped_cross_sum
        )
        residual = element_moment_residual(exact_solver, state)
        assert not residual.within_round_off
        assert residual.relative > 1e-3

    def test_the_moment_transfer_identity_catches_it_too(
        self, exact_solver: DRFTSolver, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = _torque_state(asymmetric_body_elements())
        monkeypatch.setattr(
            "bunkershot3d.solvers.drft._cross_sum", self._axis_swapped_cross_sum
        )
        residual = moment_transfer_residual(
            exact_solver, state, reference_point_m=SECOND_REFERENCE_M
        )
        assert not residual.within_round_off


class TestEnergyTruncationClass:
    """Energy under a non-symplectic scheme: the order test *is* the test."""

    def test_the_work_energy_residual_decays_at_first_order(
        self, exact_solver: DRFTSolver
    ) -> None:
        """Semi-implicit Euler leaves ``-(dt^2/2m)|F|^2`` per step.

        Summed over ``T/dt`` steps that is ``O(dt)``, so the observed
        order must be 1. Not zero, which would mean the residual is
        round-off and the check is measuring nothing; not two, which would
        mean the trace is not resolving the force history.
        """
        residuals = []
        for count in STEP_COUNTS:
            trace, step = _run_shot(exact_solver, n_steps=count)
            residuals.append(
                energy_work_residual(trace, head_mass_kg=HEAD_MASS_KG, time_step_s=step)
            )
        observed = observed_order_from_residuals(residuals)
        assert observed.order == pytest.approx(1.0, abs=0.1)
        assert observed.monotone

    def test_the_residual_is_truncation_class_and_carries_its_step(
        self, exact_solver: DRFTSolver
    ) -> None:
        trace, step = _run_shot(exact_solver, n_steps=32)
        residual = energy_work_residual(
            trace, head_mass_kg=HEAD_MASS_KG, time_step_s=step
        )
        assert residual.conservation_class is ConservationClass.TRUNCATION
        assert residual.step_size_s == pytest.approx(step)

    def test_the_residual_is_not_round_off_small(
        self, exact_solver: DRFTSolver
    ) -> None:
        """If it were, the order test would be fitting floating-point noise."""
        trace, step = _run_shot(exact_solver, n_steps=32)
        residual = energy_work_residual(
            trace, head_mass_kg=HEAD_MASS_KG, time_step_s=step
        )
        assert residual.relative > 1e-6


class TestClassDisciplineIsEnforced:
    """The two classes cannot be tested each other's way."""

    def test_a_truncation_residual_refuses_a_fixed_tolerance(self) -> None:
        residual = ConservationResidual(
            name="energy",
            conservation_class=ConservationClass.TRUNCATION,
            residual=1e-3,
            scale=1.0,
            step_size_s=1e-4,
        )
        with pytest.raises(ConservationClassError, match="observed order"):
            _ = residual.within_round_off

    def test_a_round_off_residual_refuses_an_order_test(self) -> None:
        residuals = [
            ConservationResidual(
                name="linear momentum",
                conservation_class=ConservationClass.ROUND_OFF,
                residual=1e-16 * scale,
                scale=1.0,
            )
            for scale in (1.0, 2.0)
        ]
        with pytest.raises(ConservationClassError, match="NO order test"):
            observed_order_from_residuals(residuals)

    def test_a_truncation_residual_without_a_step_is_refused(self) -> None:
        with pytest.raises(ConservationClassError, match="carries no step size"):
            ConservationResidual(
                name="energy",
                conservation_class=ConservationClass.TRUNCATION,
                residual=1.0,
                scale=1.0,
            )

    def test_a_round_off_residual_carrying_a_step_is_refused(self) -> None:
        """Recording a step on a round-off residual invites the wrong test."""
        with pytest.raises(ConservationClassError, match="do not scale"):
            ConservationResidual(
                name="linear momentum",
                conservation_class=ConservationClass.ROUND_OFF,
                residual=1e-16,
                scale=1.0,
                step_size_s=1e-4,
            )

    def test_a_residual_without_a_scale_is_refused(self) -> None:
        with pytest.raises(VerificationError, match="nothing to be judged"):
            ConservationResidual(
                name="nothing",
                conservation_class=ConservationClass.ROUND_OFF,
                residual=1.0,
                scale=0.0,
            )


class TestDissipation:
    """The inertial term provably removes energy; the depth term does not."""

    def test_the_energy_residual_is_a_defect_not_a_gain(
        self, exact_solver: DRFTSolver
    ) -> None:
        """The head must end slower than it started, at every step count."""
        for count in STEP_COUNTS:
            trace, _ = _run_shot(exact_solver, n_steps=count)
            assert trace.exit_speed_m_s < trace.entry_speed_m_s, f"n={count}"
