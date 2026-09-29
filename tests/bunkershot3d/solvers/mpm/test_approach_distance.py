"""The F1 approach must not blow up on a near-level descent (issue #8713).

F1 has no instantaneous answer, so it builds a history: the section is
reversed along its own velocity until it is clear of the bed, then driven
back to the queried pose. Reversing along the velocity means dividing the
required height by the velocity's vertical component -- and at the deepest
sample of a real shot that component is essentially zero.

Measured on the workbench's own 25 m/s greenside record: at the deepest
sample the direction cosine on ``z`` is -0.0026, so a 24 mm climb needs a
**9.5 m** run-in. The bed is then sized to cover it, and a 9.5 m bed at a
6 mm cell is fifty thousand particles marched for twenty thousand steps --
a query that looks ordinary and never returns. Nothing caught it but the
``max_steps`` cap, and only after the bed had been allocated.

The rule below is the one the level branch already implements, applied
whenever the descending branch would be the longer of the two: run in
horizontally from beyond the body's own length. Same modelling assumption,
chosen by which one is actually shorter rather than by the sign of a
number that passes through zero mid-shot.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.sand import PlayingCondition, playing_condition
from bunkershot3d.solvers.elements import SurfaceElements
from bunkershot3d.solvers.envelope import RefusalPolicy
from bunkershot3d.solvers.mpm.constitutive import SandContinuum
from bunkershot3d.solvers.mpm.solver import PlaneStrainMPMSolver
from bunkershot3d.solvers.protocol import IntrusionState

pytestmark = pytest.mark.unit


@pytest.fixture(scope="module")
def material() -> SandContinuum:
    return SandContinuum.from_sand_state(playing_condition(PlayingCondition.FIRM))


@pytest.fixture(scope="module")
def solver(material: SandContinuum) -> PlaneStrainMPMSolver:
    return PlaneStrainMPMSolver(
        material=material,
        cell_size_m=0.006,
        effective_width_m=0.030,
        bed_depth_m=0.040,
        run_in_lengths=0.3,
        refusal_policy=RefusalPolicy.REPORT,
        max_steps=200000,
    )


def submerged_state(descent_deg: float, *, speed_m_s: float = 20.0) -> IntrusionState:
    """A 40 x 16 mm section 12 mm under the surface at a stated descent."""
    corners = np.array(
        [
            [-0.020, 0.0, -0.020],
            [0.020, 0.0, -0.020],
            [0.020, 0.0, -0.004],
            [-0.020, 0.0, -0.004],
        ]
    )
    normals = np.tile([0.0, 0.0, -1.0], (corners.shape[0], 1))
    areas = np.full(corners.shape[0], 4.0e-4)
    angle = math.radians(descent_deg)
    return IntrusionState(
        SurfaceElements(corners, normals, areas),
        (speed_m_s * math.cos(angle), 0.0, -speed_m_s * math.sin(angle)),
        free_surface_height_m=0.0,
    )


class TestANearLevelDescentDoesNotDemandAMetreOfRunIn:
    """The failure a deep sweep pose walks straight into."""

    def test_the_approach_is_bounded_by_the_horizontal_run_in(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        """0.15 deg of descent must not cost more than a level query does."""
        level, _ = solver._approach(submerged_state(0.0))
        shallow, shallow_distance = solver._approach(submerged_state(0.15))
        level_distance = solver._approach(submerged_state(0.0))[1]
        del level, shallow
        # Unbounded, this is (0.012 + 0.020) / sin(0.15 deg) = 12 m.
        assert shallow_distance < 2.0 * level_distance, (
            f"{shallow_distance:.4g} m against a level run-in of {level_distance:.4g} m"
        )
        assert shallow_distance < 0.5

    def test_a_steep_descent_still_backs_straight_out_of_the_bed(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        """The guard must not take over where the descending branch is right."""
        _, steep = solver._approach(submerged_state(60.0))
        _, level = solver._approach(submerged_state(0.0))
        assert steep < level

    def test_the_step_count_stays_the_same_order_across_the_transition(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        """No discontinuity as the descent passes through the switch."""
        counts = [
            solver._approach_steps(
                submerged_state(angle),
                solver._approach(submerged_state(angle))[1],
                solver.time_step_s(submerged_state(angle)),
            )
            for angle in (0.05, 0.5, 5.0, 30.0)
        ]
        assert max(counts) < 10 * min(counts), counts

    def test_a_shallow_descending_query_actually_runs(
        self, solver: PlaneStrainMPMSolver
    ) -> None:
        """The observable: it returns, rather than allocating a 9 m bed."""
        run = solver.run(submerged_state(0.15))
        assert run.n_steps > 0
        assert run.n_steps < 2000, run.n_steps
