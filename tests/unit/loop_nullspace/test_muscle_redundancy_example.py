"""NSX-73 (#12591): verified Hill-muscle example behind the muscle-level report.

Measures, on the existing elbow fixture (``create_elbow_muscle_system``), that a
single joint driven by three muscles has a moment-arm map of rank one and a
two-dimensional null space, and that moving along that null space leaves the net
joint torque unchanged while changing the antagonist (co-contraction) activation.
The fixture parameters belong to ``multi_muscle`` and are not modelled on data.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics.hill_muscle import MuscleState
from src.shared.python.biomechanics.multi_muscle import create_elbow_muscle_system

pytestmark = pytest.mark.unit

# Isometric states at optimal fiber length (passive force is zero there).
STATES = {"biceps": (0.15, 0.0), "brachialis": (0.12, 0.0), "triceps": (0.18, 0.0)}


@pytest.fixture(scope="module")
def elbow():
    return create_elbow_muscle_system(enable_rust=False)


def _torque_map(elbow) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return names, moment-arm row R (1 x m) and force per unit activation."""
    names = elbow.muscle_names
    muscles = {**elbow.agonist.muscles, **elbow.antagonist.muscles}
    arms = {**elbow.agonist.attachments, **elbow.antagonist.attachments}
    r = np.array([[arms[n].moment_arm for n in names]])
    f_unit = np.array(
        [
            muscles[n].compute_force(
                MuscleState(activation=1.0, l_CE=STATES[n][0], v_CE=0.0, l_MT=0.0)
            )
            for n in names
        ]
    )
    return names, r, f_unit


def test_moment_arm_map_has_two_dimensional_null_space(elbow):
    names, r, _ = _torque_map(elbow)
    assert len(names) == 3
    rank = np.linalg.matrix_rank(r)
    assert rank == 1
    assert len(names) - rank == 2


def test_activation_is_linear_at_fixed_state(elbow):
    """The premise of the pulled-back feasible set: F = a * F(1) at fixed state."""
    names, _, f_unit = _torque_map(elbow)
    muscles = {**elbow.agonist.muscles, **elbow.antagonist.muscles}
    for a in (0.0, 0.25, 0.5, 1.0):
        for n, f1 in zip(names, f_unit, strict=True):
            f = muscles[n].compute_force(
                MuscleState(activation=a, l_CE=STATES[n][0], v_CE=0.0, l_MT=0.0)
            )
            assert f == pytest.approx(a * f1, abs=1e-9)


def test_null_space_step_changes_cocontraction_not_torque(elbow):
    names, r, f_unit = _torque_map(elbow)
    gain = r * f_unit[None, :]  # N m per unit activation, 1 x m
    a0 = np.array([0.5, 0.5, 0.2])
    # Null vector of the gain row with a positive triceps component.
    null = np.array([gain[0, 2], 0.0, -gain[0, 0]])
    null = null if null[2] > 0.0 else -null
    assert abs(float((gain @ null)[0])) < 1e-9
    a1 = a0 + 0.1 * null / np.abs(null).max()
    assert np.all((a1 >= 0.0) & (a1 <= 1.0))

    def torque(a: np.ndarray) -> float:
        return elbow.compute_net_torque(
            {"biceps": a[0], "brachialis": a[1]},
            {"triceps": a[2]},
            STATES,
        )

    assert torque(a1) == pytest.approx(torque(a0), abs=1e-9)
    assert a1[2] > a0[2]
    assert names[2] == "triceps"
