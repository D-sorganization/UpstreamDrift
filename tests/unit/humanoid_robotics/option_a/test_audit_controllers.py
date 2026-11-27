"""Controller synergy audit, slice 2 (#13884): known-answer controllers.

On the #13837 plant linearised along its reference, a minimum-intervention
TVLQR (Q = 0, terminal C^T W C) must show a positive synergy index and a
null-space gain norm at round-off. A full-state LQR (Q > 0) on the same seeded
perturbation grid must show a lower index and a gain norm above epsilon.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace.controller_audit import (
    AuditInputs,
    audit_controller,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

_EPS = 1e-10


@pytest.fixture(scope="module")
def controllers():
    pytest.importorskip("mujoco")
    from src.shared.python.humanoid_robotics.option_a import audit_controllers as ac

    plant = ac.reference_linearization()
    return ac, plant, ac.minimum_intervention_gains(plant), ac.full_state_gains(plant)


def _audit(ac, plant, gains, grid):
    terminal = ac.closed_loop_terminal_states(plant, gains, grid)
    inputs = AuditInputs(terminal, ("perturbed",) * terminal.shape[0])
    return audit_controller(ac.audit_spec(), inputs, gains=gains, plant=plant)


def test_reference_linearization_matches_the_fixture_horizon(controllers) -> None:
    ac, plant, mip, full = controllers
    assert plant.n_steps == 350
    assert plant.n_state == 4
    assert mip.shape == full.shape == (350, 2, 4)
    assert np.all(np.isfinite(mip)) and np.all(np.isfinite(full))


def test_minimum_intervention_and_full_state_are_discriminated(controllers) -> None:
    ac, plant, mip, full = controllers
    grid = ac.perturbation_grid(40, seed=13884)
    lean = _audit(ac, plant, mip, grid)
    busy = _audit(ac, plant, full, grid)
    assert lean.ucm.delta_v > 0.0
    assert lean.null_space_gain_norm <= _EPS
    assert busy.ucm.delta_v < lean.ucm.delta_v
    assert busy.null_space_gain_norm > _EPS
    assert lean.tnc is not None and busy.tnc is not None


def test_perturbation_grid_is_seeded_and_validated(controllers) -> None:
    ac = controllers[0]
    np.testing.assert_array_equal(
        ac.perturbation_grid(30, seed=1), ac.perturbation_grid(30, seed=1)
    )
    assert not np.array_equal(
        ac.perturbation_grid(30, seed=1), ac.perturbation_grid(30, seed=2)
    )
    with pytest.raises(ValueError, match="n_trials"):
        ac.perturbation_grid(1, seed=1)
