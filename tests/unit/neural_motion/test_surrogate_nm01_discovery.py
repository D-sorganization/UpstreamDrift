"""NM-01 (#10616): surrogate discovery pointer re-exports neural_motion freeze."""

from __future__ import annotations

import pytest

pytestmark = pytest.mark.unit


def test_surrogate_nm01_freeze_reexports_neural_motion_surface() -> None:
    from src.shared.python.motion_matching.surrogate import nm01_freeze
    from src.shared.python import neural_motion

    assert nm01_freeze.EXPERIMENT_SCHEMA == neural_motion.EXPERIMENT_SCHEMA
    assert (
        nm01_freeze.build_default_learning_tasks
        is neural_motion.build_default_learning_tasks
    )
    assert (
        nm01_freeze.build_neural_model_roster is neural_motion.build_neural_model_roster
    )
    assert (
        nm01_freeze.freeze_benefit_experiment is neural_motion.freeze_benefit_experiment
    )
