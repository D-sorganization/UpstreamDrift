"""Discovery pointer: NM-01 learning freeze lives in ``neural_motion``.

Issue #10616 lists ``motion_matching`` fit/provider surfaces and training
config among the owned paths. The fail-closed learning-task, roster and
benefit-experiment freeze is intentionally shared under
``src/shared/python/neural_motion/`` so later NM epic children (#10603) can
extend one package. This module re-exports the public freeze surface so
surrogate callers and path-membership checks resolve honestly without
duplicating the contracts (DRY), matching the NM-00 ``nm00_audit`` pattern.
"""

from __future__ import annotations

from src.shared.python.neural_motion import (
    EXPERIMENT_SCHEMA,
    BenefitExperimentReceipt,
    BenefitExperimentSpec,
    ForwardDynamicsTask,
    InverseDynamicsTask,
    LearningTaskKind,
    MaskedTrajectoryTask,
    NeuralModelRoster,
    build_default_learning_tasks,
    build_neural_model_roster,
    freeze_benefit_experiment,
)

__all__ = [
    "EXPERIMENT_SCHEMA",
    "BenefitExperimentReceipt",
    "BenefitExperimentSpec",
    "ForwardDynamicsTask",
    "InverseDynamicsTask",
    "LearningTaskKind",
    "MaskedTrajectoryTask",
    "NeuralModelRoster",
    "build_default_learning_tasks",
    "build_neural_model_roster",
    "freeze_benefit_experiment",
]
