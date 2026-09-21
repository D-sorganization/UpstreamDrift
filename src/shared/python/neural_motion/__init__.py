"""Neural motion matching support package (epic #10603).

NM-00 (#10615) owns the fail-closed dataset/checkpoint/training-claim audit.
NM-01 (#10616) freezes learning tasks, the model roster and the benefit
experiment. Later NM children extend this package; do not invent parallel
trainers here.
"""

from __future__ import annotations

from .audit import (
    AUDIT_SCHEMA,
    ArtifactAuditReceipt,
    ParquetInspectResult,
    audit_neural_artifacts,
    inspect_parquet_bounded,
)
from .claims import classify_training_claim
from .coverage import NeuralCoverageCell, generate_neural_coverage_matrix
from .experiment import (
    BASELINE_METHODS,
    EXPERIMENT_SCHEMA,
    LATENCY_PHASES,
    NESTED_EPISODE_STAGES,
    BenefitExperimentReceipt,
    BenefitExperimentSpec,
    BenchmarkCaseKind,
    ComputeBudgetCaps,
    PromotionGates,
    break_even_queries,
    default_benefit_experiment,
    freeze_benefit_experiment,
)
from .roster import (
    NeuralModelRoster,
    NeuralRosterEntry,
    RosterStage,
    build_neural_model_roster,
    resolve_roster_entry,
)
from .tasks import (
    ConditioningSpec,
    ForwardDynamicsTask,
    InverseDynamicsTask,
    InverseLabelPolicy,
    LearningTaskKind,
    MaskedTrajectoryTask,
    TaskDimensions,
    build_default_learning_tasks,
    dimensions_from_model,
)
from .types import (
    ArtifactIdentity,
    ArtifactKind,
    ArtifactRole,
    ClaimStatus,
    Disposition,
)

__all__ = [
    "AUDIT_SCHEMA",
    "BASELINE_METHODS",
    "EXPERIMENT_SCHEMA",
    "LATENCY_PHASES",
    "NESTED_EPISODE_STAGES",
    "ArtifactAuditReceipt",
    "ArtifactIdentity",
    "ArtifactKind",
    "ArtifactRole",
    "BenefitExperimentReceipt",
    "BenefitExperimentSpec",
    "BenchmarkCaseKind",
    "ClaimStatus",
    "ComputeBudgetCaps",
    "ConditioningSpec",
    "Disposition",
    "ForwardDynamicsTask",
    "InverseDynamicsTask",
    "InverseLabelPolicy",
    "LearningTaskKind",
    "MaskedTrajectoryTask",
    "NeuralCoverageCell",
    "NeuralModelRoster",
    "NeuralRosterEntry",
    "ParquetInspectResult",
    "PromotionGates",
    "RosterStage",
    "TaskDimensions",
    "audit_neural_artifacts",
    "break_even_queries",
    "build_default_learning_tasks",
    "build_neural_model_roster",
    "classify_training_claim",
    "default_benefit_experiment",
    "dimensions_from_model",
    "freeze_benefit_experiment",
    "generate_neural_coverage_matrix",
    "inspect_parquet_bounded",
    "resolve_roster_entry",
]
