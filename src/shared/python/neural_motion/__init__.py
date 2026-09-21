"""Neural motion matching support package (epic #10603).

NM-00 (#10615) owns the fail-closed dataset/checkpoint/training-claim audit.
NM-01 (#10616) freezes learning-task contracts, TB-00-keyed dimensions and the
benefit-experiment registration. Later NM children extend this package; do not
invent parallel trainers here.
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
    BENEFIT_EXPERIMENT_SCHEMA,
    BaselineKind,
    BenefitExperimentSpec,
    BreakEvenResult,
    LatencyPhase,
    PilotScale,
    PromotionGates,
    compute_break_even,
    default_benefit_experiment,
    freeze_digest,
)
from .roster import (
    CheckpointDimensions,
    NeuralModelRosterEntry,
    dimensions_for_model,
    freeze_neural_model_roster,
)
from .tasks import (
    ForwardDynamicsTaskSpec,
    InverseDynamicsTaskSpec,
    InverseLabelMode,
    LearningTaskKind,
    MaskedTrajectoryTaskSpec,
    ObservationMask,
    TaskConditioning,
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
    "BENEFIT_EXPERIMENT_SCHEMA",
    "ArtifactAuditReceipt",
    "ArtifactIdentity",
    "ArtifactKind",
    "ArtifactRole",
    "BaselineKind",
    "BenefitExperimentSpec",
    "BreakEvenResult",
    "CheckpointDimensions",
    "ClaimStatus",
    "Disposition",
    "ForwardDynamicsTaskSpec",
    "InverseDynamicsTaskSpec",
    "InverseLabelMode",
    "LatencyPhase",
    "LearningTaskKind",
    "MaskedTrajectoryTaskSpec",
    "NeuralCoverageCell",
    "NeuralModelRosterEntry",
    "ObservationMask",
    "ParquetInspectResult",
    "PilotScale",
    "PromotionGates",
    "TaskConditioning",
    "audit_neural_artifacts",
    "classify_training_claim",
    "compute_break_even",
    "default_benefit_experiment",
    "dimensions_for_model",
    "freeze_digest",
    "freeze_neural_model_roster",
    "generate_neural_coverage_matrix",
    "inspect_parquet_bounded",
]
