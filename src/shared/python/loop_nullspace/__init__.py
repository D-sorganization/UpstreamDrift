"""Closed-loop null-space exploration (NSX) package (Epic #12593)."""

from __future__ import annotations

from src.shared.python.loop_nullspace.constrained_dynamics import (
    ConstrainedEquationsOfMotionProvider,
    compute_constrained_ztcf_zvcf,
    constrained_acceleration,
)
from src.shared.python.loop_nullspace.contracts import (
    CLOSURE_KIND_ROWS,
    AllocationDependence,
    ClosedLoopSnapshot,
    ClosureKind,
    FootContact,
    GripFrame,
    InputProvenance,
    LoopTopology,
    ModelProvenance,
    QuantityRecord,
    SpatialInertia,
    SpatialInertiaPartition,
    ValidationStatus,
)
from src.shared.python.loop_nullspace.decomposition import (
    AllocationMetric,
    Decomposition,
    ZeroSideCase,
    ZeroSideResult,
    decompose,
    min_metric_torque,
    zero_side_solve,
)
from src.shared.python.loop_nullspace.fixtures import (
    GOLDEN_FIXTURES_PATH,
    GoldenValues,
    GolferConstraintRankAudit,
    analytic_envelope_case,
    audit_golfer_constraints_rank,
    load_golden_fixtures,
    planar_golfer_snapshot,
    spatial_two_arm_club_snapshot,
    straight_arm_snapshot,
)

__all__ = [
    "CLOSURE_KIND_ROWS",
    "GOLDEN_FIXTURES_PATH",
    "AllocationDependence",
    "AllocationMetric",
    "ClosedLoopSnapshot",
    "ClosureKind",
    "ConstrainedEquationsOfMotionProvider",
    "Decomposition",
    "FootContact",
    "GoldenValues",
    "GolferConstraintRankAudit",
    "GripFrame",
    "InputProvenance",
    "LoopTopology",
    "ModelProvenance",
    "QuantityRecord",
    "SpatialInertia",
    "SpatialInertiaPartition",
    "ValidationStatus",
    "ZeroSideCase",
    "ZeroSideResult",
    "analytic_envelope_case",
    "audit_golfer_constraints_rank",
    "compute_constrained_ztcf_zvcf",
    "constrained_acceleration",
    "decompose",
    "load_golden_fixtures",
    "min_metric_torque",
    "planar_golfer_snapshot",
    "spatial_two_arm_club_snapshot",
    "straight_arm_snapshot",
    "zero_side_solve",
]
