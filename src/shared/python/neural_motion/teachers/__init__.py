"""Feasible teacher episodes and active-learning candidates (NM-04 #10619).

Generate bounded trajectories near qualified baselines with structured
perturbations; retain infeasible rollouts separately; acquire new samples
without consuming test labels. Nested stages reuse NM-01 caps/sizes.
"""

from __future__ import annotations

from .acquisition import ActiveLearningAcquirer
from .corpus import NestedTeacherCorpus, StageRunResult
from .generation import TeacherEpisodeGenerator
from .types import (
    ACQUISITION_SCHEMA,
    TEACHER_SCHEMA,
    AcquisitionEntry,
    AcquisitionLog,
    AcquisitionStrategy,
    PerturbationKind,
    RejectionEntry,
    RejectionLedger,
    TeacherOutcome,
    TeacherSpec,
)

__all__ = [
    "ACQUISITION_SCHEMA",
    "TEACHER_SCHEMA",
    "AcquisitionEntry",
    "AcquisitionLog",
    "AcquisitionStrategy",
    "ActiveLearningAcquirer",
    "NestedTeacherCorpus",
    "PerturbationKind",
    "RejectionEntry",
    "RejectionLedger",
    "StageRunResult",
    "TeacherEpisodeGenerator",
    "TeacherOutcome",
    "TeacherSpec",
]
