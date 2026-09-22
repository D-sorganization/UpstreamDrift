"""Feasible teacher episodes and active-learning candidates (NM-04 #10619)."""

from __future__ import annotations

from .acquisition import AcquisitionBatch, select_acquisition_candidates
from .backends import MockTeacherRolloutBackend, TeacherRolloutBackend
from .generation import (
    RejectedRolloutStore,
    TeacherGenerationCampaign,
    TeacherGenerationState,
    TeacherStageReceipt,
    load_generation_state,
    save_generation_state,
)
from .pilot import PILOT_MODEL_ID, build_pilot_teacher_spec
from .types import (
    TEACHER_CAMPAIGN_SCHEMA,
    AcquisitionMode,
    TeacherAnchor,
    TeacherGenerationSpec,
    TeacherRolloutRequest,
    TeacherRolloutResult,
)

__all__ = [
    "TEACHER_CAMPAIGN_SCHEMA",
    "PILOT_MODEL_ID",
    "AcquisitionBatch",
    "AcquisitionMode",
    "MockTeacherRolloutBackend",
    "RejectedRolloutStore",
    "TeacherAnchor",
    "TeacherGenerationCampaign",
    "TeacherGenerationSpec",
    "TeacherGenerationState",
    "TeacherRolloutBackend",
    "TeacherRolloutRequest",
    "TeacherRolloutResult",
    "TeacherStageReceipt",
    "build_pilot_teacher_spec",
    "load_generation_state",
    "save_generation_state",
    "select_acquisition_candidates",
]
