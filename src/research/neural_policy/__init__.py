"""Exploratory neural-policy tools (NP epic, #14475).

Concrete implementations used by issues #14478 (rank-1 intervention), #14479
(LoRA / constrained fine-tune comparator), and #14500 (NP-07 loop projection).
Kept small and dependency-free.
"""

from __future__ import annotations

from src.research.neural_policy.edits import (
    LowRankEditRecipe,
    apply_low_rank_weight_edit,
    probe_activation_direction,
)
from src.research.neural_policy.loop_projection import (
    FoldedLoopProjection,
    LoopConstraintProjector,
    LoopProjectionRecord,
    compute_loop_nullspace_projector,
)
from src.research.neural_policy.lora_adapter import LoRAAdapter, project_box

__all__ = [
    "FoldedLoopProjection",
    "LoopConstraintProjector",
    "LoopProjectionRecord",
    "LoRAAdapter",
    "LowRankEditRecipe",
    "apply_low_rank_weight_edit",
    "compute_loop_nullspace_projector",
    "probe_activation_direction",
    "project_box",
]
