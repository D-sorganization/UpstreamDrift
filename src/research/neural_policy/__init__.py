"""Exploratory neural-policy tools (NP epic, #14475).

Concrete implementations used by issues #14478 (rank-1 intervention) and #14479
(LoRA / constrained fine-tune comparator). Kept small and dependency-free.
"""

from __future__ import annotations

from src.research.neural_policy.edits import (
    LowRankEditRecipe,
    apply_low_rank_weight_edit,
    probe_activation_direction,
)
from src.research.neural_policy.lora_adapter import LoRAAdapter, project_box

__all__ = [
    "LoRAAdapter",
    "LowRankEditRecipe",
    "apply_low_rank_weight_edit",
    "probe_activation_direction",
    "project_box",
]
