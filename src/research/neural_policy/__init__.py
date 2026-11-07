"""Exploratory neural-policy tools (NP epic, #14475).

Concrete implementations used by issues #14478 (rank-1 intervention), #14479
(LoRA comparator), and #14503 (offline weight inspector and guardrails).
"""

from __future__ import annotations

from src.research.neural_policy.edits import (
    LowRankEditRecipe,
    apply_low_rank_weight_edit,
    probe_activation_direction,
)
from src.research.neural_policy.lora_adapter import LoRAAdapter, project_box
from src.research.neural_policy.weight_inspector import (
    compute_effective_rank,
    compute_input_attribution,
    compute_matrix_conditioning,
    compute_state_dict_sha256,
    count_dead_units,
    export_inspection_json,
    find_lexical_guardrail_violations,
    inspect_layer,
    inspect_policy_weights,
)

__all__ = [
    "LoRAAdapter",
    "LowRankEditRecipe",
    "apply_low_rank_weight_edit",
    "compute_effective_rank",
    "compute_input_attribution",
    "compute_matrix_conditioning",
    "compute_state_dict_sha256",
    "count_dead_units",
    "export_inspection_json",
    "find_lexical_guardrail_violations",
    "inspect_layer",
    "inspect_policy_weights",
    "probe_activation_direction",
    "project_box",
]
