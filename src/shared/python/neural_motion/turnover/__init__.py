"""Turnover package for neural motion matching models (NM-12 #10627).

Provides publication-ready model reproduction cards, clean-environment reproduction
commands, end-to-end user flow verification, and catalog persistence.
"""

from __future__ import annotations

from .catalog import (
    build_reproduction_catalog,
    load_reproduction_card,
    save_reproduction_catalog,
)
from .reproduce import (
    generate_reproduction_commands,
    verify_end_to_end_flow,
)
from .types import (
    EndToEndFlowReport,
    FlowStepOutcome,
    FlowStepStatus,
    ModelReproductionCard,
    PromotionVerdict,
)

__all__ = [
    "EndToEndFlowReport",
    "FlowStepOutcome",
    "FlowStepStatus",
    "ModelReproductionCard",
    "PromotionVerdict",
    "build_reproduction_catalog",
    "generate_reproduction_commands",
    "load_reproduction_card",
    "save_reproduction_catalog",
    "verify_end_to_end_flow",
]
