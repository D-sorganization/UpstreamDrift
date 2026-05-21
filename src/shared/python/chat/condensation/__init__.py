"""Thread condensation (Tools issue #2736).

Public surface:

* :class:`CondensationRequest`, :class:`CondensationResult` -- contracts.
* :class:`Condenser` -- orchestrator.
* :class:`CondensationStrategy` and three concrete strategies:
  :class:`KeepRecentStrategy`, :class:`SemanticSummaryStrategy`,
  :class:`PinnedAnchorStrategy`.
* :func:`estimate_tokens` -- helper used by the orchestrator.
"""

from __future__ import annotations

from .condenser import Condenser
from .contracts import CondensationRequest, CondensationResult, StrategyName
from .strategy import (
    STRATEGY_REGISTRY,
    CondensationStrategy,
    KeepRecentStrategy,
    PinnedAnchorStrategy,
    SemanticSummaryStrategy,
    SummaryProvider,
)
from .tokens import estimate_tokens

__all__ = [
    "CondensationRequest",
    "CondensationResult",
    "StrategyName",
    "Condenser",
    "CondensationStrategy",
    "KeepRecentStrategy",
    "PinnedAnchorStrategy",
    "SemanticSummaryStrategy",
    "SummaryProvider",
    "STRATEGY_REGISTRY",
    "estimate_tokens",
]
