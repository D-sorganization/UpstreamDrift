"""Sidekick agent layer (epic #5967).

Houses the self-aware, audited action surface that lets Sidekick drive its
own subtabs and host applications. See :mod:`sidekick.agent.feature_catalog`
for the self-knowledge index and :mod:`sidekick.agent.action_service` for
the dispatch service.

Both layers are headless-safe — no PyQt6 imports at module scope.
"""

from __future__ import annotations

from .action_service import (
    ActionDescriptor,
    ActionResult,
    SidekickActionHandler,
    SidekickActionService,
)
from .feature_catalog import (
    FeatureEntry,
    FeatureKind,
    build_feature_catalog,
    lookup_feature,
    search_features,
)
from .host_adapter import (
    HostActionPort,
    HostAdapter,
    HostCapability,
    HostInvocationResult,
)
from .subtab_adapter import (
    CalculatorRun,
    StateProfile,
    SubtabActionPort,
    SubtabAdapter,
    WorkspaceSnapshot,
)

__all__ = [
    "ActionDescriptor",
    "ActionResult",
    "CalculatorRun",
    "FeatureEntry",
    "FeatureKind",
    "HostActionPort",
    "HostAdapter",
    "HostCapability",
    "HostInvocationResult",
    "SidekickActionHandler",
    "SidekickActionService",
    "StateProfile",
    "SubtabActionPort",
    "SubtabAdapter",
    "WorkspaceSnapshot",
    "build_feature_catalog",
    "lookup_feature",
    "search_features",
]
