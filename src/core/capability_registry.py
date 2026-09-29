"""Thin re-export shim for the capability/feature registry.

The canonical implementation lives in
:mod:`src.shared.python.feature_registry`.  This module provides the
``src.core.capability_registry`` import path referenced in the Phase 3
install-prompt UX spec (issue #5768) without duplicating any logic.

Usage::

    from src.core.capability_registry import get_registry, refresh

    registry = get_registry()
    registry.refresh()
"""

from __future__ import annotations

from src.shared.python.feature_registry import (  # noqa: F401
    CapabilityRegistry,
    FeatureReport,
    InstallResult,
    get_registry,
    install_feature,
    refresh,
)

__all__ = [
    "CapabilityRegistry",
    "FeatureReport",
    "InstallResult",
    "get_registry",
    "install_feature",
    "refresh",
]
