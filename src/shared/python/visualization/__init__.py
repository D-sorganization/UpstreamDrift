"""Visualization helpers (Phase 3 of the Functional Swing Plane epic, #5504).

Currently exposes the :class:`FspRenderer` which draws a translucent
square plane mesh onto a viewport using the best-fit FSP plane from
:mod:`src.shared.python.biomechanics.fsp_integration`.
"""

from __future__ import annotations

from src.shared.python.visualization.fsp_renderer import (
    FspRenderConfig,
    FspRenderer,
    Viewport,
)
from src.shared.python.visualization.viewport import (
    PROVIDER_METADATA,
    ProviderAvailability,
    ViewportOverlayPayload,
    ViewportProvider,
    ViewportProviderMetadata,
    ViewportProviderStatus,
    ViewportSelection,
    evaluate_viewport_providers,
    select_viewport_provider,
    selected_viewport_decision,
)

__all__ = [
    "FspRenderConfig",
    "FspRenderer",
    "PROVIDER_METADATA",
    "ProviderAvailability",
    "Viewport",
    "ViewportOverlayPayload",
    "ViewportProvider",
    "ViewportProviderMetadata",
    "ViewportProviderStatus",
    "ViewportSelection",
    "evaluate_viewport_providers",
    "select_viewport_provider",
    "selected_viewport_decision",
]
