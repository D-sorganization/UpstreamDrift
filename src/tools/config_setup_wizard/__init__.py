"""Canonical-core setup wizard tool.

Importing this package registers its embeddable-tool adapter without importing
PyQt6. The GUI is lazy-loaded only when a host creates the widget.
"""

from __future__ import annotations

import contextlib

with contextlib.suppress(ImportError, ValueError):
    from src.shared.python.launcher_embed import (
        get_embeddable_tool,
        register_embeddable_tool,
    )

    from ._embed_adapter import ConfigSetupWizardAdapter

    _ADAPTER = ConfigSetupWizardAdapter()
    if get_embeddable_tool(_ADAPTER.tool_id) is None:
        register_embeddable_tool(_ADAPTER)

__all__ = ["ConfigSetupWizardAdapter"]
