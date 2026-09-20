"""Matched Swing Results Browser package (MS-80, #10353)."""

from __future__ import annotations

import contextlib

from src.tools.matched_swing_browser.gui import (
    MatchedSwingBrowserWidget,
    MatchedSwingBrowserWindow,
)
from src.tools.matched_swing_browser.model import (
    MatchedSwingBrowserModel,
    MatchedSwingFilter,
)

with contextlib.suppress(ImportError):
    from src.shared.python.launcher_embed import (
        get_embeddable_tool,
        register_embeddable_tool,
    )
    from src.tools.matched_swing_browser._embed_adapter import (
        _MatchedSwingBrowserEmbedAdapter,
    )

    if get_embeddable_tool(_MatchedSwingBrowserEmbedAdapter.tool_id) is None:
        register_embeddable_tool(_MatchedSwingBrowserEmbedAdapter())

__all__ = [
    "MatchedSwingBrowserModel",
    "MatchedSwingBrowserWidget",
    "MatchedSwingBrowserWindow",
    "MatchedSwingFilter",
]
