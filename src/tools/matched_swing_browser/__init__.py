"""Matched Swing Results Browser package (MS-80, #10353)."""

from __future__ import annotations

import contextlib

from src.tools.matched_swing_browser.model import (
    MatchedSwingBrowserModel,
    MatchedSwingFilter,
)


def __getattr__(name: str) -> object:
    """Lazy-load PyQt GUI symbols so API imports avoid libEGL (#10358)."""
    if name == "MatchedSwingBrowserWidget":
        from src.tools.matched_swing_browser.gui import MatchedSwingBrowserWidget

        return MatchedSwingBrowserWidget
    if name == "MatchedSwingBrowserWindow":
        from src.tools.matched_swing_browser.gui import MatchedSwingBrowserWindow

        return MatchedSwingBrowserWindow
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
