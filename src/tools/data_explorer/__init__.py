"""Data Explorer — simulation dataset workbench.

Importing this package registers the :class:`_DataExplorerEmbedAdapter`
with the embeddable-tool registry so the launcher can host the tool as
a tab or dock without spawning a separate process. Registration is
guarded so reimports (test reloads) are a quiet no-op.
"""

from __future__ import annotations

import contextlib

with contextlib.suppress(ImportError):
    # PyQt6 is optional fleet-wide; import the adapter behind a guard so
    # headless / Qt-less environments still get a usable package.
    from src.shared.python.launcher_embed import (
        get_embeddable_tool,
        register_embeddable_tool,
    )

    from ._embed_adapter import _DataExplorerEmbedAdapter

    _ADAPTER = _DataExplorerEmbedAdapter()
    if get_embeddable_tool(_ADAPTER.tool_id) is None:
        register_embeddable_tool(_ADAPTER)
