"""Sidekick — embeddable wrapper around the AI assistant chat panel.

Sidekick is the launcher-facing surface of the existing
:class:`~src.shared.python.ai.gui.assistant_panel.AIAssistantPanel`.
Wrapping it as an :class:`~src.shared.python.launcher_embed.EmbeddableTool`
lets users open the chat in a tab or dock through the standard
right-click "Launch in Tab" / "Launch in Dock" affordance instead of
relying solely on the hard-wired splitter slot in the launcher UI.

Importing this package registers the
:class:`_SidekickEmbedAdapter` with the embeddable-tool registry.
Registration is guarded against double-import (test reloads) and wrapped
in ``contextlib.suppress(ImportError)`` so headless contexts where
PyQt6 is unavailable still get a usable package. See issue #5460.
"""

from __future__ import annotations

import contextlib

with contextlib.suppress(ImportError):
    from src.shared.python.launcher_embed import (
        get_embeddable_tool,
        register_embeddable_tool,
    )

    from ._embed_adapter import _SidekickEmbedAdapter

    _ADAPTER = _SidekickEmbedAdapter()
    if get_embeddable_tool(_ADAPTER.tool_id) is None:
        register_embeddable_tool(_ADAPTER)


__all__: list[str] = []
