"""Embedded-UI resolution helpers shared by the launcher model handlers.

Split out of ``launcher_model_handlers`` (file-size budget): the ADR-0013
registry lookup and the deprecation warning for tiles still embedded via the
legacy import-and-probe protocol.
"""

from __future__ import annotations

import warnings
from typing import Any


def _registry_dockable_ui(model: Any) -> Any | None:
    """Resolve ``model``'s embedded UI through the ADR-0013 registry.

    The ``EMBEDDABLE_TOOL_REGISTRY`` is THE embedding contract (issue
    #8857): every handler consults it before falling back to the legacy
    import-and-probe protocol. Returns ``None`` when the tile id is not
    registered (or has no usable id).
    """
    tool_id = getattr(model, "id", "")
    if not (isinstance(tool_id, str) and tool_id):
        return None
    from src.shared.python.launcher_embed.registry import get_embeddable_tool

    tool = get_embeddable_tool(tool_id)
    if tool is None:
        return None
    return tool.create_main_widget(None)


def _warn_legacy_embed_fallback(model: Any, mechanism: str) -> None:
    """Emit a DeprecationWarning for a tile embedded via the legacy path.

    The legacy protocol (module-level ``get_dockable_ui`` probing and
    ``embed_adapter`` "mod::func" strings) is deprecated in favor of the
    ADR-0013 ``EmbeddableTool`` registry. The warning names the tile so
    the remaining users are enumerable (ratchet test in
    ``tests/launchers/test_embed_contract_convergence.py``).
    """
    tile_id = getattr(model, "id", None) or "<unknown>"
    warnings.warn(
        f"Tile {tile_id!r} resolved its embedded UI via the deprecated "
        f"legacy fallback ({mechanism}). Register an EmbeddableTool "
        "adapter per ADR-0013 and add it to embedded_tool_bootstrap "
        "instead; the legacy path will be removed.",
        DeprecationWarning,
        stacklevel=3,
    )
