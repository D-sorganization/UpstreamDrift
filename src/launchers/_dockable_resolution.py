"""Embedded-UI resolution helpers shared by the launcher model handlers.

Split out of ``launcher_model_handlers`` (file-size budget): the ADR-0013
registry lookup and the deprecation warning for tiles still embedded via the
legacy import-and-probe protocol.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


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


def _probe_script_dockable_ui(
    model: Any,
    repo_path: Path,
    script_path: Path,
    *,
    module_name: str,
    log_context: str,
) -> Any | None:
    """Import ``script_path`` and probe its module-level ``get_dockable_ui``.

    The deprecated import-and-probe embedding path shared by
    ``ScriptHandler`` and ``SpecialAppHandler``: temporarily extend
    ``sys.path`` with the model's python paths, import the script as a
    module, and call its ``get_dockable_ui`` if present. ``sys.path`` is
    restored unless a widget was successfully produced (the loaded module
    may need those paths for lazy imports while embedded).
    """
    import importlib.util
    import sys

    from src.launchers.launcher_model_sources import get_model_python_paths

    original_sys_path = sys.path.copy()
    success = False
    try:
        if str(repo_path) not in sys.path:
            sys.path.insert(0, str(repo_path))
        for p in get_model_python_paths(model, repo_path):
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))

        spec = importlib.util.spec_from_file_location(module_name, str(script_path))
        if spec and spec.loader:
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if hasattr(module, "get_dockable_ui"):
                ui = module.get_dockable_ui()
                if ui is not None:
                    success = True
                    _warn_legacy_embed_fallback(
                        model, f"module-level get_dockable_ui in {script_path.name}"
                    )
                    return ui
    except Exception as e:  # noqa: BLE001
        logger.debug("No dockable UI found in %s: %s", log_context, e)
    finally:
        if not success:
            sys.path = original_sys_path
    return None
