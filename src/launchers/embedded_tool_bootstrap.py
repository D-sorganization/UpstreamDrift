"""Bootstrap registration of embeddable tools for the launcher.

This module imports and registers all embeddable tools at launcher startup,
ensuring the EMBEDDABLE_TOOL_REGISTRY is populated before any context menus
or embedded host widgets are created.

Part of EPIC #4993 (Subtask 5) - addresses review feedback from #5049.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

# Registry state tracking
_bootstrap_complete = False
_registered_tools: list[str] = []


def bootstrap_embeddable_tools() -> list[str]:
    """Import and register all embeddable tools.

    This function performs lazy imports of tool adapter modules, which
    triggers their self-registration with the EMBEDDABLE_TOOL_REGISTRY.

    Returns:
        List of tool_ids that were registered

    Note:
        This function is idempotent - calling it multiple times is safe.
        Subsequent calls return the list of previously registered tools.
    """
    global _bootstrap_complete, _registered_tools

    if _bootstrap_complete:
        return _registered_tools

    # Ensure Tools repo is in sys.path so embeddable tools can be found.
    # Prioritise sibling Tools repository if checked out, falling back to vendored ud-tools.
    import sys
    from pathlib import Path

    repos_root = Path(__file__).resolve().parent.parent.parent
    sibling_tools = repos_root.parent / "Tools"
    if sibling_tools.is_dir():
        tools_src_path = str(sibling_tools / "src")
        tools_shared_py_path = str(sibling_tools / "src" / "shared" / "python")
        tools_python_src_path = str(sibling_tools / "src" / "python" / "src")
    else:
        # Path to vendor/ud-tools/src
        tools_src_path = str(repos_root / "vendor" / "ud-tools" / "src")
        tools_shared_py_path = str(Path(tools_src_path) / "shared" / "python")
        tools_python_src_path = str(Path(tools_src_path) / "python" / "src")

    # Also register UpstreamDrift's shared python folder
    ud_src_path = str(repos_root / "src")
    ud_shared_py_path = str(repos_root / "src" / "shared" / "python")

    for p in [
        ud_src_path,
        ud_shared_py_path,
        tools_python_src_path,
        tools_shared_py_path,
        tools_src_path,
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # List of tool adapter modules that self-register on import
    # Each module's __init__.py calls register_embeddable_tool()
    adapter_modules = [
        "src.tools.model_explorer._embed_adapter",
        "data_explorer._embed_adapter",  # Moved from src.tools in vendor
        "src.tools.starting_pose_matcher._embed_adapter",
        "src.tools.training_controller._embed_adapter",
        "src.tools.config_setup_wizard._embed_adapter",
        "src.tools.pose_subscriber_demo._embed_adapter",
        "src.tools.canonical_core._embed_adapter",
        "src.tools.sidekick._embed_adapter",
        "src.tools.pose_studio.gui",
        "src.tools.video_analyzer._embed_adapter",
        "src.tools.ball_flight_gui._embed_adapter",
        "src.tools.bunker_shot_gui._embed_adapter",
        "src.tools.putting_green_gui._embed_adapter",
        "src.tools.golf_environment._embed_adapter",
        "src.tools.terrain_engine._embed_adapter",
        "src.tools.golf_simulation_suite._embed_adapter",
        "src.tools.simulation_backends_launcher._embed_adapter",
        "engines.Simscape_Multibody_Models.3D_Golf_Model.python.src.apps._embed_adapter",
    ]

    from src.shared.python.launcher_embed import EMBEDDABLE_TOOL_REGISTRY

    registered = []
    for module_path in adapter_modules:
        # Diff the registry around the import so we record the tool ids
        # the adapter actually registered (an adapter may register zero,
        # one, or several tools; ids need not match the module name).
        before = set(EMBEDDABLE_TOOL_REGISTRY)
        try:
            # Import the module - it self-registers at module level
            __import__(module_path)
        except ImportError as e:
            # Tools may have optional dependencies (PyQt6, etc.)
            # Log but don't fail - the tool just won't be embeddable
            logger.warning(f"Failed to bootstrap {module_path}: {e}")
            continue
        except Exception as e:  # noqa: BLE001
            # Catch any other unexpected errors during registration
            logger.warning(f"Error bootstrapping {module_path}: {e}")
            continue
        new_ids = sorted(set(EMBEDDABLE_TOOL_REGISTRY) - before)
        if new_ids:
            registered.extend(new_ids)
            logger.debug(f"Bootstrapped embeddable tools: {new_ids}")
        else:
            logger.debug(
                "Adapter module %s imported but registered no new tools "
                "(already imported, or registration is conditional)",
                module_path,
            )

    _registered_tools = registered
    _bootstrap_complete = True

    logger.info(f"Bootstrapped {len(registered)} embeddable tools: {registered}")
    _warn_on_manifest_gaps()
    return registered


def missing_embeddable_manifest_tools(manifest: object | None = None) -> list[str]:
    """Return tool-like manifest tile ids with no registered embeddable tool.

    A tile in the launcher manifest whose category is tool-like is
    expected to be openable inside the launcher; if no embeddable tool
    is registered under its id, clicking the tile can only fall back to
    a subprocess launch (or fail). Surfacing the delta makes "I added a
    tile but forgot the adapter" loud instead of a silent dead tile.

    Args:
        manifest: Optional pre-loaded ``LauncherManifest`` (mainly for
            tests). Loaded from disk when omitted.

    Returns:
        Sorted list of tile ids that are tool-like but unregistered.
    """
    from src.shared.python.launcher_embed import EMBEDDABLE_TOOL_REGISTRY

    if manifest is None:
        from src.config.launcher_manifest_loader import LauncherManifest

        manifest = LauncherManifest.load()
    return sorted(
        tile.id
        for tile in manifest.tiles  # type: ignore[attr-defined]
        if getattr(tile, "is_tool", False) and tile.id not in EMBEDDABLE_TOOL_REGISTRY
    )


def _warn_on_manifest_gaps() -> None:
    """Log a warning for manifest tool tiles without embeddable adapters."""
    try:
        missing = missing_embeddable_manifest_tools()
    except (FileNotFoundError, ValueError):
        logger.exception("Could not validate launcher manifest coverage")
        return
    if missing:
        logger.warning(
            "Manifest tool tiles without embeddable adapters (these tiles "
            "fall back to subprocess launch or fail): %s",
            missing,
        )


def get_bootstrapped_tools() -> list[str]:
    """Return list of tool_ids that were bootstrapped.

    Returns:
        List of registered tool_ids, or empty list if bootstrap not yet run
    """
    return _registered_tools.copy()


def reset_bootstrap_state() -> None:
    """Reset bootstrap state (for testing only).

    Warning:
        This function is intended for test fixtures only. Calling this
        during normal operation will break embedded tool functionality.
    """
    global _bootstrap_complete, _registered_tools
    _bootstrap_complete = False
    _registered_tools = []
