"""Bootstrap registration of embeddable tools for the launcher.

This module imports and registers all embeddable tools at launcher startup,
ensuring the EMBEDDABLE_TOOL_REGISTRY is populated before any context menus
or embedded host widgets are created.

Part of EPIC #4993 (Subtask 5) - addresses review feedback from #5049.
"""

from __future__ import annotations

import importlib
import importlib.metadata
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING

from src.launchers.tools_repo_path import resolve_tools_source_root
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:
    pass

logger = get_logger(__name__)

EMBEDDABLE_TOOL_ENTRY_POINT_GROUP = "upstream_drift.embeddable_tools"
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Tool adapter modules that self-register on import. Kept as a fallback for
# editable installs where package metadata can lag behind the source checkout.
FALLBACK_ADAPTER_MODULES = (
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
    "src.tools.launch_monitor_analytics._embed_adapter",
    "src.tools.swing_flight_pipeline._embed_adapter",
    # Physics-engine adapters (issue #8857): these self-register the
    # mujoco_unified / drake_golf / pinocchio_golf / opensim_golf /
    # myosim_suite tool ids used by the launcher tiles.
    "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._embed_adapter",
    "src.engines.physics_engines.drake.python.src._embed_adapter",
    "src.engines.physics_engines.pinocchio.python.pinocchio_golf._embed_adapter",
    "src.engines.physics_engines.opensim.python._embed_adapter",
    "src.engines.physics_engines.myosuite.python._embed_adapter",
    # Simscape C3D viewer: the real adapter lives under a path segment
    # ("3D_Golf_Model") that is not a valid dotted-import identifier, so
    # it is reached through an importable shim (issue #8856).
    "src.launchers.adapters.simscape_embed",
    # Swing Objective Lab: the widget ships in the Tools pendulum_simulator
    # provider, so the adapter lives here rather than making the provider
    # import UpstreamDrift (Tools epic #4766).
    "src.launchers.adapters.swing_objective_lab_embed",
)

# Registry state tracking
_bootstrap_complete = False
_registered_tools: list[str] = []
_bootstrap_failures: list[tuple[str, str]] = []


def _iter_entry_point_adapter_modules() -> list[str]:
    """Return adapter module paths declared by package entry points."""
    return [
        entry_point.value
        for entry_point in importlib.metadata.entry_points(
            group=EMBEDDABLE_TOOL_ENTRY_POINT_GROUP
        )
        if entry_point.value
    ]


def _registry_declared_adapter_modules() -> list[str]:
    """Adapter modules declared as ``embed_adapter`` in models.yaml (#9412).

    The single tile registry is the source of which tools are embeddable;
    the static fallback list below only covers adapters that live outside
    the registry (e.g. Simscape). Unreadable registries degrade to nothing.
    """
    try:
        from src.config.registry_adapters import embed_adapter_modules

        return embed_adapter_modules()
    except Exception as exc:  # noqa: BLE001 - bootstrap must not abort
        logger.warning("Could not read embed adapters from the tile registry: %s", exc)
        return []


def _adapter_modules_for_bootstrap() -> list[str]:
    """Return entry-point adapters plus fallback adapters without duplicates."""
    adapter_modules: list[str] = []
    seen: set[str] = set()
    for module_path in [
        *_iter_entry_point_adapter_modules(),
        *_registry_declared_adapter_modules(),
        *FALLBACK_ADAPTER_MODULES,
    ]:
        if module_path in seen:
            continue
        adapter_modules.append(module_path)
        seen.add(module_path)
    return adapter_modules


def _bootstrap_python_paths(
    repo_root: Path | None = None,
    env_value: str | None = None,
) -> list[str]:
    """Return sys.path entries needed for launcher and Tools embeddables."""
    root = repo_root or REPO_ROOT
    tools_src = resolve_tools_source_root(root, env_value)
    return [
        str(root / "src"),
        str(root / "src" / "shared" / "python"),
        str(tools_src),
        str(tools_src / "shared" / "python"),
        str(tools_src / "python" / "src"),
    ]


def _prepend_python_paths(paths: list[str]) -> None:
    """Move each path to the front once, preserving the supplied order."""
    for path in paths:
        while path in sys.path:
            sys.path.remove(path)
    for path in reversed(paths):
        sys.path.insert(0, path)


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

    _prepend_python_paths(
        _bootstrap_python_paths(env_value=os.environ.get("TOOLS_REPO_PATH"))
    )

    from src.shared.python.launcher_embed import EMBEDDABLE_TOOL_REGISTRY

    registered = []
    for module_path in _adapter_modules_for_bootstrap():
        # Diff the registry around the import so we record the tool ids
        # the adapter actually registered (an adapter may register zero,
        # one, or several tools; ids need not match the module name).
        before = set(EMBEDDABLE_TOOL_REGISTRY)
        try:
            # Import the module - it self-registers at module level
            __import__(module_path)
        except ImportError as e:
            # Tools may have optional dependencies (PyQt6, etc.)
            # Log but don't fail - the tool just won't be embeddable.
            # Record the failure so callers (health panels, tests) can
            # surface it instead of it vanishing into the log (#8856).
            logger.warning(
                "Failed to bootstrap embeddable-tool adapter %r: %s", module_path, e
            )
            _bootstrap_failures.append((module_path, repr(e)))
            continue
        except Exception as e:  # noqa: BLE001
            # Catch any other unexpected errors during registration
            logger.warning(
                "Error bootstrapping embeddable-tool adapter %r: %s", module_path, e
            )
            _bootstrap_failures.append((module_path, repr(e)))
            continue
        new_ids = sorted(set(EMBEDDABLE_TOOL_REGISTRY) - before)
        reloadable = (
            module_path.rsplit(".", 1)[-1] == "_embed_adapter"
            or module_path == "src.launchers.adapters.simscape_embed"
        )
        if not new_ids and reloadable and module_path in sys.modules:
            # Adapters self-register at module import, so a cached import is
            # a no-op. If the registry was cleared after the first import
            # (test isolation resets it between sessions), the tool would
            # otherwise stay unregistered forever; re-executing the module
            # restores its registrations idempotently. Only thin adapter
            # shims are re-executed: reloading a full tool GUI module (e.g.
            # pose_studio.gui) would swap class identities under live
            # objects.
            try:
                importlib.reload(sys.modules[module_path])
            except Exception as e:  # noqa: BLE001
                logger.warning(
                    "Failed to re-register embeddable-tool adapter %r: %s",
                    module_path,
                    e,
                )
                _bootstrap_failures.append((module_path, repr(e)))
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


def get_bootstrap_failures() -> list[tuple[str, str]]:
    """Return ``(module_path, error_repr)`` for each adapter that failed.

    Empty until :func:`bootstrap_embeddable_tools` runs. Lets callers
    (integrations-health panels, tests) surface bootstrap failures
    instead of relying on someone reading the warning log (#8856).
    """
    return _bootstrap_failures.copy()


def reset_bootstrap_state() -> None:
    """Reset bootstrap state (for testing only).

    Warning:
        This function is intended for test fixtures only. Calling this
        during normal operation will break embedded tool functionality.
    """
    global _bootstrap_complete, _registered_tools, _bootstrap_failures
    _bootstrap_complete = False
    _registered_tools = []
    _bootstrap_failures = []
