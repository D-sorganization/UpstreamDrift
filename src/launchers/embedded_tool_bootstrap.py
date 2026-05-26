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

    # Discover sibling Tools repository to prioritize real subtabs over stubs
    import sys
    from pathlib import Path
    import os

    tools_repo = None
    env_path = os.environ.get("TOOLS_REPO_PATH")
    if env_path and Path(env_path).is_dir():
        tools_repo = Path(env_path)
    else:
        # Walk up from this file to find the repository root, then check sibling
        repo_root = Path(__file__).resolve().parent.parent.parent.parent
        p = Path(__file__).resolve()
        for _ in range(10):
            p = p.parent
            for candidate in (
                p / "Tools",
                p / "Repositories" / "Tools",
                Path.home() / "Repositories" / "Tools",
            ):
                if candidate.is_dir() and (candidate / "src").is_dir():
                    try:
                        # Skip candidate if it is nested inside our repo (e.g. the vendored copy)
                        # to prioritize a true sibling checkout.
                        try:
                            if candidate.is_relative_to(repo_root):
                                continue
                        except (ValueError, AttributeError):
                            if str(repo_root) in str(candidate.resolve()):
                                continue
                    except Exception:  # noqa: BLE001
                        pass
                    tools_repo = candidate
                    break
            if tools_repo:
                break

    if tools_repo:
        # Path to sibling Tools/src
        tools_src_path = str(tools_repo / "src")
        if tools_src_path in sys.path:
            sys.path.remove(tools_src_path)
        sys.path.insert(0, tools_src_path)

        # Path to sibling Tools/src/shared/python
        tools_shared_py_path = str(tools_repo / "src" / "shared" / "python")
        if tools_shared_py_path in sys.path:
            sys.path.remove(tools_shared_py_path)
        sys.path.insert(0, tools_shared_py_path)

        logger.info(
            "Prioritized sibling Tools repo paths in sys.path: %s and %s",
            tools_src_path,
            tools_shared_py_path,
        )

        # Append vendor paths as a lower-priority fallback to satisfy tests
        vendor_src_path = str(
            Path(__file__).resolve().parent.parent.parent.parent
            / "vendor"
            / "ud-tools"
            / "src"
        )
        if vendor_src_path not in sys.path:
            sys.path.append(vendor_src_path)

        vendor_shared_py_path = str(Path(vendor_src_path) / "shared" / "python")
        if vendor_shared_py_path not in sys.path:
            sys.path.append(vendor_shared_py_path)
    else:
        # Path to vendor/ud-tools/src
        vendor_src_path = str(
            Path(__file__).resolve().parent.parent.parent.parent
            / "vendor"
            / "ud-tools"
            / "src"
        )
        if vendor_src_path not in sys.path:
            sys.path.insert(0, vendor_src_path)

        # Path to vendor/ud-tools/src/shared/python (for 'sidekick' legacy imports)
        vendor_shared_py_path = str(Path(vendor_src_path) / "shared" / "python")
        if vendor_shared_py_path not in sys.path:
            sys.path.insert(0, vendor_shared_py_path)

    # List of tool adapter modules that self-register on import
    # Each module's __init__.py calls register_embeddable_tool()
    adapter_modules = [
        "src.tools.model_explorer._embed_adapter",
        "data_explorer._embed_adapter",  # Moved from src.tools in vendor
        "src.tools.starting_pose_matcher._embed_adapter",
        "src.tools.pose_subscriber_demo._embed_adapter",
        "src.tools.sidekick._embed_adapter",
    ]

    registered = []
    for module_path in adapter_modules:
        try:
            # Import the module - it self-registers at module level
            __import__(module_path)
            # Extract tool_id from module name for tracking
            tool_id = (
                module_path.split(".")[-2]
                if "_embed_adapter" in module_path
                else module_path.split(".")[-1]
            )

            registered.append(tool_id)
            logger.debug(f"Bootstrapped embeddable tool: {tool_id}")
        except ImportError as e:
            # Tools may have optional dependencies (PyQt6, etc.)
            # Log but don't fail - the tool just won't be embeddable
            logger.warning(f"Failed to bootstrap {module_path}: {e}")
        except Exception as e:  # noqa: BLE001
            # Catch any other unexpected errors during registration
            logger.warning(f"Error bootstrapping {module_path}: {e}")

    _registered_tools = registered
    _bootstrap_complete = True

    logger.info(f"Bootstrapped {len(registered)} embeddable tools: {registered}")
    return registered


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
