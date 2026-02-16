"""Factory for creating launcher configurations and handlers.

This module consolidates launcher logic from multiple entry points to ensure DRY.
"""
from __future__ import annotations

import importlib
import logging
import sys

logger = logging.getLogger(__name__)

# Mapping of engine names to their python module paths for direct execution
ENGINE_MODULES: dict[str, str] = {
    "mujoco": "src.engines.physics_engines.mujoco.python.humanoid_launcher",
    "drake": "src.engines.physics_engines.drake.python.src.drake_gui_app",
    "pinocchio": "src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui",
    "opensim": "src.engines.physics_engines.opensim.python.opensim_launcher",
    "myosim": "src.engines.physics_engines.myosim.python.myosim_launcher",
    "pendulum": "src.engines.pendulum_models.python.pendulum_launcher",
}


def get_engine_module(engine_name: str) -> str | None:
    """Get the python module path for a given engine."""
    return ENGINE_MODULES.get(engine_name)


def launch_engine_directly(engine: str) -> None:
    """Launch a specific engine GUI directly."""
    module_path = get_engine_module(engine)

    if not module_path:
        logger.error("Direct launch not available for %s. Use web UI instead.", engine)
        sys.exit(1)

    try:
        module = importlib.import_module(module_path)
        if hasattr(module, "main"):
            module.main()
        else:
            logger.error("Module %s has no main() function.", module_path)
            sys.exit(1)
    except ImportError as e:
        logger.error("Failed to launch %s: %s", engine, e)
        logger.info("Try using 'golf-suite' without --engine to use the web UI.")
        sys.exit(1)
