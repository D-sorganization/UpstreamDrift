"""Common launcher utilities for the unified dashboard."""

import sys

from PyQt6.QtWidgets import QApplication
from src.shared.python.gui_utils import get_qapp


from src.shared.python.dashboard.window import UnifiedDashboardWindow
from src.shared.python.interfaces import PhysicsEngine
from src.shared.python.logging_config import configure_gui_logging, get_logger

logger = get_logger(__name__)


def launch_dashboard(
    engine_class: type[PhysicsEngine],
    title: str,
    model_path: str | None = None,
    engine_args: list | None = None,
    engine_kwargs: dict | None = None,
) -> None:
    """Launches the Unified Dashboard with the specified physics engine.

    Args:
        engine_class: The class of the physics engine to instantiate.
        title: The title of the dashboard window.
        model_path: Optional path to a model file to load on startup.
        engine_args: Optional positional arguments for the engine constructor.
        engine_kwargs: Optional keyword arguments for the engine constructor.
    """
    configure_gui_logging()

    app = get_qapp()

    args = engine_args or []
    kwargs = engine_kwargs or {}

    try:
        engine = engine_class(*args, **kwargs)
    except Exception as e:
        logger.error(f"Failed to initialize engine {engine_class.__name__}: {e}")
        return

    if model_path:
        try:
            logger.info(f"Loading model: {model_path}")
            engine.load_from_path(model_path)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            # Continue with empty engine, but warn

    window = UnifiedDashboardWindow(engine, title=title)
    window.show()

    sys.exit(app.exec())
