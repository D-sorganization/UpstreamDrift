"""Common launcher utilities for the unified dashboard."""

import os
import sys
from collections.abc import Callable
from typing import Any

from src.shared.python.dashboard.window import UnifiedDashboardWindow
from src.shared.python.engine_core.interfaces import PhysicsEngine
from src.shared.python.logging_pkg.logging_config import (
    configure_gui_logging,
    get_logger,
)
from src.shared.python.ui.qt.utils import get_qapp

logger = get_logger(__name__)


def _default_event_loop_runner(qt_app: Any) -> int:
    """Run the Qt event loop, refusing to block inside a pytest session.

    A headless test must never enter a real Qt event loop: nothing will ever
    post the quit event, so the worker blocks until the suite-level timeout
    kills it and the whole job fails minutes later with a stack that points at
    Qt rather than at the mistake (issue #9183).

    There is no legitimate reason for a test to reach this function. A test
    that exercises :func:`launch_dashboard` must either patch it out or inject
    ``event_loop_runner``. Raising here converts a class of multi-minute CI
    hangs into an immediate, self-describing failure.

    Args:
        qt_app: The ``QApplication`` whose event loop should be run.

    Returns:
        The event loop's exit code.

    Raises:
        RuntimeError: If called while pytest is running a test.
    """
    if "PYTEST_CURRENT_TEST" in os.environ:
        raise RuntimeError(
            "launch_dashboard() reached the blocking Qt event loop during a "
            "pytest run. A headless test must never enter a real Qt event "
            "loop. Inject event_loop_runner=... or patch launch_dashboard on "
            "the module under test. See issue #9183."
        )
    return int(qt_app.exec())


def launch_dashboard(
    engine_class: type[PhysicsEngine],
    title: str,
    model_path: str | None = None,
    engine_args: list | None = None,
    engine_kwargs: dict | None = None,
    event_loop_runner: Callable[[Any], int] | None = None,
) -> None:
    """Launches the Unified Dashboard with the specified physics engine.

    Args:
        engine_class: The class of the physics engine to instantiate.
        title: The title of the dashboard window.
        model_path: Optional path to a model file to load on startup.
        engine_args: Optional positional arguments for the engine constructor.
        engine_kwargs: Optional keyword arguments for the engine constructor.
        event_loop_runner: Optional test hook for running the Qt event loop.
    """
    if engine_class is None:
        raise ValueError("engine_class must be provided")
    configure_gui_logging()

    app = get_qapp()

    args = engine_args or []
    kwargs = engine_kwargs or {}

    try:
        engine = engine_class(*args, **kwargs)
    except (RuntimeError, ValueError, OSError) as e:
        logger.error(f"Failed to initialize engine {engine_class.__name__}: {e}")
        return

    if model_path:
        try:
            logger.info(f"Loading model: {model_path}")
            engine.load_from_path(model_path)
        except (ValueError, RuntimeError, AttributeError) as e:
            logger.error(f"Failed to load model: {e}")
            # Continue with empty engine, but warn

    window = UnifiedDashboardWindow(engine, title=title)

    # Add AI Chat dock widget (connects to FastAPI chat server)
    try:
        from PyQt6.QtCore import Qt

        from src.shared.python.chat.chat_dock_widget import (
            ChatConnectionConfig,
            ChatDockWidget,
        )

        engine_name = (
            getattr(engine_class, "__name__", "engine")
            .lower()
            .replace("physicsengine", "")
        )
        chat_dock = ChatDockWidget(
            connection=ChatConnectionConfig(app_context=engine_name),
            parent=window,
        )
        window.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, chat_dock)
    except (ImportError, TypeError, RuntimeError) as e:
        logger.debug("AI Chat dock not available: %s", e)

    window.show()

    runner = event_loop_runner or _default_event_loop_runner
    sys.exit(runner(app))
