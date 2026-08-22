"""Window/layout persistence helpers for the UpstreamDrift launcher.

Split out of ``upstream_drift_launcher`` (file-size budget): building the
window-state payload for :meth:`LayoutManager.save_layout` and re-applying a
loaded layout (geometry clamping, option checkboxes, selection) to the
launcher window. Behavior is identical to the original in-class code.
"""

from __future__ import annotations

from typing import Any

from PyQt6.QtWidgets import QApplication

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


def save_layout_state(launcher: Any) -> None:
    """Persist ``launcher``'s current model layout and window state."""
    window_state = {
        "selected_model": launcher.selected_model,
        "geometry": {
            "x": launcher.x(),
            "y": launcher.y(),
            "width": launcher.width(),
            "height": launcher.height(),
        },
        "options": {
            "live_visualization": launcher.chk_live.isChecked(),
            "gpu_acceleration": launcher.chk_gpu.isChecked(),
            "docker_mode": launcher.chk_docker.isChecked(),
            "wsl_mode": launcher.chk_wsl.isChecked(),
        },
    }
    launcher.layout_manager.save_layout(window_state)


def load_layout_state(launcher: Any) -> None:
    """Load the saved model layout and apply it to ``launcher``."""
    layout_data = launcher.layout_manager.load_layout()

    if layout_data is None:
        launcher._rebuild_grid()
        return

    # Restore view mode checkmark
    act = launcher._viewmode_actions.get(launcher.layout_manager.current_view_mode)
    if act:
        act.setChecked(True)

    launcher.model_order = launcher.layout_manager.model_order
    launcher._sync_model_cards()

    # Restore window geometry, clamped to screen bounds
    geo = layout_data.get("window_geometry", {})
    if geo:
        x = geo.get("x", 100)
        y = geo.get("y", 100)
        w = geo.get("width", 1280)
        h = geo.get("height", 800)
        # Clamp to screen size
        screen = QApplication.primaryScreen()
        if screen:
            avail = screen.availableGeometry()
            w = min(w, avail.width() - 40)
            h = min(h, avail.height() - 40)
            x = max(avail.x(), min(x, avail.x() + avail.width() - w))
            y = max(avail.y() + 30, min(y, avail.y() + avail.height() - h))
        elif y < 30:
            y = 50
        launcher.setGeometry(x, y, w, h)
    else:
        launcher._center_window()

    # Restore options
    options = layout_data.get("options", {})
    launcher.chk_live.setChecked(options.get("live_visualization", True))
    launcher.chk_gpu.setChecked(options.get("gpu_acceleration", False))
    saved_docker = options.get("docker_mode", None)
    if saved_docker is None:
        saved_docker = launcher.orchestrator.docker_available
    launcher.chk_docker.setChecked(bool(saved_docker))
    launcher.chk_wsl.setChecked(bool(options.get("wsl_mode", False)))

    # Restore selected model
    saved_selection = layout_data.get("selected_model")
    if saved_selection and saved_selection in launcher.model_cards:
        launcher.select_model(saved_selection)

    launcher._rebuild_grid()
    logger.info("Layout loaded successfully")
