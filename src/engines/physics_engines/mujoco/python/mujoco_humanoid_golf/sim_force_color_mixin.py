"""Segment tension/compression colouring state for the MuJoCo sim widget (#13146).

The widget's ``axial_color_scale`` is the single source of truth. A
:class:`ForceColorHub` mirrors it so the settings dialog, the inline controls in
the Visualization tab and persistence stay in sync without signal loops: every
path funnels through :meth:`set_axial_color_scale`, which acts only on a change.
"""

from __future__ import annotations

from typing import Any

from src.shared.python.body_part_viz.force_color_controls import ForceColorHub
from src.shared.python.body_part_viz.force_color_settings import (
    load_force_color_state,
    open_force_color_settings,
    save_force_color_state,
)
from src.shared.python.body_part_viz.force_colors import ForceColorScale

SETTINGS_ENGINE_KEY = "mujoco"


class SimForceColorMixin:
    """Hub, persistence and dialog access for ``axial_color_scale``."""

    axial_color_scale: ForceColorScale

    def _init_force_color_hub(self: Any) -> None:
        self._force_color_hub = ForceColorHub(self.axial_color_scale, parent=self)
        self._force_color_action: Any = None
        self._restoring_force_settings = False

    @property
    def force_color_hub(self: Any) -> ForceColorHub:
        """Shared state object for inline controls (read/write via ``set_scale``)."""
        return self._force_color_hub

    def open_force_color_dialog(self: Any) -> None:
        """Show the full Segment Force Colors dialog."""
        if self._force_color_action is not None:
            self._force_color_action.trigger()

    def set_axial_color_scale(self: Any, scale: Any) -> None:
        """Apply the shared force-color settings and repaint the current frame."""
        if not isinstance(scale, ForceColorScale):
            raise TypeError("scale must be ForceColorScale")
        changed = scale != self.axial_color_scale
        self.axial_color_scale = scale
        self._force_color_hub.set_scale(scale)
        if changed:
            self._persist_force_overlay_state()
            self._render_once()

    def _persist_force_overlay_state(self: Any) -> None:
        if self._restoring_force_settings:
            return
        save_force_color_state(
            open_force_color_settings(),
            SETTINGS_ENGINE_KEY,
            self.axial_color_scale,
            {
                "show_torque_vectors": bool(self.show_torque_vectors),
                "show_force_vectors": bool(self.show_force_vectors),
            },
        )

    def restore_force_overlay_settings(self: Any) -> None:
        """Load the stored scale and vector toggles (defaults if invalid)."""
        scale, toggles = load_force_color_state(
            open_force_color_settings(), SETTINGS_ENGINE_KEY
        )
        self._restoring_force_settings = True
        try:
            self.set_axial_color_scale(scale)
            self.show_torque_vectors = toggles.get("show_torque_vectors", False)
            self.show_force_vectors = toggles.get("show_force_vectors", False)
        finally:
            self._restoring_force_settings = False
