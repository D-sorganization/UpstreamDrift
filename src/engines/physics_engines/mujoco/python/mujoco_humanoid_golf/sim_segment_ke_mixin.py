"""Stored kinetic-energy shading state for the MuJoCo sim widget (#15396)."""

from __future__ import annotations

from typing import Any

from PyQt6.QtCore import QObject

from src.shared.python.body_part_viz.force_color_settings import (
    open_force_color_settings,
)
from src.shared.python.body_part_viz.scalar_shading import ScalarColorScale
from src.shared.python.body_part_viz.segment_ke_settings import (
    load_segment_ke_state,
    save_segment_ke_state,
)
from src.shared.python.segment_kinetic_energy.shading import (
    KE_COMPONENTS,
    component_values_j,
    default_stored_ke_color_scale,
    format_stored_ke_legend,
)

SETTINGS_ENGINE_KEY = "mujoco"


class SegmentKeShadingHub(QObject):
    """Shared validated state for stored-KE inline controls."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._scale = default_stored_ke_color_scale()
        self._mode = "total"

    @property
    def scale(self) -> ScalarColorScale:
        return self._scale

    @property
    def mode(self) -> str:
        return self._mode

    def set_scale(self, scale: ScalarColorScale) -> None:
        if not isinstance(scale, ScalarColorScale):
            raise TypeError("scale must be ScalarColorScale")
        self._scale = scale

    def set_mode(self, mode: str) -> None:
        if mode not in KE_COMPONENTS:
            raise ValueError(f"mode must be one of {KE_COMPONENTS}")
        self._mode = mode


class SimSegmentKeMixin:
    """Persistence and legend for stored kinetic-energy surface shading."""

    stored_ke_color_scale: ScalarColorScale
    stored_ke_component_mode: str

    def _init_segment_ke_shading(self: Any) -> None:
        self.stored_ke_color_scale = default_stored_ke_color_scale()
        self.stored_ke_component_mode = "total"
        self._segment_ke_hub = SegmentKeShadingHub(parent=self)
        self._restoring_segment_ke_settings = False
        self.ke_legend_text = ""

    @property
    def segment_ke_hub(self: Any) -> SegmentKeShadingHub:
        return self._segment_ke_hub

    def set_stored_ke_color_scale(self: Any, scale: ScalarColorScale) -> None:
        if not isinstance(scale, ScalarColorScale):
            raise TypeError("scale must be ScalarColorScale")
        changed = scale != self.stored_ke_color_scale
        self.stored_ke_color_scale = scale
        self._segment_ke_hub.set_scale(scale)
        self._update_ke_legend()
        if changed:
            self._persist_segment_ke_state()
            self._render_once()

    def set_stored_ke_component_mode(self: Any, mode: str) -> None:
        if mode not in KE_COMPONENTS:
            raise ValueError(f"mode must be one of {KE_COMPONENTS}")
        if mode == self.stored_ke_component_mode:
            return
        self.stored_ke_component_mode = mode
        self._segment_ke_hub.set_mode(mode)
        self._update_ke_legend()
        self._persist_segment_ke_state()
        self._render_once()

    def _current_stored_ke_values(self: Any) -> dict[str, float | None] | None:
        if not self.stored_ke_color_scale.enabled:
            return None
        if self.engine is None or self.data is None:
            return None
        frame = self.engine.get_segment_kinetic_energy_frame()
        if frame is None:
            return None
        return component_values_j(frame, self.stored_ke_component_mode)

    def _update_ke_legend(self: Any) -> None:
        text = format_stored_ke_legend(
            self.stored_ke_color_scale, mode=self.stored_ke_component_mode
        )
        self.ke_legend_text = text
        signal = getattr(self, "ke_legend_changed", None)
        if signal is not None:
            signal.emit(text)

    def _persist_segment_ke_state(self: Any) -> None:
        if self._restoring_segment_ke_settings:
            return
        save_segment_ke_state(
            open_force_color_settings(),
            SETTINGS_ENGINE_KEY,
            self.stored_ke_color_scale,
            mode=self.stored_ke_component_mode,
            toggles={
                "show_ke_proxy_globes": bool(
                    getattr(self, "show_segment_ke_globes", False)
                )
            },
        )

    def restore_segment_ke_settings(self: Any) -> None:
        scale, mode, toggles = load_segment_ke_state(
            open_force_color_settings(), SETTINGS_ENGINE_KEY
        )
        self._restoring_segment_ke_settings = True
        try:
            self.stored_ke_color_scale = scale
            self.stored_ke_component_mode = mode
            self._segment_ke_hub.set_scale(scale)
            self._segment_ke_hub.set_mode(mode)
            if "show_ke_proxy_globes" in toggles:
                self.show_segment_ke_globes = toggles["show_ke_proxy_globes"]
            self._update_ke_legend()
        finally:
            self._restoring_segment_ke_settings = False
