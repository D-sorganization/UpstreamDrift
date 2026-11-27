"""Segment net-power overlay state for the MuJoCo sim widget (#15399)."""

from __future__ import annotations

from typing import Any

from src.shared.python.body_part_viz.net_power_settings import (
    load_net_power_state,
    open_net_power_settings,
    save_net_power_state,
)
from src.shared.python.body_part_viz.scalar_shading import ScalarColorScale
from src.shared.python.body_part_viz.segment_net_power import (
    default_net_power_color_scale,
    format_net_power_legend,
    net_power_frame_from_interface,
)

SETTINGS_ENGINE_KEY = "mujoco"


class SimNetPowerMixin:
    """Persisted diverging dE/dt shading and interface-power glyphs."""

    net_power_color_scale: ScalarColorScale
    show_interface_power_glyphs: bool

    def _init_net_power_state(self: Any) -> None:
        self.net_power_color_scale = default_net_power_color_scale()
        self.show_interface_power_glyphs = False
        self._restoring_net_power_settings = False
        self._net_power_legend_text = ""
        self._cached_net_power_presentation = None

    def set_net_power_color_scale(self: Any, scale: ScalarColorScale) -> None:
        if not isinstance(scale, ScalarColorScale):
            raise TypeError("scale must be ScalarColorScale")
        changed = scale != self.net_power_color_scale
        self.net_power_color_scale = scale
        if changed:
            self._persist_net_power_overlay_state()
            self._render_once()

    def set_interface_power_glyphs(self: Any, enabled: bool) -> None:
        flag = bool(enabled)
        if flag == self.show_interface_power_glyphs:
            return
        self.show_interface_power_glyphs = flag
        self._persist_net_power_overlay_state()
        self._render_once()

    def _persist_net_power_overlay_state(self: Any) -> None:
        if self._restoring_net_power_settings:
            return
        save_net_power_state(
            open_net_power_settings(),
            SETTINGS_ENGINE_KEY,
            self.net_power_color_scale,
            {"show_interface_power_glyphs": self.show_interface_power_glyphs},
        )

    def restore_net_power_overlay_settings(self: Any) -> None:
        scale, toggles = load_net_power_state(
            open_net_power_settings(), SETTINGS_ENGINE_KEY
        )
        self._restoring_net_power_settings = True
        try:
            self.net_power_color_scale = scale
            self.show_interface_power_glyphs = toggles.get(
                "show_interface_power_glyphs", False
            )
        finally:
            self._restoring_net_power_settings = False

    def _set_net_power_legend(self: Any, text: str) -> None:
        if text == getattr(self, "_net_power_legend_text", ""):
            return
        self._net_power_legend_text = text
        signal = getattr(self, "net_power_legend_changed", None)
        if signal is not None:
            signal.emit(text)

    def _refresh_net_power_legend(
        self: Any, presentation: Any | None = None, *, labels: tuple[str, ...] = ()
    ) -> None:
        if not self.net_power_color_scale.enabled:
            self._set_net_power_legend("")
            return
        if presentation is None:
            try:
                iface = self.engine.get_interface_power_frame()
            except (TypeError, ValueError, RuntimeError):
                self._set_net_power_legend("")
                return
            if iface is None:
                self._set_net_power_legend("")
                return
            presentation = net_power_frame_from_interface(
                iface, self.net_power_color_scale
            )
        self._set_net_power_legend(
            format_net_power_legend(presentation, interface_labels=labels)
        )
