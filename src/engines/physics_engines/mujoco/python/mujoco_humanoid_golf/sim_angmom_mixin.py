"""Angular-momentum arrow overlay state for the MuJoCo sim widget (#15398)."""

from __future__ import annotations

from typing import Any

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.angmom_glyph_overlay import (
    build_angmom_overlay_glyphs,
)
from src.shared.python.body_part_viz.angular_momentum_glyphs import (
    DEFAULT_SCALE_M_PER_NMS,
    AngularMomentumArrowStyle,
    angular_momentum_legend_line,
    merge_glyph_sets,
)
from src.shared.python.body_part_viz.angular_momentum_settings import (
    load_angular_momentum_state,
    open_angular_momentum_settings,
    save_angular_momentum_state,
)
from src.shared.python.force_overlay.glyphs import GlyphSet

SETTINGS_ENGINE_KEY = "mujoco"


class SimAngMomMixin:
    """Persistence and sampling for segment angular-momentum arrows."""

    show_angular_momentum_arrows: bool
    angmom_arrow_style: AngularMomentumArrowStyle

    def _init_angmom_overlay_state(self: Any) -> None:
        self.show_angular_momentum_arrows = False
        self.angmom_arrow_style = AngularMomentumArrowStyle(
            scale_m_per_nms=DEFAULT_SCALE_M_PER_NMS
        )
        self._restoring_angmom_settings = False

    def set_angular_momentum_overlay(
        self: Any,
        *,
        enabled: bool | None = None,
        style: AngularMomentumArrowStyle | None = None,
    ) -> None:
        changed = False
        if enabled is not None and enabled != self.show_angular_momentum_arrows:
            self.show_angular_momentum_arrows = bool(enabled)
            changed = True
        if style is not None and style != self.angmom_arrow_style:
            if not isinstance(style, AngularMomentumArrowStyle):
                raise TypeError("style must be AngularMomentumArrowStyle")
            self.angmom_arrow_style = style
            changed = True
        if changed:
            self._persist_angmom_overlay_state()
            self._render_once()

    def _persist_angmom_overlay_state(self: Any) -> None:
        if self._restoring_angmom_settings:
            return
        save_angular_momentum_state(
            open_angular_momentum_settings(),
            SETTINGS_ENGINE_KEY,
            self.show_angular_momentum_arrows,
            self.angmom_arrow_style,
        )

    def restore_angmom_overlay_settings(self: Any) -> None:
        enabled, style = load_angular_momentum_state(
            open_angular_momentum_settings(), SETTINGS_ENGINE_KEY
        )
        self._restoring_angmom_settings = True
        try:
            self.show_angular_momentum_arrows = enabled
            self.angmom_arrow_style = style
        finally:
            self._restoring_angmom_settings = False

    def _build_angular_momentum_glyphs(self: Any) -> GlyphSet | None:
        if not self.show_angular_momentum_arrows:
            return None
        if self.model is None or self.data is None:
            return None
        catalog = self.engine.segment_group_catalog()
        if catalog is None:
            return None
        return build_angmom_overlay_glyphs(
            self.model,
            self.data,
            catalog=catalog,
            enabled=True,
            style=self.angmom_arrow_style,
        )

    def _angmom_legend_line(self: Any) -> str:
        if not self.show_angular_momentum_arrows:
            return ""
        return angular_momentum_legend_line(self.angmom_arrow_style)

    def _merge_overlay_glyphs(
        self: Any, force_glyphs: GlyphSet | None, angmom_glyphs: GlyphSet | None
    ) -> GlyphSet | None:
        return merge_glyph_sets(force_glyphs, angmom_glyphs)

    def _build_overlay_glyphs(self: Any) -> GlyphSet | None:
        """Force/torque plus optional angular-momentum arrows for this frame."""
        return self._merge_overlay_glyphs(
            self._build_force_glyphs(),
            self._build_angular_momentum_glyphs(),
        )
