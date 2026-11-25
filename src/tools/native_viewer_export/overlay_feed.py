"""Overlay feed and glyph style for the native viewer export (#15150).

Split out of :mod:`.core` (file-size budget); ``core`` re-exports every name
here so existing imports keep working. Engine-SDK-free.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
import math
from typing import Any

from src.shared.python.force_overlay.conversions import restrict_frame
from src.shared.python.force_overlay.glyphs import ForceGlyphStyle, GlyphSet
from src.shared.python.force_overlay.renderers.meshcat_glyphs import legend_text
from src.shared.python.grip_contact.palm_thumb import PALM_THUMB_LABEL


STANDARD_GRAVITY_M_S2 = 9.80665
BODY_WEIGHT_ARROW_M = 0.5
#: HUD note on states whose section loads are blanked around ball contact
CONTACT_BLACKOUT_NOTE = (
    "Shading withheld at ball contact: the impact impulse is not modelled"
)


def default_glyph_style(body_mass_kg: float | None = None) -> ForceGlyphStyle:
    """Golfer-scale glyph style.

    With ``body_mass_kg`` the force arrows use ``body_weight`` scaling so one
    body weight is ``BODY_WEIGHT_ARROW_M`` (0.5 m) long, with a 3 m ceiling
    (6 BW) so a driver swing is never silently clamped. Torque arcs are
    capped at 0.6 m length (0.30 m radius) to prevent oversized arcs near
    impact (NV-9, #11697). Without body mass the fixed 0.7 m per kN scale is kept.

    Raises ``ValueError`` when ``body_mass_kg`` is not finite and positive.
    """
    if body_mass_kg is None:
        return ForceGlyphStyle(
            force_scale_m_per_n=0.7 / 1000.0,
            torque_scale_m_per_nm=0.5 / 250.0,
            max_length_m=0.9,
            max_torque_length_m=0.6,
            min_length_m=0.04,
            shaft_radius_m=0.015,
            magnitude_floor_nm=8.0,
            magnitude_floor_n=20.0,
        )
    if not math.isfinite(body_mass_kg) or body_mass_kg <= 0.0:
        raise ValueError(
            f"body_mass_kg must be finite and positive, got {body_mass_kg}"
        )
    return ForceGlyphStyle(
        scale_mode="body_weight",
        reference_force_n=body_mass_kg * STANDARD_GRAVITY_M_S2,
        reference_length_m=BODY_WEIGHT_ARROW_M,
        torque_scale_m_per_nm=0.5 / 250.0,
        max_length_m=3.0,
        max_torque_length_m=0.6,
        min_length_m=0.04,
        shaft_radius_m=0.015,
        magnitude_floor_nm=8.0,
        magnitude_floor_n=20.0,
    )


@dataclass
class OverlayFeed:
    """Glyph sets by state index, built lazily from an overlay provider.

    Frames are cached per index: a frame may need an engine solve (the grip
    wrench), and both the glyphs and the tracked camera read it.
    """

    frame_at: Callable[[int], Any]
    style: ForceGlyphStyle = field(default_factory=default_glyph_style)
    build: Callable[[Any, ForceGlyphStyle], GlyphSet] | None = None
    #: grip analyses at given indices (set when the grip overlay is enabled)
    grip_analyses: Callable[[Sequence[int]], list[Any]] | None = None
    #: glyph kinds drawn; the frame (and its axial loads) is never restricted
    show_force: bool = True
    show_torque: bool = True
    #: the allocated palm-on-thumb arrow (``PALM_THUMB_LABEL``), drawn only when
    #: set and independent of ``show_force``/``show_torque``
    show_palm_thumb: bool = False
    #: hand capability/mobility ellipsoids generator (HCM-3, #14004)
    hand_ellipsoids: Callable[[int], tuple[Any, ...]] | None = None
    #: hand capability/mobility HUD text lines generator (HCM-3, #14004)
    hand_hud_lines: Callable[[int], tuple[str, ...]] | None = None
    hand_panel_rows: Callable[[int], tuple[Any, ...]] | None = (
        None  # PanelRow per frame
    )
    #: ``(hand_ellipsoids, hand_hud_lines, hand_panel_rows)`` at the poses of a swing
    ellipsoids_for: Callable[[Any], tuple[Any, Any, Any]] | None = None
    #: HUD note when force/torque vectors are withheld (``kinematic_replay``)
    forces_note: str = ""
    #: extra arrows by source state (tip-to-tail ZTCF/ZVCF hand force, #15685)
    extra_arrows: Callable[[int], tuple[Any, ...]] | None = None
    #: legend text appended when ``extra_arrows`` draws anything
    extra_legend: str = ""
    #: ``(text, RGB)`` naming each extra arrow colour (#15150 Track Z, Z3b)
    colour_key: tuple[tuple[str, tuple[int, int, int]], ...] = ()
    _frames: dict[int, Any] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )

    def _frame(self, index: int) -> Any:
        if index not in self._frames:
            self._frames[index] = self.frame_at(index)
        return self._frames[index]

    def following(self, swing: Any, source_index: Callable[[int], int]) -> OverlayFeed:
        """This feed for the frames of ``swing``, a re-sampling of its source (#15150).

        Frame ``k`` reads the source frame ``source_index(k)`` (forces are not
        interpolated); ellipsoids are evaluated at the pose of ``swing`` state
        ``k``, so they sit on the body that frame shows. Raises ``ValueError``
        when the feed has ellipsoids it cannot evaluate at another swing.
        """
        source, arrows_at = self.frame_at, self.extra_arrows
        feed = replace(
            self,
            frame_at=lambda i: source(source_index(int(i))),
            extra_arrows=(
                None
                if arrows_at is None
                else (lambda i: arrows_at(source_index(int(i))))
            ),
        )
        if self.hand_ellipsoids is None and self.hand_hud_lines is None:
            return feed
        if self.ellipsoids_for is None:
            raise ValueError(
                "the overlay feed has ellipsoids indexed by source state and no "
                "ellipsoids_for to evaluate them at the clip frames"
            )
        glyphs, lines, rows = self.ellipsoids_for(swing)
        return replace(
            feed, hand_ellipsoids=glyphs, hand_hud_lines=lines, hand_panel_rows=rows
        )

    def held_after(self, last: int) -> OverlayFeed:
        """This feed repeating frame ``last`` (glyphs and HUD) at every later index."""

        def clamp(fn: Any) -> Any:
            return None if fn is None else (lambda i: fn(min(int(i), last)))

        return replace(
            self,
            frame_at=clamp(self.frame_at),
            hand_ellipsoids=clamp(self.hand_ellipsoids),
            hand_hud_lines=clamp(self.hand_hud_lines),
            hand_panel_rows=clamp(self.hand_panel_rows),
            extra_arrows=clamp(self.extra_arrows),
        )

    def glyphs_at(self, index: int) -> GlyphSet:
        build = self.build
        if build is None:
            from src.shared.python.force_overlay.glyphs import build_glyphs

            build = build_glyphs
        frame = self._frame(index)
        allocated = tuple(
            w
            for w in frame.wrenches
            if w.label == PALM_THUMB_LABEL
            or (w.label.startswith("external:") and w.label.endswith("_allocated"))
        )
        if allocated:
            frame = replace(
                frame, wrenches=tuple(w for w in frame.wrenches if w not in allocated)
            )
        if not (self.show_force and self.show_torque):
            frame = restrict_frame(
                frame, show_force=self.show_force, show_torque=self.show_torque
            )
        if allocated and self.show_palm_thumb:
            frame = replace(frame, wrenches=(*frame.wrenches, *allocated))
        gs = build(frame, self.style)
        if self.extra_arrows is not None:
            gs = replace(gs, arrows=(*gs.arrows, *self.extra_arrows(index)))
        if self.hand_ellipsoids is not None:
            extra_e = self.hand_ellipsoids(index)
            if extra_e:
                gs = replace(gs, ellipsoids=(*gs.ellipsoids, *extra_e))
        return gs

    def hud_lines_at(self, index: int) -> tuple[str, ...]:
        """HUD capability/mobility text lines at frame ``index`` (HCM-3, #14004),
        then the forces note and, where the section loads are blanked around
        ball contact, :data:`CONTACT_BLACKOUT_NOTE`."""
        lines = () if self.hand_hud_lines is None else self.hand_hud_lines(index)
        notes = (self.forces_note,) if self.forces_note else ()
        axial = self.axial_at(index)
        if axial is not None and all(v is None for v in axial.values_n.values()):
            notes = (*notes, CONTACT_BLACKOUT_NOTE)
        return (*lines, *notes)

    def axial_at(self, index: int) -> Any:
        """``AxialLoadFrame`` of state ``index``, or ``None`` when not computed."""
        return getattr(self._frame(index), "axial_loads", None)

    def draw_key(self, img: Any) -> Any:
        """``img`` with the colour key panel; ``img`` itself when there is none."""
        if not self.colour_key:
            return img
        from src.tools.native_viewer_export.compositor import draw_colour_key

        return draw_colour_key(img, self.colour_key)

    def legend_for(self, glyphs: GlyphSet | None) -> str:
        """The legend row; none on a replay that draws no vectors (#15150)."""
        drawn = glyphs is not None and (glyphs.arrows or glyphs.torque_arcs)
        text = "" if self.forces_note and not drawn else self.legend(glyphs)
        if self.extra_legend and glyphs is not None and self.extra_arrows is not None:
            text = f"{text} | {self.extra_legend}" if text else self.extra_legend
        return text

    @staticmethod
    def legend(glyphs: GlyphSet | None) -> str:
        if glyphs is None:
            return ""
        text = legend_text(glyphs)
        allocated_arrows = [
            a
            for a in glyphs.arrows
            if a.label == PALM_THUMB_LABEL
            or (a.label.startswith("external:") and a.label.endswith("_allocated"))
        ]
        if allocated_arrows:
            if (
                len(allocated_arrows) == 1
                and allocated_arrows[0].label == PALM_THUMB_LABEL
            ):
                text += " | Palm-thumb force: allocated (not measured)"
            else:
                text += " | Hand-hand contact force: allocated (not measured)"
        return text
