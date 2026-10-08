"""Overlay feed and framing for native exports (needs MuJoCo for the source).

One MuJoCo evaluation of the specification export gives every engine's viewer
the same joint-torque, ground-reaction and weight glyphs (see
``force_overlay.bundle_provider``); velocities come from the bundle reference.
"""

from __future__ import annotations

from src.shared.python.force_overlay.bundle_provider import BundleOverlayProvider
from src.tools.native_viewer_export.core import (
    BackendUnavailable,
    OverlayFeed,
    SwingInput,
    default_glyph_style,
)

LOOKAT_HEIGHT_M = 0.9


def build_overlay_feed(
    swing: SwingInput, engine: str
) -> tuple[OverlayFeed, tuple[float, float, float]]:
    """Overlay feed for ``swing`` plus the framing look-at point.

    Raises ``BackendUnavailable`` when MuJoCo is not installed.
    """
    try:
        from src.engines.physics_engines.mujoco.python.overlay_source import (
            MujocoOverlaySource,
        )
    except ImportError as exc:
        raise BackendUnavailable(
            f"mujoco is not available for overlays: {exc}"
        ) from exc
    source = MujocoOverlaySource(swing.bundle.spec_bytes)
    provider = BundleOverlayProvider(
        swing.bundle, source, source, engine=engine, q=swing.q
    )
    names = swing.bundle.coordinate_order
    com = source.center_of_mass_m(dict(zip(names, map(float, swing.q[0]), strict=True)))
    lookat = (float(com[0]), float(com[1]), LOOKAT_HEIGHT_M)
    style = default_glyph_style(body_mass_kg=float(source.total_mass_kg))
    return OverlayFeed(provider.frame_at, style), lookat
