"""Regression: compositor (PIL) tests must not corrupt matplotlib glyph rendering (#15710).

PIL's ``ImageFont.load_default`` initialises FreeType via the system libfreetype2.so.
On Linux CI, matplotlib's Agg renderer links the same shared library, so the global
FreeType state left by PIL can break font lookup in ``draw_glyphs_3d``, producing an
all-white output image.  The fix is an autouse fixture in conftest.py that clears both
PIL's module-level ``_FONTS`` cache and matplotlib's ``_findfont_cached`` LRU cache
between every test.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _run_compositor_pil_ops() -> None:
    """Trigger the PIL FreeType path that contaminates global state."""
    import numpy as _np

    from src.tools.native_viewer_export.compositor import HudInfo, draw_hud

    tile = _np.zeros((120, 240, 3), dtype=_np.uint8)
    hud = HudInfo(time_s=0.5, engine="TestEngine", club="Driver", legend="blue=GRF")
    draw_hud(tile, hud)


def _render_3d_glyphs_rgba() -> np.ndarray:
    """Return an RGBA buffer from a 3-D glyph render (same setup as the canonical test)."""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    from src.shared.python.force_overlay.contracts import (
        ForceTorqueFrame,
        OverlayWrench,
        WrenchKind,
    )
    from src.shared.python.force_overlay.glyphs import ForceGlyphStyle, build_glyphs
    from src.shared.python.force_overlay.renderers.matplotlib_glyphs import draw_glyphs_3d

    frame = ForceTorqueFrame(
        time_s=0.5,
        engine="mujoco",
        wrenches=(
            OverlayWrench(
                label="joint:lead_wrist",
                kind=WrenchKind.JOINT_REACTION,
                body="wrist",
                torque_nm=(0.0, 0.0, 20.0),
                force_n=(500.0, 0.0, 0.0),
                point_m=(0.0, 0.0, 1.0),
                source="sim",
            ),
        ),
    )
    glyphs = build_glyphs(frame, style=ForceGlyphStyle())

    fig = Figure(figsize=(4, 4), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=0, azim=0)
    ax.set_axis_off()
    draw_glyphs_3d(ax, glyphs, halo=False)
    canvas.draw()
    return np.asarray(canvas.buffer_rgba())


def test_compositor_before_glyphs_does_not_produce_blank_render() -> None:
    """PIL FreeType init in compositor must not pollute matplotlib glyph rendering.

    Regression for #15710: running compositor tests (PIL-only) before matplotlib
    glyph tests in the same process caused ``draw_glyphs_3d`` to produce an all-white
    image on Linux CI due to shared libfreetype2.so global-state corruption.
    """
    _run_compositor_pil_ops()
    rgba = _render_3d_glyphs_rgba()
    assert np.any(rgba[..., :3] < 250), (
        "3-D glyph render returned an all-white image after PIL compositor ops. "
        "This is the #15710 order-pollution regression — check the font-cache "
        "cleanup fixture in conftest.py."
    )
