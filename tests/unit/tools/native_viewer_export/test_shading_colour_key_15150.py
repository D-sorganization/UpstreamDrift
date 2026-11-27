"""The clip's colour key names the shading colours (#15150, Z4a).

Review of the Track Z render: the HUD spelled the shading colours as hex
codes. The key panel now names tension and compression in their colours,
after any arrow entries of the overlay feed.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from src.shared.python.body_part_viz.force_colors import ForceColorScale
from src.tools.native_viewer_export import core
from src.tools.native_viewer_export.overlay_feed import OverlayFeed
from src.tools.native_viewer_export.shading import COMPRESSION_KEY, TENSION_KEY

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

SCALE = ForceColorScale(
    enabled=True, tension_color="#0000ff", compression_color="#ff0000"
)


def test_shaded_clip_keys_tension_and_compression() -> None:
    settings = replace(core.ExportSettings(), shading=SCALE)
    assert core._key_entries(None, settings) == (
        (TENSION_KEY, (0, 0, 255)),
        (COMPRESSION_KEY, (255, 0, 0)),
    )


def test_feed_entries_come_first() -> None:
    feed = OverlayFeed(frame_at=lambda i: None, colour_key=(("ZTCF", (0, 114, 178)),))
    settings = replace(core.ExportSettings(), shading=SCALE)
    entries = core._key_entries(feed, settings)
    assert entries[0] == ("ZTCF", (0, 114, 178))
    assert [e[0] for e in entries[1:]] == [TENSION_KEY, COMPRESSION_KEY]


def test_unshaded_clip_has_only_the_feed_entries() -> None:
    off = replace(SCALE, enabled=False)
    assert core._key_entries(None, core.ExportSettings()) == ()
    assert core._key_entries(None, replace(core.ExportSettings(), shading=off)) == ()
