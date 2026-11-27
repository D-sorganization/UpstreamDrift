"""Pinned headless verification for native flight overlays (#15429)."""

from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from src.tools.native_viewer_export import flight_inset as fi
from src.tools.native_viewer_export import kinematic_replay as kr
from src.tools.native_viewer_export.core import SwingInput
from src.tools.native_viewer_export.overlay import OverlayOptions
from tests.unit.tools.native_viewer_export import test_flight_inset_15150 as inset
from tests.unit.tools.native_viewer_export import test_flight_tracer as base
from tests.unit.tools.native_viewer_export import test_landing_track_15150 as track

pytest_plugins = ("tests.unit.tools.native_viewer_export.test_kinematic_replay_15150",)
pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

IMPACT_S, LOOKAT = base.IMPACT_S, base.LOOKAT
SIZE = (1280, 720)
DEPTH_SIZE, DTL = (640, 360), "down_the_line"
TRACK, TEXT = track.TRACK, track.TEXT
FACE_ON, DOWN = "face_on", "down_the_line_flight"

# Pinned digest of landing tiles (inset on down-the-line flight + face-on) after address→landing.
# Re-pinned when min_ball_radius_px rose from 2 to 5 px (#15150 review: legible ball).
PINNED_LANDING_DIGEST = (
    "006f25e670143ac73aec8929b1f82e88ba99e713b3f6a034edc7b08f5f2c685f"
)


def _style():
    """Flight style with a solid landing track and a loud inset."""
    marker = replace(base.DEFAULTS.landing_marker, **track.SOLID)
    inset_cfg = replace(inset.LOUD, enabled=True)
    return replace(
        base.DEFAULTS,
        landing_marker=marker,
        inset=inset_cfg,
        ball_shading=None,
        occlude_by_scene=True,
    )


def _draw(
    view: str,
    t: float,
    *,
    size: tuple[int, int] = SIZE,
    depth: float | None = None,
) -> np.ndarray:
    """Draw one flight frame for ``view`` at time ``t``."""
    flight = base._playback(_style())
    tile = np.full((size[1], size[0], 3), 40, np.uint8)
    depth_m = (
        None
        if depth is None
        else {view: np.full((size[1], size[0]), depth, np.float32)}
    )
    return flight.draw({view: tile}, t, LOOKAT, None, depth_m)[view]


def _count(img: np.ndarray, colour: tuple[int, ...]) -> int:
    """Number of pixels in ``img`` whose RGB equals ``colour``."""
    return int((img == np.array(colour, np.uint8)).all(axis=2).sum())


def test_verification_clip_face_on_and_down_the_line_address_through_landing() -> None:
    """Ground track, carry labels, depth ball hide, and inset stay coherent."""
    trace = base.DEFAULTS.tracer_color_rgba[:3]
    address = IMPACT_S - 0.5
    mid = IMPACT_S + 2.0
    landing = IMPACT_S + base._playback().duration_s + 0.5
    for view in (FACE_ON, DOWN):
        assert _count(_draw(view, address), trace) == 0
    assert _count(_draw(DOWN, mid), trace) > 20
    assert _count(_draw(FACE_ON, mid), trace) > 5
    mid_plain = _draw(DTL, mid, size=DEPTH_SIZE)
    assert (
        _count(mid_plain, trace) > 30
        and _count(mid_plain, base.DEFAULTS.ball_color_rgb) > 3
    )
    behind = _draw(DTL, mid, size=DEPTH_SIZE, depth=0.5)
    assert (
        _count(behind, trace) == 0 and _count(behind, base.DEFAULTS.ball_color_rgb) == 0
    )
    play = base._playback(_style())
    rect = fi.layout_rect((SIZE[1], SIZE[0], 3), _style().inset, [], 40, SIZE[1])
    assert rect is not None
    landed_dtl = fi.paint(_draw(DOWN, landing), play, landing, rect, _style().inset)
    landed_fo = _draw(FACE_ON, landing)
    assert _count(landed_dtl, TRACK) > 10 and _count(landed_dtl, inset.TEXT) > 20
    assert _count(landed_fo, trace) > 5
    assert fi.shows_inset(_style().inset, DOWN)
    digest = hashlib.sha256(landed_dtl.tobytes() + landed_fo.tobytes()).hexdigest()
    assert digest == PINNED_LANDING_DIGEST


def test_verification_clip_kinematic_replay_hides_force_arrows(
    zero_effort_swing: SwingInput,
) -> None:
    """A zero-effort replay does not request force or torque arrows."""
    from src.tools.native_viewer_export.overlay import build_overlay_feed

    feed, _ = build_overlay_feed(zero_effort_swing, "mujoco")
    assert kr.is_kinematic_replay(zero_effort_swing.bundle)
    assert kr.vector_visibility(zero_effort_swing.bundle, OverlayOptions()) == (
        False,
        False,
        True,
        True,  # replay: the tuple's fourth field, added by #15150 gating
    )
    assert feed.hud_lines_at(0)[0] == kr.NOTE
    glyphs = feed.glyphs_at(0)
    assert not glyphs.arrows and not glyphs.torque_arcs
