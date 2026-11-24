"""Tip-to-tail ZTCF/ZVCF hand force in the native export (#15685, K5)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.glyphs import ForceGlyphStyle
from src.shared.python.force_overlay.tip_to_tail import tip_to_tail_arrows
from src.tools.native_viewer_export import cli
from src.tools.native_viewer_export.core import OverlayFeed
from src.tools.native_viewer_export.ztcf_feed import ztcf_zvcf_arrows

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
BASE = ["--bundle", "b.npz", "--out", "o", "--swing", "s"]
STYLE = ForceGlyphStyle(force_scale_m_per_n=0.001, max_length_m=5.0)


def _frame(_index: int) -> ForceTorqueFrame:
    wrench = OverlayWrench(
        WrenchKind.CONTACT, "contact:foot", "foot", (0, 0, 0), (0, 0, 400.0), source="t"
    )
    return ForceTorqueFrame(0.0, "mujoco", (wrench,))


def _extra(index: int):
    return tip_to_tail_arrows((0, 0, 1), (100.0 * (index + 1), 0, 0), (0, 50, 0), STYLE)


def test_the_flag_defaults_off_and_parses() -> None:
    parser = cli.build_parser()
    assert parser.parse_args(BASE).ztcf_zvcf_hand_force is False
    assert parser.parse_args([*BASE, "--ztcf-zvcf-hand-force"]).ztcf_zvcf_hand_force


def test_extra_arrows_join_the_glyphs_and_the_legend() -> None:
    feed = OverlayFeed(_frame, STYLE, extra_arrows=_extra, extra_legend="ZTCF note")
    labels = [a.label for a in feed.glyphs_at(0).arrows]
    assert "ztcf:drift" in labels and "zvcf:control" in labels and "total" in labels
    assert feed.legend_for(feed.glyphs_at(0)).endswith("ZTCF note")
    assert "ZTCF" not in OverlayFeed(_frame, STYLE).legend_for(feed.glyphs_at(0))


def _drift_magnitude(feed: OverlayFeed, index: int) -> float:
    arrows = {a.label: a for a in feed.glyphs_at(index).arrows}
    return arrows["ztcf:drift"].magnitude


def test_following_and_held_after_remap_the_extra_arrows() -> None:
    feed = OverlayFeed(_frame, STYLE, extra_arrows=_extra)
    follow = feed.following(None, lambda k: 2 * k)
    assert _drift_magnitude(follow, 1) == pytest.approx(300.0)
    held = feed.held_after(1)
    assert _drift_magnitude(held, 5) == pytest.approx(200.0)


def test_a_kinematic_replay_is_refused() -> None:
    bundle = SimpleNamespace(efforts=np.zeros((3, 4)), spec_bytes=b"{}")
    with pytest.raises(ValueError, match="kinetic match"):
        ztcf_zvcf_arrows(SimpleNamespace(bundle=bundle), STYLE)
