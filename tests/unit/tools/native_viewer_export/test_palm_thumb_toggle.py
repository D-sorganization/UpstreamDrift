"""Native-viewer toggle for the allocated palm-on-thumb force (HRG-H1, #13189)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.grip_contact.palm_thumb import PALM_THUMB_LABEL
from src.tools.native_viewer_export import cli
from src.tools.native_viewer_export.core import ExportSettings, OverlayFeed

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
BASE = ["--bundle", "b.npz", "--out", "o", "--swing", "s"]


def _settings(*extra: str) -> ExportSettings:
    return cli.build_settings(cli.build_parser().parse_args([*BASE, *extra]))


def _frame(loaded: bool = True) -> ForceTorqueFrame:
    wrenches = [
        OverlayWrench(
            WrenchKind.CONTACT,
            "contact:foot",
            "foot",
            (0, 0, 0),
            (0, 0, 400.0),
            source="t",
        )
    ]
    if loaded:
        wrenches.append(
            OverlayWrench(
                WrenchKind.GRIP,
                PALM_THUMB_LABEL,
                "club",
                (0.5, 0, 1.0),
                (0.0, 90.0, 0.0),
                source="mujoco:allocation",
            )
        )
    return ForceTorqueFrame(0.0, "mujoco", tuple(wrenches))


def _labels(feed: OverlayFeed) -> list[str]:
    return [a.label for a in feed.glyphs_at(0).arrows]


def test_flag_defaults_off_and_parses() -> None:
    assert _settings().palm_thumb_force is False
    assert _settings("--palm-thumb-force").palm_thumb_force is True
    assert "--palm-thumb-force" in cli.build_parser().format_help()


def test_flag_needs_the_overlay() -> None:
    with pytest.raises(ValueError, match="palm_thumb_force"):
        _settings("--no-overlay", "--palm-thumb-force")
    with pytest.raises(TypeError):
        ExportSettings(palm_thumb_force=1)  # type: ignore[arg-type]


def test_palm_thumb_arrow_is_hidden_unless_the_toggle_is_on() -> None:
    off = OverlayFeed(lambda _i: _frame())
    on = OverlayFeed(lambda _i: _frame(), show_palm_thumb=True)
    assert PALM_THUMB_LABEL not in _labels(off)
    assert PALM_THUMB_LABEL in _labels(on)


def test_toggle_is_independent_of_the_force_and_torque_vectors() -> None:
    feed = OverlayFeed(
        lambda _i: _frame(), show_force=False, show_torque=False, show_palm_thumb=True
    )
    assert _labels(feed) == [PALM_THUMB_LABEL]


def test_unloaded_frame_draws_nothing() -> None:
    feed = OverlayFeed(lambda _i: _frame(loaded=False), show_palm_thumb=True)
    assert PALM_THUMB_LABEL not in _labels(feed)


def test_legend_says_allocated_when_the_arrow_is_drawn() -> None:
    on = OverlayFeed(lambda _i: _frame(), show_palm_thumb=True)
    assert "allocated" in on.legend(on.glyphs_at(0)).lower()
    off = OverlayFeed(lambda _i: _frame())
    assert "allocated" not in off.legend(off.glyphs_at(0)).lower()


def test_arrow_points_along_the_given_force() -> None:
    feed = OverlayFeed(lambda _i: _frame(), show_palm_thumb=True)
    arrow = next(a for a in feed.glyphs_at(0).arrows if a.label == PALM_THUMB_LABEL)
    direction = np.array(arrow.tip_m) - np.array(arrow.tail_m)
    assert direction[1] > 0 and abs(direction[0]) < 1e-9
