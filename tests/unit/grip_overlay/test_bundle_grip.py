"""Grip wrench through the bundle overlay provider (GCV-10, #11716)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics.grip_wrench import HandWrench, analyze_grip
from src.shared.python.force_overlay.bundle_provider import BundleOverlayProvider
from src.shared.python.force_overlay.glyphs import ForceGlyphStyle, build_glyphs
from tests.unit.force_overlay.test_bundle_provider import (
    _bundle,
    _Contact,
    _Kinematics,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


class _Grip:
    """Records the state it is asked about; force grows with the effort index."""

    def __init__(self, fail_at=None):
        self.calls = []
        self.fail_at = fail_at

    def grip_analysis(self, coordinates, rates, efforts):
        self.calls.append((dict(coordinates), dict(rates), dict(efforts)))
        if self.fail_at is not None and len(self.calls) - 1 == self.fail_at:
            raise ValueError("KKT solve failed")
        k = len(self.calls)
        return analyze_grip(
            HandWrench("L", (0, 0, 1.0), (10.0 * k, 0, 0), (0, 0, 0)),
            HandWrench("R", (0, 0, 0.8), (-5.0 * k, 0, 0), (0, 0, 0)),
            split_method="constraint_multiplier",
        )


def _provider(grip):
    return BundleOverlayProvider(
        _bundle(), _Contact(), _Kinematics(), engine="fake"
    ).with_grip(grip, "fake:kkt")


def test_frames_carry_grip_wrenches_and_metadata():
    grip = _Grip()
    frame = _provider(grip).frame_at(1)
    labels = {w.label for w in frame.wrenches}
    assert {"grip:hand_left", "grip:hand_right", "grip:net_midpoint"} <= labels
    assert frame.metadata["grip_split_method"] == "constraint_multiplier"
    assert frame.metadata["grip_source"] == "fake:kkt"
    assert frame.metadata["grip_midpoint_m"] == pytest.approx([0.0, 0.0, 0.9])
    coords, rates, efforts = grip.calls[0]
    assert set(efforts) == set(coords)  # named per coordinate


def test_default_provider_has_no_grip_wrenches():
    frame = BundleOverlayProvider(
        _bundle(), _Contact(), _Kinematics(), engine="fake"
    ).frame_at(1)
    assert not [w for w in frame.wrenches if w.label.startswith("grip:")]
    assert "grip_split_method" not in frame.metadata


def test_failed_solve_is_unavailable_with_a_reason_not_zero():
    frame = _provider(_Grip(fail_at=0)).frame_at(1)
    assert not [w for w in frame.wrenches if w.label.startswith("grip:")]
    assert frame.metadata["grip_split_method"] == "unavailable"
    assert "KKT solve failed" in frame.metadata["grip_unavailable_reason"]
    glyphs = build_glyphs(frame, ForceGlyphStyle())
    assert "grip:net_midpoint" in glyphs.legend.unavailable_labels


def test_grip_glyphs_follow_the_group_toggles():
    frame = _provider(_Grip()).frame_at(0)
    hidden = build_glyphs(frame, ForceGlyphStyle(groups=frozenset({"net"})))
    assert not [a for a in hidden.arrows if a.label.startswith("grip:")]


def test_grip_source_must_have_grip_analysis():
    with pytest.raises(TypeError):
        _provider(object())
    assert np.isfinite(_provider(_Grip()).frame_at(0).time_s)
