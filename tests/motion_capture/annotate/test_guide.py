"""Guided annotation cursor (#9799)."""

from __future__ import annotations

import pytest

from src.motion_capture.annotate import KEYMAP, AnnotationSet, Guide

pytestmark = pytest.mark.unit

JOINTS = ("left_wrist", "right_wrist")


def _guide(**kw: object) -> Guide:
    store = AnnotationSet("cam", 100, 100, 10.0)
    return Guide(store, JOINTS, frame_range=(0, 10), stride=5, **kw)


def test_prompts_walk_joints_then_frames_and_report_progress() -> None:
    g = _guide()
    p = g.prompt()
    assert p is not None and (p.frame, p.joint) == (0, "left_wrist")
    assert "Frame 0 · left_wrist" in p.text and "S skip" in p.text
    assert g.progress() == (0, 6)
    g.accept(1, 2)
    assert (g.frame, g.joint) == (0, "right_wrist")
    g.skip()
    assert (g.frame, g.joint) == (5, "left_wrist")
    assert g.progress() == (2, 6)
    g.next_frame()
    assert (g.frame, g.joint) == (10, "left_wrist")
    g.accept(3, 4)
    g.accept(5, 6)
    assert g.finished and g.prompt() is None
    assert g.store.point(0, "left_wrist").x_px == 1.0
    assert g.store.is_skipped(0, "right_wrist")
    assert g.store.point(5, "left_wrist") is None  # left by next_frame
    assert g.store.point(10, "right_wrist").y_px == 6.0
    with pytest.raises(Exception, match="finished"):
        g.accept(0, 0)


def test_back_jump_and_only_missing() -> None:
    g = _guide()
    g.accept(1, 1)
    g.accept(2, 2)
    assert g.frame == 5
    g.back()
    assert (g.frame, g.joint) == (0, "right_wrist")
    g.accept(9, 9)  # re-answering replaces
    assert g.store.point(0, "right_wrist").x_px == 9.0
    g.jump(99)
    assert g.frame == 10  # clamped
    g.jump(-4)
    # Frame 0 is fully answered, so the cursor settles on the next missing one.
    assert (g.frame, g.joint) == (5, "left_wrist")
    everything = Guide(g.store, JOINTS, (0, 10), 5, only_missing=False)
    assert everything.prompt().frame == 0


def test_keymap_dispatch() -> None:
    g = _guide()
    assert KEYMAP["s"] == "skip" and KEYMAP["q"] == "finish"
    assert g.handle_key("S") and g.store.is_skipped(0, "left_wrist")
    assert g.handle_key("n") and g.frame == 5
    assert g.handle_key("b") and g.frame == 0
    assert g.handle_key("j")  # reported, needs a frame from the caller
    assert not g.handle_key("x")
    assert g.handle_key("q") and g.finished


def test_guide_preconditions() -> None:
    store = AnnotationSet("cam", 100, 100, 10.0)
    with pytest.raises(Exception, match="belong"):
        Guide(store, ("tail",), (0, 1))
    with pytest.raises(Exception, match="frame_range"):
        Guide(store, JOINTS, (5, 1))
    with pytest.raises(Exception, match="stride"):
        Guide(store, JOINTS, (0, 1), stride=0)
