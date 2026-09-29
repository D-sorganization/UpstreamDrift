"""Unit tests for the body-skeleton segment table and matplotlib renderer."""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from mpl_toolkits.mplot3d import Axes3D  # noqa: E402,F401  # registers 3d projection

from src.shared.python.motion_matching.body_skeleton import (  # noqa: E402
    BodySegment,
    default_body_segments,
)
from src.shared.python.motion_matching.diagnostics._skeleton_render import (  # noqa: E402
    draw_body_target_frame,
)

pytestmark = pytest.mark.unit


# Canonical 28-marker anatomical subset (Plug-in-Gait subset).
FULL_28_MARKERS: tuple[str, ...] = (
    "WaistLeft",
    "WaistRight",
    "WaistLBack",
    "WaistRBack",
    "BackTop",
    "BackLeft",
    "BackRight",
    "HeadTop",
    "HeadFront",
    "HeadSide",
    "LShoulderTop",
    "LShoulderBack",
    "LUArmHigh",
    "LElbowOut",
    "LWristTop",
    "RShoulderTop",
    "RShoulderBack",
    "RUArmHigh",
    "RElbowOut",
    "RWristTop",
    "LKneeOut",
    "LAnkleOut",
    "LToeIn",
    "LToeOut",
    "RKneeOut",
    "RAnkleOut",
    "RToeIn",
    "RToeOut",
)


def test_default_body_segments_full_set_returns_canonical_count() -> None:
    segs = default_body_segments(FULL_28_MARKERS)
    assert len(segs) == 26
    # Every group is represented.
    groups = {s.group for s in segs}
    assert groups == {
        "torso",
        "head",
        "left_arm",
        "right_arm",
        "left_leg",
        "right_leg",
        "pelvis",
    }


def test_default_body_segments_filters_missing_markers() -> None:
    # Drop the right shoulder cluster — every right-arm segment must vanish.
    partial = tuple(
        m for m in FULL_28_MARKERS if m not in {"RShoulderTop", "RShoulderBack"}
    )
    segs = default_body_segments(partial)
    assert all("RShoulder" not in s.a and "RShoulder" not in s.b for s in segs)
    # Other groups are unaffected.
    left_arm_segs = [s for s in segs if s.group == "left_arm"]
    assert len(left_arm_segs) == 4


def test_default_body_segments_empty_marker_list_returns_empty() -> None:
    assert default_body_segments(()) == ()
    assert default_body_segments([]) == ()


def test_body_segment_validation_rules() -> None:
    # Happy path
    BodySegment(a="HeadTop", b="HeadFront", group="head")

    # Empty endpoint
    with pytest.raises(ValueError, match="non-empty"):
        BodySegment(a="", b="HeadFront", group="head")
    with pytest.raises(ValueError, match="non-empty"):
        BodySegment(a="HeadTop", b="", group="head")

    # Same endpoint
    with pytest.raises(ValueError, match="must differ"):
        BodySegment(a="HeadTop", b="HeadTop", group="head")

    # Bad group literal
    with pytest.raises(ValueError, match="group must be one of"):
        BodySegment(a="HeadTop", b="HeadFront", group="not-a-group")  # type: ignore[arg-type]


def test_body_segment_is_frozen() -> None:
    seg = BodySegment(a="HeadTop", b="HeadFront", group="head")
    with pytest.raises(AttributeError):  # FrozenInstanceError subclasses AttributeError
        seg.a = "Other"  # type: ignore[misc]


@dataclass
class _BodyTargetStub:
    """Minimal structural stand-in for ``BodyTarget`` used in renderer tests."""

    marker_xyz: np.ndarray
    marker_names: tuple[str, ...]


def _make_synthetic_target(*, with_nan: bool = False) -> _BodyTargetStub:
    rng = np.random.default_rng(0)
    n_frames = 4
    n_markers = len(FULL_28_MARKERS)
    xyz = rng.standard_normal((n_frames, n_markers, 3)) * 0.3
    if with_nan:
        # Occlude an entire arm at frame 0.
        idx = FULL_28_MARKERS.index("RWristTop")
        xyz[0, idx, :] = np.nan
    return _BodyTargetStub(marker_xyz=xyz, marker_names=FULL_28_MARKERS)


def test_draw_body_target_frame_smoke() -> None:
    target = _make_synthetic_target()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    initial_lines = len(ax.lines)
    draw_body_target_frame(ax, target, frame_idx=2)
    assert len(ax.lines) - initial_lines >= 1
    plt.close(fig)


def test_draw_body_target_frame_skips_nan_endpoints() -> None:
    target = _make_synthetic_target(with_nan=True)
    fig = plt.figure()
    ax_full = fig.add_subplot(121, projection="3d")
    ax_nan = fig.add_subplot(122, projection="3d")
    draw_body_target_frame(ax_full, target, frame_idx=1)  # no NaN frame
    draw_body_target_frame(ax_nan, target, frame_idx=0)  # has NaN
    # NaN frame must drop at least one segment vs. the clean frame.
    assert len(ax_nan.lines) < len(ax_full.lines)
    plt.close(fig)


def test_draw_body_target_frame_group_filter() -> None:
    target = _make_synthetic_target()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    draw_body_target_frame(ax, target, frame_idx=0, segment_groups=["head"])
    # Only the 2 head segments should be drawn.
    assert len(ax.lines) == 2
    plt.close(fig)


def test_draw_body_target_frame_rejects_bad_frame_idx() -> None:
    target = _make_synthetic_target()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    with pytest.raises(ValueError, match="out of range"):
        draw_body_target_frame(ax, target, frame_idx=99)
    plt.close(fig)
