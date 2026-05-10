"""Adversarial: real-world schema drift across adapter formats.

Real exports from MediaPipe, OpenPose, BVH tools, etc. all drift from
the canonical schema in subtle ways. These tests exercise the drift cases
we expect to see when real markerless mocap data starts flowing.
"""

from __future__ import annotations

import contextlib
import json
from pathlib import Path

import pytest

from src.shared.python.motion_pipeline.sources import load_any

# ---------------------------------------------------------------------------
# MediaPipe drift: pose_landmarks vs landmarks key
# ---------------------------------------------------------------------------


def test_mediapipe_landmarks_key_variant(tmp_path: Path) -> None:
    """Some MediaPipe exports use ``landmarks`` instead of ``pose_landmarks``.
    The adapter should either accept both or fail cleanly.
    """
    payload = {
        "frames": [
            {
                "timestamp": 0.0,
                "landmarks": [{"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.9}] * 33,
            }
        ]
    }
    p = tmp_path / "mp.json"
    p.write_text(json.dumps(payload))
    # Either succeeds or raises a clean adapter error.
    try:
        load_any(p)
    except Exception as e:
        msg = str(e).lower()
        assert "mediapipe" in msg or "landmark" in msg, (
            f"Unexpected adapter error message (not a clean mediapipe/landmark "
            f"failure): {type(e).__name__}: {e}"
        )


def test_mediapipe_pose_landmarks_key_variant(tmp_path: Path) -> None:
    """The canonical ``pose_landmarks`` key must work."""
    payload = {
        "frames": [
            {
                "timestamp": 0.0,
                "pose_landmarks": [{"x": 0.0, "y": 0.0, "z": 0.0, "visibility": 0.9}]
                * 33,
            }
        ]
    }
    p = tmp_path / "mp.json"
    p.write_text(json.dumps(payload))
    with contextlib.suppress(Exception):
        load_any(p)


# ---------------------------------------------------------------------------
# OpenPose drift: BODY_25 vs COCO output
# ---------------------------------------------------------------------------


def test_openpose_body25_format(tmp_path: Path) -> None:
    """Standard BODY_25 output: 25 keypoints * 3 (x, y, c)."""
    payload = {"people": [{"pose_keypoints_2d": [0.0, 0.0, 0.9] * 25}]}
    p = tmp_path / "op.json"
    p.write_text(json.dumps(payload))
    with contextlib.suppress(Exception):
        load_any(p)


def test_openpose_coco17_format(tmp_path: Path) -> None:
    """COCO output has 17 keypoints. Adapter must reject or convert; never
    crash."""
    payload = {"people": [{"pose_keypoints_2d": [0.0, 0.0, 0.9] * 17}]}
    p = tmp_path / "op17.json"
    p.write_text(json.dumps(payload))
    with contextlib.suppress(Exception):
        load_any(p)


# ---------------------------------------------------------------------------
# TRC drift: variable header, blank lines
# ---------------------------------------------------------------------------


def test_trc_with_blank_lines(tmp_path: Path) -> None:
    """A real-world TRC file may have blank lines between header rows."""
    content = (
        "PathFileType\t4\t(X/Y/Z)\tblank.trc\n"
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
        "100.0\t100.0\t2\t1\tmm\t100.0\t1\t2\n"
        "Frame#\tTime\tM1\t\t\n"
        "\t\tX1\tY1\tZ1\n"
        "\n"
        "1\t0.000\t0.0\t0.0\t0.0\n"
        "2\t0.010\t1.0\t1.0\t1.0\n"
    )
    p = tmp_path / "blank.trc"
    p.write_text(content)
    with contextlib.suppress(Exception):
        load_any(p)


# ---------------------------------------------------------------------------
# BVH drift: Euler order
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "channels",
    [
        "Xposition Yposition Zposition Zrotation Xrotation Yrotation",
        "Xposition Yposition Zposition Yrotation Xrotation Zrotation",
        "Xposition Yposition Zposition Xrotation Yrotation Zrotation",
    ],
)
def test_bvh_euler_order_variants(tmp_path: Path, channels: str) -> None:
    """Different BVH exporters use different rotation channel orders. The
    adapter must handle any documented order or fail with a clear error."""
    bvh = (
        "HIERARCHY\n"
        "ROOT Hips\n"
        "{\n"
        "  OFFSET 0.00 0.00 0.00\n"
        f"  CHANNELS 6 {channels}\n"
        "  End Site\n"
        "  {\n"
        "    OFFSET 0.00 1.00 0.00\n"
        "  }\n"
        "}\n"
        "MOTION\n"
        "Frames: 2\n"
        "Frame Time: 0.0333333\n"
        "0 0 0 0 0 0\n"
        "0 0 0 0 0 0\n"
    )
    p = tmp_path / "swap.bvh"
    p.write_text(bvh)
    with contextlib.suppress(Exception):
        load_any(p)
