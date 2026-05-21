from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from src.shared.python.motion_matching.body_target import BodyTarget
from src.shared.python.motion_matching.club_target import SourceProvenance
from src.shared.python.motion_matching.diagnostics.body_target_video import (
    save_body_target_video,
)


def _body_target() -> BodyTarget:
    names = (
        "WaistLeft",
        "WaistRight",
        "WaistLBack",
        "WaistRBack",
        "BackTop",
    )
    time = np.arange(5, dtype=float) / 30.0
    xyz = np.zeros((time.size, len(names), 3), dtype=float)
    base = np.array(
        [
            [-0.2, 0.0, 0.9],
            [0.2, 0.0, 0.9],
            [-0.2, -0.2, 0.9],
            [0.2, -0.2, 0.9],
            [0.0, -0.1, 1.5],
        ],
        dtype=float,
    )
    for i in range(time.size):
        xyz[i] = base + np.array([0.02 * i, 0.0, 0.0])
    return BodyTarget(
        time=time,
        marker_xyz=xyz,
        marker_names=names,
        impact_idx=2,
        events=(),
        source=SourceProvenance(
            filename="synthetic.c3d",
            format="synthetic",
            subject_id="synthetic",
            trial_id="synthetic",
            sha256="0" * 64,
        ),
    )


def test_save_body_target_video_writes_nonblank_mp4(tmp_path: Path) -> None:
    out = tmp_path / "body_preview.mp4"

    result = save_body_target_video(
        _body_target(),
        out,
        frame_indices=(0, 2, 4),
        fps=12.0,
        width=320,
        height=240,
        title="synthetic",
    )

    assert result.output_path == out
    assert result.frame_count == 3
    assert out.exists()
    assert out.stat().st_size > 1_000

    cap = cv2.VideoCapture(str(out))
    try:
        ok, frame = cap.read()
        assert ok
        assert frame.std() > 0
    finally:
        cap.release()


def test_save_body_target_video_rejects_empty_frame_selection(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="at least one frame"):
        save_body_target_video(_body_target(), tmp_path / "bad.mp4", frame_indices=())


def test_save_body_target_video_requires_even_dimensions(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="even"):
        save_body_target_video(_body_target(), tmp_path / "bad.mp4", width=321)
