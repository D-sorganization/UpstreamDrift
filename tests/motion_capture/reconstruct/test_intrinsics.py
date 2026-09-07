"""Chessboard intrinsics: synthetic renders through a known camera recover K."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.motion_capture.reconstruct.intrinsics import (
    MIN_FRAMES,
    Chessboard,
    calibrate,
    calibrate_video,
    find_corners,
    write_intrinsics,
)

cv2 = pytest.importorskip("cv2")
pytestmark = pytest.mark.unit

BOARD = Chessboard(columns=9, rows=6, square_m=0.025)
SIZE = (960, 600)
K_TRUE = np.array([[820.0, 0.0, 480.0], [0.0, 820.0, 300.0], [0.0, 0.0, 1.0]])
DIST_TRUE = np.array([-0.15, 0.05, 0.0, 0.0, 0.0])


def _board_image() -> np.ndarray:
    """A chessboard texture with one square of white margin around it."""
    cols, rows = BOARD.columns + 1, BOARD.rows + 1
    px = 60
    img = np.full(((rows + 2) * px, (cols + 2) * px), 255, dtype=np.uint8)
    for r in range(rows):
        for c in range(cols):
            if (r + c) % 2 == 0:
                img[(r + 1) * px : (r + 2) * px, (c + 1) * px : (c + 2) * px] = 0
    return img


def _pose_frames(n: int, rng: np.random.Generator):
    """Board poses in front of the camera: (rvec, tvec) tuples."""
    for _ in range(n):
        rvec = rng.uniform(-0.45, 0.45, 3)
        tvec = np.array(
            [rng.uniform(-0.12, 0.12), rng.uniform(-0.08, 0.08), rng.uniform(0.45, 0.8)]
        )
        yield rvec, tvec


def _distort(undistorted: np.ndarray) -> np.ndarray:
    """Resample an ideal-pinhole image through the lens model.

    For every distorted output pixel, the source pixel in the ideal image is
    the point whose distortion lands there, which is what undistortPoints
    computes (with P = K to return pixels).
    """
    w, h = SIZE
    xs, ys = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    grid = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
    src = cv2.undistortPoints(grid, K_TRUE, DIST_TRUE, P=K_TRUE).reshape(h, w, 2)
    return cv2.remap(
        undistorted,
        src[:, :, 0].astype(np.float32),
        src[:, :, 1].astype(np.float32),
        cv2.INTER_LINEAR,
        borderValue=200,
    )


def _render(rvec, tvec) -> np.ndarray:
    """Warp the board texture into the ideal view, then apply the distortion."""
    texture = _board_image()
    px = 60
    obj = BOARD.object_points
    ideal, _ = cv2.projectPoints(obj.reshape(-1, 1, 3), rvec, tvec, K_TRUE, None)
    src = np.array(
        [
            [
                (int(round(o[0] / BOARD.square_m)) + 2) * px,
                (int(round(o[1] / BOARD.square_m)) + 2) * px,
            ]
            for o in obj
        ],
        dtype=np.float32,
    )
    h, _ = cv2.findHomography(src, ideal.reshape(-1, 2).astype(np.float32))
    warped = cv2.warpPerspective(texture, h, SIZE, borderValue=200)
    return cv2.cvtColor(_distort(warped), cv2.COLOR_GRAY2BGR)


def test_board_contract_and_object_points() -> None:
    assert BOARD.object_points.shape == (54, 3)
    assert BOARD.object_points[1].tolist() == [0.025, 0.0, 0.0]
    with pytest.raises(Exception, match="asymmetric"):
        Chessboard(columns=6, rows=6, square_m=0.02)


def test_corners_found_and_calibration_recovers_k_and_distortion() -> None:
    rng = np.random.default_rng(3)
    corner_sets = []
    for rvec, tvec in _pose_frames(14, rng):
        corners = find_corners(_render(rvec, tvec), BOARD)
        if corners is not None:
            corner_sets.append(corners)
    assert len(corner_sets) >= MIN_FRAMES
    record = calibrate("cam", corner_sets, BOARD, SIZE)
    k = np.array(record.matrix)
    assert abs(k[0, 0] - K_TRUE[0, 0]) / K_TRUE[0, 0] < 0.02
    assert abs(k[1, 1] - K_TRUE[1, 1]) / K_TRUE[1, 1] < 0.02
    assert abs(k[0, 2] - K_TRUE[0, 2]) < 15 and abs(k[1, 2] - K_TRUE[1, 2]) < 15
    assert abs(record.distortion[0] - DIST_TRUE[0]) < 0.05
    assert record.rms_px < 1.0 and record.ok


def test_too_few_frames_is_refused_and_video_path_counts_misses(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    few = [find_corners(_render(*p), BOARD) for p in _pose_frames(3, rng)]
    few = [c for c in few if c is not None]
    with pytest.raises(Exception, match="too few frames"):
        calibrate("cam", few, BOARD, SIZE)
    video = tmp_path / "board.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, SIZE)
    for rvec, tvec in _pose_frames(10, np.random.default_rng(5)):
        writer.write(_render(rvec, tvec))
    writer.write(np.full((SIZE[1], SIZE[0], 3), 90, dtype=np.uint8))  # no board
    writer.release()
    record = calibrate_video("cam_b", video, BOARD, every=1)
    assert record.frames_used >= MIN_FRAMES and record.frames_without_board >= 1
    out = write_intrinsics([record], tmp_path / "intrinsics.json")
    assert out.is_file() and '"camera_id": "cam_b"' in out.read_text(encoding="utf-8")


def test_rig_calibrate_intrinsics_command(tmp_path: Path) -> None:
    from src.motion_capture.rig import __main__ as rig_cli
    from src.motion_capture.rig.bundle import build_index, write_bundle
    from src.motion_capture.rig.plan import CameraBinding, RigPlan
    from src.motion_capture.rig.probe import RecordingProbe
    from src.motion_capture.rig.recorder import RecordingResult

    plan = RigPlan(name="cal", cameras=(CameraBinding(view="cam_b", serial="1"),))
    video = tmp_path / "cam_b_1.avi"
    writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 10.0, SIZE)
    for rvec, tvec in _pose_frames(12, np.random.default_rng(9)):
        writer.write(_render(rvec, tvec))
    writer.release()
    results = [RecordingResult("1", video, 0, video.stat().st_size)]
    index = build_index(
        plan,
        results,
        1.2,
        tmp_path,
        prober=lambda p: RecordingProbe(12, 1.2, SIZE[0], SIZE[1], 10.0),
    )
    write_bundle(tmp_path, plan, index, started_utc="2026-09-07T00:00:00+00:00")
    code = rig_cli.main(
        [
            "calibrate-intrinsics",
            "--session",
            str(tmp_path),
            "--board",
            "9x6",
            "--square",
            "0.025",
            "--every",
            "1",
        ]
    )
    assert code == 0
    payload = json.loads((tmp_path / "intrinsics.json").read_text(encoding="utf-8"))
    assert payload[0]["camera_id"] == "cam_b" and payload[0]["rms_px"] < 1.0
