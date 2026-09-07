"""One-time intrinsic calibration per camera from a printed chessboard (C2, #9622).

The only "precise" setup step: a chessboard shown to each camera once. The
board corners found in a set of frames go through OpenCV's Zhang calibration
to give ``K`` and the distortion vector; the result is stored as the
ADR-0041 :class:`CameraIntrinsics` record keyed by the camera identity so it
follows the unit across jacks and takes. Reprojection RMS and the number of
frames used are part of the record's provenance — a calibration from three
frames is reported as such, not presented as equal to one from thirty.

Nothing here downloads, invents or extrapolates: a frame where the board is
not found is counted, and fewer than ``MIN_FRAMES`` usable frames refuses to
calibrate.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from src.shared.python.core.contracts import require
from src.shared.python.logging_pkg.logging_config import get_logger

Array: TypeAlias = npt.NDArray[np.float64]
logger = get_logger(__name__)

MIN_FRAMES = 8
MAX_RMS_PX = 1.0


@dataclass(frozen=True)
class Chessboard:
    """Inner-corner grid ``(columns, rows)`` and the square size in metres."""

    columns: int
    rows: int
    square_m: float

    def __post_init__(self) -> None:
        require(
            self.columns >= 3 and self.rows >= 3, "board needs >= 3x3 inner corners"
        )
        require(self.columns != self.rows, "use an asymmetric board (columns != rows)")
        require(self.square_m > 0, "square size must be positive")

    @property
    def object_points(self) -> Array:
        """Corner coordinates on the board plane ``(N, 3)`` in metres, z = 0."""
        grid = np.mgrid[0 : self.columns, 0 : self.rows].T.reshape(-1, 2)
        return np.hstack([grid * self.square_m, np.zeros((grid.shape[0], 1))]).astype(
            np.float64
        )


class IntrinsicsRecord(BaseModel):
    """What one calibration produced, with its evidence."""

    model_config = ConfigDict(frozen=True)

    camera_id: str
    matrix: list[list[float]]
    distortion: list[float]
    image_size_px: tuple[int, int]
    rms_px: float
    frames_used: int
    frames_without_board: int
    board: dict[str, Any]

    @property
    def ok(self) -> bool:
        return self.frames_used >= MIN_FRAMES and self.rms_px <= MAX_RMS_PX


def find_corners(image_bgr: Array, board: Chessboard) -> Array | None:
    """Sub-pixel inner corners ``(N, 2)`` in the frame, or None if not found."""
    import cv2

    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    found, corners = cv2.findChessboardCorners(
        gray,
        (board.columns, board.rows),
        flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE,
    )
    if not found:
        return None
    refined = cv2.cornerSubPix(
        gray,
        corners,
        (11, 11),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3),
    )
    return np.asarray(refined, dtype=np.float64).reshape(-1, 2)


def calibrate(
    camera_id: str,
    corner_sets: Sequence[Array],
    board: Chessboard,
    image_size_px: tuple[int, int],
    *,
    frames_without_board: int = 0,
) -> IntrinsicsRecord:
    """Zhang calibration from corner sets; refuses fewer than ``MIN_FRAMES``.

    Postcondition: ``rms_px`` is OpenCV's reprojection RMS over all corners.
    """
    import cv2

    require(
        len(corner_sets) >= MIN_FRAMES,
        "too few frames with the board",
        len(corner_sets),
    )
    n = board.columns * board.rows
    for c in corner_sets:
        require(c.shape == (n, 2), "corner set shape", c.shape)
    obj = [board.object_points.astype(np.float32) for _ in corner_sets]
    img = [np.asarray(c, dtype=np.float32).reshape(-1, 1, 2) for c in corner_sets]
    # Initial K/dist are outputs here (no CALIB_USE_INTRINSIC_GUESS flag).
    rms, k, dist, _, _ = cv2.calibrateCamera(
        obj, img, image_size_px, np.zeros((3, 3)), np.zeros(5)
    )
    return IntrinsicsRecord(
        camera_id=camera_id,
        matrix=np.asarray(k, dtype=float).tolist(),
        distortion=np.asarray(dist, dtype=float).ravel().tolist(),
        image_size_px=(int(image_size_px[0]), int(image_size_px[1])),
        rms_px=float(rms),
        frames_used=len(corner_sets),
        frames_without_board=frames_without_board,
        board={
            "columns": board.columns,
            "rows": board.rows,
            "square_m": board.square_m,
        },
    )


def frames_from_video(path: Path, *, every: int = 10) -> Iterator[Any]:
    """Every ``every``-th BGR frame of a video (the board moves slowly)."""
    import cv2

    require(every >= 1, "every must be >= 1", every)
    cap = cv2.VideoCapture(str(path))
    require(cap.isOpened(), "could not open video", str(path))
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                return
            if index % every == 0:
                yield frame
            index += 1
    finally:
        cap.release()


def calibrate_video(
    camera_id: str, path: Path, board: Chessboard, *, every: int = 10
) -> IntrinsicsRecord:
    """Find the board in sampled frames of a recording and calibrate."""
    corners: list[Array] = []
    missed = 0
    size: tuple[int, int] | None = None
    for frame in frames_from_video(path, every=every):
        size = (int(frame.shape[1]), int(frame.shape[0]))
        found = find_corners(frame, board)
        if found is None:
            missed += 1
        else:
            corners.append(found)
    require(size is not None, "video has no frames", str(path))
    assert size is not None
    logger.info(
        "%s: board found in %d frames, missing in %d", camera_id, len(corners), missed
    )
    return calibrate(camera_id, corners, board, size, frames_without_board=missed)


def write_intrinsics(records: Sequence[IntrinsicsRecord], path: Path) -> Path:
    """``intrinsics.json``: a list usable by ``rig reconstruct --intrinsics``."""
    payload = [r.model_dump() for r in records]
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path
