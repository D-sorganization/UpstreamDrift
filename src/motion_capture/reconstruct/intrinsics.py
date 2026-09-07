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


@dataclass(frozen=True)
class CharucoBoard:
    """ChArUco board: ``(columns, rows)`` squares, square and marker size in metres.

    Partial views count: every detected chessboard corner carries its id, so
    frames at the edge of the field still contribute (#9679).
    """

    columns: int
    rows: int
    square_m: float
    marker_m: float
    dictionary: str = "DICT_4X4_50"

    def __post_init__(self) -> None:
        require(self.columns >= 3 and self.rows >= 3, "board needs >= 3x3 squares")
        require(self.square_m > 0, "square size must be positive")
        require(0 < self.marker_m < self.square_m, "marker must fit inside a square")

    def opencv_board(self) -> Any:
        import cv2

        dictionary = cv2.aruco.getPredefinedDictionary(
            getattr(cv2.aruco, self.dictionary)
        )
        return cv2.aruco.CharucoBoard(
            (self.columns, self.rows), self.square_m, self.marker_m, dictionary
        )

    def image(self, width_px: int) -> npt.NDArray[np.uint8]:
        """A printable board image ``width_px`` wide (grey, white margin)."""
        require(width_px >= 200, "board image width", width_px)
        height = int(round(width_px * self.rows / self.columns))
        img = self.opencv_board().generateImage((width_px, height), marginSize=20)
        return np.asarray(img, dtype=np.uint8)


Board = Chessboard | CharucoBoard


def parse_board_spec(text: str, square_m: float | None = None) -> Board:
    """``9x6`` (chessboard, needs ``square_m``) or ``charuco:7x5:0.04:0.03[:DICT]``."""
    parts = text.strip().lower().split(":")
    if parts[0] == "charuco":
        require(len(parts) >= 4, "charuco:COLSxROWS:SQUARE_M:MARKER_M[:DICT]", text)
        cols, _, rows = parts[1].partition("x")
        dictionary = parts[4].upper() if len(parts) > 4 else "DICT_4X4_50"
        return CharucoBoard(
            int(cols), int(rows), float(parts[2]), float(parts[3]), dictionary
        )
    cols, sep, rows = parts[0].partition("x")
    require(bool(sep and cols.isdigit() and rows.isdigit()), "board like 9x6", text)
    require(square_m is not None and square_m > 0, "chessboard needs --square")
    assert square_m is not None
    return Chessboard(columns=int(cols), rows=int(rows), square_m=square_m)


def board_spec(board: Board) -> dict[str, Any]:
    out: dict[str, Any] = {"kind": type(board).__name__, "columns": board.columns}
    out["rows"] = board.rows
    out["square_m"] = board.square_m
    if isinstance(board, CharucoBoard):
        out["marker_m"] = board.marker_m
        out["dictionary"] = board.dictionary
    return out


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


def find_board(image_bgr: Array, board: Board) -> tuple[Array, Array] | None:
    """``(object_points (N,3) m, image_points (N,2) px)`` or None when not found.

    A chessboard needs all its corners; a ChArUco board contributes whatever
    corners were identified (at least six, so a pose is well determined).
    """
    if isinstance(board, Chessboard):
        corners = find_corners(image_bgr, board)
        return None if corners is None else (board.object_points, corners)
    import cv2

    detector = cv2.aruco.CharucoDetector(board.opencv_board())
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    ch_corners, ids, _, _ = detector.detectBoard(gray)
    if ch_corners is None or ids is None or len(ids) < 6:
        return None
    obj, img = board.opencv_board().matchImagePoints(ch_corners, ids)
    return (
        np.asarray(obj, dtype=np.float64).reshape(-1, 3),
        np.asarray(img, dtype=np.float64).reshape(-1, 2),
    )


def calibrate(
    camera_id: str,
    corner_sets: Sequence[Array | tuple[Array, Array]],
    board: Board,
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
    obj: list[Any] = []
    img: list[Any] = []
    for c in corner_sets:
        if isinstance(c, tuple):
            points, pixels = c
        else:
            require(isinstance(board, Chessboard), "bare corners need a chessboard")
            assert isinstance(board, Chessboard)
            points, pixels = board.object_points, c
        require(pixels.shape == (points.shape[0], 2), "corner set shape", pixels.shape)
        obj.append(np.asarray(points, dtype=np.float32).reshape(-1, 1, 3))
        img.append(np.asarray(pixels, dtype=np.float32).reshape(-1, 1, 2))
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
        board=board_spec(board),
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
    camera_id: str, path: Path, board: Board, *, every: int = 10
) -> IntrinsicsRecord:
    """Find the board in sampled frames of a recording and calibrate."""
    corners: list[Array | tuple[Array, Array]] = []
    missed = 0
    size: tuple[int, int] | None = None
    for frame in frames_from_video(path, every=every):
        size = (int(frame.shape[1]), int(frame.shape[0]))
        found = find_board(frame, board)
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
