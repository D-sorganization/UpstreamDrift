"""RTMPose body pose through :mod:`onnxruntime` (SimCC decode).

RTMPose (OpenMMLab, Apache-2.0) is a top-down SimCC regressor; this estimator
runs the pinned ONNX export (#9648, see :mod:`.rtmpose_models`) on the *whole
frame* letterboxed to the model input, so it behaves like the single-person
comparison detectors — per-person crops need a detector (follow-up). It is
registered as ``rtmpose_onnx`` with ``capture_source=False`` until it is
qualified against real takes.

``onnxruntime`` is an optional dependency: it is imported only inside the
session factory (never at module import time), and the registry gates
availability on it.

Output contract matches :class:`~.openpose_dnn_estimator.OpenPoseDnnEstimator`:
``raw_keypoints`` are normalized ``(x, y)`` in ``[0, 1]`` keyed by
:attr:`LANDMARK_MAP` names (COCO-17 by default, Halpe-26 via
``keypoint_set="halpe26"``), ``raw_confidences`` are the SimCC peaks (clipped
to ``[0, 1]``; the raw SimCC value is not a calibrated probability), and
``confidence`` is their mean over detected joints. ``joint_angles`` is empty.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import numpy as np
from src.shared.python.pose_estimation.interface import (
    PoseEstimationResult,
    PoseEstimator,
    detection_result,
    estimate_video_frames,
)

from src.shared.python.core.contracts import StateError, require
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.pose_estimation.rtmpose_models import resolve_rtmpose

logger = get_logger(__name__)

SessionFactory = Callable[[Path], Any]

SIMCC_SPLIT_RATIO = 2.0
DEFAULT_MIN_SCORE = 0.3
INPUT_WIDTH = 192
INPUT_HEIGHT = 256
# mmpose RTMPose preprocessing (ImageNet-style, applied to an RGB frame).
PIXEL_MEAN_RGB = (123.675, 116.28, 103.53)
PIXEL_STD_RGB = (58.395, 57.12, 57.375)

COCO17_KEYPOINTS: tuple[str, ...] = (
    "nose",
    "left_eye",
    "right_eye",
    "left_ear",
    "right_ear",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
HALPE26_EXTRA: tuple[str, ...] = (
    "head",
    "neck",
    "hip",
    "left_big_toe",
    "right_big_toe",
    "left_small_toe",
    "right_small_toe",
    "left_heel",
    "right_heel",
)

LAYOUTS: dict[str, tuple[str, tuple[str, ...]]] = {
    "coco17": ("rtmpose_coco17", COCO17_KEYPOINTS),
    "halpe26": ("rtmpose_halpe26", COCO17_KEYPOINTS + HALPE26_EXTRA),
}


def _ort_session(model_path: Path) -> Any:
    """Create an ``onnxruntime`` CPU session; the only import of the optional dep."""
    try:
        import onnxruntime
    except ImportError as err:
        raise ImportError(
            "onnxruntime is required for the rtmpose_onnx estimator; "
            "install it with `pip install onnxruntime` (extra `pose-onnx`)"
        ) from err
    return onnxruntime.InferenceSession(
        str(model_path), providers=["CPUExecutionProvider"]
    )


class RtmposeOnnxEstimator(PoseEstimator):
    """COCO-17 or Halpe-26 keypoints from a pinned RTMPose ONNX model."""

    LAYOUT_NAME: ClassVar[str] = LAYOUTS["coco17"][0]
    LANDMARK_MAP: ClassVar[dict[int, str]] = dict(enumerate(LAYOUTS["coco17"][1]))

    def __init__(
        self,
        *,
        keypoint_set: str = "coco17",
        min_score: float = DEFAULT_MIN_SCORE,
        session_factory: SessionFactory = _ort_session,
    ) -> None:
        require(keypoint_set in LAYOUTS, "unknown keypoint_set", keypoint_set)
        require(0.0 <= min_score < 1.0, "min_score must be in [0, 1)", min_score)
        self.keypoint_set = keypoint_set
        self.min_score = min_score
        layout_name, keypoints = LAYOUTS[keypoint_set]
        # Instance-level so variant estimators name their own layout
        # (RegisteredFrameEstimator reads instance attributes first).
        self.LAYOUT_NAME = layout_name
        self.LANDMARK_MAP = dict(enumerate(keypoints))
        self.model_path: Path | None = None
        self._session_factory = session_factory
        self._session: Any = None

    def load_model(self, model_path: Path | None = None) -> None:
        """Load the ONNX session; ``model_path`` is a cache dir or an ``.onnx`` file."""
        if model_path is not None and model_path.is_file():
            onnx_path = model_path
        else:
            onnx_path = resolve_rtmpose(self.keypoint_set, model_path)
        self._session = self._session_factory(onnx_path)
        self.model_path = onnx_path
        logger.info("RTMPose %s loaded from %s", self.keypoint_set, onnx_path)

    def is_ready(self) -> bool:
        return self._session is not None

    def close(self) -> None:
        self._session = None

    def estimate_from_video(self, video_path: Path) -> list[PoseEstimationResult]:
        """Run every frame of a video file; timestamps come from the frame rate."""
        if self._session is None:
            raise StateError("load_model() must be called before estimation")
        return estimate_video_frames(video_path, self.estimate_from_image)

    def estimate_from_image(
        self, image: np.ndarray, timestamp_ms: int | None = None
    ) -> PoseEstimationResult:
        """Run the model on one BGR frame and decode the SimCC peaks."""
        if self._session is None:
            raise StateError("load_model() must be called before estimation")
        require(
            image.ndim == 3 and image.shape[2] == 3, "image must be HxWx3", image.shape
        )
        height, width = image.shape[:2]
        blob, scale, pad_x, pad_y = _letterbox_blob(image, INPUT_WIDTH, INPUT_HEIGHT)
        input_name = self._session.get_inputs()[0].name
        outputs = [np.asarray(o) for o in self._session.run(None, {input_name: blob})]
        simcc_x, simcc_y = _split_simcc(outputs, round(INPUT_WIDTH * SIMCC_SPLIT_RATIO))
        keypoints, confidences = self._decode(
            simcc_x, simcc_y, scale, pad_x, pad_y, width, height
        )
        return detection_result(keypoints, confidences, self.min_score, timestamp_ms)

    def _decode(
        self,
        simcc_x: np.ndarray,
        simcc_y: np.ndarray,
        scale: float,
        pad_x: int,
        pad_y: int,
        width: int,
        height: int,
    ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Normalized keypoints and confidences from the two SimCC tensors.

        Precondition: ``simcc_x``/``simcc_y`` have one landmark channel per
        :attr:`LANDMARK_MAP` entry. Joints whose SimCC peak is below
        :attr:`min_score` are omitted from the keypoints but kept (with their
        low value) in the confidences so the caller sees why they are absent.
        """
        count = len(self.LANDMARK_MAP)
        require(
            simcc_x.shape[0] == count and simcc_y.shape[0] == count,
            "SimCC channel count must match the landmark set",
            (simcc_x.shape, simcc_y.shape),
        )
        x_locs = np.argmax(simcc_x, axis=1) / SIMCC_SPLIT_RATIO
        y_locs = np.argmax(simcc_y, axis=1) / SIMCC_SPLIT_RATIO
        values = np.maximum(np.max(simcc_x, axis=1), np.max(simcc_y, axis=1))
        keypoints: dict[str, np.ndarray] = {}
        confidences: dict[str, float] = {}
        for idx, name in self.LANDMARK_MAP.items():
            peak = float(values[idx])
            confidences[name] = max(0.0, min(1.0, peak))
            if peak >= self.min_score:
                x_px = (float(x_locs[idx]) - pad_x) / scale
                y_px = (float(y_locs[idx]) - pad_y) / scale
                keypoints[name] = np.array([x_px / width, y_px / height], dtype=float)
        return keypoints, confidences


def _letterbox_blob(
    image: np.ndarray, input_w: int, input_h: int
) -> tuple[np.ndarray, float, int, int]:
    """BGR frame -> ``(1, 3, input_h, input_w)`` float32 blob plus geometry.

    Returns the blob and the letterbox ``(scale, pad_x, pad_y)`` that maps
    model-input coordinates back to frame pixels.
    """
    import cv2

    height, width = image.shape[:2]
    scale = min(input_w / width, input_h / height)
    resized = cv2.resize(image, (int(round(width * scale)), int(round(height * scale))))
    rh, rw = resized.shape[:2]
    pad_x = (input_w - rw) // 2
    pad_y = (input_h - rh) // 2
    canvas = np.zeros((input_h, input_w, 3), dtype=np.float32)
    canvas[pad_y : pad_y + rh, pad_x : pad_x + rw] = resized
    rgb = canvas[..., ::-1]
    normalized = (rgb - np.asarray(PIXEL_MEAN_RGB, dtype=np.float32)) / np.asarray(
        PIXEL_STD_RGB, dtype=np.float32
    )
    blob = np.transpose(normalized, (2, 0, 1))[None].astype(np.float32)
    return blob, scale, pad_x, pad_y


def _split_simcc(
    outputs: list[np.ndarray], x_width: int
) -> tuple[np.ndarray, np.ndarray]:
    """Order the session outputs as ``(simcc_x, simcc_y)`` by split width.

    Precondition: exactly two tensors, whose trailing dimensions are the x
    and y split-hat widths (the y width is inferred from the x width and the
    input aspect ratio). Batch and keypoint dimensions are squeezed.
    """
    require(len(outputs) == 2, "expected two SimCC output tensors", len(outputs))
    first, second = (np.asarray(o)[0] for o in outputs)
    if first.shape[-1] == x_width:
        return first, second
    require(second.shape[-1] == x_width, "no SimCC x output of split width", x_width)
    return second, first
