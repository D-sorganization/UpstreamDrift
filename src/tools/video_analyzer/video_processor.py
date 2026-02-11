"""
Video Processor for Golf Swing Analysis.

Handles video file loading, frame extraction, and processing
pipeline for swing analysis.
"""

import logging
from collections.abc import Callable, Generator
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .pose_estimator import PoseEstimator
from .types import PoseFrame

logger = logging.getLogger(__name__)


class VideoProcessor:
    """
    Video processing utility for golf swing analysis.

    Handles:
    - Video file loading and validation
    - Frame extraction and iteration
    - FPS detection and timing
    - Frame export functionality

    Usage:
        processor = VideoProcessor("swing.mp4")
        for frame, timestamp in processor.iterate_frames():
            # Process each frame
            pass
    """

    SUPPORTED_FORMATS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}

    def __init__(self, video_path: str | None = None) -> None:
        """
        Initialize the video processor.

        Args:
            video_path: Path to the video file (optional, can be set later).
        """
        self.video_path: Path | None = None
        self._cap: cv2.VideoCapture | None = None
        self._fps: float = 30.0
        self._frame_count: int = 0
        self._width: int = 0
        self._height: int = 0
        self._duration: float = 0.0

        if video_path:
            self.load(video_path)

    def load(self, video_path: str) -> bool:
        """
        Load a video file.

        Args:
            video_path: Path to the video file.

        Returns:
            True if video loaded successfully, False otherwise.
        """
        path = Path(video_path)

        if not path.exists():
            logger.error(f"Video file not found: {video_path}")
            return False

        if path.suffix.lower() not in self.SUPPORTED_FORMATS:
            logger.error(f"Unsupported video format: {path.suffix}")
            return False

        # Close any existing video
        self.close()

        # Open the video
        self._cap = cv2.VideoCapture(str(path))

        if not self._cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return False

        # Extract video properties
        self.video_path = path
        self._fps = self._cap.get(cv2.CAP_PROP_FPS)
        self._frame_count = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self._height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self._duration = self._frame_count / self._fps if self._fps > 0 else 0

        logger.info(
            f"Loaded video: {path.name} "
            f"({self._width}x{self._height}, {self._fps:.2f} fps, "
            f"{self._frame_count} frames, {self._duration:.2f}s)"
        )

        return True

    @property
    def fps(self) -> float:
        """Video frames per second."""
        return self._fps

    @property
    def frame_count(self) -> int:
        """Total number of frames."""
        return self._frame_count

    @property
    def width(self) -> int:
        """Video width in pixels."""
        return self._width

    @property
    def height(self) -> int:
        """Video height in pixels."""
        return self._height

    @property
    def duration(self) -> float:
        """Video duration in seconds."""
        return self._duration

    @property
    def is_loaded(self) -> bool:
        """Check if a video is currently loaded."""
        return self._cap is not None and self._cap.isOpened()

    def get_frame(self, frame_number: int) -> np.ndarray | None:
        """
        Get a specific frame by number.

        Args:
            frame_number: Zero-indexed frame number.

        Returns:
            BGR image frame, or None if frame not available.
        """
        if not self.is_loaded:
            return None

        if frame_number < 0 or frame_number >= self._frame_count:
            return None

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)  # type: ignore[union-attr]
        ret, frame = self._cap.read()  # type: ignore[union-attr]

        return frame if ret else None

    def get_frame_at_time(self, time_ms: float) -> np.ndarray | None:
        """
        Get frame at a specific timestamp.

        Args:
            time_ms: Timestamp in milliseconds.

        Returns:
            BGR image frame, or None if not available.
        """
        frame_number = int((time_ms / 1000) * self._fps)
        return self.get_frame(frame_number)

    def iterate_frames(
        self,
        start_frame: int = 0,
        end_frame: int | None = None,
        step: int = 1,
    ) -> Generator[tuple[np.ndarray, int, float], None, None]:
        """
        Iterate through video frames.

        Args:
            start_frame: Starting frame number.
            end_frame: Ending frame number (exclusive). None = end of video.
            step: Frame step (1 = every frame, 2 = every other frame, etc.)

        Yields:
            Tuple of (frame, frame_number, timestamp_ms)
        """
        if not self.is_loaded:
            return

        end = end_frame if end_frame is not None else self._frame_count
        end = min(end, self._frame_count)

        self._cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # type: ignore[union-attr]

        for frame_num in range(start_frame, end, step):
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)  # type: ignore[union-attr]
            ret, frame = self._cap.read()  # type: ignore[union-attr]

            if not ret:
                break

            timestamp = (frame_num / self._fps) * 1000  # Convert to ms
            yield frame, frame_num, timestamp

    def extract_poses(
        self,
        estimator: PoseEstimator | None = None,
        start_frame: int = 0,
        end_frame: int | None = None,
        step: int = 1,
        min_confidence: float = 0.5,
        progress_callback: Callable[[int, int], None] | None = None,
    ) -> list[PoseFrame]:
        """
        Extract pose data from all frames.

        Args:
            estimator: PoseEstimator instance (will create one if not provided).
            start_frame: Starting frame number.
            end_frame: Ending frame number.
            step: Frame step for processing.
            min_confidence: Minimum pose confidence threshold.
            progress_callback: Optional callback(current, total) for progress.

        Returns:
            List of PoseFrame objects with valid poses.
        """
        if not self.is_loaded:
            return []

        own_estimator = estimator is None
        if own_estimator:
            estimator = PoseEstimator()
            estimator.initialize()

        poses = []
        total_frames = ((end_frame or self._frame_count) - start_frame) // step

        try:
            for i, (frame, frame_num, timestamp) in enumerate(
                self.iterate_frames(start_frame, end_frame, step)
            ):
                pose = estimator.process_frame(frame, frame_num, timestamp)  # type: ignore[union-attr]

                if pose and pose.confidence >= min_confidence:
                    poses.append(pose)

                if progress_callback:
                    progress_callback(i + 1, total_frames)

        finally:
            if own_estimator and estimator is not None:
                estimator.close()

        logger.info(f"Extracted {len(poses)} valid poses from {total_frames} frames")
        return poses

    def export_frame(
        self,
        frame_number: int,
        output_path: str,
        quality: int = 95,
    ) -> bool:
        """
        Export a single frame as an image.

        Args:
            frame_number: Frame to export.
            output_path: Output file path (jpg, png, etc.)
            quality: JPEG quality (0-100).

        Returns:
            True if export successful.
        """
        frame = self.get_frame(frame_number)
        if frame is None:
            return False

        try:
            if output_path.lower().endswith(".jpg"):
                cv2.imwrite(output_path, frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
            else:
                cv2.imwrite(output_path, frame)
            return True
        except (PermissionError, OSError) as e:
            logger.error(f"Failed to export frame: {e}")
            return False

    def export_clip(
        self,
        output_path: str,
        start_frame: int = 0,
        end_frame: int | None = None,
        codec: str = "mp4v",
    ) -> bool:
        """
        Export a video clip.

        Args:
            output_path: Output video file path.
            start_frame: Starting frame.
            end_frame: Ending frame.
            codec: Video codec (e.g., 'mp4v', 'avc1', 'XVID').

        Returns:
            True if export successful.
        """
        if not self.is_loaded:
            return False

        try:
            fourcc = cv2.VideoWriter_fourcc(*codec)  # type: ignore[attr-defined]
            out = cv2.VideoWriter(
                output_path,
                fourcc,
                self._fps,
                (self._width, self._height),
            )

            for frame, _, _ in self.iterate_frames(start_frame, end_frame):
                out.write(frame)

            out.release()
            logger.info(f"Exported clip to: {output_path}")
            return True

        except (PermissionError, OSError) as e:
            logger.error(f"Failed to export clip: {e}")
            return False

    def close(self) -> None:
        """Release video resources."""
        if self._cap:
            self._cap.release()
            self._cap = None
            self.video_path = None

    def __enter__(self) -> Any:
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """Context manager exit."""
        self.close()
        return False

    def __len__(self) -> int:
        """Return frame count."""
        return self._frame_count

    def __getitem__(self, frame_number: int) -> np.ndarray | None:
        """Get frame by index."""
        return self.get_frame(frame_number)
