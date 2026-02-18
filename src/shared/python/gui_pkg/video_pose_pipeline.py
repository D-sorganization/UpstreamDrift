"""Enhanced video-based pose estimation pipeline.

This module provides a complete pipeline for processing golf swing videos:
- Multi-estimator support (MediaPipe, OpenPose)
- Automatic URDF parameter fitting
- Integration with existing marker mapping
- Batch processing capabilities
- Quality assessment and filtering
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from src.shared.python.core.contracts import StateError
from src.shared.python.data_io.io_utils import ensure_directory
from src.shared.python.data_io.marker_mapping import (
    MarkerToModelMapper,
    RegistrationResult,
)
from src.shared.python.data_io.output_manager import OutputManager
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.pose_estimation.interface import (
    PoseEstimationResult,
    PoseEstimator,
)

logger = get_logger(__name__)


@dataclass
class VideoProcessingConfig:
    """Configuration for video processing pipeline."""

    # Estimator settings
    estimator_type: str = "mediapipe"  # "mediapipe" or "openpose"
    min_confidence: float = 0.5
    enable_temporal_smoothing: bool = True

    # Quality filtering
    min_frame_confidence: float = 0.3
    outlier_detection: bool = True
    outlier_threshold: float = 2.0  # Standard deviations

    # Processing settings
    batch_size: int = 32
    max_frames: int | None = None  # None for entire video
    skip_frames: int = 0  # Process every Nth frame

    # Output settings
    export_keypoints: bool = True
    export_joint_angles: bool = True
    export_quality_metrics: bool = True
    output_format: str = "json"  # "json", "csv", "hdf5"


@dataclass
class VideoProcessingResult:
    """Result from video processing pipeline."""

    video_path: Path
    total_frames: int
    processed_frames: int
    valid_frames: int
    average_confidence: float
    pose_results: list[PoseEstimationResult]
    quality_metrics: dict[str, Any]
    registration_result: RegistrationResult | None = None


class VideoPosePipeline:
    """Complete pipeline for video-based pose estimation and model fitting."""

    def __init__(self, config: VideoProcessingConfig | None = None) -> None:
        """Initialize the video processing pipeline.

        Args:
            config: Processing configuration, uses defaults if None
        """
        self.config = config or VideoProcessingConfig()
        self.estimator: PoseEstimator | None = None
        self.mapper: MarkerToModelMapper | None = None
        self.output_manager = OutputManager()

        self._load_estimator()

    def _load_estimator(self) -> None:
        """Load the specified pose estimator."""
        try:
            if self.config.estimator_type == "mediapipe":
                from shared.python.pose_estimation.mediapipe_estimator import (
                    MediaPipeEstimator,
                )

                self.estimator = MediaPipeEstimator(
                    min_detection_confidence=self.config.min_confidence,
                    min_tracking_confidence=self.config.min_confidence,
                    enable_temporal_smoothing=self.config.enable_temporal_smoothing,
                )
            elif self.config.estimator_type == "openpose":
                from shared.python.pose_estimation.openpose_estimator import (
                    OpenPoseEstimator,
                )

                self.estimator = OpenPoseEstimator()
            else:
                raise ValueError(
                    f"Unknown estimator type: {self.config.estimator_type}"
                )

            if self.estimator is not None:
                self.estimator.load_model()
            logger.info(f"Loaded {self.config.estimator_type} estimator")

        except ImportError as e:
            logger.error(f"Failed to load estimator: {e}")
            raise

    def process_video(
        self, video_path: Path, output_dir: Path | None = None
    ) -> VideoProcessingResult:
        """Process a single video file.

        Args:
            video_path: Path to input video
            output_dir: Directory for output files (optional)

        Returns:
            VideoProcessingResult with pose estimates and quality metrics
        """
        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        if self.estimator is None:
            raise StateError("Estimator not loaded")

        logger.info(f"Processing video: {video_path}")

        # Get video info
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        # Process frames
        pose_results = []
        if self.config.max_frames:
            max_frames = min(self.config.max_frames, total_frames)
        else:
            max_frames = total_frames

        # Use estimator's batch processing if available
        if hasattr(self.estimator, "estimate_from_video"):
            pose_results = self.estimator.estimate_from_video(video_path)
        else:
            # Fallback to frame-by-frame processing
            pose_results = self._process_frames_individually(video_path, max_frames)

        # Apply quality filtering
        filtered_results = self._filter_by_quality(pose_results)

        # Calculate quality metrics
        quality_metrics = self._calculate_quality_metrics(
            pose_results, filtered_results
        )

        # Create result
        result = VideoProcessingResult(
            video_path=video_path,
            total_frames=total_frames,
            processed_frames=len(pose_results),
            valid_frames=len(filtered_results),
            average_confidence=quality_metrics["average_confidence"],
            pose_results=filtered_results,
            quality_metrics=quality_metrics,
        )

        # Export results if output directory specified
        if output_dir:
            self._export_results(result, output_dir)

        logger.info(
            f"Video processing complete: {len(filtered_results)}/{total_frames} valid frames"
        )
        return result

    def process_batch(
        self, video_paths: list[Path], output_dir: Path
    ) -> list[VideoProcessingResult]:
        """Process multiple videos in batch.

        Args:
            video_paths: List of video file paths
            output_dir: Directory for output files

        Returns:
            List of VideoProcessingResult objects
        """
        results = []

        for i, video_path in enumerate(video_paths):
            logger.info(
                f"Processing batch {i + 1}/{len(video_paths)}: {video_path.name}"
            )

            try:
                result = self.process_video(video_path, output_dir)
                results.append(result)
            except (RuntimeError, ValueError, OSError) as e:
                logger.error(f"Failed to process {video_path}: {e}")
                continue

        # Generate batch summary
        self._export_batch_summary(results, output_dir)

        return results

    def fit_to_model(
        self, pose_results: list[PoseEstimationResult], model_path: Path
    ) -> RegistrationResult:
        """Fit pose estimates to a biomechanical model.

        Implements Issue #754: End-to-end fitting with A3 pipeline.

        Args:
            pose_results: List of pose estimation results
            model_path: Path to URDF/XML model file

        Returns:
            Registration result with fitted parameters
        """
        from src.shared.python.data_io.marker_mapping import RegistrationResult

        # Convert pose keypoints to marker format
        marker_positions, marker_names, timestamps = self._convert_poses_to_markers(
            pose_results
        )

        if len(marker_positions) == 0:
            logger.warning("No marker data extracted from poses")
            return RegistrationResult(
                success=False,
                transformation=np.eye(4),
                residuals=np.array([]),
                rms_error=float("inf"),
                max_error=float("inf"),
                outlier_indices=[],
                fit_quality=0.0,
                num_markers_used=0,
                condition_number=float("inf"),
            )

        # Initialize A3 fitting pipeline
        from src.shared.python.validation_pkg.data_fitting import A3FittingPipeline

        pipeline = A3FittingPipeline()

        # Default subject mass if not known
        default_mass = 75.0  # kg

        # Fit parameters
        try:
            report = pipeline.fit_from_markers(
                marker_positions,
                marker_names,
                timestamps,
                subject_mass=default_mass,
                subject_id=model_path.stem,
            )

            # Convert A3 result to RegistrationResult for compatibility
            return RegistrationResult(
                success=report.fit_result.success,
                transformation=np.eye(4),  # A3 doesn't compute rigid transform
                residuals=report.fit_result.residuals,
                rms_error=report.fit_result.rms_error,
                max_error=(
                    float(np.max(np.abs(report.fit_result.residuals)))
                    if len(report.fit_result.residuals) > 0
                    else 0.0
                ),
                outlier_indices=[],
                fit_quality=report.fit_result.r_squared,
                num_markers_used=len(marker_names),
                condition_number=report.fit_result.condition_number,
            )
        except (ValueError, TypeError, RuntimeError) as e:
            logger.error(f"A3 fitting failed: {e}")
            return RegistrationResult(
                success=False,
                transformation=np.eye(4),
                residuals=np.array([]),
                rms_error=float("inf"),
                max_error=float("inf"),
                outlier_indices=[],
                fit_quality=0.0,
                num_markers_used=0,
                condition_number=float("inf"),
            )

    def _process_frames_individually(
        self, video_path: Path, max_frames: int
    ) -> list[PoseEstimationResult]:
        """Process video frame by frame (fallback method)."""
        results = []
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)

        frame_idx = 0
        processed_count = 0

        while frame_idx < max_frames:
            ret, frame = cap.read()
            if not ret:
                break

            # Skip frames if configured
            if frame_idx % (self.config.skip_frames + 1) != 0:
                frame_idx += 1
                continue

            # Process frame
            try:
                if self.estimator is not None:
                    result = self.estimator.estimate_from_image(frame)
                    result.timestamp = frame_idx / fps
                    results.append(result)
                    processed_count += 1

                    if processed_count % 100 == 0:
                        logger.info(f"Processed {processed_count} frames")
                else:
                    logger.error("Estimator not initialized")
                    break

            except (RuntimeError, ValueError, OSError) as e:
                logger.warning(f"Failed to process frame {frame_idx}: {e}")

            frame_idx += 1

        cap.release()
        return results

    def _filter_by_quality(
        self, pose_results: list[PoseEstimationResult]
    ) -> list[PoseEstimationResult]:
        """Filter pose results by quality metrics."""
        if not self.config.outlier_detection:
            return [
                r
                for r in pose_results
                if r.confidence >= self.config.min_frame_confidence
            ]

        # Filter by confidence first
        confident_results = [
            r for r in pose_results if r.confidence >= self.config.min_frame_confidence
        ]

        if len(confident_results) < 3:
            return confident_results

        # Outlier detection based on joint angle consistency
        filtered_results = []

        for result in confident_results:
            if self._is_outlier(result, confident_results):
                continue
            filtered_results.append(result)

        return filtered_results

    def _is_outlier(
        self, result: PoseEstimationResult, all_results: list[PoseEstimationResult]
    ) -> bool:
        """Check if a pose result is an outlier."""
        # Simple outlier detection based on joint angle deviations
        # Can be enhanced with more sophisticated methods

        if not result.joint_angles:
            return True

        # Calculate mean and std for each joint angle
        joint_stats: dict[str, list[float]] = {}
        for other_result in all_results:
            for joint_name, angle in other_result.joint_angles.items():
                if joint_name not in joint_stats:
                    joint_stats[joint_name] = []
                joint_stats[joint_name].append(angle)

        # Check if any joint angle is an outlier
        for joint_name, angle in result.joint_angles.items():
            if joint_name in joint_stats and len(joint_stats[joint_name]) > 2:
                angles = np.array(joint_stats[joint_name])
                mean_angle = np.mean(angles)
                std_angle = np.std(angles)

                if std_angle > 0:
                    z_score = abs(angle - mean_angle) / std_angle
                    if z_score > self.config.outlier_threshold:
                        return True

        return False

    def _calculate_quality_metrics(
        self,
        all_results: list[PoseEstimationResult],
        filtered_results: list[PoseEstimationResult],
    ) -> dict[str, Any]:
        """Calculate quality metrics for the processing session."""
        if not all_results:
            return {"average_confidence": 0.0, "valid_frame_ratio": 0.0}

        confidences = [r.confidence for r in all_results]

        metrics = {
            "average_confidence": float(np.mean(confidences)),
            "min_confidence": float(np.min(confidences)),
            "max_confidence": float(np.max(confidences)),
            "std_confidence": float(np.std(confidences)),
            "valid_frame_ratio": len(filtered_results) / len(all_results),
            "total_frames": len(all_results),
            "valid_frames": len(filtered_results),
            "filtered_frames": len(all_results) - len(filtered_results),
        }

        return metrics

    def _convert_poses_to_markers(
        self, pose_results: list[PoseEstimationResult]
    ) -> tuple[np.ndarray, list[str], np.ndarray]:
        """Convert pose keypoints to marker format for model fitting.

        Implements Issue #754: Complete marker conversion for A3 pipeline.

        Args:
            pose_results: List of pose estimation results

        Returns:
            Tuple of (marker_positions [frames x markers x 3],
                      marker_names [M],
                      timestamps [frames]).
        """
        from src.shared.python.validation_pkg.data_fitting import (
            convert_poses_to_markers,
        )

        if not pose_results:
            return np.array([]), [], np.array([])

        # Get keypoint names from first valid result
        first_valid = next(
            (r for r in pose_results if r.raw_keypoints is not None),
            None,
        )

        if first_valid is None or first_valid.raw_keypoints is None:
            return np.array([]), [], np.array([])

        # Infer keypoint names from raw_keypoints structure
        # Assuming raw_keypoints is dict[str, tuple[x, y, confidence]]
        keypoint_names = list(first_valid.raw_keypoints.keys())

        # Convert each frame
        all_positions = []
        timestamps = []
        marker_names: list[str] | None = None

        for result in pose_results:
            if result.raw_keypoints is None:
                continue

            # Extract positions from keypoints dict
            positions = []
            for name in keypoint_names:
                if name in result.raw_keypoints:
                    kp = result.raw_keypoints[name]
                    if isinstance(kp, (list, tuple)) and len(kp) >= 2:
                        positions.append([kp[0], kp[1], kp[2] if len(kp) > 2 else 0.0])
                    else:
                        positions.append([0.0, 0.0, 0.0])
                else:
                    positions.append([0.0, 0.0, 0.0])

            # Convert to marker format
            markers, names = convert_poses_to_markers(
                np.array(positions),
                keypoint_names,
            )

            if marker_names is None:
                marker_names = names

            all_positions.append(markers)
            timestamps.append(result.timestamp)

        if not all_positions:
            return np.array([]), [], np.array([])

        return (
            np.array(all_positions),
            marker_names or [],
            np.array(timestamps),
        )

    def _export_results(self, result: VideoProcessingResult, output_dir: Path) -> None:
        """Export processing results to files."""
        ensure_directory(output_dir)

        base_name = result.video_path.stem

        # Export pose results
        if self.config.export_keypoints or self.config.export_joint_angles:
            data_to_export: dict[str, Any] = {
                "video_info": {
                    "path": str(result.video_path),
                    "total_frames": result.total_frames,
                    "processed_frames": result.processed_frames,
                    "valid_frames": result.valid_frames,
                },
                "quality_metrics": result.quality_metrics,
                "poses": [],
            }

            for pose_result in result.pose_results:
                pose_data: dict[str, Any] = {
                    "timestamp": pose_result.timestamp,
                    "confidence": pose_result.confidence,
                }

                if self.config.export_keypoints and pose_result.raw_keypoints:
                    pose_data["keypoints"] = pose_result.raw_keypoints

                if self.config.export_joint_angles:
                    pose_data["joint_angles"] = pose_result.joint_angles

                data_to_export["poses"].append(pose_data)

            # Export using output manager
            output_path = output_dir / f"{base_name}_poses.{self.config.output_format}"
            self.output_manager.save_simulation_results(
                data_to_export, str(output_path)
            )

        # Export quality metrics separately
        if self.config.export_quality_metrics:
            metrics_path = output_dir / f"{base_name}_quality.json"
            self.output_manager.save_simulation_results(
                result.quality_metrics, str(metrics_path)
            )

    def _export_batch_summary(
        self, results: list[VideoProcessingResult], output_dir: Path
    ) -> None:
        """Export summary of batch processing results."""
        summary = {
            "batch_info": {
                "total_videos": len(results),
                "successful_videos": len([r for r in results if r.valid_frames > 0]),
            },
            "aggregate_metrics": {
                "total_frames": sum(r.total_frames for r in results),
                "total_valid_frames": sum(r.valid_frames for r in results),
                "average_confidence": float(
                    np.mean([r.average_confidence for r in results])
                ),
                "average_valid_ratio": float(
                    np.mean(
                        [
                            r.valid_frames / r.total_frames
                            for r in results
                            if r.total_frames > 0
                        ]
                    )
                ),
            },
            "per_video_summary": [
                {
                    "video": r.video_path.name,
                    "valid_frames": r.valid_frames,
                    "total_frames": r.total_frames,
                    "confidence": r.average_confidence,
                }
                for r in results
            ],
        }

        summary_path = output_dir / "batch_summary.json"
        self.output_manager.save_simulation_results(summary, str(summary_path))
        logger.info(f"Batch summary exported to: {summary_path}")
