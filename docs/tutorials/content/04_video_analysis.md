# Tutorial 4: Video Analysis and Pose Estimation

**Estimated Time:** 75 minutes
**Difficulty:** Intermediate-Advanced

## Prerequisites

- Completed [Tutorial 2: Your First Simulation](02_first_simulation.md)
- MediaPipe installed (`pip install mediapipe`)
- OpenCV installed (`pip install opencv-python`)
- Sample video of a golf swing (or any human motion)

## Learning Objectives

By the end of this tutorial, you will:

- Process video to extract human pose landmarks
- Map video markers to biomechanical model bodies
- Drive a simulation using extracted motion data
- Assess fit quality and handle outliers

## Overview

The Golf Modeling Suite includes a video-to-simulation pipeline that:

1. Extracts 2D/3D pose landmarks from video using MediaPipe
2. Maps video markers to model body segments
3. Fits rigid body poses using least squares
4. Provides quality metrics for the registration

## Step 1: Understanding the Pose Estimation Pipeline

### Available Estimators

| Estimator     | Landmarks | Speed  | Accuracy  | 3D Support             |
| ------------- | --------- | ------ | --------- | ---------------------- |
| **MediaPipe** | 33 body   | Fast   | Good      | Yes (depth estimation) |
| **OpenPose**  | 25 body   | Medium | Excellent | With multi-view        |

### MediaPipe Landmark Mapping for Golf

MediaPipe provides 33 landmarks. For golf analysis, we focus on 11 key points:

```
Landmarks used for golf swing analysis:
- Shoulders: LEFT_SHOULDER (11), RIGHT_SHOULDER (12)
- Elbows: LEFT_ELBOW (13), RIGHT_ELBOW (14)
- Wrists: LEFT_WRIST (15), RIGHT_WRIST (16)
- Hips: LEFT_HIP (23), RIGHT_HIP (24)
- Knees: LEFT_KNEE (25), RIGHT_KNEE (26)
- Ankles: LEFT_ANKLE (27), RIGHT_ANKLE (28)
```

## Step 2: Processing Video with MediaPipe

Create a video processing script:

```python
"""Extract pose landmarks from golf swing video."""

from pathlib import Path
import numpy as np

from src.shared.python.video_pose_pipeline import (
    VideoPosePipeline,
    VideoProcessingConfig,
)


def process_golf_video(video_path: Path, output_dir: Path) -> dict:
    """Process a golf swing video and extract poses.

    Args:
        video_path: Path to input video file
        output_dir: Directory to save results

    Returns:
        Dictionary containing pose data and quality metrics
    """
    # Configure the pipeline
    config = VideoProcessingConfig(
        estimator_type="mediapipe",
        min_confidence=0.5,           # Minimum detection confidence
        enable_temporal_smoothing=True,  # Kalman filter for smoothness
        outlier_threshold=2.0,        # Z-score threshold for outlier detection
        batch_size=32,                # Frames per batch
        export_format="json",         # Output format
    )

    # Initialize pipeline
    pipeline = VideoPosePipeline(config)

    print(f"Processing video: {video_path}")
    print(f"Configuration: {config}")

    # Process the video
    result = pipeline.process_video(video_path, output_dir=output_dir)

    # Print summary
    print(f"\nProcessing complete!")
    print(f"  Frames processed: {result['num_frames']}")
    print(f"  Average confidence: {result['avg_confidence']:.2f}")
    print(f"  Outliers detected: {result['outliers_removed']}")
    print(f"  Output saved to: {output_dir}")

    return result


def main():
    # Paths
    video_path = Path("data/golf_swing_sample.mp4")
    output_dir = Path("output/pose_extraction")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Process video
    result = process_golf_video(video_path, output_dir)

    # Load extracted poses
    import json
    with open(output_dir / "poses.json") as f:
        poses = json.load(f)

    print(f"\nExtracted {len(poses['frames'])} frames of pose data")

    return poses


if __name__ == "__main__":
    poses = main()
```

## Step 3: Understanding Pose Data Structure

The extracted pose data follows this structure:

```python
{
    "metadata": {
        "video_file": "golf_swing_sample.mp4",
        "fps": 30.0,
        "width": 1920,
        "height": 1080,
        "estimator": "mediapipe",
        "processing_date": "2026-02-03T10:30:00"
    },
    "frames": [
        {
            "frame_idx": 0,
            "timestamp": 0.0,
            "landmarks": {
                "LEFT_SHOULDER": {"x": 0.45, "y": 0.32, "z": 0.1, "visibility": 0.95},
                "RIGHT_SHOULDER": {"x": 0.55, "y": 0.32, "z": 0.1, "visibility": 0.97},
                # ... other landmarks
            },
            "confidence": 0.89
        },
        # ... more frames
    ]
}
```

## Step 4: Mapping Video Markers to Model Bodies

The `MarkerToModelMapper` class registers video landmarks to biomechanical model segments:

```python
"""Map video pose landmarks to biomechanical model."""

import numpy as np
from pathlib import Path

from src.shared.python.marker_mapping import (
    MarkerToModelMapper,
    MarkerMapping,
    RegistrationResult,
)
from src.shared.python.engine_manager import EngineManager, EngineType


def setup_marker_mappings() -> list:
    """Define mappings from video landmarks to model bodies.

    Returns:
        List of MarkerMapping objects
    """
    # Each mapping defines:
    # - video_name: Name of landmark in pose data
    # - body_name: Name of body in physics model
    # - local_offset: Position of marker in body-local frame (meters)

    mappings = [
        # Upper body
        MarkerMapping(
            video_name="LEFT_SHOULDER",
            body_name="left_upper_arm",
            local_offset=np.array([0.0, 0.05, 0.0]),  # 5cm lateral
        ),
        MarkerMapping(
            video_name="RIGHT_SHOULDER",
            body_name="right_upper_arm",
            local_offset=np.array([0.0, -0.05, 0.0]),
        ),
        MarkerMapping(
            video_name="LEFT_ELBOW",
            body_name="left_forearm",
            local_offset=np.array([0.0, 0.0, 0.0]),  # At joint center
        ),
        MarkerMapping(
            video_name="RIGHT_ELBOW",
            body_name="right_forearm",
            local_offset=np.array([0.0, 0.0, 0.0]),
        ),
        MarkerMapping(
            video_name="LEFT_WRIST",
            body_name="left_hand",
            local_offset=np.array([0.0, 0.0, 0.0]),
        ),
        MarkerMapping(
            video_name="RIGHT_WRIST",
            body_name="right_hand",
            local_offset=np.array([0.0, 0.0, 0.0]),
        ),

        # Lower body
        MarkerMapping(
            video_name="LEFT_HIP",
            body_name="pelvis",
            local_offset=np.array([0.0, 0.1, 0.0]),  # Left hip joint
        ),
        MarkerMapping(
            video_name="RIGHT_HIP",
            body_name="pelvis",
            local_offset=np.array([0.0, -0.1, 0.0]),  # Right hip joint
        ),
        MarkerMapping(
            video_name="LEFT_KNEE",
            body_name="left_thigh",
            local_offset=np.array([0.0, 0.0, -0.4]),  # At knee
        ),
        MarkerMapping(
            video_name="RIGHT_KNEE",
            body_name="right_thigh",
            local_offset=np.array([0.0, 0.0, -0.4]),
        ),
    ]

    return mappings


def create_mapper(model_path: Path) -> MarkerToModelMapper:
    """Create and configure a marker mapper.

    Args:
        model_path: Path to biomechanical model

    Returns:
        Configured MarkerToModelMapper
    """
    # Load physics model
    project_root = Path(__file__).parent.parent
    engine_manager = EngineManager(project_root)
    engine_manager.switch_engine(EngineType.MUJOCO)
    physics_engine = engine_manager.get_active_physics_engine()
    physics_engine.load_from_path(str(model_path))

    # Create mapper
    mapper = MarkerToModelMapper(physics_engine.model)

    # Add mappings
    mappings = setup_marker_mappings()
    mapper.add_mappings(mappings)

    print(f"Mapper configured with {len(mappings)} marker mappings")

    return mapper, physics_engine
```

## Step 5: Fitting Poses to the Model

```python
def fit_frame_to_model(
    mapper: MarkerToModelMapper,
    frame_data: dict,
    physics_engine,
) -> RegistrationResult:
    """Fit a single frame's pose data to the model.

    Args:
        mapper: Configured MarkerToModelMapper
        frame_data: Single frame from pose extraction
        physics_engine: Physics engine with loaded model

    Returns:
        RegistrationResult with fit quality metrics
    """
    # Extract marker positions from frame
    landmarks = frame_data["landmarks"]

    # Convert to numpy array (only markers we have mappings for)
    marker_names = []
    marker_positions = []

    for mapping in mapper.mappings:
        if mapping.video_name in landmarks:
            lm = landmarks[mapping.video_name]
            # Scale from normalized coords to meters (assume 2m height reference)
            pos = np.array([lm["x"] * 2.0, lm["z"] * 2.0, (1.0 - lm["y"]) * 2.0])
            marker_names.append(mapping.video_name)
            marker_positions.append(pos)

    marker_positions = np.array(marker_positions)

    # Fit torso segment first (establishes global reference)
    torso_result = mapper.fit_segment_pose(
        segment_name="torso",
        marker_positions=marker_positions[:4],  # Shoulders and hips
    )

    # Fit arm segments
    left_arm_result = mapper.fit_segment_pose(
        segment_name="left_arm",
        marker_positions=marker_positions[[0, 2, 4]],  # L shoulder, elbow, wrist
    )

    right_arm_result = mapper.fit_segment_pose(
        segment_name="right_arm",
        marker_positions=marker_positions[[1, 3, 5]],
    )

    # Combine results
    overall_rms = np.sqrt(
        (torso_result.rms_error**2 + left_arm_result.rms_error**2 + right_arm_result.rms_error**2) / 3
    )

    return RegistrationResult(
        rms_error=overall_rms,
        max_error=max(torso_result.max_error, left_arm_result.max_error, right_arm_result.max_error),
        fit_quality=min(torso_result.fit_quality, left_arm_result.fit_quality, right_arm_result.fit_quality),
        outlier_indices=[],
        condition_number=torso_result.condition_number,
    )


def process_all_frames(
    mapper: MarkerToModelMapper,
    poses: dict,
    physics_engine,
) -> list:
    """Process all frames and collect fit results.

    Args:
        mapper: Configured MarkerToModelMapper
        poses: Full pose extraction result
        physics_engine: Physics engine instance

    Returns:
        List of RegistrationResult for each frame
    """
    results = []
    frames = poses["frames"]

    print(f"Fitting {len(frames)} frames to model...")

    for i, frame in enumerate(frames):
        result = fit_frame_to_model(mapper, frame, physics_engine)
        results.append(result)

        if (i + 1) % 100 == 0:
            avg_rms = np.mean([r.rms_error for r in results])
            print(f"  Processed {i + 1}/{len(frames)} frames (avg RMS: {avg_rms:.4f}m)")

    return results
```

## Step 6: Quality Assessment

```python
def assess_fit_quality(results: list) -> dict:
    """Assess overall fit quality across all frames.

    Args:
        results: List of RegistrationResult objects

    Returns:
        Quality assessment dictionary
    """
    rms_errors = [r.rms_error for r in results]
    max_errors = [r.max_error for r in results]
    qualities = [r.fit_quality for r in results]

    assessment = {
        "num_frames": len(results),
        "rms_error": {
            "mean": np.mean(rms_errors),
            "std": np.std(rms_errors),
            "min": np.min(rms_errors),
            "max": np.max(rms_errors),
        },
        "max_error": {
            "mean": np.mean(max_errors),
            "max": np.max(max_errors),
        },
        "fit_quality": {
            "mean": np.mean(qualities),
            "min": np.min(qualities),
        },
        "good_frames": sum(1 for q in qualities if q > 0.8),
        "acceptable_frames": sum(1 for q in qualities if q > 0.5),
        "poor_frames": sum(1 for q in qualities if q <= 0.5),
    }

    # Print summary
    print("\n" + "=" * 60)
    print("FIT QUALITY ASSESSMENT")
    print("=" * 60)
    print(f"Total frames: {assessment['num_frames']}")
    print(f"Mean RMS error: {assessment['rms_error']['mean']:.4f}m")
    print(f"Mean fit quality: {assessment['fit_quality']['mean']:.2f}")
    print(f"Good frames (>0.8): {assessment['good_frames']} ({100*assessment['good_frames']/len(results):.1f}%)")
    print(f"Acceptable (>0.5): {assessment['acceptable_frames']} ({100*assessment['acceptable_frames']/len(results):.1f}%)")
    print(f"Poor frames (<=0.5): {assessment['poor_frames']} ({100*assessment['poor_frames']/len(results):.1f}%)")

    return assessment
```

## Step 7: Complete Video-to-Simulation Pipeline

```python
"""Complete video analysis pipeline."""

from pathlib import Path
import numpy as np
import json

from src.shared.python.video_pose_pipeline import VideoPosePipeline, VideoProcessingConfig
from src.shared.python.marker_mapping import MarkerToModelMapper, MarkerMapping
from src.shared.python.engine_manager import EngineManager, EngineType


def main():
    # Paths
    project_root = Path(__file__).parent.parent
    video_path = Path("data/golf_swing_sample.mp4")
    model_path = project_root / "shared/models/mujoco/humanoid/humanoid.xml"
    output_dir = Path("output/video_analysis")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Step 1: Extract poses from video
    print("Step 1: Extracting poses from video...")
    config = VideoProcessingConfig(
        estimator_type="mediapipe",
        min_confidence=0.5,
        enable_temporal_smoothing=True,
    )
    pipeline = VideoPosePipeline(config)
    extraction_result = pipeline.process_video(video_path, output_dir=output_dir)

    # Load extracted poses
    with open(output_dir / "poses.json") as f:
        poses = json.load(f)

    # Step 2: Setup model and mapper
    print("\nStep 2: Setting up marker mapper...")
    engine_manager = EngineManager(project_root)
    engine_manager.switch_engine(EngineType.MUJOCO)
    physics_engine = engine_manager.get_active_physics_engine()
    physics_engine.load_from_path(str(model_path))

    mapper = MarkerToModelMapper(physics_engine.model)
    mapper.add_mappings(setup_marker_mappings())

    # Step 3: Fit all frames
    print("\nStep 3: Fitting poses to model...")
    fit_results = process_all_frames(mapper, poses, physics_engine)

    # Step 4: Assess quality
    print("\nStep 4: Assessing fit quality...")
    assessment = assess_fit_quality(fit_results)

    # Step 5: Export results
    print("\nStep 5: Exporting results...")
    with open(output_dir / "fit_assessment.json", "w") as f:
        json.dump(assessment, f, indent=2)

    print(f"\nPipeline complete! Results saved to {output_dir}/")

    return assessment


if __name__ == "__main__":
    assessment = main()
```

## Troubleshooting

### Low Detection Confidence

- Ensure good lighting in video
- Use higher resolution video (1080p+)
- Subject should be fully visible in frame
- Avoid loose/baggy clothing

### High RMS Fit Error

- Check marker mapping local offsets match your model
- Verify video calibration (camera intrinsics)
- Consider multi-view setup for better 3D accuracy

### Temporal Jitter in Results

- Increase `outlier_threshold` to filter more aggressively
- Enable temporal smoothing (Kalman filter)
- Use higher frame rate video (60fps+)

### Missing Landmarks

- Some poses may occlude landmarks (e.g., wrist during backswing)
- Use `visibility` score to filter unreliable detections
- Consider adding redundant markers for robustness

## Next Steps

- [API Reference: VideoPosePipeline](../../api/video_pose_pipeline.md)
- [API Reference: MarkerToModelMapper](../../api/marker_mapping.md)
- [Guideline A2: Marker Mapping Protocol](../../guidelines/A2_marker_mapping.md)
- [Example: Multi-View Reconstruction](../examples/multi_view_reconstruction.md)
