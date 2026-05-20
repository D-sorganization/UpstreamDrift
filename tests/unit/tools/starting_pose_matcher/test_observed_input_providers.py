"""Tests for observed-input providers (OpenPose and MediaPipe)."""

import pytest
import json

# =============================================================================
# OpenPose Provider Tests
# =============================================================================


def test_openpose_import():
    """Test that OpenPose provider can be imported."""
    from src.tools.starting_pose_matcher.skeleton_extractors.openpose import (
        OpenPoseProvider,
        OpenPoseProviderError,
        create_provider,
        OPENPOSE_COCO_INDICES,
        OPENPOSE_TO_MATCHER_VOCAB,
    )

    assert hasattr(OpenPoseProvider, "get_skeleton")
    assert hasattr(OpenPoseProvider, "get_confidence_map")
    assert hasattr(OpenPoseProvider, "get_missing_keypoints")


def test_openpose_provider_error_no_data():
    """Test that OpenPoseProviderError is raised when no data provided."""
    from src.tools.starting_pose_matcher.skeleton_extractors.openpose import (
        OpenPoseProvider,
        OpenPoseProviderError,
    )

    with pytest.raises(OpenPoseProviderError):
        OpenPoseProvider(json_path=None, json_data=None)


def test_openpose_parse_json_fixture():
    """Test parsing OpenPose JSON fixture."""
    from src.tools.starting_pose_matcher.skeleton_extractors.openpose import OpenPoseProvider

    # Create minimal OpenPose JSON fixture
    fixture = {
        "people": [
            {
                "pose_keypoints_2d": [
                    # nose (0)
                    100,
                    100,
                    0.9,
                    # neck (1)
                    100,
                    120,
                    0.9,
                    # right_shoulder (2)
                    120,
                    130,
                    0.9,
                    # right_elbow (3)
                    130,
                    150,
                    0.9,
                    # right_wrist (4)
                    140,
                    170,
                    0.9,
                    # left_shoulder (5)
                    80,
                    130,
                    0.9,
                    # left_elbow (6)
                    70,
                    150,
                    0.9,
                    # left_wrist (7)
                    60,
                    170,
                    0.9,
                    # right_hip (8)
                    115,
                    200,
                    0.9,
                    # right_knee (9)
                    120,
                    250,
                    0.5,
                    # right_ankle (10)
                    125,
                    300,
                    0.5,
                    # left_hip (11)
                    85,
                    200,
                    0.9,
                    # left_knee (12)
                    80,
                    250,
                    0.5,
                    # left_ankle (13)
                    75,
                    300,
                    0.5,
                    # right_eye (14)
                    105,
                    95,
                    0.8,
                    # left_eye (15)
                    95,
                    95,
                    0.8,
                    # right_ear (16)
                    110,
                    90,
                    0.7,
                    # left_ear (17)
                    90,
                    90,
                    0.7,
                ]
            }
        ]
    }

    provider = OpenPoseProvider(json_data=fixture)

    assert len(provider.frames) == 1
    assert len(provider.frames[0].keypoints) > 0


def test_openpose_get_skeleton():
    """Test getting skeleton from OpenPose provider."""
    from src.tools.starting_pose_matcher.skeleton_extractors.openpose import OpenPoseProvider

    fixture = {
        "people": [
            {
                "pose_keypoints_2d": [
                    100,
                    100,
                    0.9,  # nose
                    100,
                    120,
                    0.9,  # neck
                    120,
                    130,
                    0.9,  # right_shoulder
                    130,
                    150,
                    0.9,  # right_elbow
                    140,
                    170,
                    0.9,  # right_wrist
                    80,
                    130,
                    0.9,  # left_shoulder
                    70,
                    150,
                    0.9,  # left_elbow
                    60,
                    170,
                    0.9,  # left_wrist
                    115,
                    200,
                    0.9,  # right_hip
                    120,
                    250,
                    0.5,  # right_knee
                    125,
                    300,
                    0.5,  # right_ankle
                    85,
                    200,
                    0.9,  # left_hip
                    80,
                    250,
                    0.5,  # left_knee
                    75,
                    300,
                    0.5,  # left_ankle
                    105,
                    95,
                    0.8,  # right_eye
                    95,
                    95,
                    0.8,  # left_eye
                    110,
                    90,
                    0.7,  # right_ear
                    90,
                    90,
                    0.7,  # left_ear
                ]
            }
        ]
    }

    provider = OpenPoseProvider(json_data=fixture)
    skeleton = provider.get_skeleton()

    # Check required upper-body vocabulary
    required = ["ls", "rs", "le", "re", "lw", "rw", "hip"]
    for name in required:
        assert name in skeleton, f"Missing {name}"

    # Check derived keypoints
    assert "mp" in skeleton  # midpoint from wrists
    assert "torso" in skeleton  # torso from shoulders


def test_openpose_confidence_thresholding():
    """Test that confidence thresholding works correctly."""
    from src.tools.starting_pose_matcher.skeleton_extractors.openpose import OpenPoseProvider

    fixture = {
        "people": [
            {
                "pose_keypoints_2d": [
                    100,
                    100,
                    0.1,  # nose - low confidence
                    100,
                    120,
                    0.1,  # neck - low confidence
                    120,
                    130,
                    0.9,  # right_shoulder - high confidence
                    130,
                    150,
                    0.1,  # right_elbow - low confidence
                    140,
                    170,
                    0.9,  # right_wrist - high confidence
                    80,
                    130,
                    0.9,  # left_shoulder - high confidence
                    70,
                    150,
                    0.1,  # left_elbow - low confidence
                    60,
                    170,
                    0.9,  # left_wrist - high confidence
                    115,
                    200,
                    0.9,  # right_hip - high confidence
                    120,
                    250,
                    0.1,  # right_knee - low confidence
                    125,
                    300,
                    0.1,  # right_ankle - low confidence
                    85,
                    200,
                    0.9,  # left_hip - high confidence
                    80,
                    250,
                    0.1,  # left_knee - low confidence
                    75,
                    300,
                    0.1,  # left_ankle - low confidence
                    105,
                    95,
                    0.1,  # right_eye - low confidence
                    95,
                    95,
                    0.1,  # left_eye - low confidence
                    110,
                    90,
                    0.1,  # right_ear - low confidence
                    90,
                    90,
                    0.1,  # left_ear - low confidence
                ]
            }
        ]
    }

    provider = OpenPoseProvider(json_data=fixture, confidence_threshold=0.5)
    skeleton = provider.get_skeleton()

    # Low confidence keypoints should be excluded
    assert "le" not in skeleton  # left elbow has low confidence
    assert "re" not in skeleton  # right elbow has low confidence

    # High confidence keypoints should be included
    assert "ls" in skeleton
    assert "rs" in skeleton
    assert "lw" in skeleton
    assert "rw" in skeleton


def test_openpose_missing_keypoints():
    """Test that missing keypoints are reported correctly."""
    from src.tools.starting_pose_matcher.skeleton_extractors.openpose import OpenPoseProvider

    fixture = {
        "people": [
            {
                "pose_keypoints_2d": [
                    100,
                    100,
                    0.9,  # nose
                    100,
                    120,
                    0.9,  # neck
                    120,
                    130,
                    0.9,  # right_shoulder
                    130,
                    150,
                    0.9,  # right_elbow
                    140,
                    170,
                    0.9,  # right_wrist
                    80,
                    130,
                    0.9,  # left_shoulder
                    70,
                    150,
                    0.9,  # left_elbow
                    60,
                    170,
                    0.9,  # left_wrist
                    115,
                    200,
                    0.9,  # right_hip
                    120,
                    250,
                    0.5,  # right_knee
                    125,
                    300,
                    0.5,  # right_ankle
                    85,
                    200,
                    0.9,  # left_hip
                    80,
                    250,
                    0.5,  # left_knee
                    75,
                    300,
                    0.5,  # left_ankle
                    105,
                    95,
                    0.8,  # right_eye
                    95,
                    95,
                    0.8,  # left_eye
                    110,
                    90,
                    0.7,  # right_ear
                    90,
                    90,
                    0.7,  # left_ear
                ]
            }
        ]
    }

    provider = OpenPoseProvider(json_data=fixture)
    missing = provider.get_missing_keypoints()

    # Clubhead should always be missing (not observable from body)
    assert "ch" in missing


def test_openpose_create_provider():
    """Test create_provider function."""
    from src.tools.starting_pose_matcher.skeleton_extractors.openpose import create_provider

    fixture = {"people": [{"pose_keypoints_2d": []}]}
    provider = create_provider(json_data=fixture, confidence_threshold=0.5)

    assert provider.confidence_threshold == 0.5


# =============================================================================
# MediaPipe Provider Tests
# =============================================================================


def test_mediapipe_import():
    """Test that MediaPipe provider can be imported."""
    from src.tools.starting_pose_matcher.skeleton_extractors.mediapipe import (
        MediaPipeProvider,
        MediaPipeProviderError,
        create_provider,
        MEDIAPIPE_POSE_LANDMARKS,
        MEDIAPIPE_TO_MATCHER_VOCAB,
    )

    assert hasattr(MediaPipeProvider, "get_skeleton")
    assert hasattr(MediaPipeProvider, "get_visibility_map")
    assert hasattr(MediaPipeProvider, "get_missing_landmarks")


def test_mediapipe_provider_error_no_data():
    """Test that MediaPipeProviderError is raised when no data provided."""
    from src.tools.starting_pose_matcher.skeleton_extractors.mediapipe import (
        MediaPipeProvider,
        MediaPipeProviderError,
    )

    with pytest.raises(MediaPipeProviderError):
        MediaPipeProvider(landmarks_data=None)


class MockLandmark:
    """Mock MediaPipe landmark for testing."""

    def __init__(self, x=0.0, y=0.0, z=0.0, visibility=1.0, presence=1.0):
        self.x = x
        self.y = y
        self.z = z
        self.visibility = visibility
        self.presence = presence


def test_mediapipe_parse_landmarks():
    """Test parsing MediaPipe landmarks."""
    from src.tools.starting_pose_matcher.skeleton_extractors.mediapipe import MediaPipeProvider

    # Create mock landmarks for all 33 MediaPipe Pose landmarks
    landmarks = [MockLandmark(0.5, 0.5, 0.0, 0.9, 0.9) for _ in range(33)]

    provider = MediaPipeProvider(landmarks_data=[landmarks])

    assert len(provider.frames) == 1
    assert len(provider.frames[0].landmarks) == 33


def test_mediapipe_get_skeleton():
    """Test getting skeleton from MediaPipe provider."""
    from src.tools.starting_pose_matcher.skeleton_extractors.mediapipe import MediaPipeProvider

    # Create mock landmarks
    landmarks = [MockLandmark(0.5, 0.5, 0.0, 0.9, 0.9) for _ in range(33)]

    provider = MediaPipeProvider(landmarks_data=[landmarks])
    skeleton = provider.get_skeleton()

    # Check required upper-body vocabulary
    required = ["ls", "rs", "le", "re", "lw", "rw", "hip"]
    for name in required:
        assert name in skeleton, f"Missing {name}"

    # Check derived keypoints
    assert "mp" in skeleton  # midpoint from wrists
    assert "torso" in skeleton  # torso from shoulders


def test_mediapipe_visibility_thresholding():
    """Test that visibility thresholding works correctly."""
    from src.tools.starting_pose_matcher.skeleton_extractors.mediapipe import MediaPipeProvider

    # Create mock landmarks with varying visibility
    landmarks = []
    for i in range(33):
        if i == 13 or i == 14:  # left_elbow
            landmarks.append(MockLandmark(0.5, 0.5, 0.0, 0.1, 0.9))  # low visibility
        else:
            landmarks.append(MockLandmark(0.5, 0.5, 0.0, 0.9, 0.9))  # high visibility

    provider = MediaPipeProvider(landmarks_data=[landmarks], visibility_threshold=0.5)
    skeleton = provider.get_skeleton()

    # Low visibility landmarks should be excluded
    assert "le" not in skeleton
    assert "re" not in skeleton


def test_mediapipe_missing_landmarks():
    """Test that missing landmarks are reported correctly."""
    from src.tools.starting_pose_matcher.skeleton_extractors.mediapipe import MediaPipeProvider

    # Create mock landmarks
    landmarks = [MockLandmark(0.5, 0.5, 0.0, 0.9, 0.9) for _ in range(33)]

    provider = MediaPipeProvider(landmarks_data=[landmarks])
    missing = provider.get_missing_landmarks()

    # Clubhead should always be missing (not observable from body)
    assert "ch" in missing


def test_mediapipe_create_provider():
    """Test create_provider function."""
    from src.tools.starting_pose_matcher.skeleton_extractors.mediapipe import create_provider

    landmarks = [MockLandmark() for _ in range(33)]
    provider = create_provider(landmarks_data=[landmarks], visibility_threshold=0.7)

    assert provider.visibility_threshold == 0.7


# =============================================================================
# Integration Tests
# =============================================================================


def test_observed_input_providers_distinct_from_physics():
    """Test that observed-input providers are distinct from physics providers."""
    from src.tools.starting_pose_matcher.skeleton_extractors.openpose import OpenPoseProvider
    from src.tools.starting_pose_matcher.skeleton_extractors.mediapipe import MediaPipeProvider

    # Verify these are different classes
    assert OpenPoseProvider is not MediaPipeProvider

    # Verify they have different source identifiers
    from src.tools.starting_pose_matcher.skeleton_extractors.openpose import KeypointObservation
    from src.tools.starting_pose_matcher.skeleton_extractors.mediapipe import LandmarkObservation

    obs = KeypointObservation(name="test")
    assert obs.source == "openpose"

    lm_obs = LandmarkObservation(name="test")
    assert lm_obs.source == "mediapipe"
