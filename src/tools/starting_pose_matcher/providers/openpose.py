"""OpenPose observed-input provider for starting-pose matcher.

This module provides an OpenPose-based observed-input provider that maps
OpenPose keypoints to the shared matcher skeleton vocabulary.

OpenPose provides observed human keypoints from images/video, not physics
simulation. This provider normalizes OpenPose JSON output into matcher
target coordinates.

Required vocabulary (upper-body focused for OpenPose):
    hip, spine, torso, hub, ls, rs, le, re, lw, rw, mp, ch

Note: OpenPose may not provide all required vocabulary points. Missing
or low-confidence landmarks are tracked explicitly.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# OpenPose COCO format body part indices
# https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/output.md
OPENPOSE_COCO_INDICES: dict[str, int] = {
    "nose": 0,
    "neck": 1,
    "right_shoulder": 2,
    "right_elbow": 3,
    "right_wrist": 4,
    "left_shoulder": 5,
    "left_elbow": 6,
    "left_wrist": 7,
    "right_hip": 8,
    "right_knee": 9,
    "right_ankle": 10,
    "left_hip": 11,
    "left_knee": 12,
    "left_ankle": 13,
    "right_eye": 14,
    "left_eye": 15,
    "right_ear": 16,
    "left_ear": 17,
}

# Mapping from OpenPose keypoints to matcher vocabulary
# OpenPose provides observed landmarks, not physics bodies
OPENPOSE_TO_MATCHER_VOCAB: dict[str, str | None] = {
    # Shoulders map directly
    "left_shoulder": "ls",
    "right_shoulder": "rs",
    # Elbows map directly
    "left_elbow": "le",
    "right_elbow": "re",
    # Wrists map directly
    "left_wrist": "lw",
    "right_wrist": "rw",
    # Neck can approximate spine/torso
    "neck": "spine",
    # Hips can be averaged from left/right
    "left_hip": "hip",
    "right_hip": "hip",
    # These don't have direct OpenPose equivalents
    "hub": None,  # Must be derived
    "midpoint": None,  # Can be computed from wrists
    "clubhead": None,  # Not observable from body keypoints
    "torso": None,  # Can be derived from spine/shoulders
}

# Reverse mapping
MATCHER_TO_OPENPOSE: dict[str, list[str]] = {}
for openpose_name, matcher_name in OPENPOSE_TO_MATCHER_VOCAB.items():
    if matcher_name is not None:
        if matcher_name not in MATCHER_TO_OPENPOSE:
            MATCHER_TO_OPENPOSE[matcher_name] = []
        MATCHER_TO_OPENPOSE[matcher_name].append(openpose_name)


@dataclass
class KeypointObservation:
    """Represents an observed keypoint with confidence."""

    name: str
    position: tuple[float, float, float] | None = None
    confidence: float = 0.0
    is_observed: bool = False
    source: str = "openpose"


@dataclass
class OpenPoseFrame:
    """Represents a single frame of OpenPose observations."""

    frame_index: int
    keypoints: dict[str, KeypointObservation] = field(default_factory=dict)
    camera_metadata: dict[str, Any] | None = None


class OpenPoseProviderError(Exception):
    """Raised when there's an error with the OpenPose provider."""


class OpenPoseProvider:
    """Provides observed skeleton data from OpenPose JSON output.

    This provider loads OpenPose JSON files and extracts keypoints
    to map them to the shared matcher skeleton vocabulary.

    Attributes:
        json_path: Path to the OpenPose JSON file.
        confidence_threshold: Minimum confidence for accepting keypoints.
    """

    def __init__(
        self,
        json_path: str | None = None,
        json_data: dict | None = None,
        confidence_threshold: float = 0.3,
    ):
        """Initialize the OpenPose provider.

        Args:
            json_path: Path to the OpenPose JSON file.
            json_data: Optional pre-loaded JSON data (alternative to path).
            confidence_threshold: Minimum confidence for accepting keypoints.

        Raises:
            OpenPoseProviderError: If neither json_path nor json_data is provided.
        """
        if json_path is None and json_data is None:
            raise OpenPoseProviderError(
                "Either json_path or json_data must be provided"
            )

        self.confidence_threshold = confidence_threshold
        self.frames: list[OpenPoseFrame] = []

        if json_data is not None:
            self._parse_json_data(json_data)
        else:
            with open(json_path) as f:
                data = json.load(f)
            self._parse_json_data(data)

    def _parse_json_data(self, data: dict) -> None:
        """Parse OpenPose JSON data into frames.

        Args:
            data: The loaded OpenPose JSON data.
        """
        # OpenPose JSON format: {"people": [{"pose_keypoints_2d": [...]}], ...}
        people = data.get("people", [])

        for person_idx, person in enumerate(people):
            keypoints_2d = person.get("pose_keypoints_2d", [])
            if not keypoints_2d:
                continue

            # Keypoints are in format [x, y, confidence, x, y, confidence, ...]
            num_keypoints = len(keypoints_2d) // 3

            frame = OpenPoseFrame(frame_index=person_idx)

            for kp_idx in range(num_keypoints):
                x = keypoints_2d[kp_idx * 3]
                y = keypoints_2d[kp_idx * 3 + 1]
                conf = keypoints_2d[kp_idx * 3 + 2]

                # Find the keypoint name
                kp_name = None
                for name, idx in OPENPOSE_COCO_INDICES.items():
                    if idx == kp_idx:
                        kp_name = name
                        break

                if kp_name is None:
                    continue

                is_observed = conf >= self.confidence_threshold
                position = (float(x), float(y), 0.0) if is_observed else None

                frame.keypoints[kp_name] = KeypointObservation(
                    name=kp_name,
                    position=position,
                    confidence=float(conf),
                    is_observed=is_observed,
                )

            self.frames.append(frame)

    def get_skeleton(
        self,
        frame_index: int = 0,
        person_index: int = 0,
    ) -> dict[str, NDArray[np.float64]]:
        """Get skeleton keypoints from OpenPose observations.

        Args:
            frame_index: Index of the frame to extract.
            person_index: Index of the person in the frame.

        Returns:
            Dictionary mapping matcher vocabulary names to 3D positions.
            Missing or low-confidence keypoints are excluded.
        """
        import numpy as np

        if frame_index >= len(self.frames):
            raise OpenPoseProviderError(
                f"Frame index {frame_index} out of range (0-{len(self.frames) - 1})"
            )

        frame = self.frames[frame_index]
        skeleton: dict[str, NDArray[np.float64]] = {}

        # Map observed keypoints to matcher vocabulary
        for kp_name, kp_obs in frame.keypoints.items():
            if not kp_obs.is_observed:
                continue

            matcher_name = OPENPOSE_TO_MATCHER_VOCAB.get(kp_name)
            if matcher_name is None:
                continue

            # For hip, average left and right if both available
            if matcher_name == "hip":
                if kp_name == "left_hip":
                    # Check if right_hip is also available
                    if (
                        "right_hip" in frame.keypoints
                        and frame.keypoints["right_hip"].is_observed
                    ):
                        # Average will be done when processing right_hip
                        continue
                    pos = kp_obs.position
                elif kp_name == "right_hip":
                    left_hip = frame.keypoints.get("left_hip")
                    if left_hip and left_hip.is_observed:
                        # Average both hips
                        pos = (
                            (kp_obs.position[0] + left_hip.position[0]) / 2,
                            (kp_obs.position[1] + left_hip.position[1]) / 2,
                            (kp_obs.position[2] + left_hip.position[2]) / 2,
                        )
                    else:
                        pos = kp_obs.position
                else:
                    continue
            else:
                pos = kp_obs.position

            skeleton[matcher_name] = np.array(pos, dtype=np.float64)

        # Compute derived keypoints if possible
        # Midpoint between wrists
        if "lw" in skeleton and "rw" in skeleton:
            skeleton["mp"] = (skeleton["lw"] + skeleton["rw"]) / 2

        # Torso as midpoint between shoulders
        if "ls" in skeleton and "rs" in skeleton:
            skeleton["torso"] = (skeleton["ls"] + skeleton["rs"]) / 2

        # Hub can be derived from spine and torso
        if "spine" in skeleton and "torso" in skeleton:
            skeleton["hub"] = (skeleton["spine"] + skeleton["torso"]) / 2

        return skeleton

    def get_confidence_map(
        self,
        frame_index: int = 0,
    ) -> dict[str, float]:
        """Get confidence values for all matcher vocabulary keypoints.

        Args:
            frame_index: Index of the frame to extract.

        Returns:
            Dictionary mapping matcher vocabulary names to confidence values.
        """
        if frame_index >= len(self.frames):
            raise OpenPoseProviderError(
                f"Frame index {frame_index} out of range (0-{len(self.frames) - 1})"
            )

        frame = self.frames[frame_index]
        confidence_map: dict[str, float] = {}

        for kp_name, kp_obs in frame.keypoints.items():
            matcher_name = OPENPOSE_TO_MATCHER_VOCAB.get(kp_name)
            if matcher_name is None:
                continue

            if matcher_name not in confidence_map:
                confidence_map[matcher_name] = kp_obs.confidence
            elif matcher_name == "hip":
                # Average confidence for hip
                confidence_map[matcher_name] = (
                    confidence_map[matcher_name] + kp_obs.confidence
                ) / 2

        return confidence_map

    def get_missing_keypoints(
        self,
        frame_index: int = 0,
    ) -> list[str]:
        """Get list of missing or low-confidence keypoints.

        Args:
            frame_index: Index of the frame to extract.

        Returns:
            List of matcher vocabulary names that are missing or low-confidence.
        """
        confidence_map = self.get_confidence_map(frame_index)
        required = [
            "hip",
            "spine",
            "torso",
            "hub",
            "ls",
            "rs",
            "le",
            "re",
            "lw",
            "rw",
            "mp",
            "ch",
        ]
        return [
            name
            for name in required
            if name not in confidence_map
            or confidence_map[name] < self.confidence_threshold
        ]


def create_provider(
    json_path: str | None = None,
    json_data: dict | None = None,
    confidence_threshold: float = 0.3,
) -> OpenPoseProvider:
    """Create an OpenPose observed-input provider.

    Args:
        json_path: Path to the OpenPose JSON file.
        json_data: Optional pre-loaded JSON data.
        confidence_threshold: Minimum confidence for accepting keypoints.

    Returns:
        A configured OpenPoseProvider instance.
    """
    return OpenPoseProvider(
        json_path=json_path,
        json_data=json_data,
        confidence_threshold=confidence_threshold,
    )
