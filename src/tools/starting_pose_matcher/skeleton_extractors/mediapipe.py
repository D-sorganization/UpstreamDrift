"""MediaPipe observed-input provider for starting-pose matcher.

This module provides a MediaPipe-based observed-input provider that maps
MediaPipe Pose landmarks to the shared matcher skeleton vocabulary.

MediaPipe provides observed human pose landmarks from images/video, not
physics simulation. This provider normalizes MediaPipe output into matcher
target coordinates.

Required vocabulary (upper-body focused for MediaPipe):
    hip, spine, torso, hub, ls, rs, le, re, lw, rw, mp, ch

Note: MediaPipe may not provide all required vocabulary points. Missing
or low-confidence landmarks are tracked explicitly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import numpy as np
    from numpy.typing import NDArray

# MediaPipe Pose landmarks indices
# https://google.github.io/mediapipe/solutions/pose.html
MEDIAPIPE_POSE_LANDMARKS: dict[str, int] = {
    "nose": 0,
    "left_eye_inner": 1,
    "left_eye": 2,
    "left_eye_outer": 3,
    "right_eye_inner": 4,
    "right_eye": 5,
    "right_eye_outer": 6,
    "left_ear": 7,
    "right_ear": 8,
    "mouth_left": 9,
    "mouth_right": 10,
    "left_shoulder": 11,
    "right_shoulder": 12,
    "left_elbow": 13,
    "right_elbow": 14,
    "left_wrist": 15,
    "right_wrist": 16,
    "left_pinky": 17,
    "right_pinky": 18,
    "left_index": 19,
    "right_index": 20,
    "left_thumb": 21,
    "right_thumb": 22,
    "left_hip": 23,
    "right_hip": 24,
    "left_knee": 25,
    "right_knee": 26,
    "left_ankle": 27,
    "right_ankle": 28,
    "left_heel": 29,
    "right_heel": 30,
    "left_foot_index": 31,
    "right_foot_index": 32,
}

# Mapping from MediaPipe landmarks to matcher vocabulary
MEDIAPIPE_TO_MATCHER_VOCAB: dict[str, str | None] = {
    # Shoulders map directly
    "left_shoulder": "ls",
    "right_shoulder": "rs",
    # Elbows map directly
    "left_elbow": "le",
    "right_elbow": "re",
    # Wrists map directly
    "left_wrist": "lw",
    "right_wrist": "rw",
    # Hips can be averaged from left/right
    "left_hip": "hip",
    "right_hip": "hip",
    # These don't have direct MediaPipe equivalents
    "spine": None,  # Can be derived from shoulders/hips
    "torso": None,  # Can be derived from shoulders
    "hub": None,  # Can be derived
    "midpoint": None,  # Can be computed from wrists
    "clubhead": None,  # Not observable from body landmarks
}

# Reverse mapping
MATCHER_TO_MEDIAPIPE: dict[str, list[str]] = {}
for mp_name, matcher_name in MEDIAPIPE_TO_MATCHER_VOCAB.items():
    if matcher_name is not None:
        if matcher_name not in MATCHER_TO_MEDIAPIPE:
            MATCHER_TO_MEDIAPIPE[matcher_name] = []
        MATCHER_TO_MEDIAPIPE[matcher_name].append(mp_name)


@dataclass
class LandmarkObservation:
    """Represents an observed landmark with visibility and presence."""

    name: str
    position: tuple[float, float, float] | None = None
    visibility: float = 0.0
    presence: float = 0.0
    is_observed: bool = False
    source: str = "mediapipe"


@dataclass
class MediaPipeFrame:
    """Represents a single frame of MediaPipe observations."""

    frame_index: int
    landmarks: dict[str, LandmarkObservation] = field(default_factory=dict)
    camera_metadata: dict[str, Any] | None = None


class MediaPipeProviderError(Exception):
    """Raised when there's an error with the MediaPipe provider."""


class MediaPipeProvider:
    """Provides observed skeleton data from MediaPipe Pose landmarks.

    This provider loads MediaPipe Pose detection results and extracts landmarks
    to map them to the shared matcher skeleton vocabulary.

    Attributes:
        landmarks_data: List of MediaPipe landmark lists.
        visibility_threshold: Minimum visibility for accepting landmarks.
    """

    def __init__(
        self,
        landmarks_data: list | None = None,
        visibility_threshold: float = 0.5,
        presence_threshold: float = 0.5,
    ):
        """Initialize the MediaPipe provider.

        Args:
            landmarks_data: List of MediaPipe landmark lists (from solutions.pose.PoseLandmark).
                           Each element is a list of landmarks with x, y, z, visibility, presence.
            visibility_threshold: Minimum visibility for accepting landmarks.
            presence_threshold: Minimum presence for accepting landmarks.

        Raises:
            MediaPipeProviderError: If landmarks_data is not provided.
        """
        if landmarks_data is None:
            raise MediaPipeProviderError("landmarks_data must be provided")

        self.visibility_threshold = visibility_threshold
        self.presence_threshold = presence_threshold
        self.frames: list[MediaPipeFrame] = []

        self._parse_landmarks_data(landmarks_data)

    def _parse_landmarks_data(self, landmarks_data: list) -> None:
        """Parse MediaPipe landmarks data into frames.

        Args:
            landmarks_data: List of landmark lists from MediaPipe.
        """
        for frame_idx, frame_landmarks in enumerate(landmarks_data):
            frame = MediaPipeFrame(frame_index=frame_idx)

            for lm_idx, landmark in enumerate(frame_landmarks):
                # Find the landmark name
                lm_name = None
                for name, idx in MEDIAPIPE_POSE_LANDMARKS.items():
                    if idx == lm_idx:
                        lm_name = name
                        break

                if lm_name is None:
                    continue

                # Extract landmark data
                # MediaPipe landmarks have: x, y, z, visibility, presence
                x = getattr(landmark, "x", 0.0)
                y = getattr(landmark, "y", 0.0)
                z = getattr(landmark, "z", 0.0)
                visibility = getattr(landmark, "visibility", 0.0)
                presence = getattr(landmark, "presence", 0.0)

                is_observed = (
                    visibility >= self.visibility_threshold
                    and presence >= self.presence_threshold
                )
                position = (float(x), float(y), float(z)) if is_observed else None

                frame.landmarks[lm_name] = LandmarkObservation(
                    name=lm_name,
                    position=position,
                    visibility=float(visibility),
                    presence=float(presence),
                    is_observed=is_observed,
                )

            self.frames.append(frame)

    def get_skeleton(
        self,
        frame_index: int = 0,
    ) -> dict[str, NDArray[np.float64]]:
        """Get skeleton landmarks from MediaPipe observations.

        Args:
            frame_index: Index of the frame to extract.

        Returns:
            Dictionary mapping matcher vocabulary names to 3D positions.
            Missing or low-visibility landmarks are excluded.
        """
        import numpy as np

        if frame_index >= len(self.frames):
            raise MediaPipeProviderError(
                f"Frame index {frame_index} out of range (0-{len(self.frames) - 1})"
            )

        frame = self.frames[frame_index]
        skeleton: dict[str, NDArray[np.float64]] = {}

        # Map observed landmarks to matcher vocabulary
        for lm_name, lm_obs in frame.landmarks.items():
            if not lm_obs.is_observed:
                continue

            matcher_name = MEDIAPIPE_TO_MATCHER_VOCAB.get(lm_name)
            if matcher_name is None:
                continue

            # For hip, average left and right if both available
            if matcher_name == "hip":
                if lm_name == "left_hip":
                    # Check if right_hip is also available
                    if (
                        "right_hip" in frame.landmarks
                        and frame.landmarks["right_hip"].is_observed
                    ):
                        # Average will be done when processing right_hip
                        continue
                    pos = lm_obs.position
                elif lm_name == "right_hip":
                    left_hip = frame.landmarks.get("left_hip")
                    if left_hip and left_hip.is_observed:
                        # Average both hips
                        pos = (
                            (lm_obs.position[0] + left_hip.position[0]) / 2,
                            (lm_obs.position[1] + left_hip.position[1]) / 2,
                            (lm_obs.position[2] + left_hip.position[2]) / 2,
                        )
                    else:
                        pos = lm_obs.position
                else:
                    continue
            else:
                pos = lm_obs.position

            skeleton[matcher_name] = np.array(pos, dtype=np.float64)

        # Compute derived landmarks if possible
        # Midpoint between wrists
        if "lw" in skeleton and "rw" in skeleton:
            skeleton["mp"] = (skeleton["lw"] + skeleton["rw"]) / 2

        # Torso as midpoint between shoulders
        if "ls" in skeleton and "rs" in skeleton:
            skeleton["torso"] = (skeleton["ls"] + skeleton["rs"]) / 2

        # Spine as midpoint between hips
        if "hip" in skeleton:
            # Use hip position as approximate spine base
            skeleton["spine"] = skeleton["hip"].copy()
            # Adjust spine upward (approximate)
            skeleton["spine"][2] += 0.3  # ~30cm up

        # Hub can be derived from spine and torso
        if "spine" in skeleton and "torso" in skeleton:
            skeleton["hub"] = (skeleton["spine"] + skeleton["torso"]) / 2

        return skeleton

    def get_visibility_map(
        self,
        frame_index: int = 0,
    ) -> dict[str, float]:
        """Get visibility values for all matcher vocabulary landmarks.

        Args:
            frame_index: Index of the frame to extract.

        Returns:
            Dictionary mapping matcher vocabulary names to visibility values.
        """
        if frame_index >= len(self.frames):
            raise MediaPipeProviderError(
                f"Frame index {frame_index} out of range (0-{len(self.frames) - 1})"
            )

        frame = self.frames[frame_index]
        visibility_map: dict[str, float] = {}

        for lm_name, lm_obs in frame.landmarks.items():
            matcher_name = MEDIAPIPE_TO_MATCHER_VOCAB.get(lm_name)
            if matcher_name is None:
                continue

            if matcher_name not in visibility_map:
                visibility_map[matcher_name] = lm_obs.visibility
            elif matcher_name == "hip":
                # Average visibility for hip
                visibility_map[matcher_name] = (
                    visibility_map[matcher_name] + lm_obs.visibility
                ) / 2

        return visibility_map

    def get_missing_landmarks(
        self,
        frame_index: int = 0,
    ) -> list[str]:
        """Get list of missing or low-visibility landmarks.

        Args:
            frame_index: Index of the frame to extract.

        Returns:
            List of matcher vocabulary names that are missing or low-visibility.
        """
        visibility_map = self.get_visibility_map(frame_index)
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
            if name not in visibility_map
            or visibility_map[name] < self.visibility_threshold
        ]


def create_provider(
    landmarks_data: list | None = None,
    visibility_threshold: float = 0.5,
    presence_threshold: float = 0.5,
) -> MediaPipeProvider:
    """Create a MediaPipe observed-input provider.

    Args:
        landmarks_data: List of MediaPipe landmark lists.
        visibility_threshold: Minimum visibility for accepting landmarks.
        presence_threshold: Minimum presence for accepting landmarks.

    Returns:
        A configured MediaPipeProvider instance.
    """
    return MediaPipeProvider(
        landmarks_data=landmarks_data,
        visibility_threshold=visibility_threshold,
        presence_threshold=presence_threshold,
    )
