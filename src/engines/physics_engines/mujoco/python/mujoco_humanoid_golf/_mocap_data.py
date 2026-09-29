from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class MotionCaptureFrame:
    """Single frame of motion capture data."""

    time: float
    marker_positions: dict[str, np.ndarray]  # marker_name -> position [3]
    marker_velocities: dict[str, np.ndarray] | None = None
    body_orientations: dict[str, np.ndarray] | None = (
        None  # body_name -> quaternion [4]
    )
    joint_angles: np.ndarray | None = None  # If available from mocap system


@dataclass
class MotionCaptureSequence:
    """Complete motion capture sequence."""

    frames: list[MotionCaptureFrame]
    frame_rate: float
    marker_names: list[str]
    metadata: dict = field(default_factory=dict)

    @property
    def num_frames(self) -> int:
        """Get number of frames."""
        return len(self.frames)

    @property
    def duration(self) -> float:
        """Get sequence duration in seconds."""
        if len(self.frames) < 2:
            return 0.0
        return self.frames[-1].time - self.frames[0].time

    def get_marker_trajectory(self, marker_name: str) -> tuple[np.ndarray, np.ndarray]:
        """Get trajectory for a specific marker.

        Args:
            marker_name: Name of marker

        Returns:
            Tuple of (times [N], positions [N x 3])
        """
        if marker_name is None:
            raise ValueError("marker_name must be provided")
        times = []
        positions = []

        for frame in self.frames:
            if marker_name in frame.marker_positions:
                times.append(frame.time)
                positions.append(frame.marker_positions[marker_name])

        return np.array(times), np.array(positions)


@dataclass
class MarkerSet:
    """Marker set configuration for motion capture."""

    markers: dict[str, str]  # marker_name -> body_name
    marker_offsets: dict[str, np.ndarray]  # marker_name -> offset from body origin [3]

    @classmethod
    def golf_swing_marker_set(cls) -> MarkerSet:
        """Standard marker set for golf swing capture.

        Based on common motion capture protocols for golf biomechanics.
        """
        markers = {
            # Head
            "HEAD_TOP": "head",
            "HEAD_FRONT": "head",
            # Torso
            "C7": "upper_torso",  # 7th cervical vertebra
            "T10": "lower_torso",  # 10th thoracic vertebra
            "STERN": "upper_torso",  # Sternum
            "CLAV": "upper_torso",  # Clavicle
            # Pelvis
            "SACR": "pelvis",  # Sacrum
            "LASI": "pelvis",  # Left anterior superior iliac spine
            "RASI": "pelvis",  # Right ASIS
            "LPSI": "pelvis",  # Left posterior superior iliac spine
            "RPSI": "pelvis",  # Right PSIS
            # Left arm
            "LSHO": "left_upper_arm",  # Left shoulder
            "LELB": "left_forearm",  # Left elbow
            "LWRA": "left_hand",  # Left wrist radial
            "LWRU": "left_hand",  # Left wrist ulnar
            "LFIN": "left_hand",  # Left finger
            # Right arm
            "RSHO": "right_upper_arm",
            "RELB": "right_forearm",
            "RWRA": "right_hand",
            "RWRU": "right_hand",
            "RFIN": "right_hand",
            # Left leg
            "LKNE": "left_shin",  # Left knee
            "LANK": "left_foot",  # Left ankle
            "LHEE": "left_foot",  # Left heel
            "LTOE": "left_foot",  # Left toe
            # Right leg
            "RKNE": "right_shin",
            "RANK": "right_foot",
            "RHEE": "right_foot",
            "RTOE": "right_foot",
            # Club
            "CLUB_GRIP_TOP": "club",
            "CLUB_GRIP_MID": "club",
            "CLUB_HEAD": "club_head",
        }

        # Approximate marker offsets (these should be measured for each subject)
        offsets = {}
        for marker_name in markers:
            offsets[marker_name] = np.zeros(3)  # Will be calibrated

        return cls(markers=markers, marker_offsets=offsets)
