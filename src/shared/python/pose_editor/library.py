"""Pose library with save/load/export/import/interpolate functionality.

Provides a comprehensive pose management system for storing, retrieving,
and interpolating between poses across all physics engines.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.logging_config import get_logger

logger = get_logger(__name__)


class PresetPoseCategory(Enum):
    """Categories for preset poses."""

    GOLF_SWING = "Golf Swing"
    NEUTRAL = "Neutral"
    ATHLETIC = "Athletic"
    TESTING = "Testing"
    CUSTOM = "Custom"


@dataclass
class StoredPose:
    """A stored pose configuration."""

    name: str
    joint_positions: np.ndarray  # q vector
    joint_velocities: np.ndarray | None = None  # v vector (optional)

    # Metadata
    description: str = ""
    category: PresetPoseCategory = PresetPoseCategory.CUSTOM
    created_at: str = ""
    modified_at: str = ""
    tags: list[str] = field(default_factory=list)

    # Optional: specific joint values by name for portability
    named_positions: dict[str, float | list[float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Initialize timestamps if not set."""
        if not self.created_at:
            self.created_at = datetime.now().isoformat()
        if not self.modified_at:
            self.modified_at = self.created_at

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "joint_positions": self.joint_positions.tolist(),
            "joint_velocities": (
                self.joint_velocities.tolist()
                if self.joint_velocities is not None
                else None
            ),
            "description": self.description,
            "category": self.category.value,
            "created_at": self.created_at,
            "modified_at": self.modified_at,
            "tags": self.tags,
            "named_positions": self.named_positions,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StoredPose:
        """Create from dictionary."""
        category = PresetPoseCategory.CUSTOM
        if "category" in data:
            for cat in PresetPoseCategory:
                if cat.value == data["category"]:
                    category = cat
                    break

        return cls(
            name=data.get("name", "Unnamed"),
            joint_positions=np.array(data.get("joint_positions", [])),
            joint_velocities=(
                np.array(data["joint_velocities"])
                if data.get("joint_velocities")
                else None
            ),
            description=data.get("description", ""),
            category=category,
            created_at=data.get("created_at", ""),
            modified_at=data.get("modified_at", ""),
            tags=data.get("tags", []),
            named_positions=data.get("named_positions", {}),
        )


class PoseInterpolator:
    """Interpolates between poses for smooth transitions."""

    @staticmethod
    def linear(
        pose_a: StoredPose,
        pose_b: StoredPose,
        alpha: float,
    ) -> np.ndarray:
        """Linear interpolation between two poses.

        Args:
            pose_a: Starting pose
            pose_b: Ending pose
            alpha: Interpolation factor (0 = pose_a, 1 = pose_b)

        Returns:
            Interpolated joint positions
        """
        alpha = np.clip(alpha, 0.0, 1.0)

        # Handle size mismatch
        len_a = len(pose_a.joint_positions)
        len_b = len(pose_b.joint_positions)

        if len_a != len_b:
            logger.warning(
                "Pose size mismatch: %d vs %d. Using minimum size.", len_a, len_b
            )
            min_len = min(len_a, len_b)
            pos_a = pose_a.joint_positions[:min_len]
            pos_b = pose_b.joint_positions[:min_len]
        else:
            pos_a = pose_a.joint_positions
            pos_b = pose_b.joint_positions

        return (1 - alpha) * pos_a + alpha * pos_b

    @staticmethod
    def slerp_scalar(
        angle_a: float,
        angle_b: float,
        alpha: float,
    ) -> float:
        """Spherical linear interpolation for angles.

        Properly handles angle wrapping for rotation joints.

        Args:
            angle_a: Starting angle (radians)
            angle_b: Ending angle (radians)
            alpha: Interpolation factor

        Returns:
            Interpolated angle
        """
        # Normalize angle difference
        diff = angle_b - angle_a

        # Take shortest path
        while diff > np.pi:
            diff -= 2 * np.pi
        while diff < -np.pi:
            diff += 2 * np.pi

        return angle_a + alpha * diff

    @staticmethod
    def cubic_bezier(
        pose_a: StoredPose,
        pose_b: StoredPose,
        control_a: np.ndarray,
        control_b: np.ndarray,
        alpha: float,
    ) -> np.ndarray:
        """Cubic Bezier interpolation for smooth motion.

        Args:
            pose_a: Starting pose
            pose_b: Ending pose
            control_a: First control point
            control_b: Second control point
            alpha: Interpolation factor

        Returns:
            Interpolated joint positions
        """
        alpha = np.clip(alpha, 0.0, 1.0)
        t = alpha
        t2 = t * t
        t3 = t2 * t

        p0 = pose_a.joint_positions
        p1 = control_a
        p2 = control_b
        p3 = pose_b.joint_positions

        # Bezier formula
        return (
            (1 - t) ** 3 * p0
            + 3 * (1 - t) ** 2 * t * p1
            + 3 * (1 - t) * t2 * p2
            + t3 * p3
        )

    @staticmethod
    def sequence(
        poses: list[StoredPose],
        alpha: float,
    ) -> np.ndarray:
        """Interpolate through a sequence of poses.

        Args:
            poses: List of poses to interpolate through
            alpha: Overall progress (0 to 1)

        Returns:
            Interpolated joint positions
        """
        if len(poses) < 2:
            return poses[0].joint_positions if poses else np.array([])

        # Map alpha to segment
        num_segments = len(poses) - 1
        scaled_alpha = alpha * num_segments
        segment_idx = int(scaled_alpha)
        segment_alpha = scaled_alpha - segment_idx

        # Handle edge cases
        if segment_idx >= num_segments:
            return poses[-1].joint_positions
        if segment_idx < 0:
            return poses[0].joint_positions

        return PoseInterpolator.linear(
            poses[segment_idx],
            poses[segment_idx + 1],
            segment_alpha,
        )


class PoseLibrary:
    """Library for storing and managing poses.

    Provides functionality for saving, loading, exporting, and importing
    pose configurations.
    """

    def __init__(self) -> None:
        """Initialize the pose library."""
        self._poses: dict[str, StoredPose] = {}
        self._interpolator = PoseInterpolator()

    def save_pose(
        self,
        name: str,
        positions: np.ndarray,
        velocities: np.ndarray | None = None,
        description: str = "",
        category: PresetPoseCategory = PresetPoseCategory.CUSTOM,
        tags: list[str] | None = None,
        named_positions: dict[str, float | list[float]] | None = None,
    ) -> StoredPose:
        """Save a pose to the library.

        Args:
            name: Unique name for the pose
            positions: Joint positions (q vector)
            velocities: Optional joint velocities (v vector)
            description: Human-readable description
            category: Pose category
            tags: Optional tags for filtering
            named_positions: Optional named joint positions for portability

        Returns:
            The saved StoredPose
        """
        # Check for existing pose
        existing = self._poses.get(name)
        created_at = existing.created_at if existing else datetime.now().isoformat()

        pose = StoredPose(
            name=name,
            joint_positions=positions.copy(),
            joint_velocities=velocities.copy() if velocities is not None else None,
            description=description,
            category=category,
            created_at=created_at,
            modified_at=datetime.now().isoformat(),
            tags=tags or [],
            named_positions=named_positions or {},
        )

        self._poses[name] = pose
        logger.info("Saved pose: %s", name)
        return pose

    def load_pose(self, name: str) -> StoredPose | None:
        """Load a pose from the library.

        Args:
            name: Pose name

        Returns:
            StoredPose or None if not found
        """
        return self._poses.get(name)

    def delete_pose(self, name: str) -> bool:
        """Delete a pose from the library.

        Args:
            name: Pose name

        Returns:
            True if deleted, False if not found
        """
        if name in self._poses:
            del self._poses[name]
            logger.info("Deleted pose: %s", name)
            return True
        return False

    def rename_pose(self, old_name: str, new_name: str) -> bool:
        """Rename a pose.

        Args:
            old_name: Current name
            new_name: New name

        Returns:
            True if renamed successfully
        """
        if old_name not in self._poses:
            return False
        if new_name in self._poses:
            logger.warning("Cannot rename: '%s' already exists", new_name)
            return False

        pose = self._poses.pop(old_name)
        pose.name = new_name
        pose.modified_at = datetime.now().isoformat()
        self._poses[new_name] = pose
        return True

    def list_poses(self) -> list[str]:
        """Get list of all pose names.

        Returns:
            List of pose names
        """
        return list(self._poses.keys())

    def list_poses_by_category(self, category: PresetPoseCategory) -> list[StoredPose]:
        """Get all poses in a category.

        Args:
            category: Category to filter by

        Returns:
            List of poses in the category
        """
        return [p for p in self._poses.values() if p.category == category]

    def list_poses_by_tag(self, tag: str) -> list[StoredPose]:
        """Get all poses with a specific tag.

        Args:
            tag: Tag to filter by

        Returns:
            List of poses with the tag
        """
        return [p for p in self._poses.values() if tag in p.tags]

    def get_all_poses(self) -> list[StoredPose]:
        """Get all poses in the library.

        Returns:
            List of all stored poses
        """
        return list(self._poses.values())

    def clear(self) -> None:
        """Clear all poses from the library."""
        self._poses.clear()
        logger.info("Cleared pose library")

    def interpolate(
        self,
        pose_name_a: str,
        pose_name_b: str,
        alpha: float,
    ) -> np.ndarray | None:
        """Interpolate between two poses.

        Args:
            pose_name_a: First pose name
            pose_name_b: Second pose name
            alpha: Interpolation factor (0 = a, 1 = b)

        Returns:
            Interpolated positions or None if poses not found
        """
        pose_a = self._poses.get(pose_name_a)
        pose_b = self._poses.get(pose_name_b)

        if pose_a is None or pose_b is None:
            return None

        return self._interpolator.linear(pose_a, pose_b, alpha)

    def export_to_json(self, file_path: str | Path) -> int:
        """Export the library to a JSON file.

        Args:
            file_path: Path to output file

        Returns:
            Number of poses exported
        """
        file_path = Path(file_path)
        data = {
            "version": "1.0",
            "exported_at": datetime.now().isoformat(),
            "poses": [pose.to_dict() for pose in self._poses.values()],
        }

        with open(file_path, "w") as f:
            json.dump(data, f, indent=2)

        logger.info("Exported %d poses to %s", len(self._poses), file_path)
        return len(self._poses)

    def import_from_json(
        self,
        file_path: str | Path,
        overwrite: bool = False,
    ) -> int:
        """Import poses from a JSON file.

        Args:
            file_path: Path to input file
            overwrite: Whether to overwrite existing poses with same name

        Returns:
            Number of poses imported
        """
        file_path = Path(file_path)

        with open(file_path) as f:
            data = json.load(f)

        imported = 0
        poses_data = data.get("poses", [])

        for pose_data in poses_data:
            pose = StoredPose.from_dict(pose_data)

            if pose.name in self._poses and not overwrite:
                logger.debug("Skipping existing pose: %s", pose.name)
                continue

            self._poses[pose.name] = pose
            imported += 1

        logger.info("Imported %d poses from %s", imported, file_path)
        return imported

    def merge_library(
        self,
        other: PoseLibrary,
        overwrite: bool = False,
    ) -> int:
        """Merge another library into this one.

        Args:
            other: Library to merge from
            overwrite: Whether to overwrite existing poses

        Returns:
            Number of poses merged
        """
        merged = 0
        for name, pose in other._poses.items():
            if name in self._poses and not overwrite:
                continue
            self._poses[name] = pose
            merged += 1

        logger.info("Merged %d poses from another library", merged)
        return merged


# -------- Preset Poses --------

PRESET_POSES: dict[str, dict[str, float]] = {
    # Golf Swing Presets
    "Address": {
        "description": "Standard address position for golf swing",
        "category": "Golf Swing",
        # Joint angles (will be mapped by name)
        "lowerbackrx": 0.35,
        "upperbackrx": 0.15,
        "lhumerusrx": 0.6,
        "lhumerusry": 0.3,
        "rhumerusrx": 0.6,
        "rhumerusry": -0.3,
        "lelbow": -0.3,
        "relbow": -0.3,
        "lhiprx": 0.2,
        "rhiprx": 0.2,
        "lknee": 0.15,
        "rknee": 0.15,
    },
    "Top of Backswing": {
        "description": "Top of backswing position",
        "category": "Golf Swing",
        "lowerbackrz": -0.4,
        "upperbackrz": -0.4,
        "rhumerusrx": -1.2,
        "rhumerusry": 0.8,
        "rhumerusrz": 0.5,
        "lhumerusrx": -0.8,
        "lhumerusry": -0.6,
        "relbow": -1.8,
        "lelbow": -0.5,
        "lhiprx": 0.3,
        "rhiprx": 0.25,
        "lhipry": 0.3,
        "rhipry": -0.2,
    },
    "Impact": {
        "description": "Impact position at ball contact",
        "category": "Golf Swing",
        "lowerbackrz": 0.6,
        "upperbackrz": 0.4,
        "rhumerusrx": 0.2,
        "rhumerusry": -0.3,
        "lhumerusrx": 0.3,
        "lhumerusry": 0.1,
        "relbow": -0.15,
        "lelbow": -0.1,
        "lhiprz": -0.5,
        "rhiprz": 0.3,
    },
    "Follow Through": {
        "description": "Full follow through position",
        "category": "Golf Swing",
        "lowerbackrz": 1.0,
        "upperbackrz": 0.8,
        "rhumerusrx": -0.5,
        "rhumerusry": -0.8,
        "rhumerusrz": -0.6,
        "lhumerusrx": -1.0,
        "lhumerusry": 0.5,
        "lhumerusrz": -0.4,
        "relbow": -1.2,
        "lelbow": -0.8,
    },
    # Neutral Poses
    "T-Pose": {
        "description": "Arms extended horizontally",
        "category": "Neutral",
        "lhumerusry": 1.57,
        "rhumerusry": -1.57,
        "lelbow": 0.0,
        "relbow": 0.0,
    },
    "A-Pose": {
        "description": "Arms at 45 degrees",
        "category": "Neutral",
        "lhumerusry": 0.785,
        "rhumerusry": -0.785,
        "lelbow": 0.0,
        "relbow": 0.0,
    },
    "Neutral Standing": {
        "description": "Relaxed standing position",
        "category": "Neutral",
    },
    # Athletic Poses
    "Athletic Ready": {
        "description": "Athletic ready stance",
        "category": "Athletic",
        "lowerbackrx": 0.2,
        "lhiprx": 0.3,
        "rhiprx": 0.3,
        "lknee": 0.4,
        "rknee": 0.4,
        "lankle": -0.1,
        "rankle": -0.1,
    },
}


def get_preset_pose(name: str) -> dict[str, Any] | None:
    """Get a preset pose by name.

    Args:
        name: Preset pose name

    Returns:
        Pose data dictionary or None if not found
    """
    return PRESET_POSES.get(name)


def list_preset_poses() -> list[str]:
    """Get list of all preset pose names.

    Returns:
        List of preset pose names
    """
    return list(PRESET_POSES.keys())


def list_preset_poses_by_category(category: str) -> list[str]:
    """Get preset poses in a category.

    Args:
        category: Category name

    Returns:
        List of pose names in the category
    """
    return [
        name
        for name, data in PRESET_POSES.items()
        if data.get("category", "") == category
    ]
