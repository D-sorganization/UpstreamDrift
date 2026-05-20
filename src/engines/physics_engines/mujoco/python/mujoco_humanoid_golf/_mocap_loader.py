from __future__ import annotations

import json
from typing import TYPE_CHECKING  # noqa: ICN003

if TYPE_CHECKING:
    from pathlib import Path

import numpy as np

from ._mocap_data import MotionCaptureFrame, MotionCaptureSequence


class MotionCaptureLoader:
    """Load motion capture data from various file formats."""

    @staticmethod
    def load_csv(
        filepath: str | Path,
        frame_rate: float = 120.0,
        marker_names: list[str] | None = None,
    ) -> MotionCaptureSequence:
        """Load motion capture data from CSV file.

        Expected format:
        time, marker1_x, marker1_y, marker1_z, marker2_x, marker2_y, marker2_z, ...

        Args:
            filepath: Path to CSV file
            frame_rate: Frame rate in Hz
            marker_names: List of marker names (if None, auto-detect from header)

        Returns:
            MotionCaptureSequence
        """
        if filepath is None:
            raise ValueError("filepath must be provided")
        data = np.loadtxt(filepath, delimiter=",", skiprows=1)

        # Parse header for marker names if not provided
        if marker_names is None:
            with open(filepath) as f:
                header = f.readline().strip().split(",")
                # Extract marker names from column headers (e.g., "LSHO_x", "LSHO_y")
                marker_names = []
                for i in range(1, len(header), 3):
                    marker_name = header[i].rsplit("_", 1)[0]
                    if marker_name not in marker_names:
                        marker_names.append(marker_name)

        frames = []
        for row in data:
            time = row[0]
            marker_positions = {}

            for i, marker_name in enumerate(marker_names):
                idx = 1 + i * 3
                if idx + 2 < len(row):
                    position = row[idx : idx + 3]
                    marker_positions[marker_name] = position

            frame = MotionCaptureFrame(time=time, marker_positions=marker_positions)
            frames.append(frame)

        return MotionCaptureSequence(
            frames=frames,
            frame_rate=frame_rate,
            marker_names=marker_names,
        )

    @staticmethod
    def load_json(filepath: str | Path) -> MotionCaptureSequence:
        """Load motion capture data from JSON file.

        Expected format:
        {
            "frame_rate": 120.0,
            "marker_names": ["LSHO", "RSHO", ...],
            "frames": [
                {
                    "time": 0.0,
                    "markers": {
                        "LSHO": [x, y, z],
                        # ...
                        # ...
                    }
                },
                # ...
            ]
        }

        Args:
            filepath: Path to JSON file

        Returns:
            MotionCaptureSequence
        """
        with open(filepath) as f:
            data = json.load(f)

        frames = []
        for frame_data in data["frames"]:
            frame = MotionCaptureFrame(
                time=frame_data["time"],
                marker_positions={
                    name: np.array(pos) for name, pos in frame_data["markers"].items()
                },
            )
            frames.append(frame)

        return MotionCaptureSequence(
            frames=frames,
            frame_rate=data.get("frame_rate", 120.0),
            marker_names=data.get(
                "marker_names",
                list(frames[0].marker_positions.keys()),
            ),
            metadata=data.get("metadata", {}),
        )

    @staticmethod
    def load_bvh(filepath: str | Path) -> MotionCaptureSequence | None:
        """Load motion capture data from BVH file.

        BVH (Biovision Hierarchy) is a common format for motion capture.
        This is a simplified parser - for production use, consider using
        a dedicated BVH library.

        Args:
            filepath: Path to BVH file

        Returns:
            MotionCaptureSequence
        """
        # Placeholder for BVH parsing
        # In production, use a library like 'bvh' or 'scikit-kinematics'
        return None
