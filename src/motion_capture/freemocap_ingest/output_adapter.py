"""
FreeMoCap Output Adapter - Parse and convert FreeMoCap output to internal schema.

This module handles reading FreeMoCap's output files (CSV/numpy) and converting
them to UpstreamDrift's internal landmark format for downstream processing.
"""

import csv
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# MediaPipe Holistic landmark names (body + hands + face)
MEDIAPIPE_LANDMARKS = [
    # Body (33 landmarks, 0-32)
    "nose",
    "left_eye_inner",
    "left_eye",
    "left_eye_outer",
    "right_eye_inner",
    "right_eye",
    "right_eye_outer",
    "left_ear",
    "right_ear",
    "mouth_left",
    "mouth_right",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_pinky",
    "right_pinky",
    "left_index",
    "right_index",
    "left_thumb",
    "right_thumb",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
    "left_heel",
    "right_heel",
    "left_foot_index",
    "right_foot_index",
    # Hands (42 landmarks each, 33-74 left, 75-116 right)
    # Simplified: just track key hand points
    "left_wrist_mp",
    "left_thumb_cmc",
    "left_index_mcp",
    "left_pinky_mcp",
    "right_wrist_mp",
    "right_thumb_cmc",
    "right_index_mcp",
    "right_pinky_mcp",
    # Face simplified (just key points)
    "face_center",
]


@dataclass
class LandmarkPoint:
    """A single 3D landmark point."""

    name: str
    x: float
    y: float
    z: float
    confidence: float = 1.0
    visible: bool = True


@dataclass
class LandmarkFrame:
    """A frame of landmark data (all points at one timestep)."""

    frame_number: int
    timestamp: float
    points: list[LandmarkPoint] = field(default_factory=list)

    def get_point(self, name: str) -> LandmarkPoint | None:
        """Get a landmark point by name."""
        for p in self.points:
            if p.name == name:
                return p
        return None

    def to_array(self) -> np.ndarray:
        """
        Convert to numpy array of shape (N, 4) where columns are [x, y, z, confidence].

        Returns:
            Numpy array with landmark coordinates.
        """
        if not self.points:
            return np.empty((0, 4))
        return np.array([[p.x, p.y, p.z, p.confidence] for p in self.points])


@dataclass
class LandmarkSession:
    """Complete session of landmark data."""

    session_id: str
    frames: list[LandmarkFrame] = field(default_factory=list)
    calibration: dict | None = None
    metadata: dict = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        """
        Convert to numpy array of shape (T, N, 4) where T is frames, N is landmarks.

        Returns:
            Numpy array with all frame data.
        """
        if not self.frames:
            return np.empty((0, 0, 4))
        # All frames must have the same number of points (#6639 F5). Validate
        # explicitly so a ragged session raises a clear error instead of an
        # opaque ``np.stack`` failure.
        point_counts = {len(f.points) for f in self.frames}
        if len(point_counts) > 1:
            raise ValueError(
                "Ragged landmark session: frames have differing point counts "
                f"{sorted(point_counts)}; expected a uniform landmark schema."
            )
        frame_arrays = [f.to_array() for f in self.frames]
        return np.stack(frame_arrays, axis=0)


class FreeMoCapOutputAdapter:
    """
    Adapter for parsing FreeMoCap output files.

    FreeMoCap outputs:
    - 3D landmark data as CSV and/or numpy files
    - Calibration data as JSON
    - Diagnostic plots and metadata

    This adapter reads those files and converts to our internal schema.
    """

    # Expected output structure from FreeMoCap
    LANDMARKS_CSV_PATTERN = "freemocap_3d_landmarks_*.csv"
    CALIBRATION_FILE = "camera_calibration.json"
    METADATA_FILE = "recording_metadata.json"

    def __init__(self, output_dir: Path):
        """
        Initialize the adapter.

        Args:
            output_dir: Directory containing FreeMoCap output files.
        """
        self.output_dir = Path(output_dir).expanduser().resolve()
        self._session: LandmarkSession | None = None

    def _find_landmarks_csv(self) -> list[Path]:
        """Find landmark CSV files in output directory."""
        return list(self.output_dir.glob(self.LANDMARKS_CSV_PATTERN))

    def _load_calibration(self) -> dict | None:
        """Load camera calibration data."""
        calib_file = self.output_dir / self.CALIBRATION_FILE
        if calib_file.exists():
            with open(calib_file) as f:
                return json.load(f)
        return None

    def _load_metadata(self) -> dict:
        """Load recording metadata."""
        meta_file = self.output_dir / self.METADATA_FILE
        if meta_file.exists():
            with open(meta_file) as f:
                return json.load(f)
        return {}

    def _parse_landmark_csv(self, csv_path: Path) -> list[LandmarkFrame]:
        """
        Parse a FreeMoCap landmark CSV file.

        Expected format: columns are frame_number, timestamp, then for each landmark:
        {landmark_name}_x, {landmark_name}_y, {landmark_name}_z, {landmark_name}_conf

        Args:
            csv_path: Path to CSV file.

        Returns:
            List of LandmarkFrame objects.
        """
        frames = []

        with open(csv_path, newline="") as f:
            reader = csv.reader(f)
            header = next(reader)

            # Parse header to find landmark columns
            # Format: frame_num, timestamp, nose_x, nose_y, nose_z, nose_conf, ...
            landmark_data: dict[str, dict[str, int]] = {}
            for i, col in enumerate(header[2:], start=2):
                parts = col.rsplit("_", 1)
                if len(parts) == 2:
                    name, coord = parts
                    if name not in landmark_data:
                        landmark_data[name] = {}
                    landmark_data[name][coord] = i

            # Build a FIXED landmark schema from the header (#6639 F5): the set
            # and order of landmarks is fixed once, so every frame emits the
            # same number of points in the same column order. Missing or
            # low-confidence cells become NaN coordinates rather than being
            # dropped, which previously produced ragged frames and crashed
            # ``to_array``/``np.stack`` or silently shifted every column.
            schema: list[str] = [
                name
                for name, coords in landmark_data.items()
                if {"x", "y", "z"} <= coords.keys()
            ]

            # Read data rows
            for row in reader:
                if not row:
                    continue

                frame_num = int(row[0])
                timestamp = float(row[1])

                points = []
                for name in schema:
                    coords = landmark_data[name]
                    conf_idx = coords.get("conf")
                    try:
                        conf = (
                            float(row[conf_idx])
                            if conf_idx is not None and conf_idx < len(row)
                            else 1.0
                        )
                    except (ValueError, IndexError, TypeError):
                        conf = 0.0

                    # Drop coordinates that fail to parse or are low-confidence:
                    # represent them as NaN so the row stays fixed-width.
                    try:
                        if conf <= 0.5:
                            raise ValueError("low confidence")
                        x = float(row[coords["x"]])
                        y = float(row[coords["y"]])
                        z = float(row[coords["z"]])
                        visible = True
                    except (ValueError, IndexError, TypeError):
                        x = y = z = float("nan")
                        visible = False

                    points.append(
                        LandmarkPoint(
                            name=name,
                            x=x,
                            y=y,
                            z=z,
                            confidence=conf,
                            visible=visible,
                        )
                    )

                frames.append(
                    LandmarkFrame(
                        frame_number=frame_num,
                        timestamp=timestamp,
                        points=points,
                    )
                )

        return frames

    def parse(self, session_id: str | None = None) -> LandmarkSession:
        """
        Parse all output files and return a complete session.

        Args:
            session_id: Optional session ID override.

        Returns:
            LandmarkSession with all parsed data.

        Raises:
            FileNotFoundError: If no landmark CSV files found.
        """
        if not self.output_dir.exists():
            raise FileNotFoundError(f"Output directory not found: {self.output_dir}")

        # Find landmark files
        landmark_files = self._find_landmarks_csv()
        if not landmark_files:
            raise FileNotFoundError(f"No landmark CSV files found in {self.output_dir}")

        # Use first landmark file (typically the main body tracking)
        main_file = landmark_files[0]
        logger.info(f"Parsing landmark file: {main_file}")

        frames = self._parse_landmark_csv(main_file)
        logger.info(f"Parsed {len(frames)} frames")

        # Load optional calibration and metadata
        calibration = self._load_calibration()
        metadata = self._load_metadata()

        # Determine session ID
        if session_id is None:
            session_id = self.output_dir.parent.name or "unknown_session"

        session = LandmarkSession(
            session_id=session_id,
            frames=frames,
            calibration=calibration,
            metadata=metadata,
        )

        self._session = session
        return session

    def get_session(self) -> LandmarkSession | None:
        """Get the most recently parsed session."""
        return self._session

    def export_to_numpy(self, output_path: Path) -> np.ndarray:
        """
        Export session data to numpy format.

        Args:
            output_path: Path to save .npy file.

        Returns:
            Numpy array of shape (T, N, 4).
        """
        if self._session is None:
            raise ValueError("No session loaded. Call parse() first.")

        data = self._session.to_array()
        np.save(output_path, data)
        return data

    def export_to_csv(self, output_path: Path) -> None:
        """
        Export session data to CSV format.

        Args:
            output_path: Path to save CSV file.
        """
        if self._session is None:
            raise ValueError("No session loaded. Call parse() first.")

        with open(output_path, "w", newline="") as f:
            writer = csv.writer(f)

            # Write header
            header = ["frame_number", "timestamp"]
            if self._session.frames:
                for p in self._session.frames[0].points:
                    header.extend(
                        [f"{p.name}_x", f"{p.name}_y", f"{p.name}_z", f"{p.name}_conf"]
                    )
            writer.writerow(header)

            # Write data
            for frame in self._session.frames:
                row = [frame.frame_number, frame.timestamp]
                for p in frame.points:
                    row.extend([p.x, p.y, p.z, p.confidence])
                writer.writerow(row)


def main():
    """CLI entrypoint for output adapter."""
    import argparse

    parser = argparse.ArgumentParser(description="Parse FreeMoCap output files")
    parser.add_argument(
        "output_dir",
        type=Path,
        help="FreeMoCap output directory",
    )
    parser.add_argument(
        "--export-npy",
        type=Path,
        help="Export to numpy file",
    )
    parser.add_argument(
        "--export-csv",
        type=Path,
        help="Export to CSV file",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    adapter = FreeMoCapOutputAdapter(args.output_dir)
    session = adapter.parse()

    if session.frames:
        pass

    if args.export_npy:
        adapter.export_to_numpy(args.export_npy)

    if args.export_csv:
        adapter.export_to_csv(args.export_csv)


if __name__ == "__main__":
    main()
