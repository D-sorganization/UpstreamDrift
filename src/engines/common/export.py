"""Common Export Interfaces — video and dataset export protocols.

Provides engine-agnostic interfaces for exporting simulation results
as video files or structured datasets. Each engine implements these
protocols according to its native rendering/recording capabilities.

Design by Contract:
    Preconditions:
        - Engine must be initialized before creating exporters
        - Output paths must be writable
    Postconditions:
        - Exported files are valid and complete
        - No partial files left on error (cleanup on failure)
    Invariants:
        - Original simulation state is never modified by export
"""

from __future__ import annotations

import csv
import json
import logging
from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.core.constants import DEFAULT_FPS, HD_HEIGHT, HD_WIDTH

logger = logging.getLogger(__name__)


# =============================================================================
# Video Export
# =============================================================================


@dataclass
class VideoConfig:
    """Configuration for video export.

    Attributes:
        width: Frame width in pixels
        height: Frame height in pixels
        fps: Frames per second
        format: Output format ('mp4', 'avi', 'gif')
        codec: Video codec (None = auto-detect)
        show_overlays: Whether to render metric overlays
    """

    width: int = HD_WIDTH
    height: int = HD_HEIGHT
    fps: int = DEFAULT_FPS
    format: str = "mp4"
    codec: str | None = None
    show_overlays: bool = True


class VideoExportProtocol(ABC):
    """Abstract interface for engine video export.

    Each engine implements this to render its native visualization
    to video frames. The interface is designed for reversibility —
    exporting does not modify simulation state.
    """

    @abstractmethod
    def start_recording(self, output_path: Path, config: VideoConfig) -> bool:
        """Begin recording video frames.

        Args:
            output_path: Path for output video file
            config: Video configuration

        Returns:
            True if recording started successfully
        """

    @abstractmethod
    def capture_frame(
        self,
        overlay_callback: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> bool:
        """Capture the current simulation state as a video frame.

        Args:
            overlay_callback: Optional function to add overlays to the frame

        Returns:
            True if frame was captured successfully
        """

    @abstractmethod
    def finish_recording(self) -> Path | None:
        """Finalize the recording and write the video file.

        Returns:
            Path to the completed video file, or None on failure
        """

    @property
    @abstractmethod
    def is_recording(self) -> bool:
        """Check if recording is currently active."""

    @property
    @abstractmethod
    def frame_count(self) -> int:
        """Number of frames captured so far."""


# =============================================================================
# Dataset Export
# =============================================================================


@dataclass
class DatasetRecord:
    """A single timestep record for dataset export.

    Attributes:
        time: Simulation time [s]
        positions: Generalized coordinates (n_q,)
        velocities: Generalized velocities (n_v,)
        accelerations: Generalized accelerations (n_v,), if available
        forces: Applied forces/torques (n_u,), if available
        metadata: Extra per-timestep data
    """

    time: float
    positions: np.ndarray
    velocities: np.ndarray
    accelerations: np.ndarray | None = None
    forces: np.ndarray | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DatasetExporter:
    """Engine-agnostic dataset exporter.

    Collects simulation records and exports them in multiple formats.
    This is a concrete class (not abstract) because the export logic
    is the same for all engines — only the data collection differs.

    Design by Contract:
        Preconditions:
            - Records must be added in chronological order
        Postconditions:
            - Exported files are complete and valid
        Invariants:
            - Records are immutable once added
    """

    def __init__(self, joint_names: list[str] | None = None) -> None:
        """Initialize dataset exporter.

        Args:
            joint_names: Optional joint names for column headers
        """
        self._records: list[DatasetRecord] = []
        self._joint_names = joint_names or []
        self._metadata: dict[str, Any] = {}

    @property
    def record_count(self) -> int:
        """Number of records collected."""
        return len(self._records)

    def set_metadata(self, key: str, value: Any) -> None:
        """Set export metadata (engine name, model, etc.).

        Args:
            key: Metadata key
            value: Metadata value
        """
        self._metadata[key] = value

    def add_record(self, record: DatasetRecord) -> None:
        """Add a timestep record.

        Args:
            record: DatasetRecord for the current timestep

        Raises:
            ValueError: If record time is not monotonically increasing
        """
        if self._records and record.time < self._records[-1].time:
            raise ValueError(
                f"Records must be chronological: {record.time} < {self._records[-1].time}"
            )
        self._records.append(record)

    def add_from_state(
        self,
        time: float,
        q: np.ndarray,
        v: np.ndarray,
        qacc: np.ndarray | None = None,
        forces: np.ndarray | None = None,
    ) -> None:
        """Convenience method to add a record from raw state arrays.

        Args:
            time: Simulation time
            q: Positions
            v: Velocities
            qacc: Accelerations (optional)
            forces: Applied forces (optional)
        """
        self.add_record(
            DatasetRecord(
                time=time,
                positions=q.copy(),
                velocities=v.copy(),
                accelerations=qacc.copy() if qacc is not None else None,
                forces=forces.copy() if forces is not None else None,
            )
        )

    def export_csv(self, output_path: Path) -> Path:
        """Export collected records as CSV.

        Args:
            output_path: Path for output CSV file

        Returns:
            Path to the written file

        Raises:
            ValueError: If no records have been collected
        """
        if not self._records:
            raise ValueError("No records to export")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        nq = len(self._records[0].positions)
        nv = len(self._records[0].velocities)

        # Build column headers
        headers = ["time"]
        q_names = (
            [f"q_{name}" for name in self._joint_names[:nq]]
            if self._joint_names
            else [f"q_{i}" for i in range(nq)]
        )
        v_names = (
            [f"v_{name}" for name in self._joint_names[:nv]]
            if self._joint_names
            else [f"v_{i}" for i in range(nv)]
        )
        headers.extend(q_names)
        headers.extend(v_names)

        has_accel = self._records[0].accelerations is not None
        if has_accel:
            a_names = [f"a_{i}" for i in range(nv)]
            headers.extend(a_names)

        has_forces = self._records[0].forces is not None
        if has_forces:
            nu = len(self._records[0].forces)  # type: ignore[arg-type]
            f_names = [f"f_{i}" for i in range(nu)]
            headers.extend(f_names)

        with open(output_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(headers)

            for rec in self._records:
                row: list[float] = [rec.time]
                row.extend(rec.positions.tolist())
                row.extend(rec.velocities.tolist())
                if has_accel and rec.accelerations is not None:
                    row.extend(rec.accelerations.tolist())
                if has_forces and rec.forces is not None:
                    row.extend(rec.forces.tolist())
                writer.writerow(row)

        logger.info("Exported %d records to CSV: %s", len(self._records), output_path)
        return output_path

    def export_json(self, output_path: Path) -> Path:
        """Export collected records as JSON.

        Args:
            output_path: Path for output JSON file

        Returns:
            Path to the written file

        Raises:
            ValueError: If no records have been collected
        """
        if not self._records:
            raise ValueError("No records to export")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        data: dict[str, Any] = {
            "metadata": self._metadata,
            "joint_names": self._joint_names,
            "record_count": len(self._records),
            "records": [],
        }

        for rec in self._records:
            record_dict: dict[str, Any] = {
                "time": rec.time,
                "positions": rec.positions.tolist(),
                "velocities": rec.velocities.tolist(),
            }
            if rec.accelerations is not None:
                record_dict["accelerations"] = rec.accelerations.tolist()
            if rec.forces is not None:
                record_dict["forces"] = rec.forces.tolist()
            if rec.metadata:
                record_dict["metadata"] = rec.metadata
            data["records"].append(record_dict)

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info("Exported %d records to JSON: %s", len(self._records), output_path)
        return output_path

    def export_hdf5(self, output_path: Path) -> Path:
        """Export collected records as HDF5.

        Requires the ``h5py`` package. Stores arrays in columnar format
        for efficient random access and analysis.

        Args:
            output_path: Path for output HDF5 file

        Returns:
            Path to the written file

        Raises:
            ValueError: If no records have been collected
            ImportError: If h5py is not installed
        """
        if not self._records:
            raise ValueError("No records to export")

        import h5py

        output_path.parent.mkdir(parents=True, exist_ok=True)

        times = np.array([r.time for r in self._records])
        positions = np.array([r.positions for r in self._records])
        velocities = np.array([r.velocities for r in self._records])

        with h5py.File(output_path, "w") as f:
            f.create_dataset("time", data=times)
            f.create_dataset("positions", data=positions)
            f.create_dataset("velocities", data=velocities)

            if self._records[0].accelerations is not None:
                accel = np.array(
                    [
                        r.accelerations
                        for r in self._records
                        if r.accelerations is not None
                    ]
                )
                f.create_dataset("accelerations", data=accel)

            if self._records[0].forces is not None:
                forces = np.array(
                    [r.forces for r in self._records if r.forces is not None]
                )
                f.create_dataset("forces", data=forces)

            # Store metadata as HDF5 attributes
            for key, value in self._metadata.items():
                f.attrs[key] = value

            if self._joint_names:
                f.attrs["joint_names"] = self._joint_names

        logger.info("Exported %d records to HDF5: %s", len(self._records), output_path)
        return output_path

    def clear(self) -> None:
        """Clear all collected records (for reuse)."""
        self._records.clear()
