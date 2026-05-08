# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

"""Utilities for loading and interpreting C3D motion-capture files.

Migrated from Golf_Modeling_Suite.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from ...utils.logging import get_logger
from ._c3d_analog import (
    build_analog_dataframe,
    build_force_plate_dataframe,
    build_force_plate_dataframe_from_calibration,
    detect_force_plate_channels,
    force_plate_columns,
)
from ._c3d_io import (
    C3DMapping,
    build_metadata,
    export_dataframe,
    load_c3d,
    unit_scale,
)
from ._c3d_markers import build_points_dataframe
from ._c3d_models import (
    BIOMECHANICAL_MARKER_MAX_M,
    BIOMECHANICAL_MARKER_MIN_M,
    SCHEMA_VERSION,
    C3DEvent,
    C3DMetadata,
    ForcePlateCalibration,
)

logger = get_logger(__name__)

__all__ = [
    "C3DDataReader",
    "C3DEvent",
    "C3DMapping",
    "C3DMetadata",
    "ForcePlateCalibration",
    "BIOMECHANICAL_MARKER_MAX_M",
    "BIOMECHANICAL_MARKER_MIN_M",
    "SCHEMA_VERSION",
]


class C3DDataReader:
    """Loads marker trajectories and metadata from a C3D file."""

    def __init__(self, file_path: Path | str) -> None:
        """Initialize the C3D data reader with a file path."""
        self.file_path = Path(file_path)
        self._c3d_data: C3DMapping | None = None
        self._metadata: C3DMetadata | None = None

    def get_metadata(self) -> C3DMetadata:
        """Return metadata describing marker labels, frame count, rate, and units."""
        if self._metadata is None:
            self._metadata = build_metadata(self._load(), self.file_path)
        return self._metadata

    def points_dataframe(
        self,
        include_time: bool = True,
        markers: Sequence[str] | None = None,
        residual_nan_threshold: float | None = None,
        target_units: str | None = None,
    ) -> pd.DataFrame:
        """Return marker trajectories as a tidy DataFrame.

        Args:
            include_time: Whether to include a time column calculated from the frame
                index and the frame rate reported in the C3D header.
            markers: Optional list of marker names to retain. All markers are
                returned when ``None``.
            residual_nan_threshold: If provided, coordinates with residuals above
                the threshold are replaced with ``NaN`` to make downstream QA
                easier in visualization tools.
            target_units: Optional unit string (``"m"`` or ``"mm"``) for the point
                coordinates. A no-op when ``None`` or when the requested units match
                the file's native units.

        Returns:
            DataFrame with columns ``frame``, ``marker``, ``x``, ``y``, ``z``,
            ``residual`` (EzC3D stores residuals in the fourth point channel), and
            an optional ``time`` column in seconds.
        """
        c3d_data = self._load()
        metadata = self.get_metadata()
        scale = unit_scale(metadata.units, target_units)
        return build_points_dataframe(
            c3d_data,
            metadata,
            self.file_path.name,
            scale,
            include_time,
            markers,
            residual_nan_threshold,
            target_units,
        )

    def analog_dataframe(self, include_time: bool = True) -> pd.DataFrame:
        """Return analog channels as a tidy DataFrame.

        Rows are ordered by sample index and channel name so downstream GUI
        components can easily plot synchronized sensor traces.
        """
        return build_analog_dataframe(self._load(), self.get_metadata(), include_time)

    def export_points(
        self,
        output_path: Path | str,
        *,
        include_time: bool = True,
        markers: Sequence[str] | None = None,
        residual_nan_threshold: float | None = None,
        target_units: str | None = None,
        file_format: str | None = None,
    ) -> Path:
        """Export marker trajectories to a tabular file.

        Supported formats are CSV, JSON (records orientation), and NPZ. The
        format is inferred from the file extension when ``file_format`` is not
        provided.

        Args:
            output_path: Destination file path.
            include_time: Include a time column in the output.
            markers: Filter for specific markers.
            residual_nan_threshold: Threshold to filter noisy data.
            target_units: Unit conversion (e.g. 'm', 'mm').
            file_format: Explicit format ('csv', 'json', 'npz').

        Note:
            CSV output is automatically sanitized to prevent Excel Formula Injection.
        """
        dataframe = self.points_dataframe(
            include_time=include_time,
            markers=markers,
            residual_nan_threshold=residual_nan_threshold,
            target_units=target_units,
        )
        return export_dataframe(
            dataframe,
            output_path,
            file_format,
            self.file_path.name,
            self.get_metadata().units,
            sanitize=True,
        )

    def export_analog(
        self,
        output_path: Path | str,
        *,
        include_time: bool = True,
        file_format: str | None = None,
    ) -> Path:
        """Export analog channels to a tabular file.

        Supports the same formats as :meth:`export_points`. Empty analog data
        produces an output file with headers so downstream automation can rely
        on the presence of the export artifact.

        Args:
            output_path: Destination file path.
            include_time: Include a time column in the output.
            file_format: Explicit format ('csv', 'json', 'npz').

        Note:
            CSV output is automatically sanitized to prevent Excel Formula Injection.
        """
        dataframe = self.analog_dataframe(include_time=include_time)
        return export_dataframe(
            dataframe,
            output_path,
            file_format,
            self.file_path.name,
            self.get_metadata().units,
            sanitize=True,
        )

    def get_force_plate_channels(self) -> dict[int, dict[str, str]]:
        """Detect and map force plate channels by plate number.

        Force plate channels are identified by common naming conventions:
        - Fx1, Fy1, Fz1, Mx1, My1, Mz1 (standard)
        - Force.Fx1, Force.Fy1, etc. (prefixed)
        - FP1Force1, FP1Force2, etc. (Vicon-style)

        Returns:
            Dictionary mapping plate number (1-indexed) to channel names:
            {1: {'fx': 'Fx1', 'fy': 'Fy1', 'fz': 'Fz1',
                 'mx': 'Mx1', 'my': 'My1', 'mz': 'Mz1'}, ...}
        """
        return detect_force_plate_channels(self.get_metadata().analog_labels)

    def force_plate_dataframe(
        self,
        plate_number: int | None = None,
        include_time: bool = True,
        compute_cop: bool = True,
        ground_height: float = 0.0,
    ) -> pd.DataFrame:
        """Extract force plate data as a wide-format DataFrame.

        Implements Guideline E5: Ground Reaction Forces.

        Args:
            plate_number: Specific plate to extract (1-indexed), or None for all.
            include_time: Whether to include a time column.
            compute_cop: Whether to compute center of pressure.
            ground_height: Height of ground plane for COP z-coordinate [m].

        Returns:
            DataFrame with columns:
            - sample: Sample index
            - time: Time in seconds (if include_time=True)
            - plate: Force plate number (1-indexed)
            - fx, fy, fz: Force components [N]
            - mx, my, mz: Moment components [N·m]
            - cop_x, cop_y, cop_z: COP position [m] (if compute_cop=True)
        """
        metadata = self.get_metadata()
        if metadata.force_plates:
            c3d_data = self._load()
            return build_force_plate_dataframe_from_calibration(
                c3d_data["data"]["analogs"],
                metadata.force_plates,
                metadata.analog_rate,
                self.file_path.name,
                plate_number,
                include_time,
                compute_cop,
                ground_height,
            )
        plate_channels = self.get_force_plate_channels()
        analog_df = self.analog_dataframe(include_time=False)
        return build_force_plate_dataframe(
            plate_channels,
            analog_df,
            metadata.analog_rate,
            self.file_path.name,
            plate_number,
            include_time,
            compute_cop,
            ground_height,
        )

    def get_force_plate_count(self) -> int:
        """Return the number of detected force plates.

        Prefers ``FORCE_PLATFORM.USED`` from the C3D parameter group when
        available; falls back to analog-channel-name regex detection otherwise.
        """
        metadata = self.get_metadata()
        if metadata.force_plates:
            return len(metadata.force_plates)
        return len(self.get_force_plate_channels())

    def _load(self) -> C3DMapping:
        """Load the C3D file if not already loaded."""
        if self._c3d_data is None:
            self._c3d_data = load_c3d(self.file_path)
        return self._c3d_data

    @staticmethod
    def _force_plate_columns(
        include_time: bool,
        compute_cop: bool,
    ) -> list[str]:
        """Return column names for an empty force plate DataFrame."""
        return force_plate_columns(include_time, compute_cop)
