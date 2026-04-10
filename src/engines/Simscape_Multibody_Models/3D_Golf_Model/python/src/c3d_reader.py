"""Utilities for loading and interpreting C3D motion-capture files."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    from ._c3d_export import export_dataframe, unit_scale
    from ._c3d_force_plates import (
        detect_force_plate_channels,
        extract_force_plate_dataframe,
        force_plate_columns,
        pivot_analog_to_wide,
    )
    from ._c3d_io import build_metadata, load_c3d_file
    from ._c3d_models import (
        BIOMECHANICAL_MARKER_MAX_M,
        BIOMECHANICAL_MARKER_MIN_M,
        SCHEMA_VERSION,
        C3DEvent,
        C3DMapping,
        C3DMetadata,
    )
    from .logger_utils import get_logger
except ImportError:
    from _c3d_export import export_dataframe, unit_scale  # type: ignore[no-redef]
    from _c3d_force_plates import (  # type: ignore[no-redef]
        detect_force_plate_channels,
        extract_force_plate_dataframe,
        force_plate_columns,
        pivot_analog_to_wide,
    )
    from _c3d_io import build_metadata, load_c3d_file  # type: ignore[no-redef]
    from _c3d_models import (  # type: ignore[no-redef]
        BIOMECHANICAL_MARKER_MAX_M,
        BIOMECHANICAL_MARKER_MIN_M,
        SCHEMA_VERSION,
        C3DEvent,
        C3DMapping,
        C3DMetadata,
    )
    from logger_utils import get_logger  # type: ignore[no-redef]

logger = get_logger(__name__)

__all__ = [
    "C3DEvent",
    "C3DMetadata",
    "C3DMapping",
    "C3DDataReader",
    "SCHEMA_VERSION",
    "BIOMECHANICAL_MARKER_MIN_M",
    "BIOMECHANICAL_MARKER_MAX_M",
    "load_tour_average_reader",
]


class C3DDataReader:
    """Loads marker trajectories and metadata from a C3D file."""

    def __init__(self, file_path: Path | str) -> None:
        """Initialize the C3D data reader with a file path."""
        if not (file_path is not None):
            raise ValueError("file_path must be provided")
        if not (file_path is not None):
            raise ValueError("file_path must be provided")
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
        if not (include_time is not None):
            raise ValueError("include_time must be provided")
        if not (include_time is not None):
            raise ValueError("include_time must be provided")
        c3d_data = self._load()
        metadata = self.get_metadata()
        points = c3d_data["data"]["points"]

        marker_labels = np.array(metadata.marker_labels)

        if markers:
            mask = np.isin(marker_labels, list(markers))
            marker_labels = marker_labels[mask]
            points = points[:, mask, :]

        sorted_labels, points = self._sort_markers(marker_labels, points)
        coordinates = self._compute_coordinates(points, metadata, target_units)

        self._validate_marker_positions(coordinates, metadata, target_units)

        residuals = points[3, :, :].T.reshape(-1)
        if residual_nan_threshold is not None:
            too_noisy = residuals > residual_nan_threshold
            coordinates[too_noisy, :] = np.nan

        dataframe = self._build_points_dataframe(
            sorted_labels, coordinates, residuals, metadata, include_time
        )

        logger.info(
            "Loaded %s frames for %s markers from %s",
            metadata.frame_count,
            len(sorted_labels),
            self.file_path.name,
        )
        return dataframe

    @staticmethod
    def _sort_markers(
        marker_labels: np.ndarray, points: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        sort_indices = np.argsort(marker_labels)
        return marker_labels[sort_indices], points[:, sort_indices, :]

    def _compute_coordinates(
        self,
        points: np.ndarray,
        metadata: C3DMetadata,
        target_units: str | None,
    ) -> np.ndarray:
        raw_coordinates = np.transpose(points[:3, :, :], axes=(2, 1, 0)).reshape(-1, 3)
        return raw_coordinates * unit_scale(metadata.units, target_units)

    @staticmethod
    def _validate_marker_positions(
        coordinates: np.ndarray,
        metadata: C3DMetadata,
        target_units: str | None,
    ) -> None:
        if coordinates.size == 0:
            return

        min_pos = np.nanmin(coordinates)
        max_pos = np.nanmax(coordinates)

        if np.isnan(min_pos) or np.isnan(max_pos):
            logger.warning(
                "All marker coordinates are NaN or non-finite; skipping unit "
                "range validation (Guideline P1). Verify upstream data quality "
                "and missing-data handling."
            )
            return

        if min_pos < BIOMECHANICAL_MARKER_MIN_M:
            logger.warning(
                "Suspiciously small marker positions detected (< 1mm). "
                f"Min position: {min_pos:.6f}m. "
                f"Source units: {metadata.units}, target: "
                f"{target_units or 'unchanged'}. "
                "Guideline P1: Verify unit conversion is correct to "
                "avoid 1000x errors."
            )

        if max_pos > BIOMECHANICAL_MARKER_MAX_M:
            logger.error(
                "Unrealistic marker positions detected (> 10m). "
                f"Max position: {max_pos:.2f}m. "
                f"Source units: {metadata.units}, target: "
                f"{target_units or 'unchanged'}. "
                "Guideline P1 VIOLATION: Likely unit conversion error."
            )
            raise ValueError(
                f"Marker positions exceed {BIOMECHANICAL_MARKER_MAX_M}m "
                f"(max: {max_pos:.2f}m) - likely unit error. "
                f"Check that source units '{metadata.units}' are correct. "
                "Common issue: mm labeled as m or vice versa."
            )

    @staticmethod
    def _build_points_dataframe(
        sorted_labels: np.ndarray,
        coordinates: np.ndarray,
        residuals: np.ndarray,
        metadata: C3DMetadata,
        include_time: bool,
    ) -> pd.DataFrame:
        if not (sorted_labels is not None):
            raise ValueError("sorted_labels must be provided")
        if not (sorted_labels is not None):
            raise ValueError("sorted_labels must be provided")
        current_marker_count = len(sorted_labels)
        frame_indices = np.repeat(np.arange(metadata.frame_count), current_marker_count)
        marker_names = np.tile(sorted_labels, metadata.frame_count)

        data: dict[str, Any] = {
            "frame": frame_indices,
            "marker": marker_names,
            "x": coordinates[:, 0],
            "y": coordinates[:, 1],
            "z": coordinates[:, 2],
            "residual": residuals,
        }

        if include_time:
            if metadata.frame_rate > 0:
                data["time"] = frame_indices / metadata.frame_rate
            else:
                logger.warning(
                    "Frame rate is 0. Time column will be omitted "
                    "despite include_time=True."
                )

        return pd.DataFrame(data).reset_index(drop=True)

    def analog_dataframe(self, include_time: bool = True) -> pd.DataFrame:
        """Return analog channels as a tidy DataFrame.

        Rows are ordered by sample index and channel name so downstream GUI
        components can easily plot synchronized sensor traces.
        """

        if not (include_time is not None):
            raise ValueError("include_time must be provided")
        if not (include_time is not None):
            raise ValueError("include_time must be provided")
        c3d_data = self._load()
        metadata = self.get_metadata()
        analog_array = c3d_data["data"]["analogs"]
        subframes, channel_count, frame_count = analog_array.shape
        analog_rate = metadata.analog_rate

        columns = ["sample", "channel", "value"]
        if include_time and analog_rate:
            columns = ["sample", "time", "channel", "value"]

        if channel_count == 0:
            return pd.DataFrame(columns=columns)

        values = analog_array.transpose(2, 0, 1).reshape(
            frame_count * subframes, channel_count
        )
        sample_indices = np.arange(values.shape[0])
        channel_names = np.array(
            metadata.analog_labels
            or [f"Analog_{idx + 1}" for idx in range(channel_count)]
        )

        dataframe = pd.DataFrame(
            {
                "sample": np.repeat(sample_indices, channel_count),
                "channel": np.tile(channel_names, values.shape[0]),
                "value": values.reshape(-1),
            }
        )

        if include_time and analog_rate:
            dataframe.insert(1, "time", dataframe["sample"] / analog_rate)

        return dataframe

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

        if not (output_path is not None):
            raise ValueError("output_path must be provided")
        if not (output_path is not None):
            raise ValueError("output_path must be provided")
        dataframe = self.points_dataframe(
            include_time=include_time,
            markers=markers,
            residual_nan_threshold=residual_nan_threshold,
            target_units=target_units,
        )
        metadata = self.get_metadata()
        result = export_dataframe(
            dataframe,
            output_path,
            file_format,
            sanitize=True,
            source_file_name=self.file_path.name,
            units=metadata.units,
        )
        logger.info("Exported %s rows to %s", len(dataframe), result)
        return result

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

        if not (output_path is not None):
            raise ValueError("output_path must be provided")
        if not (output_path is not None):
            raise ValueError("output_path must be provided")
        dataframe = self.analog_dataframe(include_time=include_time)
        metadata = self.get_metadata()
        result = export_dataframe(
            dataframe,
            output_path,
            file_format,
            sanitize=True,
            source_file_name=self.file_path.name,
            units=metadata.units,
        )
        logger.info("Exported %s rows to %s", len(dataframe), result)
        return result

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
        metadata = self.get_metadata()
        return detect_force_plate_channels(metadata.analog_labels)

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
        if not (include_time is not None):
            raise ValueError("include_time must be provided")
        if not (include_time is not None):
            raise ValueError("include_time must be provided")
        plate_channels = self.get_force_plate_channels()

        if not plate_channels:
            logger.warning(
                "No force plate channels detected in C3D file. "
                "Expected channels like Fx1, Fy1, Fz1, Mx1, My1, Mz1."
            )
            return pd.DataFrame(columns=force_plate_columns(include_time, compute_cop))

        if plate_number is not None:
            if plate_number not in plate_channels:
                raise ValueError(
                    f"Force plate {plate_number} not found. "
                    f"Available plates: {list(plate_channels.keys())}"
                )
            plate_channels = {plate_number: plate_channels[plate_number]}

        analog_wide = pivot_analog_to_wide(self.analog_dataframe(include_time=False))
        metadata = self.get_metadata()

        return extract_force_plate_dataframe(
            plate_channels,
            analog_wide,
            metadata.analog_rate,
            include_time,
            compute_cop,
            ground_height,
            self.file_path.name,
        )

    def get_force_plate_count(self) -> int:
        """Return the number of detected force plates."""
        return len(self.get_force_plate_channels())

    def _load(self) -> C3DMapping:
        """Load the C3D file if not already loaded."""
        if self._c3d_data is None:
            self._c3d_data = load_c3d_file(self.file_path)
        return self._c3d_data


def load_tour_average_reader(base_directory: Path | None = None) -> C3DDataReader:
    """Convenience loader for the repository's Tour average capture.

    Args:
        base_directory: Optional base directory containing the repository files. If
            omitted, the repository root is derived from this module's location.

    Returns:
        A configured :class:`C3DDataReader` pointing to the Tour average capture file.
    """

    base_path = base_directory or Path(__file__).resolve().parents[2]
    default_path = (
        base_path / "matlab" / "Data" / "Optical C3D Files" / "CaptureExport Tour average.c3d"
    )
    return C3DDataReader(default_path)
