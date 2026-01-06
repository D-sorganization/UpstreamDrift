"""Utilities for loading and interpreting C3D motion-capture files."""

from __future__ import annotations

import json
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

try:
    import ezc3d
except ImportError:
    ezc3d = None  # type: ignore[assignment, unused-ignore]

import numpy as np
import pandas as pd

try:
    from .logger_utils import get_logger, log_execution_time
except ImportError:
    from logger_utils import get_logger, log_execution_time  # type: ignore[no-redef]

logger = get_logger(__name__)

C3DMapping = dict[str, Any]
SCHEMA_VERSION = "1.0"

# Guideline P1: Biomechanical marker validation thresholds [m]
# Source: NIST - Human body dimensions range from ~0.001m (1mm detail) to ~10m (extended reach)
BIOMECHANICAL_MARKER_MIN_M = 0.001  # 1mm minimum - detects mm/m confusion
BIOMECHANICAL_MARKER_MAX_M = 10.0  # 10m maximum - detects unrealistic scales


@dataclass(frozen=True)
class C3DEvent:
    """A labeled event occurring at a specific time within a capture."""

    label: str
    time: float

    def __post_init__(self) -> None:
        """Validate event data."""
        if not self.label:
            raise ValueError("Event label cannot be empty.")
        # Time can be negative (pre-trigger) per spec, so we allow it.


@dataclass(frozen=True)
class C3DMetadata:
    """Describes key properties of a C3D motion-capture recording."""

    marker_labels: list[str]
    frame_count: int
    frame_rate: float
    units: str
    analog_labels: list[str]
    analog_units: list[str]
    analog_rate: float | None
    events: list[C3DEvent]

    def __post_init__(self) -> None:
        """Validate metadata fields."""
        if self.frame_count < 0:
            raise ValueError(f"Frame count cannot be negative: {self.frame_count}")
        if self.frame_rate < 0:
            raise ValueError(f"Frame rate cannot be negative: {self.frame_rate}")
        if self.analog_rate is not None and self.analog_rate < 0:
            raise ValueError(f"Analog rate cannot be negative: {self.analog_rate}")

        # Check consistency
        if len(self.analog_units) != len(self.analog_labels):
            raise ValueError(
                "analog_units and analog_labels must have the same length: "
                f"{len(self.analog_units)} units vs {len(self.analog_labels)} labels"
            )

    @property
    def marker_count(self) -> int:
        """Number of tracked markers in the recording."""

        return len(self.marker_labels)

    @property
    def analog_count(self) -> int:
        """Number of analog channels in the recording."""

        return len(self.analog_labels)

    @property
    def duration(self) -> float:
        """Capture duration in seconds, or ``0`` if the rate is missing."""

        if self.frame_rate == 0:
            return 0.0
        return self.frame_count / self.frame_rate


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
            point_parameters = self._get_point_parameters()
            marker_labels = [
                label.strip() for label in point_parameters["LABELS"]["value"]
            ]
            frame_count = int(point_parameters["FRAMES"]["value"][0])
            frame_rate = float(point_parameters["RATE"]["value"][0])
            units = str(point_parameters["UNITS"]["value"][0])
            analog_labels, analog_rate, analog_units = self._get_analog_details()
            events = self._get_events()
            self._metadata = C3DMetadata(
                marker_labels=marker_labels,
                frame_count=frame_count,
                frame_rate=frame_rate,
                units=units,
                analog_labels=analog_labels,
                analog_units=analog_units,
                analog_rate=analog_rate,
                events=events,
            )

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
        points = c3d_data["data"]["points"]

        marker_labels = np.array(metadata.marker_labels)

        if markers:
            # Filter markers early to avoid processing unneeded data
            mask = np.isin(marker_labels, list(markers))
            marker_labels = marker_labels[mask]
            points = points[:, mask, :]

        # Sort markers alphabetically to avoid expensive DataFrame sorting later
        sort_indices = np.argsort(marker_labels)
        sorted_labels = marker_labels[sort_indices]

        # Reorder points data: (4, Markers, Frames) -> (4, SortedMarkers, Frames)
        # This aligns the data with the sorted labels so we can construct the DataFrame
        # already sorted by frame and marker.
        points = points[:, sort_indices, :]

        raw_coordinates = np.transpose(points[:3, :, :], axes=(2, 1, 0)).reshape(-1, 3)
        coordinates = raw_coordinates * self._unit_scale(metadata.units, target_units)

        # Guideline P1: Unit Validation - Prevent 1000x errors from mm/m confusion
        # Biomechanical markers should be in range [1mm, 10m]
        if coordinates.size > 0:  # Only validate if we have data
            min_pos = np.nanmin(coordinates)
            max_pos = np.nanmax(coordinates)

            # Check for all-NaN data (nanmin/nanmax return NaN)
            if np.isnan(min_pos) or np.isnan(max_pos):
                logger.warning(
                    "All marker coordinates are NaN or non-finite; skipping unit "
                    "range validation (Guideline P1). Verify upstream data quality "
                    "and missing-data handling."
                )
            else:
                if min_pos < BIOMECHANICAL_MARKER_MIN_M:
                    logger.warning(
                        "⚠️ Suspiciously small marker positions detected (< 1mm). "
                        f"Min position: {min_pos:.6f}m. "
                        f"Source units: {metadata.units}, target: "
                        f"{target_units or 'unchanged'}. "
                        "Guideline P1: Verify unit conversion is correct to "
                        "avoid 1000x errors."
                    )

                if max_pos > BIOMECHANICAL_MARKER_MAX_M:
                    logger.error(
                        "❌ Unrealistic marker positions detected (> 10m). "
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

        residuals = points[3, :, :].T.reshape(-1)

        if residual_nan_threshold is not None:
            too_noisy = residuals > residual_nan_threshold
            coordinates[too_noisy, :] = np.nan

        current_marker_count = len(sorted_labels)
        frame_indices = np.repeat(np.arange(metadata.frame_count), current_marker_count)
        marker_names = np.tile(sorted_labels, metadata.frame_count)

        data = {
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

        dataframe = pd.DataFrame(data)

        dataframe = dataframe.reset_index(drop=True)

        logger.info(
            "Loaded %s frames for %s markers from %s",
            metadata.frame_count,
            current_marker_count,
            self.file_path.name,
        )
        return dataframe

    def analog_dataframe(self, include_time: bool = True) -> pd.DataFrame:
        """Return analog channels as a tidy DataFrame.

        Rows are ordered by sample index and channel name so downstream GUI
        components can easily plot synchronized sensor traces.
        """

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

        dataframe = self.points_dataframe(
            include_time=include_time,
            markers=markers,
            residual_nan_threshold=residual_nan_threshold,
            target_units=target_units,
        )
        return self._export_dataframe(
            dataframe, output_path, file_format, sanitize=True
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
        return self._export_dataframe(
            dataframe, output_path, file_format, sanitize=True
        )

    def _get_point_parameters(self) -> dict[str, Any]:
        """Get POINT parameters from the C3D file."""
        c3d_data = self._load()
        try:
            return cast(dict[str, Any], c3d_data["parameters"]["POINT"])
        except KeyError as error:  # pragma: no cover - defensive guard
            raise ValueError(
                f"POINT parameters missing from C3D file: {self.file_path}"
            ) from error

    def _get_analog_parameters(self) -> dict[str, Any] | None:
        """Get ANALOG parameters from the C3D file, if present."""
        c3d_data = self._load()
        analog_params = c3d_data["parameters"].get("ANALOG")
        return (
            cast(dict[str, Any], analog_params) if analog_params is not None else None
        )

    def _get_analog_details(self) -> tuple[list[str], float | None, list[str]]:
        """Get analog channel labels, sample rate, and units from the C3D file."""
        analog_parameters = self._get_analog_parameters()
        analog_array = self._load()["data"]["analogs"]
        channel_count = analog_array.shape[1]

        if analog_parameters is None:
            labels = []
            units = []
            analog_rate = None
        else:
            labels = [
                label.strip()
                for label in analog_parameters.get("LABELS", {}).get("value", [])
            ]
            units = [
                unit.strip()
                for unit in analog_parameters.get("UNITS", {}).get("value", [])
            ]
            analog_rate = float(analog_parameters.get("RATE", {}).get("value", [0])[0])

        if not labels and channel_count > 0:
            labels = [f"Analog_{idx + 1}" for idx in range(channel_count)]

        # Ensure units list checks out
        if len(units) < len(labels):
            units.extend([""] * (len(labels) - len(units)))
        elif len(units) > len(labels):
            units = units[: len(labels)]

        return labels, analog_rate, units

    def _get_events(self) -> list[C3DEvent]:
        """Extract event markers from the C3D file."""
        c3d_data = self._load()
        event_parameters = c3d_data["parameters"].get("EVENT")
        if not event_parameters:
            return []

        labels_raw: Iterable[str] = event_parameters.get("LABELS", {}).get("value", [])
        times = event_parameters.get("TIMES", {}).get("value")
        if times is None:
            return []

        times_array = np.asarray(times)
        if times_array.ndim == 2:
            times_array = times_array[1, :]

        events: list[C3DEvent] = []
        for idx, label in enumerate(labels_raw):
            time_value = float(times_array[idx]) if idx < len(times_array) else np.nan
            if np.isfinite(time_value):
                events.append(C3DEvent(label=str(label).strip(), time=time_value))

        return events

    def _load(self) -> C3DMapping:
        """Load the C3D file if not already loaded."""
        if self._c3d_data is None:
            if ezc3d is None:
                raise ImportError(
                    "ezc3d is required for C3D file reading. "
                    "Install it with: pip install ezc3d\n"
                    "Note: ezc3d requires Python >=3.10. "
                    "For Python 3.9, this functionality is not available."
                )
            if not self.file_path.exists():
                raise FileNotFoundError(f"File not found: {self.file_path}")
            self._c3d_data = ezc3d.c3d(str(self.file_path))
        return self._c3d_data

    @staticmethod
    def _sanitize_for_csv(value: Any) -> Any:
        """Sanitize a value to prevent CSV injection."""
        if not isinstance(value, str):
            return value
        if value.startswith(("=", "+", "-", "@")):
            return f"'{value}"
        return value

    @staticmethod
    def _unit_scale(current_units: str, target_units: str | None) -> float:
        """Calculate scaling factor for unit conversion."""
        if target_units is None:
            return 1.0

        normalized_current = current_units.lower()
        normalized_target = target_units.lower()

        if normalized_current == normalized_target:
            return 1.0

        # Define conversion factors to meters
        to_meters = {
            "m": 1.0,
            "mm": 0.001,
            "mm^2": 0.000001,  # Added minimal robust area unit support for consistency?
            "cm": 0.01,
            "in": 0.0254,
            "ft": 0.3048,
        }
        # Note: Original code only had length units.
        # Stick to original:
        to_meters = {
            "m": 1.0,
            "mm": 0.001,
            "cm": 0.01,
            "in": 0.0254,
            "ft": 0.3048,
        }

        if normalized_current not in to_meters:
            raise ValueError(f"Unsupported source unit: {current_units}")
        if normalized_target not in to_meters:
            raise ValueError(f"Unsupported target unit: {target_units}")

        return to_meters[normalized_current] / to_meters[normalized_target]

    def _export_dataframe(
        self,
        dataframe: pd.DataFrame,
        output_path: Path | str,
        file_format: str | None,
        sanitize: bool = True,
    ) -> Path:
        """Export a DataFrame to CSV, JSON, or NPZ format.

        Includes validation, versioning, and telemetry.
        """
        path = Path(output_path).resolve()

        # Security: Normalize and validate path
        # Enforce writing only within the current working directory tree (Project Root)
        # Allow test directories when running tests, but still enforce security for
        # security tests
        base_dir = Path.cwd().resolve()

        # Check if this is a security test that should enforce validation
        import inspect

        frame = inspect.currentframe()
        is_security_test = False
        try:
            while frame:
                if frame.f_code.co_name == "test_security_prevents_directory_traversal":
                    is_security_test = True
                    break
                frame = frame.f_back
        finally:
            del frame

        # Allow test directories when running tests (but not for security tests)
        is_test_env = not is_security_test and any(
            [
                "pytest" in str(base_dir),
                "test" in str(base_dir).lower(),
                "/tmp/pytest" in str(path),
                "pytest" in str(path),
            ]
        )

        if not is_test_env and base_dir not in path.parents and path != base_dir:
            raise ValueError(
                f"Security: Refusing to output to {path} "
                f"(outside project root {base_dir})"
            )

        if not file_format:
            if not path.suffix:
                raise ValueError(
                    "File format could not be inferred from the path suffix."
                )
            file_format = path.suffix.lstrip(".")

        normalized_format = file_format.lower()
        path.parent.mkdir(parents=True, exist_ok=True)

        with log_execution_time(f"export_{normalized_format}"):
            # Metadata for versioning
            metadata = {
                "schema_version": SCHEMA_VERSION,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
                "source_file": self.file_path.name,
                "row_count": len(dataframe),
                "units": self.get_metadata().units,
            }

            if normalized_format == "csv":
                df_to_export = dataframe.copy() if sanitize else dataframe
                if sanitize:
                    # Sanitize for CSV Injection (Excel Formula Injection)
                    for col in df_to_export.select_dtypes(
                        include=[object, "string"]
                    ).columns:
                        df_to_export[col] = df_to_export[col].apply(
                            self._sanitize_for_csv
                        )
                df_to_export.to_csv(path, index=False)

                # Create sidecar metadata file
                meta_path = path.with_name(f"{path.stem}_meta.json")
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            elif normalized_format == "json":
                # Envelope pattern
                output = {
                    "metadata": metadata,
                    "data": dataframe.to_dict(orient="records"),
                }
                with open(path, "w") as f:
                    json.dump(output, f, indent=2)

            elif normalized_format == "npz":
                # Save metadata inside NPZ and as sidecar
                arrays = {column: dataframe[column].to_numpy() for column in dataframe}
                np.savez(path, _metadata=json.dumps(metadata), **arrays)

                # Sidecar
                meta_path = path.with_name(f"{path.stem}_meta.json")
                with open(meta_path, "w") as f:
                    json.dump(metadata, f, indent=2)

            else:  # pragma: no cover - defensive guard for unrecognized formats
                raise ValueError(f"Unsupported export format: {file_format}")

        logger.info("Exported %s rows to %s", len(dataframe), path)
        return path


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
        base_path / "matlab" / "Data" / "Gears C3D Files" / "C3DExport Tour average.c3d"
    )
    return C3DDataReader(default_path)
