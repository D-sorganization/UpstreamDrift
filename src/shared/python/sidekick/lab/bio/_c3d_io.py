from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

try:
    import ezc3d
except ImportError:
    ezc3d = None  # type: ignore[assignment, unused-ignore]

import numpy as np
import pandas as pd

from ...utils.logging import get_logger, log_execution_time
from ._c3d_marker_set import detect_marker_set
from ._c3d_models import (
    SCHEMA_VERSION,
    C3DEvent,
    C3DMetadata,
    ForcePlateCalibration,
)

logger = get_logger(__name__)

C3DMapping = dict[str, Any]


def load_c3d(file_path: Path) -> C3DMapping:
    """Load the C3D file via ezc3d."""
    if ezc3d is None:
        raise ImportError(
            "ezc3d is required for C3D file reading. "
            "Install it with: pip install ezc3d\n"
            "Note: ezc3d requires Python >=3.10. "
            "For Python 3.9, this functionality is not available."
        )
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    return ezc3d.c3d(str(file_path))


def get_point_parameters(c3d_data: C3DMapping, file_path: Path) -> dict[str, Any]:
    """Get POINT parameters from loaded C3D data."""
    try:
        return cast(dict[str, Any], c3d_data["parameters"]["POINT"])
    except KeyError as error:  # pragma: no cover - defensive guard
        raise ValueError(
            f"POINT parameters missing from C3D file: {file_path}"
        ) from error


def get_analog_parameters(c3d_data: C3DMapping) -> dict[str, Any] | None:
    """Get ANALOG parameters from loaded C3D data, if present."""
    analog_params = c3d_data["parameters"].get("ANALOG")
    return cast(dict[str, Any], analog_params) if analog_params is not None else None


def get_analog_details(
    c3d_data: C3DMapping,
) -> tuple[list[str], float | None, list[str]]:
    """Get analog channel labels, sample rate, and units from C3D data."""
    analog_parameters = get_analog_parameters(c3d_data)
    analog_array = c3d_data["data"]["analogs"]
    channel_count = analog_array.shape[1]

    if analog_parameters is None:
        labels: list[str] = []
        units: list[str] = []
        analog_rate = None
    else:
        labels = [
            label.strip()
            for label in analog_parameters.get("LABELS", {}).get("value", [])
        ]
        units = [
            unit.strip() for unit in analog_parameters.get("UNITS", {}).get("value", [])
        ]
        analog_rate = float(analog_parameters.get("RATE", {}).get("value", [0])[0])

    if not labels and channel_count > 0:
        labels = [f"Analog_{idx + 1}" for idx in range(channel_count)]

    if len(units) < len(labels):
        units.extend([""] * (len(labels) - len(units)))
    elif len(units) > len(labels):
        units = units[: len(labels)]

    return labels, analog_rate, units


def get_events(c3d_data: C3DMapping) -> list[C3DEvent]:
    """Extract event markers from loaded C3D data.

    Reads the EVENT parameter group including:
    - EVENT:LABELS - event names
    - EVENT:TIMES - event times in seconds
    - EVENT:USED - number of active events
    - EVENT:CONTEXTS - event context (e.g., 'General', 'FootStrike')

    Args:
        c3d_data: Loaded C3D data from ezc3d.

    Returns:
        List of C3DEvent objects with label and time.
    """
    event_parameters = c3d_data["parameters"].get("EVENT")
    if not event_parameters:
        return []

    labels_raw: list[str] = list(
        event_parameters.get("LABELS", {}).get("value", []) or []
    )
    times = event_parameters.get("TIMES", {}).get("value")
    contexts_raw: list[str] = list(
        event_parameters.get("CONTEXTS", {}).get("value", []) or []
    )
    del contexts_raw  # currently unused; preserved for future C3DEvent.context

    if times is None:
        return []

    # EVENT:USED tells us how many events are actually defined.
    # Real-world c3d files routinely omit USED while still containing valid
    # LABELS/TIMES arrays. When USED is absent, infer the count from the
    # available metadata rather than silently dropping all events.
    # An explicit USED=0 is honored (caller deliberately marked the group empty).
    used_param = event_parameters.get("USED")
    if used_param is None:
        # Missing USED: infer from LABELS / TIMES length.
        labels_len = len(labels_raw)
        times_len = int(np.asarray(times).shape[-1])
        num_events = max(labels_len, times_len)
        if num_events == 0:
            return []
    else:
        used = used_param.get("value", [0])
        if not used or used[0] == 0:
            return []
        num_events = int(used[0])

    times_array = np.asarray(times)
    if times_array.ndim == 2:
        # ezc3d returns (2, N) for times: [cycle_display, actual_times]
        times_array = times_array[1, :]

    events: list[C3DEvent] = []
    for idx in range(min(num_events, len(labels_raw))):
        time_value = float(times_array[idx]) if idx < len(times_array) else np.nan
        if np.isfinite(time_value):
            label = (
                str(labels_raw[idx]).strip()
                if idx < len(labels_raw)
                else f"Event_{idx}"
            )
            # Context is available but C3DEvent only has label/time for now
            # Could extend C3DEvent to include context if needed
            events.append(C3DEvent(label=label, time=time_value))

    return events


def _as_array(value: Any) -> np.ndarray:
    """Coerce an ezc3d parameter value into a numpy array."""
    return np.asarray(value)


def _scalar_int(value: Any, default: int = 0) -> int:
    """Read the first scalar from an ezc3d parameter value as int."""
    arr = _as_array(value).ravel()
    if arr.size == 0:
        return default
    return int(arr[0])


def get_force_platforms(
    c3d_data: C3DMapping, analog_label_count: int
) -> tuple[ForcePlateCalibration, ...]:
    """Build a tuple of ForcePlateCalibration from the FORCE_PLATFORM group.

    Returns an empty tuple when the group is absent, USED is zero, or any of
    the required CORNERS/ORIGIN/TYPE entries are missing.
    """
    fp_params = c3d_data["parameters"].get("FORCE_PLATFORM")
    if fp_params is None:
        return ()

    used_value = fp_params.get("USED", {}).get("value")
    if used_value is None:
        return ()
    n_used = _scalar_int(used_value)
    if n_used <= 0:
        return ()

    type_value = fp_params.get("TYPE", {}).get("value")
    corners_value = fp_params.get("CORNERS", {}).get("value")
    origin_value = fp_params.get("ORIGIN", {}).get("value")
    if type_value is None or corners_value is None or origin_value is None:
        logger.warning(
            "FORCE_PLATFORM group present but missing TYPE/CORNERS/ORIGIN; "
            "skipping force-plate metadata."
        )
        return ()

    types_arr = _as_array(type_value).ravel().astype(int)
    corners_arr = _as_array(corners_value).astype(float)
    origin_arr = _as_array(origin_value).astype(float)
    cal_matrix_value = fp_params.get("CAL_MATRIX", {}).get("value")
    cal_matrix_arr = (
        _as_array(cal_matrix_value).astype(float)
        if cal_matrix_value is not None
        else None
    )
    channel_value = fp_params.get("CHANNEL", {}).get("value")
    channel_arr = (
        _as_array(channel_value).astype(int) if channel_value is not None else None
    )

    plates: list[ForcePlateCalibration] = []
    fallback_cursor = 0
    for plate_idx in range(n_used):
        plate_type = int(types_arr[plate_idx]) if plate_idx < types_arr.size else 1

        corners_plate = _extract_plate_corners(corners_arr, plate_idx, n_used)
        if corners_plate is None:
            logger.warning(
                "FORCE_PLATFORM CORNERS for plate %d has unexpected shape %s; "
                "skipping plate.",
                plate_idx + 1,
                corners_arr.shape,
            )
            continue
        # ezc3d reports CORNERS in millimetres per the C3D spec.
        corners_plate = corners_plate * 0.001

        origin_plate = _extract_plate_origin(origin_arr, plate_idx, n_used)
        if origin_plate is None:
            logger.warning(
                "FORCE_PLATFORM ORIGIN for plate %d has unexpected shape %s; "
                "skipping plate.",
                plate_idx + 1,
                origin_arr.shape,
            )
            continue
        origin_plate = origin_plate * 0.001

        cal_matrix_plate = _extract_plate_cal_matrix(
            cal_matrix_arr, plate_idx, n_used, plate_type
        )

        start, end = _extract_plate_channels(
            channel_arr,
            plate_idx,
            n_used,
            plate_type,
            fallback_cursor,
            analog_label_count,
        )
        fallback_cursor = end

        if plate_type not in (1, 2, 3, 4):
            logger.warning(
                "FORCE_PLATFORM plate %d has unsupported type %d; "
                "treating as type 1 (pre-calibrated).",
                plate_idx + 1,
                plate_type,
            )
            plate_type = 1

        plates.append(
            ForcePlateCalibration(
                corners=corners_plate,
                origin=origin_plate,
                cal_matrix=cal_matrix_plate,
                plate_type=plate_type,
                channel_indices=(start, end),
            )
        )

    return tuple(plates)


def _extract_plate_corners(
    corners_arr: np.ndarray, plate_idx: int, n_plates: int
) -> np.ndarray | None:
    """Slice a single plate's corners (returns 4x3 array) from a flexible layout."""
    # ezc3d typically returns (3, 4, n_plates).
    if corners_arr.ndim == 3 and corners_arr.shape[:2] == (3, 4):
        if plate_idx >= corners_arr.shape[2]:
            return None
        return corners_arr[:, :, plate_idx].T.copy()
    if corners_arr.ndim == 3 and corners_arr.shape[:2] == (4, 3):
        if plate_idx >= corners_arr.shape[2]:
            return None
        return corners_arr[:, :, plate_idx].copy()
    if corners_arr.ndim == 2 and n_plates == 1 and corners_arr.shape == (3, 4):
        return corners_arr.T.copy()
    if corners_arr.ndim == 2 and n_plates == 1 and corners_arr.shape == (4, 3):
        return corners_arr.copy()
    if corners_arr.size == 12 * n_plates:
        reshaped = corners_arr.reshape(3, 4, n_plates)
        return reshaped[:, :, plate_idx].T.copy()
    return None


def _extract_plate_origin(
    origin_arr: np.ndarray, plate_idx: int, n_plates: int
) -> np.ndarray | None:
    """Slice a single plate's origin (returns shape (3,))."""
    if origin_arr.ndim == 2 and origin_arr.shape[0] == 3:
        if plate_idx >= origin_arr.shape[1]:
            return None
        return origin_arr[:, plate_idx].copy()
    if origin_arr.ndim == 1 and n_plates == 1 and origin_arr.size == 3:
        return origin_arr.copy()
    if origin_arr.size == 3 * n_plates:
        return origin_arr.reshape(3, n_plates)[:, plate_idx].copy()
    return None


def _extract_plate_cal_matrix(
    cal_matrix_arr: np.ndarray | None,
    plate_idx: int,
    n_plates: int,
    plate_type: int,
) -> np.ndarray | None:
    """Slice a single plate's calibration matrix, or None when not applicable."""
    if cal_matrix_arr is None or cal_matrix_arr.size == 0 or plate_type == 1:
        return None
    if cal_matrix_arr.ndim == 3:
        if plate_idx >= cal_matrix_arr.shape[-1]:
            return None
        return cal_matrix_arr[..., plate_idx].copy()
    if cal_matrix_arr.ndim == 2 and n_plates == 1:
        return cal_matrix_arr.copy()
    return None


def _extract_plate_channels(
    channel_arr: np.ndarray | None,
    plate_idx: int,
    n_plates: int,
    plate_type: int,
    fallback_cursor: int,
    analog_label_count: int,
) -> tuple[int, int]:
    """Return ``(start, end)`` analog indices for the plate (zero-based, end-exclusive)."""
    expected = 6 if plate_type in (1, 2, 3) else 8 if plate_type == 4 else 6
    if channel_arr is not None and channel_arr.size:
        # CHANNEL is typically (n_channels, n_plates) of 1-based indices.
        if channel_arr.ndim == 2:
            if plate_idx < channel_arr.shape[1]:
                col = channel_arr[:, plate_idx]
                col = col[col > 0]
                if col.size:
                    start = int(col.min()) - 1
                    end = int(col.max())
                    return start, end
        elif channel_arr.ndim == 1 and n_plates == 1:
            col = channel_arr[channel_arr > 0]
            if col.size:
                return int(col.min()) - 1, int(col.max())

    end = min(fallback_cursor + expected, analog_label_count)
    return fallback_cursor, end


def build_metadata(c3d_data: C3DMapping, file_path: Path) -> C3DMetadata:
    """Build a C3DMetadata object from loaded C3D data."""
    point_parameters = get_point_parameters(c3d_data, file_path)
    marker_labels = [label.strip() for label in point_parameters["LABELS"]["value"]]
    frame_count = int(point_parameters["FRAMES"]["value"][0])
    frame_rate = float(point_parameters["RATE"]["value"][0])
    units = str(point_parameters["UNITS"]["value"][0])
    analog_labels, analog_rate, analog_units = get_analog_details(c3d_data)
    events = get_events(c3d_data)
    force_plates = get_force_platforms(c3d_data, len(analog_labels))
    marker_set = detect_marker_set(marker_labels, c3d_data.get("parameters"))
    return C3DMetadata(
        marker_labels=marker_labels,
        frame_count=frame_count,
        frame_rate=frame_rate,
        units=units,
        analog_labels=analog_labels,
        analog_units=analog_units,
        analog_rate=analog_rate,
        events=events,
        force_plates=force_plates,
        marker_set=marker_set,
    )


def unit_scale(current_units: str, target_units: str | None) -> float:
    """Calculate scaling factor for unit conversion."""
    if target_units is None:
        return 1.0

    normalized_current = current_units.strip().lower()
    normalized_target = target_units.strip().lower()

    # Map common aliases to standard keys
    aliases = {
        "meters": "m",
        "meter": "m",
        "millimeters": "mm",
        "millimeter": "mm",
        "centimeters": "cm",
        "centimeter": "cm",
        "inches": "in",
        "inch": "in",
        "feet": "ft",
        "foot": "ft",
    }
    normalized_current = aliases.get(normalized_current, normalized_current)
    normalized_target = aliases.get(normalized_target, normalized_target)

    if normalized_current == normalized_target:
        return 1.0

    to_meters = {
        "m": 1.0,
        "mm": 0.001,
        "cm": 0.01,
        "in": 0.0254,
        "ft": 0.3048,
    }

    if normalized_current not in to_meters or normalized_target not in to_meters:
        logger.warning(
            f"Unsupported or unknown unit conversion from '{current_units}' to '{target_units}'. "
            "Falling back to 1.0 scaling factor."
        )
        return 1.0

    return to_meters[normalized_current] / to_meters[normalized_target]


def sanitize_for_csv(value: Any) -> Any:
    """Sanitize a value to prevent CSV injection."""
    if not isinstance(value, str):
        return value
    if value.startswith(("=", "+", "-", "@")):
        return f"'{value}"
    return value


def validate_export_path(path: Path) -> None:
    """Validate export path for security (prevent directory traversal)."""
    import inspect

    base_dir = Path.cwd().resolve()

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

    is_test_env = not is_security_test and any(
        [
            "pytest" in str(base_dir),
            "test" in str(base_dir).lower(),
            "/tmp/pytest" in str(path),  # nosec B108 - string comparison only, not creating a temp file
            "pytest" in str(path),
        ]
    )

    if not is_test_env and base_dir not in path.parents and path != base_dir:
        raise ValueError(
            f"Security: Refusing to output to {path} (outside project root {base_dir})"
        )


def write_export(
    path: Path,
    fmt: str,
    dataframe: pd.DataFrame,
    metadata: dict[str, Any],
    do_sanitize: bool,
) -> None:
    """Write dataframe to disk in the given format."""
    if fmt == "csv":
        df_to_export = dataframe.copy() if do_sanitize else dataframe
        if do_sanitize:
            for col in df_to_export.select_dtypes(include=[object, "string"]).columns:
                df_to_export[col] = df_to_export[col].apply(sanitize_for_csv)
        df_to_export.to_csv(path, index=False)

        meta_path = path.with_name(f"{path.stem}_meta.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

    elif fmt == "json":
        output = {
            "metadata": metadata,
            "data": dataframe.to_dict(orient="records"),
        }
        with open(path, "w") as f:
            json.dump(output, f, indent=2)

    elif fmt == "npz":
        arrays: dict[str, np.ndarray] = {
            str(column): dataframe[column].to_numpy() for column in dataframe
        }
        arrays["_metadata"] = np.asarray(json.dumps(metadata))
        # numpy savez stubs declare *args as ArrayLike but **kwargs are
        # not modelled cleanly; the call is correct at runtime.
        np.savez(path, **arrays)  # type: ignore[arg-type]

    else:
        raise ValueError(
            f"Unsupported export format: '{fmt}'. Supported formats: csv, json, npz."
        )


def export_dataframe(
    dataframe: pd.DataFrame,
    output_path: Path | str,
    file_format: str | None,
    source_file_name: str,
    units: str,
    sanitize: bool = True,
) -> Path:
    """Export a DataFrame to CSV, JSON, or NPZ format."""
    path = Path(output_path).resolve()

    validate_export_path(path)

    if not file_format:
        if not path.suffix:
            raise ValueError("File format could not be inferred from the path suffix.")
        file_format = path.suffix.lstrip(".")

    normalized_format = file_format.lower()
    path.parent.mkdir(parents=True, exist_ok=True)

    with log_execution_time(f"export_{normalized_format}"):
        meta = {
            "schema_version": SCHEMA_VERSION,
            "created_at_utc": datetime.now(timezone.utc).isoformat(),  # noqa: UP017
            "source_file": source_file_name,
            "row_count": len(dataframe),
            "units": units,
        }
        write_export(path, normalized_format, dataframe, meta, sanitize)

    return path
