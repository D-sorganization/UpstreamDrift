from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from ...utils.logging import get_logger
from ._c3d_models import C3DMetadata, ForcePlateCalibration

logger = get_logger(__name__)


def build_analog_dataframe(
    c3d_data: Any,
    metadata: C3DMetadata,
    include_time: bool,
) -> pd.DataFrame:
    """Return analog channels as a tidy DataFrame."""
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
        metadata.analog_labels or [f"Analog_{idx + 1}" for idx in range(channel_count)]
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


def detect_force_plate_channels(labels: list[str]) -> dict[int, dict[str, str]]:
    """Detect and map force plate channels by plate number."""
    plate_channels: dict[int, dict[str, str]] = {}

    standard_pattern = re.compile(r"^(?:Force\.)?([FfMm])([xyzXYZ])(\d+)$")
    prefix_pattern = re.compile(r"^(?:FP|fp)?(\d+)[_.]?([FfMm])([xyzXYZ])$")

    for label in labels:
        label_stripped = label.strip()

        match = standard_pattern.match(label_stripped)
        if match:
            force_or_moment = match.group(1).lower()
            axis = match.group(2).lower()
            plate_num = int(match.group(3))

            if plate_num not in plate_channels:
                plate_channels[plate_num] = {}

            key = f"{force_or_moment}{axis}"
            plate_channels[plate_num][key] = label
            continue

        match = prefix_pattern.match(label_stripped)
        if match:
            plate_num = int(match.group(1))
            force_or_moment = match.group(2).lower()
            axis = match.group(3).lower()

            if plate_num not in plate_channels:
                plate_channels[plate_num] = {}

            key = f"{force_or_moment}{axis}"
            plate_channels[plate_num][key] = label

    return plate_channels


def force_plate_columns(include_time: bool, compute_cop: bool) -> list[str]:
    """Return column names for an empty force plate DataFrame."""
    columns = ["sample", "plate", "fx", "fy", "fz", "mx", "my", "mz"]
    if include_time:
        columns.insert(1, "time")
    if compute_cop:
        columns.extend(["cop_x", "cop_y", "cop_z"])
    return columns


def build_plate_dataframe(
    plate_num: int,
    channels: dict[str, str],
    required_keys: set[str],
    analog_wide: pd.DataFrame,
    compute_cop: bool,
    ground_height: float,
) -> pd.DataFrame | None:
    """Build a DataFrame for a single force plate, or None if channels missing."""
    missing_keys = required_keys - set(channels.keys())
    if missing_keys:
        logger.warning(
            f"Force plate {plate_num} missing channels: {missing_keys}. Skipping."
        )
        return None

    plate_df = pd.DataFrame(
        {
            "sample": analog_wide["sample"],
            "plate": plate_num,
            "fx": analog_wide[channels["fx"]].to_numpy(),
            "fy": analog_wide[channels["fy"]].to_numpy(),
            "fz": analog_wide[channels["fz"]].to_numpy(),
            "mx": analog_wide[channels["mx"]].to_numpy(),
            "my": analog_wide[channels["my"]].to_numpy(),
            "mz": analog_wide[channels["mz"]].to_numpy(),
        }
    )

    if compute_cop:
        fz = plate_df["fz"].to_numpy()
        mx = plate_df["mx"].to_numpy()
        my = plate_df["my"].to_numpy()

        min_force_threshold = 10.0
        valid_contact = np.abs(fz) > min_force_threshold

        plate_df["cop_x"] = np.where(valid_contact, -my / fz, np.nan)
        plate_df["cop_y"] = np.where(valid_contact, mx / fz, np.nan)
        plate_df["cop_z"] = np.where(valid_contact, ground_height, np.nan)

    return plate_df


def _plate_channel_samples(
    analog_array: np.ndarray, channel_indices: tuple[int, int]
) -> np.ndarray:
    """Return analog samples for a plate as ``(n_samples, n_channels)``.

    ``analog_array`` is the raw ezc3d ``data['analogs']`` array shaped
    ``(subframes, channels, frames)``; we flatten to one row per analog sample.
    """
    start, end = channel_indices
    if end <= start:
        return np.zeros((0, 0))
    subframes, _, frame_count = analog_array.shape
    plate_slice = analog_array[:, start:end, :]
    # (subframes, n_channels, frames) -> (frames, subframes, n_channels)
    samples = plate_slice.transpose(2, 0, 1).reshape(
        frame_count * subframes, end - start
    )
    return samples


def _apply_calibration(
    raw: np.ndarray, cal_matrix: np.ndarray | None, plate_type: int
) -> np.ndarray:
    """Convert raw voltages to forces/moments via the cal matrix when needed.

    Type 1 plates report calibrated forces/moments directly. Type 2/3/4 plates
    require multiplication by the calibration matrix. Returns an
    ``(n_samples, 6)`` array of ``(fx, fy, fz, mx, my, mz)``.
    """
    n_samples, n_channels = raw.shape
    if plate_type == 1 or cal_matrix is None:
        if n_channels < 6:
            padded = np.zeros((n_samples, 6))
            padded[:, :n_channels] = raw
            return padded
        return raw[:, :6].copy()

    # cal_matrix shape is (6, n_channels). Multiply per sample.
    if cal_matrix.shape[0] != 6 or cal_matrix.shape[1] != n_channels:
        logger.warning(
            "Calibration matrix shape %s incompatible with %d analog channels; "
            "returning uncalibrated channels.",
            cal_matrix.shape,
            n_channels,
        )
        if n_channels < 6:
            padded = np.zeros((n_samples, 6))
            padded[:, :n_channels] = raw
            return padded
        return raw[:, :6].copy()

    # (n_samples, n_channels) @ (n_channels, 6) -> (n_samples, 6)
    return raw @ cal_matrix.T


def _cop_local_to_lab(
    fx: np.ndarray,
    fy: np.ndarray,
    fz: np.ndarray,
    mx: np.ndarray,
    my: np.ndarray,
    calibration: ForcePlateCalibration,
    ground_height: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute centre of pressure in lab frame using corners + origin.

    Plate-local CoP from forces/moments at the plate origin:
        cop_x_local = (-my - fx * z0) / fz
        cop_y_local = ( mx - fy * z0) / fz
    where ``z0`` is the z component of the plate origin (typically negative,
    measured from plate top surface to sensor centre).

    The plate-local frame is rebuilt from the four corners:
        +x axis: midpoint(c1, c4) - midpoint(c2, c3) (corners are +x+y, -x+y,
                  -x-y, +x-y in C3D order)
        +y axis: midpoint(c1, c2) - midpoint(c3, c4)
        +z axis: x_axis x y_axis
        origin: mean of the four corners (plate top surface centre)
    """
    z0 = float(calibration.origin[2])
    safe_fz = np.where(np.abs(fz) > 1e-9, fz, np.nan)
    cop_local_x = (-my - fx * z0) / safe_fz
    cop_local_y = (mx - fy * z0) / safe_fz
    cop_local_z = np.zeros_like(cop_local_x)

    corners = calibration.corners  # (4, 3) in metres
    centre = corners.mean(axis=0)
    x_axis = ((corners[0] + corners[3]) / 2.0) - ((corners[1] + corners[2]) / 2.0)
    y_axis = ((corners[0] + corners[1]) / 2.0) - ((corners[2] + corners[3]) / 2.0)
    x_norm = np.linalg.norm(x_axis)
    y_norm = np.linalg.norm(y_axis)
    if x_norm > 0:
        x_axis = x_axis / x_norm
    if y_norm > 0:
        y_axis = y_axis / y_norm
    z_axis = np.cross(x_axis, y_axis)
    z_norm = np.linalg.norm(z_axis)
    if z_norm > 0:
        z_axis = z_axis / z_norm

    rotation = np.column_stack([x_axis, y_axis, z_axis])  # (3, 3)
    local = np.column_stack([cop_local_x, cop_local_y, cop_local_z])  # (n, 3)
    lab = local @ rotation.T + centre

    cop_x = lab[:, 0]
    cop_y = lab[:, 1]
    cop_z = np.where(np.isfinite(cop_x), np.full_like(cop_x, ground_height), np.nan)
    # Preserve NaN where fz was too low.
    cop_x = np.where(np.isfinite(cop_local_x), cop_x, np.nan)
    cop_y = np.where(np.isfinite(cop_local_y), cop_y, np.nan)
    return cop_x, cop_y, cop_z


def build_force_plate_dataframe_from_calibration(
    analog_array: np.ndarray,
    calibrations: tuple[ForcePlateCalibration, ...],
    analog_rate: float | None,
    file_name: str,
    plate_number: int | None,
    include_time: bool,
    compute_cop: bool,
    ground_height: float,
) -> pd.DataFrame:
    """Build a force-plate dataframe using FORCE_PLATFORM calibration metadata.

    Honours plate type (1: pre-calibrated; 2/3/4: cal_matrix applied) and
    transforms CoP from plate-local to lab frame using corners + origin.
    """
    if not calibrations:
        return pd.DataFrame(columns=force_plate_columns(include_time, compute_cop))

    selected: list[tuple[int, ForcePlateCalibration]]
    if plate_number is not None:
        if plate_number < 1 or plate_number > len(calibrations):
            raise ValueError(
                f"Force plate {plate_number} not found. "
                f"Available plates: {list(range(1, len(calibrations) + 1))}"
            )
        selected = [(plate_number, calibrations[plate_number - 1])]
    else:
        selected = [(i + 1, cal) for i, cal in enumerate(calibrations)]

    frames: list[pd.DataFrame] = []
    for plate_num, calibration in selected:
        raw = _plate_channel_samples(analog_array, calibration.channel_indices)
        if raw.size == 0:
            logger.warning(
                "Force plate %d has no analog samples in channels %s; skipping.",
                plate_num,
                calibration.channel_indices,
            )
            continue
        wrench = _apply_calibration(raw, calibration.cal_matrix, calibration.plate_type)
        n_samples = wrench.shape[0]
        fx, fy, fz = wrench[:, 0], wrench[:, 1], wrench[:, 2]
        mx, my, mz = wrench[:, 3], wrench[:, 4], wrench[:, 5]

        plate_df = pd.DataFrame(
            {
                "sample": np.arange(n_samples),
                "plate": plate_num,
                "fx": fx,
                "fy": fy,
                "fz": fz,
                "mx": mx,
                "my": my,
                "mz": mz,
            }
        )

        if compute_cop:
            cop_x, cop_y, cop_z = _cop_local_to_lab(
                fx, fy, fz, mx, my, calibration, ground_height
            )
            plate_df["cop_x"] = cop_x
            plate_df["cop_y"] = cop_y
            plate_df["cop_z"] = cop_z

        frames.append(plate_df)

    if not frames:
        return pd.DataFrame(columns=force_plate_columns(include_time, compute_cop))

    result = pd.concat(frames, ignore_index=True)
    if include_time and analog_rate:
        result.insert(1, "time", result["sample"] / analog_rate)

    logger.info(
        "Extracted force plate data via FORCE_PLATFORM group for %d plates, "
        "%d samples from %s",
        len(selected),
        len(result),
        file_name,
    )

    return result


def build_force_plate_dataframe(
    plate_channels: dict[int, dict[str, str]],
    analog_df: pd.DataFrame,
    analog_rate: float | None,
    file_name: str,
    plate_number: int | None,
    include_time: bool,
    compute_cop: bool,
    ground_height: float,
) -> pd.DataFrame:
    """Assemble force plate data into a wide-format DataFrame."""
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

    analog_wide = analog_df.pivot(
        index="sample", columns="channel", values="value"
    ).reset_index()

    result_dfs = []
    required_keys = {"fx", "fy", "fz", "mx", "my", "mz"}

    for plate_num, channels in sorted(plate_channels.items()):
        plate_df = build_plate_dataframe(
            plate_num,
            channels,
            required_keys,
            analog_wide,
            compute_cop,
            ground_height,
        )
        if plate_df is not None:
            result_dfs.append(plate_df)

    if not result_dfs:
        return pd.DataFrame(columns=force_plate_columns(include_time, compute_cop))

    result = pd.concat(result_dfs, ignore_index=True)

    if include_time and analog_rate:
        result.insert(1, "time", result["sample"] / analog_rate)

    logger.info(
        "Extracted force plate data for %d plates, %d samples from %s",
        len(plate_channels),
        len(result),
        file_name,
    )

    return result
