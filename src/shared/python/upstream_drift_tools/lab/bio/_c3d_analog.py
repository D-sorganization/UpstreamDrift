from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

from ...utils.logging import get_logger
from ._c3d_models import C3DMetadata

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
    if not (include_time is not None):
        raise ValueError("include_time must be provided")
    if not (include_time is not None):
        raise ValueError("include_time must be provided")
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
    if not (plate_num is not None):
        raise ValueError("plate_num must be provided")
    if not (plate_num is not None):
        raise ValueError("plate_num must be provided")
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
