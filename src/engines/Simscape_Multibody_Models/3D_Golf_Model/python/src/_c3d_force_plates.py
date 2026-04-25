"""Force plate channel detection and data extraction."""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

try:
    from .logger_utils import get_logger
except ImportError:
    from logger_utils import get_logger  # type: ignore[no-redef]

logger = get_logger(__name__)


def force_plate_columns(include_time: bool, compute_cop: bool) -> list[str]:
    if not (include_time is not None):
        raise ValueError("include_time must be provided")
    columns = ["sample", "plate", "fx", "fy", "fz", "mx", "my", "mz"]
    if include_time:
        columns.insert(1, "time")
    if compute_cop:
        columns.extend(["cop_x", "cop_y", "cop_z"])
    return columns


def add_cop_columns(plate_df: pd.DataFrame, ground_height: float) -> None:
    if not (plate_df is not None):
        raise ValueError("plate_df must be provided")
    if not (plate_df is not None):
        raise ValueError("plate_df must be provided")
    fz = plate_df["fz"].to_numpy()
    mx = plate_df["mx"].to_numpy()
    my = plate_df["my"].to_numpy()

    min_force_threshold = 10.0
    valid_contact = np.abs(fz) > min_force_threshold

    plate_df["cop_x"] = np.where(valid_contact, -my / fz, np.nan)
    plate_df["cop_y"] = np.where(valid_contact, mx / fz, np.nan)
    plate_df["cop_z"] = np.where(valid_contact, ground_height, np.nan)


def detect_force_plate_channels(
    analog_labels: list[str],
) -> dict[int, dict[str, str]]:
    plate_channels: dict[int, dict[str, str]] = {}

    standard_pattern = re.compile(r"^(?:Force\.)?([FfMm])([xyzXYZ])(\d+)$")
    prefix_pattern = re.compile(r"^(?:FP|fp)?(\d+)[_.]?([FfMm])([xyzXYZ])$")

    for label in analog_labels:
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


def build_plate_dataframes(
    plate_channels: dict[int, dict[str, str]],
    analog_wide: pd.DataFrame,
    compute_cop: bool,
    ground_height: float,
) -> list[pd.DataFrame]:
    if not (plate_channels is not None):
        raise ValueError("plate_channels must be provided")
    if not (plate_channels is not None):
        raise ValueError("plate_channels must be provided")
    required_keys = {"fx", "fy", "fz", "mx", "my", "mz"}
    result_dfs: list[pd.DataFrame] = []

    for plate_num, channels in sorted(plate_channels.items()):
        missing_keys = required_keys - set(channels.keys())
        if missing_keys:
            logger.warning(
                f"Force plate {plate_num} missing channels: {missing_keys}. Skipping."
            )
            continue

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
            add_cop_columns(plate_df, ground_height)

        result_dfs.append(plate_df)

    return result_dfs


def pivot_analog_to_wide(analog_df: pd.DataFrame) -> pd.DataFrame:
    return analog_df.pivot(
        index="sample", columns="channel", values="value"
    ).reset_index()


def extract_force_plate_dataframe(
    plate_channels: dict[int, dict[str, Any]],
    analog_wide: pd.DataFrame,
    analog_rate: float | None,
    include_time: bool,
    compute_cop: bool,
    ground_height: float,
    file_name: str,
) -> pd.DataFrame:
    result_dfs = build_plate_dataframes(
        plate_channels, analog_wide, compute_cop, ground_height
    )

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
