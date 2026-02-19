"""
Data loading and parsing for motion capture and Simscape data.

Extracted from MotionCapturePlotter to respect SRP:
data parsing logic is independent of visualization and UI layout.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# Conversion factor: inches to meters
INCHES_TO_METERS = 0.0254


def safe_float(value: Any, default: float = 0.0) -> float:
    """Safely convert a value to float, returning default on failure."""
    if pd.isna(value):
        return default
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def parse_excel_row(row: pd.Series, row_index: int) -> dict[str, float] | None:
    """Parse a single Excel row into a frame data dict.

    Returns a dict with mid-hands and club head position/orientation data,
    or None if the row has insufficient columns.
    """
    if len(row) < 25:
        return None

    return {
        "time": safe_float(row[1], row_index),  # Time is in column 1
        # Mid-hands position (convert inches to meters) and orientation
        "mid_X": safe_float(row[2]) * INCHES_TO_METERS,
        "mid_Y": safe_float(row[3]) * INCHES_TO_METERS,
        "mid_Z": safe_float(row[4]) * INCHES_TO_METERS,
        "mid_Xx": safe_float(row[5]),  # Direction cosines (unitless)
        "mid_Xy": safe_float(row[6]),
        "mid_Xz": safe_float(row[7]),
        "mid_Yx": safe_float(row[8]),
        "mid_Yy": safe_float(row[9]),
        "mid_Yz": safe_float(row[10]),
        "mid_Zx": safe_float(row[11]),
        "mid_Zy": safe_float(row[12]),
        "mid_Zz": safe_float(row[13]),
        # Club head position (convert inches to meters) and orientation
        "club_X": safe_float(row[14]) * INCHES_TO_METERS,
        "club_Y": safe_float(row[15]) * INCHES_TO_METERS,
        "club_Z": safe_float(row[16]) * INCHES_TO_METERS,
        "club_Xx": safe_float(row[17]),  # Direction cosines (unitless)
        "club_Xy": safe_float(row[18]),
        "club_Xz": safe_float(row[19]),
        "club_Yx": safe_float(row[20]),
        "club_Yy": safe_float(row[21]),
        "club_Yz": safe_float(row[22]),
        "club_Zx": safe_float(row[23]),
        "club_Zy": safe_float(row[24]),
        "club_Zz": safe_float(row[25]),
    }


def process_excel_sheet(filename: str, sheet_name: str) -> pd.DataFrame | None:
    """Process a single Excel sheet and return parsed frame data.

    Returns:
        DataFrame with parsed frame data, or None if sheet is too small.
    """
    df = pd.read_excel(filename, sheet_name=sheet_name, header=None)

    if len(df) <= 3:
        return None

    data = []
    for i in range(3, len(df)):
        frame_data = parse_excel_row(df.iloc[i], i - 3)
        if frame_data is not None:
            data.append(frame_data)

    if data:
        logger.debug(f"Successfully loaded {len(data)} frames for {sheet_name}")
        return pd.DataFrame(data)
    return None


def get_simscape_joint_positions() -> dict[str, list[str]]:
    """Return the mapping of joint names to their CSV column names."""
    return {
        "club_head": [
            "ClubLogs_CHGlobalPosition_1",
            "ClubLogs_CHGlobalPosition_2",
            "ClubLogs_CHGlobalPosition_3",
        ],
        "left_hand": [
            "LWLogs_LHGlobalPosition_1",
            "LWLogs_LHGlobalPosition_2",
            "LWLogs_LHGlobalPosition_3",
        ],
        "right_hand": [
            "RWLogs_RHGlobalPosition_1",
            "RWLogs_RHGlobalPosition_2",
            "RWLogs_RHGlobalPosition_3",
        ],
        "left_shoulder": [
            "LSLogs_GlobalPosition_1",
            "LSLogs_GlobalPosition_2",
            "LSLogs_GlobalPosition_3",
        ],
        "right_shoulder": [
            "RSLogs_GlobalPosition_1",
            "RSLogs_GlobalPosition_2",
            "RSLogs_GlobalPosition_3",
        ],
        "left_elbow": [
            "LELogs_LArmonLForearmFGlobal_1",
            "LELogs_LArmonLForearmFGlobal_2",
            "LELogs_LArmonLForearmFGlobal_3",
        ],
        "right_elbow": [
            "RELogs_RArmonLForearmFGlobal_1",
            "RELogs_RArmonLForearmFGlobal_2",
            "RELogs_RArmonLForearmFGlobal_3",
        ],
        "hub": [
            "HipLogs_HUBGlobalPosition_1",
            "HipLogs_HUBGlobalPosition_2",
            "HipLogs_HUBGlobalPosition_3",
        ],
        "spine": [
            "SpineLogs_GlobalPosition_1",
            "SpineLogs_GlobalPosition_2",
            "SpineLogs_GlobalPosition_3",
        ],
        "hip": [
            "HipLogs_HipGlobalPosition_dim1",
            "HipLogs_HipGlobalPosition_dim2",
            "HipLogs_HipGlobalPosition_dim3",
        ],
    }


def find_available_joints(
    joint_positions: dict[str, list[str]], df_columns: pd.Index
) -> dict[str, list[str]]:
    """Check which joint positions are available in the CSV columns.

    Returns a dict of available joint names to their column lists.
    """
    available_joints: dict[str, list[str]] = {}
    for joint_name, columns in joint_positions.items():
        if all(col in df_columns for col in columns):
            available_joints[joint_name] = columns
            logger.info(f"  {joint_name}: AVAILABLE")
        else:
            missing_cols = [col for col in columns if col not in df_columns]
            logger.warning(f"  {joint_name}: MISSING {len(missing_cols)} columns")
    return available_joints


def parse_simscape_csv(filename: str) -> pd.DataFrame:
    """Parse a Simscape CSV file into a standardized DataFrame.

    Args:
        filename: Path to the CSV file.

    Returns:
        DataFrame with time and joint position columns.

    Raises:
        ValueError: If no valid joint position data is found.
    """
    df = pd.read_csv(filename)
    logger.debug(
        f"Successfully loaded CSV with {len(df)} rows "
        f"and {len(df.columns)} columns"
    )
    logger.info(
        f"Time range: {df['time'].min():.3f} to {df['time'].max():.3f} seconds"
    )

    joint_positions = get_simscape_joint_positions()
    available_joints = find_available_joints(joint_positions, df.columns)

    if not available_joints:
        raise ValueError("No valid joint position data found in the CSV file")

    data = []
    for _i, row in df.iterrows():
        frame_data: dict[str, float] = {"time": row["time"]}
        for joint_name, columns in available_joints.items():
            if all(col in df.columns for col in columns):
                frame_data[f"{joint_name}_X"] = row[columns[0]]
                frame_data[f"{joint_name}_Y"] = row[columns[1]]
                frame_data[f"{joint_name}_Z"] = row[columns[2]]
        data.append(frame_data)

    return pd.DataFrame(data)
