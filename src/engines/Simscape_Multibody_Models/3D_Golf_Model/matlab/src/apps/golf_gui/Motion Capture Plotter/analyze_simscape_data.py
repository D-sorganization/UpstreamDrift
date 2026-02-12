#!/usr/bin/env python3
"""
Analyze Simscape CSV data to identify joint center positions
for golf swing visualization.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def _load_csv_data(csv_file):
    """Load and validate Simscape CSV data.

    Returns the DataFrame on success, or None on failure.
    """
    try:
        df = pd.read_csv(csv_file)
        logger.info(
            f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns"
        )
        logger.info("Time range: %s to %s seconds", df["time"].min(), df["time"].max())
        logger.info("")
        return df
    except (RuntimeError, ValueError, OSError) as e:
        logger.error("Error reading CSV file: %s", e)
        return None


def _categorize_columns(columns):
    """Categorize CSV columns by data type (position, rotation, etc.).

    Returns a dict mapping category names to lists of column names.
    """
    categories = {
        "position": [],
        "rotation": [],
        "velocity": [],
        "force": [],
        "torque": [],
        "other": [],
    }

    for col in columns:
        if col == "time":
            continue
        elif "GlobalPosition" in col:
            categories["position"].append(col)
        elif "Rotation_Transform" in col:
            categories["rotation"].append(col)
        elif "GlobalVelocity" in col:
            categories["velocity"].append(col)
        elif "Force" in col and "Local" in col:
            categories["force"].append(col)
        elif "Torque" in col and "Local" in col:
            categories["torque"].append(col)
        else:
            categories["other"].append(col)

    logger.info("COLUMN ANALYSIS:")
    logger.info("Position columns: %s", len(categories["position"]))
    logger.info("Rotation columns: %s", len(categories["rotation"]))
    logger.info("Velocity columns: %s", len(categories["velocity"]))
    logger.info("Force columns: %s", len(categories["force"]))
    logger.info("Torque columns: %s", len(categories["torque"]))
    logger.info("Other columns: %s", len(categories["other"]))
    logger.info("")

    return categories


def _identify_joint_positions(position_columns):
    """Identify key joint center positions from position columns.

    Returns a dict mapping joint group names to lists of matching columns.
    """
    logger.info("KEY JOINT CENTER POSITIONS:")
    logger.info("%s", "-" * 50)

    joint_positions = {}

    # Each entry: (joint_key, label, filter function)
    joint_definitions = [
        ("club", "Club", lambda col: "Club" in col),
        (
            "hands",
            "Hand",
            lambda col: any(x in col for x in ["LHGlobalPosition", "RHGlobalPosition"]),
        ),
        (
            "wrists",
            "Wrist",
            lambda col: any(x in col for x in ["LWLogs", "RWLogs"]),
        ),
        (
            "elbows",
            "Elbow",
            lambda col: any(x in col for x in ["LELogs", "RELogs"]),
        ),
        (
            "shoulders",
            "Shoulder",
            lambda col: any(x in col for x in ["LSLogs", "RSLogs"]),
        ),
        (
            "scapulae",
            "Scapula",
            lambda col: any(x in col for x in ["LScapLogs", "RScapLogs"]),
        ),
        ("hips", "Hip", lambda col: "HipLogs" in col),
        ("spine", "Spine", lambda col: "SpineLogs" in col),
        ("torso", "Torso", lambda col: "TorsoLogs" in col),
        ("hub", "Hub", lambda col: "HUB" in col),
    ]

    for joint_key, label, filter_fn in joint_definitions:
        matching = [col for col in position_columns if filter_fn(col)]
        if matching:
            logger.info("\n%s positions found:", label)
            for pos in matching:
                logger.info("  %s", pos)
            joint_positions[joint_key] = matching

    return joint_positions


def _build_segment_definitions():
    """Build the dictionary of visualization segment definitions.

    Each segment maps a name to six column names (start XYZ + end XYZ).
    """
    return {
        "midpoint_to_clubhead": [
            "MidpointCalcsLogs_MPGlobalPosition_1",
            "MidpointCalcsLogs_MPGlobalPosition_2",
            "MidpointCalcsLogs_MPGlobalPosition_3",
            "ClubLogs_CHGlobalPosition_1",
            "ClubLogs_CHGlobalPosition_2",
            "ClubLogs_CHGlobalPosition_3",
        ],
        "left_hand_to_midpoint": [
            "LHCalcsLogs_LeftHandPostion_1",
            "LHCalcsLogs_LeftHandPostion_2",
            "LHCalcsLogs_LeftHandPostion_3",
            "MidpointCalcsLogs_MPGlobalPosition_1",
            "MidpointCalcsLogs_MPGlobalPosition_2",
            "MidpointCalcsLogs_MPGlobalPosition_3",
        ],
        "right_hand_to_midpoint": [
            "RHCalcsLogs_RightHandPostion_1",
            "RHCalcsLogs_RightHandPostion_2",
            "RHCalcsLogs_RightHandPostion_3",
            "MidpointCalcsLogs_MPGlobalPosition_1",
            "MidpointCalcsLogs_MPGlobalPosition_2",
            "MidpointCalcsLogs_MPGlobalPosition_3",
        ],
        "left_hand_to_left_elbow": [
            "LHCalcsLogs_LeftHandPostion_1",
            "LHCalcsLogs_LeftHandPostion_2",
            "LHCalcsLogs_LeftHandPostion_3",
            "LELogs_LArmonLForearmFGlobal_1",
            "LELogs_LArmonLForearmFGlobal_2",
            "LELogs_LArmonLForearmFGlobal_3",
        ],
        "right_hand_to_right_elbow": [
            "RHCalcsLogs_RightHandPostion_1",
            "RHCalcsLogs_RightHandPostion_2",
            "RHCalcsLogs_RightHandPostion_3",
            "RELogs_RArmonLForearmFGlobal_1",
            "RELogs_RArmonLForearmFGlobal_2",
            "RELogs_RArmonLForearmFGlobal_3",
        ],
        "left_elbow_to_left_shoulder": [
            "LELogs_LArmonLForearmFGlobal_1",
            "LELogs_LArmonLForearmFGlobal_2",
            "LELogs_LArmonLForearmFGlobal_3",
            "LSLogs_GlobalPosition_1",
            "LSLogs_GlobalPosition_2",
            "LSLogs_GlobalPosition_3",
        ],
        "right_elbow_to_right_shoulder": [
            "RELogs_RArmonLForearmFGlobal_1",
            "RELogs_RArmonLForearmFGlobal_2",
            "RELogs_RArmonLForearmFGlobal_3",
            "RSLogs_GlobalPosition_1",
            "RSLogs_GlobalPosition_2",
            "RSLogs_GlobalPosition_3",
        ],
        "left_shoulder_to_hub": [
            "LSLogs_GlobalPosition_1",
            "LSLogs_GlobalPosition_2",
            "LSLogs_GlobalPosition_3",
            "HipLogs_HUBGlobalPosition_1",
            "HipLogs_HUBGlobalPosition_2",
            "HipLogs_HUBGlobalPosition_3",
        ],
        "right_shoulder_to_hub": [
            "RSLogs_GlobalPosition_1",
            "RSLogs_GlobalPosition_2",
            "RSLogs_GlobalPosition_3",
            "HipLogs_HUBGlobalPosition_1",
            "HipLogs_HUBGlobalPosition_2",
            "HipLogs_HUBGlobalPosition_3",
        ],
        "hub_to_spine": [
            "HipLogs_HUBGlobalPosition_1",
            "HipLogs_HUBGlobalPosition_2",
            "HipLogs_HUBGlobalPosition_3",
            "SpineLogs_GlobalPosition_1",
            "SpineLogs_GlobalPosition_2",
            "SpineLogs_GlobalPosition_3",
        ],
        "spine_to_hips": [
            "SpineLogs_GlobalPosition_1",
            "SpineLogs_GlobalPosition_2",
            "SpineLogs_GlobalPosition_3",
            "HipLogs_HipGlobalPosition_dim1",
            "HipLogs_HipGlobalPosition_dim2",
            "HipLogs_HipGlobalPosition_dim3",
        ],
    }


def _check_segment_availability(segments, columns):
    """Check which segments have all required columns available.

    Returns a dict of available segment names to their column lists.
    """
    available_segments = {}
    for segment_name, required_cols in segments.items():
        available_cols = [col for col in required_cols if col in columns]
        if len(available_cols) == len(required_cols):
            available_segments[segment_name] = available_cols
            logger.info("✓ %s: AVAILABLE", segment_name)
        else:
            missing_cols = [col for col in required_cols if col not in columns]
            logger.info("✗ %s: MISSING %s columns", segment_name, len(missing_cols))
            if len(missing_cols) <= 3:  # Show missing columns if not too many
                for col in missing_cols:
                    logger.info("    Missing: %s", col)
    return available_segments


def _log_data_sample(df, available_segments):
    """Log a sample of data from the available segments."""
    logger.info("%s", "\n" + "=" * 80)
    logger.info("DATA SAMPLE (first 3 rows):")
    logger.info("%s", "-" * 40)

    if available_segments:
        sample_cols = []
        for segment_cols in available_segments.values():
            sample_cols.extend(
                segment_cols[:3]
            )  # Take first 3 columns from each segment

        # Remove duplicates and limit to reasonable number
        sample_cols = list(set(sample_cols))[:15]
        sample_cols.insert(0, "time")  # Always include time

        logger.info("%s", df[sample_cols].head(3).to_string())


def analyze_simscape_data(csv_file) -> tuple | None:
    """Analyze the Simscape CSV file and identify key joint positions."""

    logger.info("Analyzing Simscape data file: %s", csv_file)
    logger.info("%s", "=" * 80)

    df = _load_csv_data(csv_file)
    if df is None:
        return

    columns = df.columns.tolist()

    categories = _categorize_columns(columns)
    position_columns = categories["position"]

    joint_positions = _identify_joint_positions(position_columns)

    logger.info("%s", "\n" + "=" * 80)
    logger.info("RECOMMENDED SEGMENTS FOR GOLF SWING VISUALIZATION:")
    logger.info("%s", "-" * 60)

    segments = _build_segment_definitions()
    available_segments = _check_segment_availability(segments, columns)

    _log_data_sample(df, available_segments)

    return joint_positions, available_segments


if __name__ == "__main__":
    csv_file = "trial_001_20250802_204903.csv"
    analyze_simscape_data(csv_file)
