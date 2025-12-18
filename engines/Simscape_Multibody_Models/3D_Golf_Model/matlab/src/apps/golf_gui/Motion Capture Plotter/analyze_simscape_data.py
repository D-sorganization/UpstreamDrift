#!/usr/bin/env python3
"""
Analyze Simscape CSV data to identify joint center positions
for golf swing visualization.
"""

import pandas as pd


def analyze_simscape_data(csv_file):
    """Analyze the Simscape CSV file and identify key joint positions."""

    print(f"Analyzing Simscape data file: {csv_file}")
    print("=" * 80)

    # Read the CSV file
    try:
        df = pd.read_csv(csv_file)
        print(
            f"Successfully loaded CSV with {len(df)} rows and {len(df.columns)} columns"
        )
        print(f"Time range: {df['time'].min():.3f} to {df['time'].max():.3f} seconds")
        print()
    except Exception as e:
        print(f"Error reading CSV file: {e}")
        return

    # Get all column names
    columns = df.columns.tolist()

    # Categorize columns by type
    position_columns = []
    rotation_columns = []
    velocity_columns = []
    force_columns = []
    torque_columns = []
    other_columns = []

    for col in columns:
        if col == "time":
            continue
        elif "GlobalPosition" in col:
            position_columns.append(col)
        elif "Rotation_Transform" in col:
            rotation_columns.append(col)
        elif "GlobalVelocity" in col:
            velocity_columns.append(col)
        elif "Force" in col and "Local" in col:
            force_columns.append(col)
        elif "Torque" in col and "Local" in col:
            torque_columns.append(col)
        else:
            other_columns.append(col)

    print("COLUMN ANALYSIS:")
    print(f"Position columns: {len(position_columns)}")
    print(f"Rotation columns: {len(rotation_columns)}")
    print(f"Velocity columns: {len(velocity_columns)}")
    print(f"Force columns: {len(force_columns)}")
    print(f"Torque columns: {len(torque_columns)}")
    print(f"Other columns: {len(other_columns)}")
    print()

    # Identify key joint centers for golf swing
    print("KEY JOINT CENTER POSITIONS:")
    print("-" * 50)

    # Look for specific joint center positions
    joint_positions = {}

    # Club-related positions
    club_positions = [col for col in position_columns if "Club" in col]
    if club_positions:
        print("Club positions found:")
        for pos in club_positions:
            print(f"  {pos}")
        joint_positions["club"] = club_positions

    # Hand positions
    hand_positions = [
        col
        for col in position_columns
        if any(x in col for x in ["LHGlobalPosition", "RHGlobalPosition"])
    ]
    if hand_positions:
        print("\nHand positions found:")
        for pos in hand_positions:
            print(f"  {pos}")
        joint_positions["hands"] = hand_positions

    # Wrist positions
    wrist_positions = [
        col for col in position_columns if any(x in col for x in ["LWLogs", "RWLogs"])
    ]
    if wrist_positions:
        print("\nWrist positions found:")
        for pos in wrist_positions:
            print(f"  {pos}")
        joint_positions["wrists"] = wrist_positions

    # Elbow positions
    elbow_positions = [
        col for col in position_columns if any(x in col for x in ["LELogs", "RELogs"])
    ]
    if elbow_positions:
        print("\nElbow positions found:")
        for pos in elbow_positions:
            print(f"  {pos}")
        joint_positions["elbows"] = elbow_positions

    # Shoulder positions
    shoulder_positions = [
        col for col in position_columns if any(x in col for x in ["LSLogs", "RSLogs"])
    ]
    if shoulder_positions:
        print("\nShoulder positions found:")
        for pos in shoulder_positions:
            print(f"  {pos}")
        joint_positions["shoulders"] = shoulder_positions

    # Scapula positions
    scapula_positions = [
        col
        for col in position_columns
        if any(x in col for x in ["LScapLogs", "RScapLogs"])
    ]
    if scapula_positions:
        print("\nScapula positions found:")
        for pos in scapula_positions:
            print(f"  {pos}")
        joint_positions["scapulae"] = scapula_positions

    # Hip positions
    hip_positions = [col for col in position_columns if "HipLogs" in col]
    if hip_positions:
        print("\nHip positions found:")
        for pos in hip_positions:
            print(f"  {pos}")
        joint_positions["hips"] = hip_positions

    # Spine positions
    spine_positions = [col for col in position_columns if "SpineLogs" in col]
    if spine_positions:
        print("\nSpine positions found:")
        for pos in spine_positions:
            print(f"  {pos}")
        joint_positions["spine"] = spine_positions

    # Torso positions
    torso_positions = [col for col in position_columns if "TorsoLogs" in col]
    if torso_positions:
        print("\nTorso positions found:")
        for pos in torso_positions:
            print(f"  {pos}")
        joint_positions["torso"] = torso_positions

    # Hub positions
    hub_positions = [col for col in position_columns if "HUB" in col]
    if hub_positions:
        print("\nHub positions found:")
        for pos in hub_positions:
            print(f"  {pos}")
        joint_positions["hub"] = hub_positions

    print("\n" + "=" * 80)
    print("RECOMMENDED SEGMENTS FOR GOLF SWING VISUALIZATION:")
    print("-" * 60)

    # Define the segments we want to visualize
    segments = {
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

    # Check which segments are available
    available_segments = {}
    for segment_name, required_cols in segments.items():
        available_cols = [col for col in required_cols if col in columns]
        if len(available_cols) == len(required_cols):
            available_segments[segment_name] = available_cols
            print(f"✓ {segment_name}: AVAILABLE")
        else:
            missing_cols = [col for col in required_cols if col not in columns]
            print(f"✗ {segment_name}: MISSING {len(missing_cols)} columns")
            if len(missing_cols) <= 3:  # Show missing columns if not too many
                for col in missing_cols:
                    print(f"    Missing: {col}")

    print("\n" + "=" * 80)
    print("DATA SAMPLE (first 3 rows):")
    print("-" * 40)

    # Show sample data for available segments
    if available_segments:
        sample_cols = []
        for segment_cols in available_segments.values():
            sample_cols.extend(
                segment_cols[:3]
            )  # Take first 3 columns from each segment

        # Remove duplicates and limit to reasonable number
        sample_cols = list(set(sample_cols))[:15]
        sample_cols.insert(0, "time")  # Always include time

        print(df[sample_cols].head(3).to_string())

    return joint_positions, available_segments


if __name__ == "__main__":
    csv_file = "trial_001_20250802_204903.csv"
    analyze_simscape_data(csv_file)
