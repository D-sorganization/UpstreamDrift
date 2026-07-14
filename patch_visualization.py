import re

file_path = "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/src/apps/golf_gui/Motion Capture Plotter/motion_capture_plotter_visualization.py"

with open(file_path, "r") as f:
    content = f.read()

# Replace block 1 (mid-hands path)
search1 = """            trajectory = np.array(
                [
                    [
                        -row["mid_X"] * self.motion_scale,
                        row["mid_Y"] * self.motion_scale,
                        row["mid_Z"] * self.motion_scale,
                    ]
                    for _, row in data.iterrows()
                ]
            )"""
replace1 = """            # ⚡ Bolt: Vectorized array construction is much faster than row-by-row iterrows()
            trajectory = np.column_stack(
                [
                    -data["mid_X"].values * self.motion_scale,
                    data["mid_Y"].values * self.motion_scale,
                    data["mid_Z"].values * self.motion_scale,
                ]
            )"""
content = content.replace(search1, replace1)

# Replace block 2 (club path)
search2 = """            club_path = np.array(
                [
                    [
                        -row["club_X"] * self.motion_scale,
                        row["club_Y"] * self.motion_scale,
                        row["club_Z"] * self.motion_scale,
                    ]
                    for _, row in data.iterrows()
                ]
            )"""
replace2 = """            # ⚡ Bolt: Vectorized array construction is much faster than row-by-row iterrows()
            club_path = np.column_stack(
                [
                    -data["club_X"].values * self.motion_scale,
                    data["club_Y"].values * self.motion_scale,
                    data["club_Z"].values * self.motion_scale,
                ]
            )"""
content = content.replace(search2, replace2)

# Replace block 3 (club_head trajectory)
search3 = """            club_trajectory = np.array(
                [
                    [
                        -row["club_head_X"]
                        * self.motion_scale,  # Flip X for right-handed swing  # noqa: E501
                        row["club_head_Y"] * self.motion_scale,
                        row["club_head_Z"] * self.motion_scale,
                    ]
                    for _, row in data.iterrows()
                    if "club_head_X" in row
                ]
            )"""
replace3 = """            # ⚡ Bolt: Vectorized array construction is much faster than row-by-row iterrows()
            # Column check happens once outside the loop since dataframes have uniform columns
            if "club_head_X" in data.columns:
                club_trajectory = np.column_stack(
                    [
                        -data["club_head_X"].values
                        * self.motion_scale,  # Flip X for right-handed swing
                        data["club_head_Y"].values * self.motion_scale,
                        data["club_head_Z"].values * self.motion_scale,
                    ]
                )
            else:
                club_trajectory = np.array([])"""
content = content.replace(search3, replace3)


# Replace block 4 (hands_trajectory)
search4 = """            hands_trajectory = np.array(
                [
                    [
                        -row["left_hand_X"]
                        * self.motion_scale,  # Flip X for right-handed swing  # noqa: E501
                        row["left_hand_Y"] * self.motion_scale,
                        row["left_hand_Z"] * self.motion_scale,
                    ]
                    for _, row in data.iterrows()
                    if "left_hand_X" in row
                ]
            )"""
replace4 = """            # ⚡ Bolt: Vectorized array construction is much faster than row-by-row iterrows()
            # Column check happens once outside the loop since dataframes have uniform columns
            if "left_hand_X" in data.columns:
                hands_trajectory = np.column_stack(
                    [
                        -data["left_hand_X"].values
                        * self.motion_scale,  # Flip X for right-handed swing
                        data["left_hand_Y"].values * self.motion_scale,
                        data["left_hand_Z"].values * self.motion_scale,
                    ]
                )
            else:
                hands_trajectory = np.array([])"""
content = content.replace(search4, replace4)


# Replace block 5 (segment_trajectory)
search5 = """                # Create trajectory for this segment
                segment_trajectory = np.array(
                    [
                        [
                            -row[f"{segment_key}_X"]
                            * self.motion_scale,  # Flip X for right-handed swing
                            row[f"{segment_key}_Y"] * self.motion_scale,
                            row[f"{segment_key}_Z"] * self.motion_scale,
                        ]
                        for _, row in data.iterrows()
                        if f"{segment_key}_X" in row
                    ]
                )"""
replace5 = """                # Create trajectory for this segment
                # ⚡ Bolt: Vectorized array construction is much faster than row-by-row iterrows()
                # Column check happens once outside the loop since dataframes have uniform columns
                if f"{segment_key}_X" in data.columns:
                    segment_trajectory = np.column_stack(
                        [
                            -data[f"{segment_key}_X"].values
                            * self.motion_scale,  # Flip X for right-handed swing
                            data[f"{segment_key}_Y"].values * self.motion_scale,
                            data[f"{segment_key}_Z"].values * self.motion_scale,
                        ]
                    )
                else:
                    segment_trajectory = np.array([])"""
content = content.replace(search5, replace5)


with open(file_path, "w") as f:
    f.write(content)

print("Patched.")
