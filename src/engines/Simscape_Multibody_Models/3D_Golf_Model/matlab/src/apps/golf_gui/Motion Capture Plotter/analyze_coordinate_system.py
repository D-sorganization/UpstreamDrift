import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def analyze_coordinate_system():
    """Analyze the coordinate system and golfer orientation in the data"""

    # Load the Excel file
    filename = "Wiffle_ProV1_club_3D_data.xlsx"
    sheet_name = "TW_wiffle"  # Use one of the available sheets

    logger.info("Analyzing coordinate system for %s", sheet_name)
    logger.info("%s", "=" * 50)

    # Read the data
    df = pd.read_excel(filename, sheet_name=sheet_name, header=None)

    # Get data starting from row 3 - analyze more frames
    if len(df) > 3:
        data = []
        for i in range(3, len(df)):
            row = df.iloc[i]
            if len(row) >= 25:
                try:
                    frame_data = {
                        "time": float(row[1]) if pd.notna(row[1]) else i - 3,
                        "mid_X": float(row[2]) * 0.0254,  # inches to meters
                        "mid_Y": float(row[3]) * 0.0254,
                        "mid_Z": float(row[4]) * 0.0254,
                        "club_X": float(row[14]) * 0.0254,
                        "club_Y": float(row[15]) * 0.0254,
                        "club_Z": float(row[16]) * 0.0254,
                        # Direction cosines for mid-hands
                        "mid_Xx": float(row[5]),
                        "mid_Xy": float(row[6]),
                        "mid_Xz": float(row[7]),
                        "mid_Yx": float(row[8]),
                        "mid_Yy": float(row[9]),
                        "mid_Yz": float(row[10]),
                        "mid_Zx": float(row[11]),
                        "mid_Zy": float(row[12]),
                        "mid_Zz": float(row[13]),
                        # Direction cosines for club head
                        "club_Xx": float(row[17]),
                        "club_Xy": float(row[18]),
                        "club_Xz": float(row[19]),
                        "club_Yx": float(row[20]),
                        "club_Yy": float(row[21]),
                        "club_Yz": float(row[22]),
                        "club_Zx": float(row[23]),
                        "club_Zy": float(row[24]),
                        "club_Zz": float(row[25]),
                    }
                    data.append(frame_data)
                except (ValueError, TypeError):
                    continue

    if not data:
        logger.info("No data found!")
        return

    logger.info("Analyzing %s frames", len(data))

    # Analyze overall motion patterns
    logger.info("\nOverall motion analysis:")
    logger.info("%s", "-" * 30)

    mid_x_range = [min(d["mid_X"] for d in data), max(d["mid_X"] for d in data)]
    mid_y_range = [min(d["mid_Y"] for d in data), max(d["mid_Y"] for d in data)]
    mid_z_range = [min(d["mid_Z"] for d in data), max(d["mid_Z"] for d in data)]

    club_x_range = [min(d["club_X"] for d in data), max(d["club_X"] for d in data)]
    club_y_range = [min(d["club_Y"] for d in data), max(d["club_Y"] for d in data)]
    club_z_range = [min(d["club_Z"] for d in data), max(d["club_Z"] for d in data)]

    logger.info("Mid-hands motion ranges:")
    logger.info(
        f"  X: {mid_x_range[0]:.3f} to {mid_x_range[1]:.3f} "
        f"(range: {mid_x_range[1] - mid_x_range[0]:.3f})"
    )
    logger.info(
        f"  Y: {mid_y_range[0]:.3f} to {mid_y_range[1]:.3f} "
        f"(range: {mid_y_range[1] - mid_y_range[0]:.3f})"
    )
    logger.info(
        f"  Z: {mid_z_range[0]:.3f} to {mid_z_range[1]:.3f} "
        f"(range: {mid_z_range[1] - mid_z_range[0]:.3f})"
    )

    logger.info("Club head motion ranges:")
    logger.info(
        f"  X: {club_x_range[0]:.3f} to {club_x_range[1]:.3f} "
        f"(range: {club_x_range[1] - club_x_range[0]:.3f})"
    )
    logger.info(
        f"  Y: {club_y_range[0]:.3f} to {club_y_range[1]:.3f} "
        f"(range: {club_y_range[1] - club_y_range[0]:.3f})"
    )
    logger.info(
        f"  Z: {club_z_range[0]:.3f} to {club_z_range[1]:.3f} "
        f"(range: {club_z_range[1] - club_z_range[0]:.3f})"
    )

    # Determine primary motion direction
    mid_motion_ranges = [
        mid_x_range[1] - mid_x_range[0],
        mid_y_range[1] - mid_y_range[0],
        mid_z_range[1] - mid_z_range[0],
    ]

    club_motion_ranges = [
        club_x_range[1] - club_x_range[0],
        club_y_range[1] - club_y_range[0],
        club_z_range[1] - club_z_range[0],
    ]

    logger.info("\nMotion analysis:")
    # Determine the axis with largest motion range using explicit if-elif-else
    # for clarity (avoiding nested ternary operators)
    max_mid_range = max(mid_motion_ranges)
    if mid_motion_ranges[0] == max_mid_range:
        largest_mid_axis = "X"
    elif mid_motion_ranges[1] == max_mid_range:
        largest_mid_axis = "Y"
    else:
        largest_mid_axis = "Z"

    max_club_range = max(club_motion_ranges)
    if club_motion_ranges[0] == max_club_range:
        largest_club_axis = "X"
    elif club_motion_ranges[1] == max_club_range:
        largest_club_axis = "Y"
    else:
        largest_club_axis = "Z"
    logger.info("  Mid-hands largest motion: %s", largest_mid_axis)
    logger.info("  Club head largest motion: %s", largest_club_axis)

    # Check if this looks like a golf swing
    logger.info("\nGolf swing interpretation:")
    if max(club_motion_ranges) > max(mid_motion_ranges) * 1.5:
        logger.info(
            "  ✓ Club head has larger motion than hands (typical of golf swing)"
        )
    else:
        logger.info("  ✗ Club head motion similar to hands (unusual for golf swing)")

    # Determine target line direction
    if max(club_motion_ranges) == club_motion_ranges[0]:  # X direction
        logger.info("  Primary swing motion is in X direction")
        logger.info(
            "  If X is target line: Face-on view should look at +X, "
            "Down-the-line should look at -Y"
        )
    elif max(club_motion_ranges) == club_motion_ranges[1]:  # Y direction
        logger.info("  Primary swing motion is in Y direction")
        logger.info(
            "  If Y is target line: Face-on view should look at +Y, "
            "Down-the-line should look at -X"
        )
    else:  # Z direction
        logger.info("  Primary swing motion is in Z direction (vertical)")
        logger.info("  This seems unusual for a golf swing")

    # Look at key frames (start, middle, end)
    logger.info("\nKey frame analysis:")
    logger.info("%s", "-" * 30)

    # Find frames at different times
    start_frame = data[0]
    mid_frame = data[len(data) // 2]
    end_frame = data[-1]

    for name, frame in [
        ("Start", start_frame),
        ("Middle", mid_frame),
        ("End", end_frame),
    ]:
        logger.info("%s frame (t=%ss):", name, frame["time"])
        logger.info(
            f"  Mid-hands: X={frame['mid_X']:.3f}, Y={frame['mid_Y']:.3f}, "
            f"Z={frame['mid_Z']:.3f}"
        )
        logger.info(
            f"  Club head: X={frame['club_X']:.3f}, Y={frame['club_Y']:.3f}, "
            f"Z={frame['club_Z']:.3f}"
        )

        # Calculate club direction vector
        club_vector = np.array(
            [
                frame["club_X"] - frame["mid_X"],
                frame["club_Y"] - frame["mid_Y"],
                frame["club_Z"] - frame["mid_Z"],
            ]
        )
        club_length = np.linalg.norm(club_vector)
        logger.info("  Club vector: %s", club_vector)
        logger.info("  Club length: %sm", club_length)

        # Analyze direction cosines for mid-hands
        logger.info("  Mid-hands direction cosines:")
        logger.info(
            f"    X-axis: [{frame['mid_Xx']:.3f}, {frame['mid_Xy']:.3f}, "
            f"{frame['mid_Xz']:.3f}]"
        )
        logger.info(
            f"    Y-axis: [{frame['mid_Yx']:.3f}, {frame['mid_Yy']:.3f}, "
            f"{frame['mid_Yz']:.3f}]"
        )
        logger.info(
            f"    Z-axis: [{frame['mid_Zx']:.3f}, {frame['mid_Zy']:.3f}, "
            f"{frame['mid_Zz']:.3f}]"
        )

        # Check if direction cosines form a proper rotation matrix
        X_vec = np.array([frame["mid_Xx"], frame["mid_Xy"], frame["mid_Xz"]])
        Y_vec = np.array([frame["mid_Yx"], frame["mid_Yy"], frame["mid_Yz"]])
        Z_vec = np.array([frame["mid_Zx"], frame["mid_Zy"], frame["mid_Zz"]])

        # Check orthogonality
        dot_XY = np.dot(X_vec, Y_vec)
        dot_XZ = np.dot(X_vec, Z_vec)
        dot_YZ = np.dot(Y_vec, Z_vec)

        logger.info(
            f"    Orthogonality checks (should be ~0): XY={dot_XY:.3f}, "
            f"XZ={dot_XZ:.3f}, YZ={dot_YZ:.3f}"
        )

        # Check if this is a right-handed coordinate system
        cross_product = np.cross(X_vec, Y_vec)
        dot_cross_Z = np.dot(cross_product, Z_vec)
        logger.info("    Right-handed check (should be ~1): %s", dot_cross_Z)
        logger.info("")


if __name__ == "__main__":
    analyze_coordinate_system()
