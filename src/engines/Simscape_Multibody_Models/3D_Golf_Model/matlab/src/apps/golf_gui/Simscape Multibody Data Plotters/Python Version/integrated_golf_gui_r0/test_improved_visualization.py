#!/usr/bin/env python3
"""
Test script for improved golf visualization
Demonstrates:
- Proper camera views (face-on and down-the-line 90° apart)
- Ground level at lowest Z point
- Club face normal vector
- Realistic golf club appearance
- Ball positioned for center strike
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add the current directory to Python path

from golf_gui_application import GolfVisualizerMainWindow
from PyQt6.QtWidgets import QApplication
import logging


logger = logging.getLogger(__name__)

def create_sample_data():
    """Create sample golf swing data for testing"""
    num_frames = 100

    # Create time vector
    time_vector = np.linspace(0, 2.0, num_frames)  # 2 second swing

    # Create sample positions (simplified golf swing)
    base_data = []

    for _i, t in enumerate(time_vector):
        # Simple swing motion
        swing_phase = t / 2.0  # 0 to 1

        # Club positions
        club_length = 1.0
        club_angle = swing_phase * np.pi  # 0 to pi (backswing to follow-through)

        # Clubhead position (circular motion)
        clubhead_x = np.cos(club_angle) * club_length
        clubhead_y = 0.5 + np.sin(club_angle) * club_length * 0.3
        clubhead_z = -0.1 + np.sin(club_angle) * 0.1  # Slight up/down motion

        # Butt position (grip)
        butt_x = clubhead_x * 0.1  # Grip near origin
        butt_y = 1.2  # Grip height
        butt_z = -0.05  # Slightly above ground

        # Midpoint
        midpoint_x = (clubhead_x + butt_x) / 2
        midpoint_y = (clubhead_y + butt_y) / 2
        midpoint_z = (clubhead_z + butt_z) / 2

        # Body positions (simplified)
        left_wrist = np.array([butt_x - 0.1, butt_y - 0.1, butt_z])
        left_elbow = np.array([butt_x - 0.2, butt_y + 0.1, butt_z])
        left_shoulder = np.array([butt_x - 0.3, butt_y + 0.3, butt_z])

        right_wrist = np.array([butt_x + 0.1, butt_y - 0.1, butt_z])
        right_elbow = np.array([butt_x + 0.2, butt_y + 0.1, butt_z])
        right_shoulder = np.array([butt_x + 0.3, butt_y + 0.3, butt_z])

        hub = np.array([0, 1.5, 0])  # Head position

        # Create frame data
        frame_data = {
            "Butt": [butt_x, butt_y, butt_z],
            "Clubhead": [clubhead_x, clubhead_y, clubhead_z],
            "MidPoint": [midpoint_x, midpoint_y, midpoint_z],
            "LeftWrist": [left_wrist[0], left_wrist[1], left_wrist[2]],
            "LeftElbow": [left_elbow[0], left_elbow[1], left_elbow[2]],
            "LeftShoulder": [left_shoulder[0], left_shoulder[1], left_shoulder[2]],
            "RightWrist": [right_wrist[0], right_wrist[1], right_wrist[2]],
            "RightElbow": [right_elbow[0], right_elbow[1], right_elbow[2]],
            "RightShoulder": [right_shoulder[0], right_shoulder[1], right_shoulder[2]],
            "Hub": [hub[0], hub[1], hub[2]],
            "TotalHandForceGlobal": [0, 0, 0],  # No forces for demo
            "EquivalentMidpointCoupleGlobal": [0, 0, 0],  # No torques for demo
        }

        base_data.append(frame_data)

    # Create DataFrames
    baseq_df = pd.DataFrame(base_data)
    ztcfq_df = baseq_df.copy()  # Same data for demo
    deltaq_df = baseq_df.copy()  # Same data for demo

    return baseq_df, ztcfq_df, deltaq_df


def main():
    """Main test function"""
    logger.info("🎯 Testing Improved Golf Visualization")
    logger.info("=" * 50)

    # Create sample data
    logger.info("📊 Creating sample golf swing data...")
    baseq_df, ztcfq_df, deltaq_df = create_sample_data()
    logger.info(f"✅ Created {len(baseq_df)} frames of sample data")

    # Create Qt application
    app = QApplication(sys.argv)

    # Create main window
    logger.info("🖥️ Creating main window...")
    window = GolfVisualizerMainWindow()

    # Load sample data into the motion capture tab
    logger.info("📥 Loading sample data...")
    motion_tab = window.tab_widget.widget(0)  # First tab is motion capture
    if hasattr(motion_tab, "opengl_widget"):
        motion_tab.opengl_widget.load_data_from_dataframes(
            (baseq_df, ztcfq_df, deltaq_df)
        )
        logger.info("✅ Sample data loaded successfully")

        # Set initial camera view
        logger.info("📷 Setting initial camera view...")
        motion_tab.opengl_widget.set_face_on_view()

        # Print usage instructions
        logger.info("\n🎮 Usage Instructions:")
        logger.info("   Press 1: Face-on view")
        logger.info("   Press 2: Down-the-line view (90° from face-on)")
        logger.info("   Press 3: Behind view")
        logger.info("   Press 4: Overhead view")
        logger.info("   Press R: Reset camera")
        logger.info("   Mouse: Rotate camera")
        logger.info("   Mouse wheel: Zoom")
        logger.info("   Space: Toggle playback")
        logger.info("\n🎯 Features to test:")
        logger.info("   ✓ Face-on and down-the-line views are 90° apart")
        logger.info("   ✓ Ground level is at lowest Z point")
        logger.info("   ✓ Red face normal vector shows club face direction")
        logger.info("   ✓ White ball positioned for center strike")
        logger.info("   ✓ More realistic golf club proportions")
    else:
        logger.info("❌ Could not find OpenGL widget")

    # Show window
    window.show()
    logger.info("✅ Window displayed - test the visualization!")

    # Run application
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
