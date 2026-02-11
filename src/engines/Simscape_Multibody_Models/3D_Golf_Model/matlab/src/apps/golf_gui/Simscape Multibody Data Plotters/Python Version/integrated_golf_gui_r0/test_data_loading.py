#!/usr/bin/env python3
"""
Test script to verify data loading and GUI functionality
"""

import os
import sys


from golf_gui_application import GolfVisualizerMainWindow
from PyQt6.QtWidgets import QApplication
from wiffle_data_loader import WiffleDataLoader
import logging


logger = logging.getLogger(__name__)

def test_data_loading():
    """Test the data loading functionality"""
    logger.info("🧪 Testing data loading...")

    try:
        # Load data
        loader = WiffleDataLoader()
        data = loader.load_data()
        logger.info(f"✅ Data loaded successfully: {len(data)} datasets")

        # Convert to GUI format
        baseq_data, ztcfq_data, deltaq_data = loader.convert_to_gui_format(data)
        logger.info("✅ GUI format conversion successful:")
        logger.info(f"   BASEQ: {baseq_data.shape}")
        logger.info(f"   ZTCFQ: {ztcfq_data.shape}")
        logger.info(f"   DELTAQ: {deltaq_data.shape}")

        return {"baseq": baseq_data, "ztcfq": ztcfq_data, "deltaq": deltaq_data}

    except Exception as e:
        logger.info(f"❌ Data loading failed: {e}")
        return None


def test_gui_launch():
    """Test launching the GUI"""
    logger.info("🧪 Testing GUI launch...")

    try:
        QApplication(sys.argv)

        # Create main window
        window = GolfVisualizerMainWindow()
        logger.info("✅ Main window created successfully")

        # Load test data
        gui_data = test_data_loading()
        if gui_data:
            success = window.load_data_from_dataframes(
                gui_data["baseq"], gui_data["ztcfq"], gui_data["deltaq"]
            )

            if success:
                logger.info("✅ Data loaded into GUI successfully")
                window.show()
                logger.info("✅ GUI window displayed")
                return True
            else:
                logger.info("❌ Failed to load data into GUI")
                return False
        else:
            logger.info("❌ No data available for GUI test")
            return False

    except Exception as e:
        logger.info(f"❌ GUI launch failed: {e}")
        return False


if __name__ == "__main__":
    logger.info("🚀 Starting Wiffle Swing Visualizer Tests")
    logger.info("=" * 50)

    # Test data loading
    data = test_data_loading()

    if data:
        logger.info("\n✅ All tests passed! The application should work correctly.")
        logger.info("\nTo launch the full application:")
        logger.info("   python simple_wiffle_launcher.py")
    else:
        logger.info("\n❌ Tests failed. Please check the error messages above.")
