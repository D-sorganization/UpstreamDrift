#!/usr/bin/env python3
"""
Test script to verify data loading and GUI functionality
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from golf_gui_application import GolfVisualizerMainWindow
from PyQt6.QtWidgets import QApplication
from wiffle_data_loader import WiffleDataLoader


def test_data_loading():
    """Test the data loading functionality"""
    print("🧪 Testing data loading...")

    try:
        # Load data
        loader = WiffleDataLoader()
        data = loader.load_data()
        print(f"✅ Data loaded successfully: {len(data)} datasets")

        # Convert to GUI format
        baseq_data, ztcfq_data, deltaq_data = loader.convert_to_gui_format(data)
        print("✅ GUI format conversion successful:")
        print(f"   BASEQ: {baseq_data.shape}")
        print(f"   ZTCFQ: {ztcfq_data.shape}")
        print(f"   DELTAQ: {deltaq_data.shape}")

        return {"baseq": baseq_data, "ztcfq": ztcfq_data, "deltaq": deltaq_data}

    except Exception as e:
        print(f"❌ Data loading failed: {e}")
        return None


def test_gui_launch():
    """Test launching the GUI"""
    print("🧪 Testing GUI launch...")

    try:
        app = QApplication(sys.argv)

        # Create main window
        window = GolfVisualizerMainWindow()
        print("✅ Main window created successfully")

        # Load test data
        gui_data = test_data_loading()
        if gui_data:
            success = window.load_data_from_dataframes(
                gui_data["baseq"], gui_data["ztcfq"], gui_data["deltaq"]
            )

            if success:
                print("✅ Data loaded into GUI successfully")
                window.show()
                print("✅ GUI window displayed")
                return True
            else:
                print("❌ Failed to load data into GUI")
                return False
        else:
            print("❌ No data available for GUI test")
            return False

    except Exception as e:
        print(f"❌ GUI launch failed: {e}")
        return False


if __name__ == "__main__":
    print("🚀 Starting Wiffle Swing Visualizer Tests")
    print("=" * 50)

    # Test data loading
    data = test_data_loading()

    if data:
        print("\n✅ All tests passed! The application should work correctly.")
        print("\nTo launch the full application:")
        print("   python simple_wiffle_launcher.py")
    else:
        print("\n❌ Tests failed. Please check the error messages above.")
