#!/usr/bin/env python3
"""
Test script to check signal bus compatibility with the GUI
This script will analyze the current MATLAB data structure 
and test if the GUI can handle it.
"""

import os
import sys
from pathlib import Path

import numpy as np
import scipy.io

# Add the current directory to the path so we can import the golf modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from golf_data_core import MatlabDataLoader


def test_matlab_data_structure():
    """Test the current MATLAB data structure to understand what's available"""
    print("=== Testing MATLAB Data Structure ===")

    # Check if the data files exist
    baseq_file = "BASEQ.mat"
    ztcfq_file = "ZTCFQ.mat"
    delta_file = "DELTAQ.mat"

    files_exist = all(Path(f).exists() for f in [baseq_file, ztcfq_file, delta_file])
    print(f"Data files exist: {files_exist}")

    if not files_exist:
        print("❌ Required data files not found!")
        return False

    # Load and analyze each file
    for filename in [baseq_file, ztcfq_file, delta_file]:
        print(f"\n--- Analyzing {filename} ---")
        try:
            mat_data = scipy.io.loadmat(filename)

            print(f"Keys in {filename}:")
            for key in mat_data.keys():
                if not key.startswith("__"):  # Skip metadata keys
                    value = mat_data[key]
                    if isinstance(value, np.ndarray):
                        print(
                            f"  {key}: {type(value).__name__} with shape {value.shape}"
                        )
                        if value.ndim == 2 and value.shape[1] > 10:
                            print(
                                f"    Sample columns: "
                                f"{list(range(min(5, value.shape[1])))}"
                            )
                    else:
                        print(f"  {key}: {type(value).__name__}")

        except Exception as e:
            print(f"❌ Error loading {filename}: {e}")
            return False

    return True


def test_data_loader():
    """Test the MatlabDataLoader with the current data structure"""
    print("\n=== Testing MatlabDataLoader ===")

    try:
        loader = MatlabDataLoader()
        datasets = loader.load_datasets("BASEQ.mat", "ZTCFQ.mat", "DELTAQ.mat")

        baseq_df, ztcfq_df, delta_df = datasets

        print(f"BASEQ shape: {baseq_df.shape}")
        print(f"ZTCFQ shape: {ztcfq_df.shape}")
        print(f"DELTAQ shape: {delta_df.shape}")

        print(f"\nBASEQ columns (first 10): {list(baseq_df.columns[:10])}")
        print(f"ZTCFQ columns (first 10): {list(ztcfq_df.columns[:10])}")
        print(f"DELTAQ columns (first 10): {list(delta_df.columns[:10])}")

        # Check for required columns
        required_columns = ["CHx", "CHy", "CHz", "MPx", "MPy", "MPz"]
        for df_name, df in [
            ("BASEQ", baseq_df),
            ("ZTCFQ", ztcfq_df),
            ("DELTAQ", delta_df),
        ]:
            missing_cols = [col for col in required_columns if col not in df.columns]
            if missing_cols:
                print(f"⚠️  {df_name} missing columns: {missing_cols}")
            else:
                print(f"✅ {df_name} has all required columns")

        return True

    except Exception as e:
        print(f"❌ Error in data loader: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_frame_processor():
    """Test the FrameProcessor with the current data"""
    print("\n=== Testing FrameProcessor ===")

    try:
        from golf_data_core import FrameProcessor, RenderConfig

        loader = MatlabDataLoader()
        datasets = loader.load_datasets("BASEQ.mat", "ZTCFQ.mat", "DELTAQ.mat")

        config = RenderConfig()
        processor = FrameProcessor(datasets, config)

        print(f"Number of frames: {processor.get_num_frames()}")
        print(f"Time vector length: {len(processor.get_time_vector())}")

        # Test getting a few frames
        for frame_idx in [
            0,
            processor.get_num_frames() // 2,
            processor.get_num_frames() - 1,
        ]:
            frame_data = processor.get_frame_data(frame_idx)
            print(f"Frame {frame_idx}:")
            print(f"  Clubhead: {frame_data.clubhead}")
            print(f"  Midpoint: {frame_data.midpoint}")
            print(f"  Shaft length: {frame_data.shaft_length:.3f}")
            print(f"  Valid: {frame_data.is_valid}")

        return True

    except Exception as e:
        print(f"❌ Error in frame processor: {e}")
        import traceback

        traceback.print_exc()
        return False


def analyze_signal_bus_structure():
    """Analyze the signal bus structure to understand the new logging setup"""
    print("\n=== Analyzing Signal Bus Structure ===")

    try:
        # Load one of the files to see the structure
        mat_data = scipy.io.loadmat("BASEQ.mat")

        # Look for signal bus related fields
        signal_bus_indicators = []
        for key in mat_data.keys():
            if not key.startswith("__"):
                value = mat_data[key]
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    # Check if this looks like signal bus data
                    if value.shape[1] > 20:  # Many columns suggest signal bus
                        signal_bus_indicators.append((key, value.shape))

        print("Potential signal bus data:")
        for key, shape in signal_bus_indicators:
            print(f"  {key}: {shape}")

        # Check for specific signal patterns
        if "BASEQ" in mat_data:
            baseq_data = mat_data["BASEQ"]
            if isinstance(baseq_data, np.ndarray) and baseq_data.ndim == 2:
                print(f"\nBASEQ data shape: {baseq_data.shape}")

                # Try to identify column patterns
                if baseq_data.shape[1] > 10:
                    print("Sample column analysis:")
                    for i in range(min(10, baseq_data.shape[1])):
                        col_data = baseq_data[:, i]
                        print(
                            f"  Column {i}: range [{col_data.min():.3f}, "
                            f"{col_data.max():.3f}], mean {col_data.mean():.3f}"
                        )

        return True

    except Exception as e:
        print(f"❌ Error analyzing signal bus structure: {e}")
        return False


def main():
    """Main test function"""
    print("🚀 Starting Signal Bus Compatibility Test")
    print("=" * 50)

    # Change to the script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Run tests
    tests = [
        ("MATLAB Data Structure", test_matlab_data_structure),
        ("Data Loader", test_data_loader),
        ("Frame Processor", test_frame_processor),
        ("Signal Bus Structure", analyze_signal_bus_structure),
    ]

    results = {}
    for test_name, test_func in tests:
        print(f"\n{'='*20} {test_name} {'='*20}")
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results[test_name] = False

    # Summary
    print(f"\n{'='*50}")
    print("TEST SUMMARY")
    print("=" * 50)

    all_passed = True
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name}: {status}")
        if not passed:
            all_passed = False

    print(
        f"\nOverall: {'✅ ALL TESTS PASSED' if all_passed else '❌ SOME TESTS FAILED'}"
    )

    if all_passed:
        print(
            "\n🎉 The GUI should be compatible with the current signal bus structure!"
        )
        print("Recommendations:")
        print("1. The current data structure appears to be compatible")
        print(
            "2. Consider adding a GUI option to disable Simscape Results "
            "Explorer for speed"
        )
        print("3. Test with a full simulation run to verify all data is captured")
    else:
        print("\n⚠️  Some compatibility issues detected. Review the errors above.")

    return all_passed


if __name__ == "__main__":
    main()
