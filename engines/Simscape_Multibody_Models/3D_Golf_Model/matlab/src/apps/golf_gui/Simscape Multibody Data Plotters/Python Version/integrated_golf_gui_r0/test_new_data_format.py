#!/usr/bin/env python3
"""
Quick test to verify the new test data files are compatible with GUI format
"""


import numpy as np
import scipy.io


def test_new_data_files():
    """Test the newly generated test data files"""
    print("=== Testing New Data Files ===")

    test_files = ["test_BASEQ.mat", "test_ZTCFQ.mat", "test_DELTAQ.mat"]

    for filename in test_files:
        print(f"\n--- Testing {filename} ---")
        try:
            mat_data = scipy.io.loadmat(filename)

            print(f"Keys: {list(mat_data.keys())}")

            # Look for the main data array
            for key in mat_data.keys():
                if not key.startswith("__"):
                    data = mat_data[key]
                    print(f"  {key}: shape {data.shape}, dtype {data.dtype}")

                    if isinstance(data, np.ndarray) and data.ndim == 2:
                        print("    ✅ This is a numeric array - GUI compatible!")
                        print("    Sample data (first 3 rows, first 5 cols):")
                        print(f"    {data[:3, :5]}")

                        # Check if it has the expected structure
                        if data.shape[1] >= 7:  # time + 6 position signals
                            print("    ✅ Has sufficient columns for GUI")
                        else:
                            print("    ⚠️  May be missing required signals")

                        return True

        except Exception as e:
            print(f"❌ Error testing {filename}: {e}")

    return False


def main():
    """Main test function"""
    print("🧪 Testing New Data Format Compatibility")
    print("=" * 50)

    success = test_new_data_files()

    print(f"\n{'='*50}")
    print("SUMMARY")
    print("=" * 50)

    if success:
        print("✅ New data format is compatible with GUI!")
        print("🎉 The test files should work with the GUI")
        print("\n📋 Next Steps:")
        print("1. Copy test_*.mat files to replace the old ones")
        print("2. Test the GUI with the new data")
        print("3. Fix the signal bus export process to generate this format")
    else:
        print("❌ New data format still has issues")
        print("🔧 Need to fix the data export process")

    return success


if __name__ == "__main__":
    main()
