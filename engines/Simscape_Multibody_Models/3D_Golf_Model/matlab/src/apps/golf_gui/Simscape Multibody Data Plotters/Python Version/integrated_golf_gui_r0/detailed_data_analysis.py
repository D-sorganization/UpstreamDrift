#!/usr/bin/env python3
"""
Detailed analysis of MATLAB data structure
This script will explore the actual structure of the .mat files
to understand the data format.
"""

import os
from pathlib import Path

import numpy as np
import scipy.io


def deep_analyze_matlab_file(filename):
    """Deep analysis of a MATLAB file structure"""
    print(f"\n=== Deep Analysis of {filename} ===")

    try:
        mat_data = scipy.io.loadmat(filename)

        print(f"File: {filename}")
        print(f"Keys: {list(mat_data.keys())}")

        for key, value in mat_data.items():
            if key.startswith("__"):
                continue

            print(f"\nKey: {key}")
            print(f"  Type: {type(value)}")
            print(f"  Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}")
            print(f"  Dtype: {value.dtype if hasattr(value, 'dtype') else 'N/A'}")

            if isinstance(value, np.ndarray):
                if value.dtype.names:  # Structured array
                    print(f"  Structured array with fields: {value.dtype.names}")
                    for field_name in value.dtype.names:
                        field_data = value[field_name]
                        shape_info = (
                            field_data.shape if hasattr(field_data, 'shape') else 'N/A'
                        )
                        print(
                            f"    {field_name}: {type(field_data)}, "
                            f"shape {shape_info}"
                        )

                        # If it's an object array, try to explore further
                        if hasattr(
                            field_data, "dtype"
                        ) and field_data.dtype == np.dtype("O"):
                            print(f"      Object array with {len(field_data)} elements")
                            for i, obj in enumerate(field_data[:3]):  # Show first 3
                                obj_shape = (
                                    obj.shape if hasattr(obj, 'shape') else 'N/A'
                                )
                                print(
                                    f"        Element {i}: {type(obj)}, "
                                    f"shape {obj_shape}"
                                )
                                if hasattr(obj, "dtype"):
                                    print(f"        Dtype: {obj.dtype}")

                elif value.dtype == np.dtype("O"):  # Object array
                    print(f"  Object array with {len(value)} elements")
                    for i, obj in enumerate(value[:3]):  # Show first 3
                        print(
                            f"    Element {i}: {type(obj)}, "
                            f"shape {obj.shape if hasattr(obj, 'shape') else 'N/A'}"
                        )
                        if hasattr(obj, "dtype"):
                            print(f"    Dtype: {obj.dtype}")

                else:  # Regular numeric array
                    print("  Numeric array")
                    if value.size > 0:
                        print(
                            f"    Min: {value.min()}, Max: {value.max()}, "
                            f"Mean: {value.mean()}"
                        )
                        if value.ndim <= 2 and value.size <= 20:
                            print(f"    Data: {value}")

        return True

    except Exception as e:
        print(f"❌ Error analyzing {filename}: {e}")
        import traceback

        traceback.print_exc()
        return False


def extract_actual_data(filename):
    """Try to extract the actual data from the MATLAB file"""
    print(f"\n=== Extracting Data from {filename} ===")

    try:
        mat_data = scipy.io.loadmat(filename)

        # Look for the main data structure
        for key, value in mat_data.items():
            if key.startswith("__"):
                continue

            if isinstance(value, np.ndarray) and value.dtype.names:
                print(f"Found structured array in key '{key}'")

                # Try to extract data from structured array
                for field_name in value.dtype.names:
                    field_data = value[field_name]
                    print(f"  Field '{field_name}': {type(field_data)}")

                    if hasattr(field_data, "dtype") and field_data.dtype == np.dtype(
                        "O"
                    ):
                        # Object array - this might contain the actual data
                        print(f"    Object array with {len(field_data)} elements")

                        for i, obj in enumerate(field_data):
                            if hasattr(obj, "shape") and len(obj.shape) == 2:
                                print(f"      Element {i}: shape {obj.shape}")
                                if (
                                    obj.shape[1] > 10
                                ):  # Many columns suggest signal data
                                    print("        This looks like signal data!")
                                    print(
                                        "        Sample (first 3 rows, first 5 cols):"
                                    )
                                    print(f"        {obj[:3, :5]}")

                                    # Check if this has the expected structure
                                    if obj.shape[0] > 100:  # Many time points
                                        print(
                                            "        ✅ This appears to be "
                                            "the main dataset!"
                                        )
                                        return obj

                    elif hasattr(field_data, "shape") and len(field_data.shape) == 2:
                        print(f"    Direct array: shape {field_data.shape}")
                        if field_data.shape[1] > 10:
                            print("      This looks like signal data!")
                            return field_data

        return None

    except Exception as e:
        print(f"❌ Error extracting data from {filename}: {e}")
        return None


def main():
    """Main analysis function"""
    print("🔍 Detailed MATLAB Data Structure Analysis")
    print("=" * 60)

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    files = ["BASEQ.mat", "ZTCFQ.mat", "DELTAQ.mat"]

    # Deep analyze each file
    for filename in files:
        if Path(filename).exists():
            deep_analyze_matlab_file(filename)

    # Try to extract actual data
    print("\n" + "=" * 60)
    print("EXTRACTING ACTUAL DATA")
    print("=" * 60)

    extracted_data = {}
    for filename in files:
        if Path(filename).exists():
            data = extract_actual_data(filename)
            if data is not None:
                extracted_data[filename] = data
                print(f"✅ Successfully extracted data from {filename}: {data.shape}")
            else:
                print(f"❌ Could not extract data from {filename}")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("=" * 60)

    if extracted_data:
        print("✅ Data extraction successful!")
        print("The files contain structured data that can be accessed.")
        print("\nData shapes:")
        for filename, data in extracted_data.items():
            print(f"  {filename}: {data.shape}")

        print("\n🎉 RECOMMENDATIONS:")
        print("1. The data structure is compatible with signal bus logging")
        print("2. The GUI should be able to handle this data format")
        print("3. Consider adding a data extraction function to handle this structure")
        print("4. Test the GUI with this data to verify compatibility")

    else:
        print("❌ Could not extract usable data from the files")
        print("The data structure may need special handling in the GUI")

    return len(extracted_data) > 0


if __name__ == "__main__":
    main()
