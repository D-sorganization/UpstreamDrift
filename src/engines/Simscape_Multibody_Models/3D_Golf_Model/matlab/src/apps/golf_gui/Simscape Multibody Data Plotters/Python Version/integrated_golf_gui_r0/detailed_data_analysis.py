#!/usr/bin/env python3
"""
Detailed analysis of MATLAB data structure
This script will explore the actual structure of the .mat files
to understand the data format.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import scipy.io

logger = logging.getLogger(__name__)


def deep_analyze_matlab_file(filename) -> bool:
    """Deep analysis of a MATLAB file structure"""
    logger.debug(f"\n=== Deep Analysis of {filename} ===")

    try:
        mat_data = scipy.io.loadmat(filename)

        logger.info(f"File: {filename}")
        logger.info(f"Keys: {list(mat_data.keys())}")

        for key, value in mat_data.items():
            if key.startswith("__"):
                continue

            logger.info(f"\nKey: {key}")
            logger.debug(f"  Type: {type(value)}")
            logger.debug(
                f"  Shape: {value.shape if hasattr(value, 'shape') else 'N/A'}"
            )
            logger.debug(
                f"  Dtype: {value.dtype if hasattr(value, 'dtype') else 'N/A'}"
            )

            if isinstance(value, np.ndarray):
                if value.dtype.names:  # Structured array
                    logger.info(f"  Structured array with fields: {value.dtype.names}")
                    for field_name in value.dtype.names:
                        field_data = value[field_name]
                        shape_info = (
                            field_data.shape if hasattr(field_data, "shape") else "N/A"
                        )
                        logger.info(
                            f"    {field_name}: {type(field_data)}, shape {shape_info}"
                        )

                        # If it's an object array, try to explore further
                        if hasattr(
                            field_data, "dtype"
                        ) and field_data.dtype == np.dtype("O"):
                            logger.info(
                                f"      Object array with {len(field_data)} elements"
                            )
                            for i, obj in enumerate(field_data[:3]):  # Show first 3
                                obj_shape = (
                                    obj.shape if hasattr(obj, "shape") else "N/A"
                                )
                                logger.info(
                                    f"        Element {i}: {type(obj)}, "
                                    f"shape {obj_shape}"
                                )
                                if hasattr(obj, "dtype"):
                                    logger.debug(f"        Dtype: {obj.dtype}")

                elif value.dtype == np.dtype("O"):  # Object array
                    logger.info(f"  Object array with {len(value)} elements")
                    for i, obj in enumerate(value[:3]):  # Show first 3
                        logger.info(
                            f"    Element {i}: {type(obj)}, "
                            f"shape {obj.shape if hasattr(obj, 'shape') else 'N/A'}"
                        )
                        if hasattr(obj, "dtype"):
                            logger.debug(f"    Dtype: {obj.dtype}")

                else:  # Regular numeric array
                    logger.info("  Numeric array")
                    if value.size > 0:
                        logger.info(
                            f"    Min: {value.min()}, Max: {value.max()}, "
                            f"Mean: {value.mean()}"
                        )
                        if value.ndim <= 2 and value.size <= 20:
                            logger.info(f"    Data: {value}")

        return True

    except (ValueError, TypeError, RuntimeError) as e:
        logger.error(f"❌ Error analyzing {filename}: {e}")
        import traceback

        traceback.print_exc()
        return False


def extract_actual_data(filename) -> np.ndarray | None:
    """Try to extract the actual data from the MATLAB file"""
    logger.debug(f"\n=== Extracting Data from {filename} ===")

    try:
        mat_data = scipy.io.loadmat(filename)

        # Look for the main data structure
        for key, value in mat_data.items():
            if key.startswith("__"):
                continue

            if isinstance(value, np.ndarray) and value.dtype.names:
                logger.info(f"Found structured array in key '{key}'")

                # Try to extract data from structured array
                for field_name in value.dtype.names:
                    field_data = value[field_name]
                    logger.info(f"  Field '{field_name}': {type(field_data)}")

                    if hasattr(field_data, "dtype") and field_data.dtype == np.dtype(
                        "O"
                    ):
                        # Object array - this might contain the actual data
                        logger.info(f"    Object array with {len(field_data)} elements")

                        for i, obj in enumerate(field_data):
                            if hasattr(obj, "shape") and len(obj.shape) == 2:
                                logger.info(f"      Element {i}: shape {obj.shape}")
                                if (
                                    obj.shape[1] > 10
                                ):  # Many columns suggest signal data
                                    logger.info("        This looks like signal data!")
                                    logger.info(
                                        "        Sample (first 3 rows, first 5 cols):"
                                    )
                                    logger.info(f"        {obj[:3, :5]}")

                                    # Check if this has the expected structure
                                    if obj.shape[0] > 100:  # Many time points
                                        logger.info(
                                            "        ✅ This appears to be "
                                            "the main dataset!"
                                        )
                                        return obj

                    elif hasattr(field_data, "shape") and len(field_data.shape) == 2:
                        logger.info(f"    Direct array: shape {field_data.shape}")
                        if field_data.shape[1] > 10:
                            logger.info("      This looks like signal data!")
                            return field_data

        return None

    except ImportError as e:
        logger.error(f"❌ Error extracting data from {filename}: {e}")
        return None


def main() -> None:
    """Main analysis function"""
    logger.info("🔍 Detailed MATLAB Data Structure Analysis")
    logger.info("=" * 60)

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    files = ["BASEQ.mat", "ZTCFQ.mat", "DELTAQ.mat"]

    # Deep analyze each file
    for filename in files:
        if Path(filename).exists():
            deep_analyze_matlab_file(filename)

    # Try to extract actual data
    logger.info("\n" + "=" * 60)
    logger.info("EXTRACTING ACTUAL DATA")
    logger.info("=" * 60)

    extracted_data = {}
    for filename in files:
        if Path(filename).exists():
            data = extract_actual_data(filename)
            if data is not None:
                extracted_data[filename] = data
                logger.info(
                    f"✅ Successfully extracted data from {filename}: {data.shape}"
                )
            else:
                logger.info(f"❌ Could not extract data from {filename}")

    # Summary
    logger.info(f"\n{'=' * 60}")
    logger.info("SUMMARY")
    logger.info("=" * 60)

    if extracted_data:
        logger.info("✅ Data extraction successful!")
        logger.info("The files contain structured data that can be accessed.")
        logger.info("\nData shapes:")
        for filename, data in extracted_data.items():
            logger.info(f"  {filename}: {data.shape}")

        logger.info("\n🎉 RECOMMENDATIONS:")
        logger.info("1. The data structure is compatible with signal bus logging")
        logger.info("2. The GUI should be able to handle this data format")
        logger.info(
            "3. Consider adding a data extraction function to handle this structure"
        )
        logger.info("4. Test the GUI with this data to verify compatibility")

    else:
        logger.info("❌ Could not extract usable data from the files")
        logger.info("The data structure may need special handling in the GUI")

    return len(extracted_data) > 0


if __name__ == "__main__":
    main()
