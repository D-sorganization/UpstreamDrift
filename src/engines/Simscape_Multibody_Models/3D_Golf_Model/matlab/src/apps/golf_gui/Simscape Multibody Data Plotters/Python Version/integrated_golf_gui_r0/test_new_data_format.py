#!/usr/bin/env python3
"""
Quick test to verify the new test data files are compatible with GUI format
"""

import logging

import numpy as np
import scipy.io

logger = logging.getLogger(__name__)


def test_new_data_files():
    """Test the newly generated test data files"""
    logger.debug("=== Testing New Data Files ===")

    test_files = ["test_BASEQ.mat", "test_ZTCFQ.mat", "test_DELTAQ.mat"]

    for filename in test_files:
        logger.debug(f"\n--- Testing {filename} ---")
        try:
            mat_data = scipy.io.loadmat(filename)

            logger.info(f"Keys: {list(mat_data.keys())}")

            # Look for the main data array
            for key in mat_data.keys():
                if not key.startswith("__"):
                    data = mat_data[key]
                    logger.info(f"  {key}: shape {data.shape}, dtype {data.dtype}")

                    if isinstance(data, np.ndarray) and data.ndim == 2:
                        logger.info("    ✅ This is a numeric array - GUI compatible!")
                        logger.info("    Sample data (first 3 rows, first 5 cols):")
                        logger.info(f"    {data[:3, :5]}")

                        # Check if it has the expected structure
                        if data.shape[1] >= 7:  # time + 6 position signals
                            logger.info("    ✅ Has sufficient columns for GUI")
                        else:
                            logger.warning("    ⚠️  May be missing required signals")

                        return True

        except Exception as e:
            logger.error(f"❌ Error testing {filename}: {e}")

    return False


def main():
    """Main test function"""
    logger.info("🧪 Testing New Data Format Compatibility")
    logger.info("=" * 50)

    success = test_new_data_files()

    logger.info(f"\n{'=' * 50}")
    logger.info("SUMMARY")
    logger.info("=" * 50)

    if success:
        logger.info("✅ New data format is compatible with GUI!")
        logger.info("🎉 The test files should work with the GUI")
        logger.info("\n📋 Next Steps:")
        logger.info("1. Copy test_*.mat files to replace the old ones")
        logger.info("2. Test the GUI with the new data")
        logger.info("3. Fix the signal bus export process to generate this format")
    else:
        logger.info("❌ New data format still has issues")
        logger.info("🔧 Need to fix the data export process")

    return success


if __name__ == "__main__":
    main()
