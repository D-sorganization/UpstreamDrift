#!/usr/bin/env python3
"""
Simple test script to analyze MATLAB data structure for signal bus compatibility
This script uses only basic Python libraries to avoid dependency issues.
"""

import os
from pathlib import Path

import numpy as np
import scipy.io
import logging


logger = logging.getLogger(__name__)

def analyze_matlab_files() -> bool:
    """Analyze the MATLAB data files to understand the current structure"""
    logger.info("=== Analyzing MATLAB Data Files ===")

    # Check if files exist
    files = ["BASEQ.mat", "ZTCFQ.mat", "DELTAQ.mat"]
    existing_files = [f for f in files if Path(f).exists()]

    logger.info(f"Found {len(existing_files)} data files: {existing_files}")

    if not existing_files:
        logger.info("❌ No data files found!")
        return False

    # Analyze each file
    for filename in existing_files:
        logger.info(f"\n--- Analyzing {filename} ---")
        try:
            mat_data = scipy.io.loadmat(filename)

            logger.info(f"Keys in {filename}:")
            for key in mat_data.keys():
                if not key.startswith("__"):  # Skip metadata
                    value = mat_data[key]
                    if isinstance(value, np.ndarray):
                        logger.info(f"  {key}: shape {value.shape}, dtype {value.dtype}")
                        if value.ndim == 2 and value.shape[1] > 10:
                            logger.info(
                                f"    Large dataset: {value.shape[0]} rows, "
                                f"{value.shape[1]} columns"
                            )
                            # Show sample of first few columns
                            logger.info("    Sample data (first 3 rows, first 5 cols):")
                            logger.info(f"    {value[:3, :5]}")
                    else:
                        logger.info(f"  {key}: {type(value).__name__}")

        except (ValueError, TypeError, RuntimeError) as e:
            logger.info(f"❌ Error loading {filename}: {e}")
            return False

    return True


def check_signal_bus_structure() -> bool:
    """Check if the data structure suggests signal bus logging"""
    logger.info("\n=== Checking Signal Bus Structure ===")

    try:
        # Load BASEQ to analyze structure
        mat_data = scipy.io.loadmat("BASEQ.mat")

        if "BASEQ" in mat_data:
            baseq_data = mat_data["BASEQ"]
            if isinstance(baseq_data, np.ndarray) and baseq_data.ndim == 2:
                logger.info(
                    f"BASEQ data: {baseq_data.shape[0]} time points, "
                    f"{baseq_data.shape[1]} signals"
                )

                # Check if this looks like signal bus data
                if baseq_data.shape[1] > 50:
                    logger.info("✅ This appears to be signal bus data (many columns)")
                    logger.info("   Signal bus logging is likely being used")

                    # Try to identify signal types
                    logger.info("\nSignal analysis:")
                    for i in range(min(10, baseq_data.shape[1])):
                        col_data = baseq_data[:, i]
                        logger.info(
                            f"  Signal {i}: range [{col_data.min():.3f}, "
                            f"{col_data.max():.3f}], mean {col_data.mean():.3f}"
                        )

                    return True
                else:
                    logger.info("⚠️  This appears to be traditional logging (fewer columns)")
                    return False

        return False

    except (ValueError, TypeError, RuntimeError) as e:
        logger.info(f"❌ Error analyzing signal bus structure: {e}")
        return False


def check_required_signals() -> bool:
    """Check if required signals for GUI are present"""
    logger.info("\n=== Checking Required Signals ===")

    try:
        # Load all three files
        baseq_data = scipy.io.loadmat("BASEQ.mat")
        ztcfq_data = scipy.io.loadmat("ZTCFQ.mat")
        delta_data = scipy.io.loadmat("DELTAQ.mat")

        # Check for required signals (these would be column indices in the data)

        logger.info("Checking for required signals in each dataset:")

        datasets = [
            ("BASEQ", baseq_data),
            ("ZTCFQ", ztcfq_data),
            ("DELTAQ", delta_data),
        ]

        for name, data in datasets:
            if "BASEQ" in data or "ZTCFQ" in data or "DELTAQ" in data:
                # Get the actual data array
                key = name
                if key in data:
                    dataset = data[key]
                    logger.info(
                        f"  {name}: {dataset.shape[0]} time points, "
                        f"{dataset.shape[1]} signals"
                    )

                    # In signal bus format, signals are columns,
                    # so we need to check if we have enough
                    if dataset.shape[1] >= 6:  # At least 6 signals for positions
                        logger.info(f"    ✅ {name} has sufficient signals for GUI")
                    else:
                        logger.info(f"    ⚠️  {name} may be missing required signals")

        return True

    except (RuntimeError, ValueError, OSError) as e:
        logger.info(f"❌ Error checking required signals: {e}")
        return False


def main() -> Any:
    """Main analysis function"""
    logger.info("🚀 Starting Signal Bus Analysis")
    logger.info("=" * 50)

    # Change to script directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)

    # Run analyses
    tests = [
        ("MATLAB Files", analyze_matlab_files),
        ("Signal Bus Structure", check_signal_bus_structure),
        ("Required Signals", check_required_signals),
    ]

    results = {}
    for test_name, test_func in tests:
        logger.info(f"\n{'=' * 20} {test_name} {'=' * 20}")
        try:
            results[test_name] = test_func()
        except (RuntimeError, ValueError, OSError) as e:
            logger.info(f"❌ {test_name} failed: {e}")
            results[test_name] = False

    # Summary and recommendations
    logger.info(f"\n{'=' * 50}")
    logger.info("ANALYSIS SUMMARY")
    logger.info("=" * 50)

    all_passed = all(results.values())

    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        logger.info(f"{test_name}: {status}")

    logger.info(f"\nOverall: {'✅ COMPATIBLE' if all_passed else '⚠️  POTENTIAL ISSUES'}")

    if all_passed:
        logger.info("\n🎉 RECOMMENDATIONS:")
        logger.info("1. ✅ The current signal bus structure appears compatible with the GUI")
        logger.info("2. ✅ All required data files are present and readable")
        logger.info("3. ✅ Signal bus logging is likely being used (many columns)")
        logger.info(
            "4. 🔧 Consider adding GUI option to disable Simscape Results "
            "Explorer for speed"
        )
        logger.info("5. 🧪 Test with a full simulation run to verify all data is captured")

        logger.info("\n📋 NEXT STEPS:")
        logger.info("- Run a test simulation to generate new data")
        logger.info("- Verify the GUI can load and display the new data")
        logger.info("- Add the Simscape Results Explorer toggle option to the GUI")

    else:
        logger.info("\n⚠️  ISSUES DETECTED:")
        logger.info("- Review the errors above")
        logger.info("- Check if signal bus logging is properly configured")
        logger.info("- Verify all required signals are being logged")

    return all_passed


if __name__ == "__main__":
    main()
