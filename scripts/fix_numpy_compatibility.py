#!/usr/bin/env python3
"""
NumPy Compatibility Fixer for Golf Modeling Suite

This script resolves NumPy 2.x compatibility issues by ensuring all packages
are compatible with the installed NumPy version.
"""

import subprocess
import sys


def run_command(cmd: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and return the result."""
    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if check and result.returncode != 0:
        print(f"Error: {result.stderr}")
        sys.exit(1)
    return result


def check_numpy_version() -> str | None:
    """Check current NumPy version."""
    try:
        import numpy

        version: str = numpy.__version__
        print(f"Current NumPy version: {version}")
        return version
    except ImportError:
        print("NumPy not installed")
        return None


def fix_numpy_compatibility() -> None:
    """Fix NumPy compatibility issues."""
    print("🔧 Fixing NumPy compatibility issues...")

    # Check current NumPy version
    numpy_version = check_numpy_version()
    if not numpy_version:
        print("Installing NumPy...")
        run_command([sys.executable, "-m", "pip", "install", "numpy<2.0.0"])
        return

    # If NumPy 2.x is installed, we need to either:
    # 1. Downgrade to NumPy 1.x (safer for compatibility)
    # 2. Upgrade all dependent packages to NumPy 2.x compatible versions

    if numpy_version.startswith("2."):
        print("NumPy 2.x detected. Downgrading to 1.x for compatibility...")

        # Uninstall problematic packages first
        packages_to_reinstall = [
            "scipy",
            "pandas",
            "matplotlib",
            "scikit-learn",
        ]

        print("Uninstalling packages that may have NumPy compatibility issues...")
        for package in packages_to_reinstall:
            run_command(
                [sys.executable, "-m", "pip", "uninstall", "-y", package], check=False
            )

        # Downgrade NumPy
        print("Downgrading NumPy to 1.x...")
        run_command(
            [
                sys.executable,
                "-m",
                "pip",
                "install",
                "numpy>=1.26.4,<2.0.0",
                "--force-reinstall",
            ]
        )

        # Reinstall packages
        print("Reinstalling packages with NumPy 1.x compatibility...")
        for package in packages_to_reinstall:
            run_command([sys.executable, "-m", "pip", "install", package], check=False)

        # Install project dependencies
        print("Installing project dependencies...")
        run_command(
            [sys.executable, "-m", "pip", "install", "-e", ".", "--force-reinstall"]
        )

    print("✅ NumPy compatibility fix complete!")


def verify_installation() -> bool:
    """Verify that the installation works."""
    print("🧪 Verifying installation...")

    import importlib.util

    test_imports = [
        "numpy",
        "scipy",
        "pandas",
        "matplotlib",
    ]

    for module in test_imports:
        try:
            __import__(module)
            print(f"✅ {module} imports successfully")
        except ImportError as e:
            print(f"❌ {module} failed to import: {e}")
            return False

        # Test shared module import
        if importlib.util.find_spec("shared.python"):
            print("✅ shared.python imports successfully")
        else:
            print("❌ shared.python failed to import")
            return False

    return True


if __name__ == "__main__":
    print("🚀 Golf Modeling Suite - NumPy Compatibility Fixer")
    print("=" * 50)

    fix_numpy_compatibility()

    if verify_installation():
        print("\n🎉 All compatibility issues resolved!")
    else:
        print("\n❌ Some issues remain. Please check the error messages above.")
        sys.exit(1)
