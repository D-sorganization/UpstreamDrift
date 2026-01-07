#!/usr/bin/env python3
"""Script to rebuild the Docker image with updated dependencies."""

import subprocess
import sys
from pathlib import Path


def rebuild_docker_image():
    """Rebuild the robotics_env Docker image."""
    print("🔧 Rebuilding Docker image with updated dependencies...")

    # Get the directory containing this script (should be repo root)
    script_dir = Path(__file__).parent
    docker_dir = script_dir / "docker"

    if not docker_dir.exists():
        print(f"❌ Docker directory not found at {docker_dir}")
        return False

    print(f"📁 Building from: {docker_dir}")

    # Build command
    cmd = ["docker", "build", "-t", "robotics_env", "."]

    try:
        print(f"🚀 Running: {' '.join(cmd)}")
        print("📝 This may take several minutes...")

        # Run the build process
        subprocess.run(
            cmd,
            cwd=docker_dir,
            check=True,
            text=True,
            capture_output=False,  # Show output in real-time
        )

        print("✅ Docker image rebuilt successfully!")
        print("🎯 You can now run simulations with the updated dependencies.")
        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Docker build failed with exit code {e.returncode}")
        return False
    except FileNotFoundError:
        print("❌ Docker not found. Please install Docker Desktop.")
        return False


def main():
    """Main entry point."""
    print("🐳 MuJoCo Golf Model - Docker Image Rebuilder")
    print("=" * 50)

    success = rebuild_docker_image()

    if success:
        print("\n🎉 Build completed successfully!")
        print("💡 You can now launch simulations from the GUI.")
    else:
        print("\n💥 Build failed!")
        print("🔍 Check the error messages above for details.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
