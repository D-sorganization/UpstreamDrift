#!/usr/bin/env python3
"""Script to add defusedxml to the existing robotics_env Docker image."""

import os
import subprocess
import sys
import tempfile


def create_minimal_dockerfile() -> str:
    """Create a minimal Dockerfile to add defusedxml to robotics_env."""
    dockerfile_content = """# Add defusedxml to existing robotics_env
FROM robotics_env:latest

# Install missing dependencies in the existing virtual environment
RUN /opt/mujoco-env/bin/pip install "defusedxml>=0.7.1" "PyQt6>=6.6.0"

# Update PATH to use robotics_env by default
ENV PATH="/opt/mujoco-env/bin:$PATH"
ENV VIRTUAL_ENV="/opt/mujoco-env"
"""
    return dockerfile_content


def update_robotics_env() -> bool:
    """Update the robotics_env image with defusedxml."""
    print("🔧 Adding defusedxml to existing robotics_env Docker image...")

    # Create temporary directory for Dockerfile
    with tempfile.TemporaryDirectory() as temp_dir:
        dockerfile_path = os.path.join(temp_dir, "Dockerfile")

        # Write the minimal Dockerfile
        with open(dockerfile_path, "w") as f:
            f.write(create_minimal_dockerfile())

        print(f"📝 Created temporary Dockerfile: {dockerfile_path}")

        # Build the updated image
        cmd = ["docker", "build", "-t", "robotics_env", "."]

        try:
            print(f"🚀 Running: {' '.join(cmd)}")
            print("📦 This should be quick since we're just adding one package...")

            subprocess.run(cmd, cwd=temp_dir, check=True, text=True)

            print("✅ Successfully added defusedxml to robotics_env!")
            return True

        except subprocess.CalledProcessError as e:
            print(f"❌ Failed to update robotics_env: {e}")
            return False
        except FileNotFoundError:
            print("❌ Docker not found. Please install Docker Desktop.")
            return False


def test_updated_environment() -> bool:
    """Test that defusedxml is now available in the updated environment."""
    print("\n🧪 Testing updated robotics_env...")

    try:
        # Test defusedxml import
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "robotics_env",
                "python",
                "-c",
                "import defusedxml; print('✅ defusedxml available')",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        print(result.stdout.strip())

        # Test defusedxml.ElementTree import
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "robotics_env",
                "python",
                "-c",
                "import defusedxml.ElementTree; "
                "print('✅ defusedxml.ElementTree available')",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        print(result.stdout.strip())

        # Show what robotics libraries are available
        print("\n📚 Available robotics libraries:")
        result = subprocess.run(
            ["docker", "run", "--rm", "robotics_env", "pip", "list"],
            capture_output=True,
            text=True,
            check=True,
        )

        # Filter for robotics-related packages
        lines = result.stdout.split("\n")
        robotics_packages = []
        for line in lines:
            if any(
                pkg in line.lower()
                for pkg in [
                    "mujoco",
                    "drake",
                    "pinocchio",
                    "defusedxml",
                    "dm-control",
                    "jax",
                ]
            ):
                robotics_packages.append(line)

        for pkg in robotics_packages:
            if pkg.strip():
                print(f"  {pkg}")

        return True

    except subprocess.CalledProcessError as e:
        print(f"❌ Test failed: {e.stderr}")
        return False


def main() -> int:
    """Main function."""
    print("🤖 Robotics Environment Updater")
    print("=" * 50)

    # Update the environment
    success = update_robotics_env()

    if success:
        # Test the updated environment
        test_success = test_updated_environment()

        if test_success:
            print("\n🎉 Success! The robotics_env now has all required dependencies.")
            print("💡 You can now run MuJoCo, Drake, and Pinocchio simulations!")
        else:
            print("\n⚠️  Update completed but tests failed. Check the output above.")
    else:
        print("\n💥 Failed to update robotics_env. Check error messages above.")

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
