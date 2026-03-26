#!/usr/bin/env python3
"""Script to add defusedxml to the existing upstream-drift Docker image."""

import logging
import os
import subprocess
import sys
import tempfile

logger = logging.getLogger(__name__)


def create_minimal_dockerfile() -> str:
    """Create a minimal Dockerfile to add defusedxml to upstream-drift."""
    dockerfile_content = """# Add defusedxml to existing upstream-drift
FROM upstream-drift:engine

# Install missing dependencies in the existing virtual environment
RUN /opt/mujoco-env/bin/pip install "defusedxml>=0.7.1" "PyQt6>=6.6.0"

# Update PATH to use upstream-drift by default
ENV PATH="/opt/mujoco-env/bin:$PATH"
ENV VIRTUAL_ENV="/opt/mujoco-env"
"""
    return dockerfile_content


def update_upstream_drift() -> bool:
    """Update the upstream-drift image with defusedxml."""
    logger.info("🔧 Adding defusedxml to existing upstream-drift Docker image...")

    # Create temporary directory for Dockerfile
    with tempfile.TemporaryDirectory() as temp_dir:
        dockerfile_path = os.path.join(temp_dir, "Dockerfile")

        # Write the minimal Dockerfile
        with open(dockerfile_path, "w") as f:
            f.write(create_minimal_dockerfile())

        logger.info("📝 Created temporary Dockerfile: %s", dockerfile_path)

        # Build the updated image
        cmd = ["docker", "build", "-t", "upstream-drift:engine", "."]

        try:
            logger.info("🚀 Running: %s", " ".join(cmd))
            logger.info(
                "📦 This should be quick since we're just adding one package..."
            )  # noqa: E501

            subprocess.run(cmd, cwd=temp_dir, check=True, text=True)

            logger.info("✅ Successfully added defusedxml to upstream-drift!")
            return True

        except subprocess.CalledProcessError as e:
            logger.error("❌ Failed to update upstream-drift: %s", e)
            return False
        except FileNotFoundError:
            logger.info("❌ Docker not found. Please install Docker Desktop.")
            return False


def test_updated_environment() -> bool:
    """Test that defusedxml is now available in the updated environment."""
    logger.info("\n🧪 Testing updated upstream-drift...")

    try:
        # Test defusedxml import
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "upstream-drift:engine",
                "python",
                "-c",
                "import defusedxml; print('✅ defusedxml available')",
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        logger.info("%s", result.stdout.strip())

        # Test defusedxml.ElementTree import
        result = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "upstream-drift:engine",
                "python",
                "-c",
                "import defusedxml.ElementTree; "
                "print('✅ defusedxml.ElementTree available')",  # noqa: E501
            ],
            capture_output=True,
            text=True,
            check=True,
        )

        logger.info("%s", result.stdout.strip())

        # Show what robotics libraries are available
        logger.info("\n📚 Available robotics libraries:")
        result = subprocess.run(
            ["docker", "run", "--rm", "upstream-drift:engine", "pip", "list"],
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
                logger.info("  %s", pkg)

        return True

    except subprocess.CalledProcessError as e:
        logger.error("❌ Test failed: %s", e.stderr)
        return False


def main() -> int:
    """Main function."""
    logger.info("🤖 Robotics Environment Updater")
    logger.info("%s", "=" * 50)

    # Update the environment
    success = update_upstream_drift()

    if success:
        # Test the updated environment
        test_success = test_updated_environment()

        if test_success:
            logger.info(
                "\n🎉 Success! The upstream-drift now has all required dependencies."
            )  # noqa: E501
            logger.info("💡 You can now run MuJoCo, Drake, and Pinocchio simulations!")
        else:
            logger.error(
                "\n⚠️  Update completed but tests failed. Check the output above."
            )  # noqa: E501
    else:
        logger.error(
            "\n💥 Failed to update upstream-drift. Check error messages above."
        )  # noqa: E501

    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
