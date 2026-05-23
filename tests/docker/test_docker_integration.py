#!/usr/bin/env python3
"""
Docker integration tests for Golf Modeling Suite launcher.

Tests Docker container setup, PYTHONPATH configuration, and module accessibility.
"""

import subprocess
import unittest
from unittest.mock import MagicMock, Mock

import pytest
from src.shared.python.data_io.path_utils import get_repo_root, get_src_root

# Docker launch command tests are broken after the launcher refactoring to
# mixin-based architecture (launcher_simulation.py, launcher_dialogs.py).
# The tests assume a single Popen call but the refactored code makes multiple
# subprocess calls (VcXsrv, docker). Needs full test rewrite.
_DOCKER_CMD_XFAIL = pytest.mark.xfail(
    reason="Docker command tests need rewrite for mixin-based launcher architecture",
    strict=False,
)


def _is_docker_available() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(["docker", "--version"], capture_output=True, timeout=5)
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class TestDockerBuild(unittest.TestCase):
    """Test Docker image building and configuration."""

    def test_dockerfile_syntax(self) -> None:
        """Test that Dockerfile has valid syntax."""
        dockerfile_path = get_repo_root() / "Dockerfile"
        self.assertTrue(
            dockerfile_path.exists(), f"Dockerfile not found at {dockerfile_path}"
        )

        content = dockerfile_path.read_text()

        # Check for required components (multi-stage slim Python runtime)
        self.assertRegex(
            content, r"FROM python:3\.12-slim(?:@sha256:[0-9a-f]{64})? AS builder"
        )
        self.assertRegex(
            content, r"FROM python:3\.12-slim(?:@sha256:[0-9a-f]{64})? AS runtime"
        )
        self.assertIn('PYTHONPATH="/workspace"', content)
        self.assertIn("WORKDIR /workspace", content)

    def test_dockerfile_pythonpath_setup(self) -> None:
        """Test that Dockerfile sets up PYTHONPATH correctly."""
        dockerfile_path = get_repo_root() / "Dockerfile"
        content = dockerfile_path.read_text()

        # Verify PYTHONPATH is set to the workspace root
        # The multi-stage Dockerfile sets PYTHONPATH="/workspace" in the runtime stage,
        pythonpath_line = [
            line for line in content.split("\n") if "PYTHONPATH=" in line
        ][0]
        self.assertIn("/workspace", pythonpath_line)
        self.assertEqual(pythonpath_line.strip(), 'PYTHONPATH="/workspace"')


class TestDockerRuntimeEntrypoint(unittest.TestCase):
    """Regression tests for the hardened runtime API entrypoint (#2786).

    Salvaged from stale PR #2723: the runtime image must default to the
    FastAPI server (not an interactive shell), bound to 0.0.0.0:8001, and
    must carry production hardening flags (proxy headers, single worker,
    access logs) that match the documented SPEC behavior.
    """

    def setUp(self):
        self.content = (get_repo_root() / "Dockerfile").read_text()

    def test_runtime_cmd_invokes_uvicorn_api_server(self):
        """CMD must launch src.api.server:app via uvicorn."""
        self.assertIn('"python3", "-m", "uvicorn"', self.content)
        self.assertIn('"src.api.server:app"', self.content)

    def test_runtime_cmd_binds_public_host_and_port(self):
        """CMD must bind 0.0.0.0:8001 to match EXPOSE/HEALTHCHECK."""
        self.assertIn('"--host", "0.0.0.0"', self.content)
        self.assertIn('"--port", "8001"', self.content)

    def test_runtime_cmd_single_worker_for_healthcheck(self):
        """Single worker keeps in-process state + HEALTHCHECK aligned."""
        self.assertIn('"--workers", "1"', self.content)

    def test_runtime_cmd_proxy_headers_hardening(self):
        """Proxy-aware flags must be present for reverse-proxy deployments."""
        self.assertIn('"--proxy-headers"', self.content)
        self.assertIn('"--forwarded-allow-ips"', self.content)
        self.assertIn('"--access-log"', self.content)

    def test_runtime_healthcheck_hits_health_endpoint(self):
        """HEALTHCHECK must probe /health on the same port as CMD."""
        self.assertIn("curl -f http://localhost:8001/health", self.content)

    def test_runtime_does_not_default_to_interactive_shell(self):
        """Runtime stage must not default CMD to /bin/bash."""
        # Extract the runtime stage (between `AS runtime` and the next `FROM`).
        runtime_start = self.content.index("AS runtime")
        next_from = self.content.find("\nFROM ", runtime_start)
        runtime_block = self.content[
            runtime_start : next_from if next_from != -1 else None
        ]
        self.assertNotIn('CMD ["/bin/bash"]', runtime_block)


class TestDockerLaunchCommands(unittest.TestCase):
    """Test Docker container launch command generation."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        # Mock launcher components
        self.mock_launcher = Mock()
        self.mock_launcher.chk_live = Mock()
        self.mock_launcher.chk_gpu = Mock()
        self.mock_launcher._launch_docker_container = Mock()

    @staticmethod
    def _configure_launcher_mocks(launcher):
        """Add required mock attributes for launcher docker commands.

        The launcher was refactored to use mixins (launcher_simulation.py),
        so __new__-created instances need these attributes manually set.
        """
        launcher.docker_launcher = MagicMock(
            spec=["check_image_exists", "build_image", "run_container"]
        )
        launcher.docker_launcher.check_image_exists.return_value = True
        launcher.process_manager = MagicMock(
            spec=["start_process", "stop_process", "is_running"]
        )
        launcher.lbl_status = MagicMock(spec=["setText", "text"])
        launcher.toast_manager = None  # Prevent show_toast from crashing
        launcher.show_toast = MagicMock()  # Mock the toast display method


class TestContainerEnvironment(unittest.TestCase):
    """Test container environment setup and module accessibility."""

    def test_pythonpath_environment_variable(self) -> None:
        """Test PYTHONPATH environment variable setup."""
        dockerfile_path = get_repo_root() / "Dockerfile"
        content = dockerfile_path.read_text()

        # Find PYTHONPATH line
        pythonpath_lines = [
            line for line in content.split("\n") if "PYTHONPATH=" in line
        ]
        self.assertEqual(
            len(pythonpath_lines), 1, "Should have exactly one PYTHONPATH definition"
        )

        pythonpath_line = pythonpath_lines[0]
        # so that "from src.xxx" imports work inside the container.
        self.assertEqual(pythonpath_line.strip(), 'PYTHONPATH="/workspace"')

    def test_workspace_directory_creation(self) -> None:
        """Test workspace directory structure creation."""
        dockerfile_path = get_repo_root() / "Dockerfile"
        content = dockerfile_path.read_text()

        # Check for workspace directory creation
        # The multi-stage Dockerfile creates /workspace and sets ownership
        # (source code is COPYed in, so subdirectories are implicit)
        self.assertIn("mkdir -p /workspace", content)
        self.assertIn("WORKDIR /workspace", content)

    def test_conda_environment_setup(self) -> None:
        """Test conda environment configuration."""
        dockerfile_path = get_repo_root() / "Dockerfile"
        content = dockerfile_path.read_text()

        # Verify base image (multi-stage slim Python build) and package installation
        self.assertRegex(
            content, r"FROM python:3\.12-slim(?:@sha256:[0-9a-f]{64})? AS builder"
        )
        self.assertRegex(
            content, r"FROM python:3\.12-slim(?:@sha256:[0-9a-f]{64})? AS runtime"
        )
        self.assertIn("python -m venv /opt/venv", content)
        self.assertIn(
            "python -m pip install --upgrade --no-cache-dir pip==26.1", content
        )
        self.assertIn("pip install -r /tmp/requirements.lock", content)

        # Check for required packages that are installed directly by the Dockerfile.
        required_packages = [
            "pandas",
            "matplotlib",
            "sympy",
            "defusedxml",
            "pin",
            "pin-pink",
            "qpsolvers",
            "meshcat",
        ]
        for package in required_packages:
            self.assertIn(package, content, f"Should install {package}")

        # matplotlib is installed directly in the image for shared-code imports.
        self.assertIn('"matplotlib==3.10.8"', content)

    def test_container_security_pins_clear_trivy_findings(self) -> None:
        """Docker runtime pins must stay at or above the Trivy fixed versions."""
        dockerfile_path = get_repo_root() / "Dockerfile"
        content = dockerfile_path.read_text()
        requirements_lock = (get_repo_root() / "requirements.lock").read_text()

        self.assertIn("pip install --upgrade pip==26.1", content)
        self.assertIn('"PyJWT==2.12.0"', content)
        self.assertIn('"cryptography==46.0.7"', content)
        self.assertIn("idna==3.15", requirements_lock)


class TestModuleAccessibility(unittest.TestCase):
    """Test that modules will be accessible in Docker containers."""

    def test_shared_module_structure(self) -> None:
        """Test shared module directory structure."""
        shared_path = get_src_root() / "shared" / "python"
        self.assertTrue(shared_path.exists(), "Shared python directory should exist")

        # Check for key modules that live in shared/python/ subpackages
        # Modules were reorganized into subpackages (config/, engine_core/, data_io/)
        key_modules = [
            ("config", "configuration_manager.py"),
            ("engine_core", "engine_manager.py"),
            ("data_io", "common_utils.py"),
            ("", "__init__.py"),
        ]

        for subdir, module in key_modules:
            if subdir:
                module_path = shared_path / subdir / module
            else:
                module_path = shared_path / module
            self.assertTrue(
                module_path.exists(), f"Key module {subdir}/{module} should exist"
            )

    def test_engine_directory_structure(self) -> None:
        """Test engine directory structure."""
        engines_path = get_src_root() / "engines"
        self.assertTrue(engines_path.exists(), "Engines directory should exist")

        # Check for physics engines
        physics_engines_path = engines_path / "physics_engines"
        self.assertTrue(
            physics_engines_path.exists(), "Physics engines directory should exist"
        )

        # Check for specific engines
        expected_engines = ["mujoco", "drake", "pinocchio"]
        for engine in expected_engines:
            engine_path = physics_engines_path / engine
            if engine_path.exists():  # Not all engines may be installed
                python_path = engine_path / "python"
                self.assertTrue(
                    python_path.exists(), f"{engine} should have python directory"
                )

    def test_mujoco_module_accessibility(self) -> None:
        """Test MuJoCo module structure for container access."""
        mujoco_python_path = (
            get_src_root() / "engines" / "physics_engines" / "mujoco" / "python"
        )

        if mujoco_python_path.exists():
            # Check for humanoid launcher
            humanoid_launcher = mujoco_python_path / "humanoid_launcher.py"
            self.assertTrue(
                humanoid_launcher.exists(), "Humanoid launcher should exist"
            )

            # Check for module package
            module_path = mujoco_python_path / "mujoco_humanoid_golf"
            self.assertTrue(
                module_path.exists(), "MuJoCo humanoid golf module should exist"
            )

            main_file = module_path / "__main__.py"
            self.assertTrue(main_file.exists(), "Module should have __main__.py")


if __name__ == "__main__":
    # Run tests with detailed output
    unittest.main(verbosity=2)
