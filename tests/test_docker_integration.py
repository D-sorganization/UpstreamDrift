#!/usr/bin/env python3
"""
Docker integration tests for Golf Modeling Suite launcher.

Tests Docker container setup, PYTHONPATH configuration, and module accessibility.
"""

import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock, patch


def _is_docker_available() -> bool:
    """Check if Docker is available."""
    try:
        result = subprocess.run(
            ["docker", "--version"], capture_output=True, timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


class TestDockerBuild(unittest.TestCase):
    """Test Docker image building and configuration."""

    def test_dockerfile_syntax(self):
        """Test that Dockerfile has valid syntax."""
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile"
        self.assertTrue(dockerfile_path.exists())

        content = dockerfile_path.read_text()

        # Check for required components
        self.assertIn("FROM continuumio/miniconda3:latest", content)
        self.assertIn("ENV PYTHONPATH=", content)
        self.assertIn("/workspace:/workspace/shared/python:/workspace/engines", content)
        self.assertIn("WORKDIR /workspace", content)

    def test_dockerfile_pythonpath_setup(self):
        """Test that Dockerfile sets up PYTHONPATH correctly."""
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile"
        content = dockerfile_path.read_text()

        # Verify PYTHONPATH includes all required directories
        pythonpath_line = [
            line for line in content.split("\n") if "PYTHONPATH=" in line
        ][0]
        self.assertIn("/workspace", pythonpath_line)
        self.assertIn("/workspace/shared/python", pythonpath_line)
        self.assertIn("/workspace/engines", pythonpath_line)

    @unittest.skipUnless(_is_docker_available(), "Docker not available")
    def test_docker_available(self):
        """Test that Docker is available for building."""
        result = subprocess.run(["docker", "--version"], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("Docker version", result.stdout)


class TestDockerLaunchCommands(unittest.TestCase):
    """Test Docker container launch command generation."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock launcher components
        self.mock_launcher = Mock()
        self.mock_launcher.chk_live = Mock()
        self.mock_launcher.chk_gpu = Mock()

    def test_mujoco_humanoid_command(self):
        """Test MuJoCo humanoid Docker command generation."""
        from launchers.golf_launcher import GolfLauncher

        # Create launcher instance without full initialization
        launcher = GolfLauncher.__new__(GolfLauncher)
        launcher.chk_live = Mock()
        launcher.chk_live.isChecked.return_value = True
        launcher.chk_gpu = Mock()
        launcher.chk_gpu.isChecked.return_value = False

        # Mock model
        mock_model = Mock()
        mock_model.type = "custom_humanoid"

        # Mock path
        mock_path = Path("/test/suite/path")

        with (
            patch("subprocess.Popen") as mock_popen,
            patch("os.name", "nt"),
            patch("launchers.golf_launcher.logger"),
        ):
            launcher._launch_docker_container(mock_model, mock_path)

            # Verify subprocess was called
            mock_popen.assert_called_once()

            # Get the command arguments
            call_args = mock_popen.call_args[0][0]
            command_str = " ".join(call_args)

            # Verify command structure
            self.assertIn("docker run", command_str)
            self.assertIn("--rm -it", command_str)
            self.assertIn("-v /test/suite/path:/workspace", command_str)
            self.assertIn(
                "-e PYTHONPATH=/workspace:/workspace/shared/python:/workspace/engines",
                command_str,
            )
            self.assertIn(
                "cd /workspace/engines/physics_engines/mujoco/python", command_str
            )
            self.assertIn("python humanoid_launcher.py", command_str)

    def test_drake_command(self):
        """Test Drake Docker command generation."""
        from launchers.golf_launcher import GolfLauncher

        launcher = GolfLauncher.__new__(GolfLauncher)
        launcher.chk_live = Mock()
        launcher.chk_live.isChecked.return_value = True
        launcher.chk_gpu = Mock()
        launcher.chk_gpu.isChecked.return_value = False

        mock_model = Mock()
        mock_model.type = "drake"

        mock_path = Path("/test/suite/path")

        with (
            patch("subprocess.Popen") as mock_popen,
            patch("os.name", "nt"),
            patch("launchers.golf_launcher.logger"),
            patch("launchers.golf_launcher.threading.Thread"),
        ):
            launcher._launch_docker_container(mock_model, mock_path)

            call_args = mock_popen.call_args[0][0]
            command_str = " ".join(call_args)

            # Verify Drake-specific components
            self.assertIn("-p 7000-7010:7000-7010", command_str)
            self.assertIn("-e MESHCAT_HOST=0.0.0.0", command_str)
            self.assertIn(
                "cd /workspace/engines/physics_engines/drake/python", command_str
            )
            self.assertIn("python -m src.drake_gui_app", command_str)

    def test_pinocchio_command(self):
        """Test Pinocchio Docker command generation."""
        from launchers.golf_launcher import GolfLauncher

        launcher = GolfLauncher.__new__(GolfLauncher)
        launcher.chk_live = Mock()
        launcher.chk_live.isChecked.return_value = True
        launcher.chk_gpu = Mock()
        launcher.chk_gpu.isChecked.return_value = False

        mock_model = Mock()
        mock_model.type = "pinocchio"

        mock_path = Path("/test/suite/path")

        with (
            patch("subprocess.Popen") as mock_popen,
            patch("os.name", "nt"),
            patch("launchers.golf_launcher.logger"),
            patch("launchers.golf_launcher.threading.Thread"),
        ):
            launcher._launch_docker_container(mock_model, mock_path)

            call_args = mock_popen.call_args[0][0]
            command_str = " ".join(call_args)

            # Verify Pinocchio-specific components
            self.assertIn("-p 7000-7010:7000-7010", command_str)
            self.assertIn("-e MESHCAT_HOST=0.0.0.0", command_str)
            self.assertIn(
                "cd /workspace/engines/physics_engines/pinocchio/python", command_str
            )
            self.assertIn("python pinocchio_golf/gui.py", command_str)

    def test_display_configuration_windows(self):
        """Test Windows display configuration."""
        from launchers.golf_launcher import GolfLauncher

        launcher = GolfLauncher.__new__(GolfLauncher)
        launcher.chk_live = Mock()
        launcher.chk_live.isChecked.return_value = True
        launcher.chk_gpu = Mock()
        launcher.chk_gpu.isChecked.return_value = False

        mock_model = Mock()
        mock_model.type = "custom_humanoid"
        mock_path = Path("/test/path")

        with (
            patch("subprocess.Popen") as mock_popen,
            patch("os.name", "nt"),
            patch("launchers.golf_launcher.logger"),
        ):
            launcher._launch_docker_container(mock_model, mock_path)

            call_args = mock_popen.call_args[0][0]
            command_str = " ".join(call_args)

            # Verify Windows-specific display setup
            self.assertIn("-e DISPLAY=host.docker.internal:0", command_str)
            self.assertIn("-e MUJOCO_GL=glfw", command_str)
            self.assertIn("-e LIBGL_ALWAYS_INDIRECT=1", command_str)

    def test_gpu_acceleration_option(self):
        """Test GPU acceleration option."""
        from launchers.golf_launcher import GolfLauncher

        launcher = GolfLauncher.__new__(GolfLauncher)
        launcher.chk_live = Mock()
        launcher.chk_live.isChecked.return_value = False
        launcher.chk_gpu = Mock()
        launcher.chk_gpu.isChecked.return_value = True  # Enable GPU

        mock_model = Mock()
        mock_model.type = "custom_humanoid"
        mock_path = Path("/test/path")

        with (
            patch("subprocess.Popen") as mock_popen,
            patch("os.name", "nt"),
            patch("launchers.golf_launcher.logger"),
        ):
            launcher._launch_docker_container(mock_model, mock_path)

            call_args = mock_popen.call_args[0][0]
            command_str = " ".join(call_args)

            # Verify GPU option is included
            self.assertIn("--gpus=all", command_str)


class TestContainerEnvironment(unittest.TestCase):
    """Test container environment setup and module accessibility."""

    def test_pythonpath_environment_variable(self):
        """Test PYTHONPATH environment variable setup."""
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile"
        content = dockerfile_path.read_text()

        # Find PYTHONPATH line
        pythonpath_lines = [
            line for line in content.split("\n") if "PYTHONPATH=" in line
        ]
        self.assertEqual(
            len(pythonpath_lines), 1, "Should have exactly one PYTHONPATH definition"
        )

        pythonpath_line = pythonpath_lines[0]
        expected_paths = [
            "/workspace",
            "/workspace/shared/python",
            "/workspace/engines",
        ]

        for path in expected_paths:
            self.assertIn(path, pythonpath_line, f"PYTHONPATH should include {path}")

    def test_workspace_directory_creation(self):
        """Test workspace directory structure creation."""
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile"
        content = dockerfile_path.read_text()

        # Check for workspace directory creation
        self.assertIn("mkdir -p /workspace/shared/python /workspace/engines", content)
        self.assertIn("WORKDIR /workspace", content)

    def test_conda_environment_setup(self):
        """Test conda environment configuration."""
        dockerfile_path = Path(__file__).parent.parent / "Dockerfile"
        content = dockerfile_path.read_text()

        # Verify base image and package installation
        self.assertIn("FROM continuumio/miniconda3:latest", content)
        self.assertIn("conda install", content)
        self.assertIn("python=3.11", content)

        # Check for required packages
        required_packages = ["numpy", "scipy", "matplotlib", "pandas", "pyqt"]
        for package in required_packages:
            self.assertIn(package, content, f"Should install {package}")


class TestModuleAccessibility(unittest.TestCase):
    """Test that modules will be accessible in Docker containers."""

    def test_shared_module_structure(self):
        """Test shared module directory structure."""
        shared_path = Path(__file__).parent.parent / "shared" / "python"
        self.assertTrue(shared_path.exists(), "Shared python directory should exist")

        # Check for key modules
        key_modules = [
            "configuration_manager.py",
            "process_worker.py",
            "engine_manager.py",
            "__init__.py",
        ]

        for module in key_modules:
            module_path = shared_path / module
            self.assertTrue(module_path.exists(), f"Key module {module} should exist")

    def test_engine_directory_structure(self):
        """Test engine directory structure."""
        engines_path = Path(__file__).parent.parent / "engines"
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

    def test_mujoco_module_accessibility(self):
        """Test MuJoCo module structure for container access."""
        mujoco_python_path = (
            Path(__file__).parent.parent
            / "engines"
            / "physics_engines"
            / "mujoco"
            / "python"
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
