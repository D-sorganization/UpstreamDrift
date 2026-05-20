"""Tests for secure subprocess utilities."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

# Note: Import paths for the `src` package are configured at the test runner /
# package level (e.g., via pyproject.toml or conftest.py), so no manual
from src.shared.python.security.secure_subprocess import (
    SecureSubprocessError,
    secure_popen,
    secure_run,
    validate_executable,
    validate_script_path,
)


class TestSecureSubprocess(unittest.TestCase):
    """Test cases for secure subprocess utilities."""

    def setUp(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.suite_root = Path(self.temp_dir)

        # Create allowed directories
        (self.suite_root / "engines").mkdir()
        (self.suite_root / "launchers").mkdir()
        (self.suite_root / "tools").mkdir()
        (self.suite_root / "src").mkdir()

        # Create test scripts
        self.test_script = self.suite_root / "engines" / "test_script.py"
        self.test_script.write_text("print('hello')")

        self.malicious_script = Path(self.temp_dir) / ".." / "malicious.py"
        self.malicious_script.parent.mkdir(exist_ok=True)
        self.malicious_script.write_text("print('malicious')")

    def test_validate_executable_allowed(self) -> None:
        """Test that allowed executables pass validation."""
        # sys.executable should always be allowed
        result = validate_executable(sys.executable)
        self.assertEqual(result, sys.executable)

        # Standard allowed executables
        result = validate_executable("python")
        self.assertEqual(result, "python")

        result = validate_executable("docker")
        self.assertEqual(result, "docker")

    def test_validate_executable_disallowed(self) -> None:
        """Test that disallowed executables are rejected."""
        with self.assertRaises(SecureSubprocessError):
            validate_executable("rm")

        with self.assertRaises(SecureSubprocessError):
            validate_executable("curl")

        with self.assertRaises(SecureSubprocessError):
            validate_executable("/bin/sh")

    def test_validate_script_path_allowed(self) -> None:
        """Test that scripts in allowed directories pass validation."""
        # Should not raise exception
        validate_script_path(self.test_script, self.suite_root)

    def test_validate_script_path_outside_suite(self) -> None:
        """Test that scripts outside suite directory are rejected."""
        with self.assertRaises(SecureSubprocessError):
            validate_script_path(self.malicious_script, self.suite_root)

    def test_validate_script_path_disallowed_directory(self) -> None:
        """Test that scripts in disallowed directories are rejected."""
        bad_script = self.suite_root / "bad_dir" / "script.py"
        bad_script.parent.mkdir()
        bad_script.write_text("print('bad')")

        with self.assertRaises(SecureSubprocessError):
            validate_script_path(bad_script, self.suite_root)

    def test_validate_script_path_nonexistent(self) -> None:
        """Test that nonexistent scripts are rejected."""
        nonexistent = self.suite_root / "engines" / "nonexistent.py"

        with self.assertRaises(SecureSubprocessError):
            validate_script_path(nonexistent, self.suite_root)

    @patch("src.shared.python.security.secure_subprocess.subprocess.Popen")
    def test_secure_popen_valid_command(self, mock_popen) -> None:
        """Test secure_popen with valid command."""
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        result = secure_popen(
            [sys.executable, str(self.test_script)],
            cwd=str(self.suite_root),
            suite_root=self.suite_root,
        )

        self.assertEqual(result, mock_process)
        mock_popen.assert_called_once()

    @patch("src.shared.python.security.secure_subprocess.subprocess.Popen")
    def test_secure_popen_hides_windows_by_default_on_windows(self, mock_popen) -> None:
        """Background Python probes must not flash console windows on Windows."""
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        with patch("src.shared.python.security.secure_subprocess.os.name", "nt"):
            secure_popen(
                [sys.executable, str(self.test_script)],
                cwd=str(self.suite_root),
                suite_root=self.suite_root,
            )

        kwargs = mock_popen.call_args.kwargs
        self.assertEqual(kwargs["creationflags"], 0x08000000)

    @patch("src.shared.python.security.secure_subprocess.subprocess.Popen")
    def test_secure_popen_preserves_explicit_creationflags(self, mock_popen) -> None:
        """Interactive launch paths can still request their own console flags."""
        mock_process = MagicMock()
        mock_popen.return_value = mock_process

        with patch("src.shared.python.security.secure_subprocess.os.name", "nt"):
            secure_popen(
                [sys.executable, str(self.test_script)],
                cwd=str(self.suite_root),
                suite_root=self.suite_root,
                creationflags=0x10,
            )

        kwargs = mock_popen.call_args.kwargs
        self.assertEqual(kwargs["creationflags"], 0x10)

    def test_secure_popen_empty_command(self) -> None:
        """Test secure_popen with empty command."""
        with self.assertRaises(SecureSubprocessError):
            secure_popen([], suite_root=self.suite_root)

    def test_secure_popen_shell_not_allowed(self) -> None:
        """Test that shell=True is rejected."""
        with self.assertRaises(SecureSubprocessError):
            secure_popen(["echo", "test"], shell=True, suite_root=self.suite_root)  # nosec B604

    @patch("src.shared.python.security.secure_subprocess.subprocess.run")
    def test_secure_run_valid_command(self, mock_run) -> None:
        """Test secure_run with valid command."""
        mock_result = MagicMock()
        mock_run.return_value = mock_result

        result = secure_run(
            ["python", "--version"], timeout=5.0, suite_root=self.suite_root
        )

        self.assertEqual(result, mock_result)
        mock_run.assert_called_once()

    @patch("src.shared.python.security.secure_subprocess.subprocess.run")
    def test_secure_run_hides_windows_by_default_on_windows(self, mock_run) -> None:
        """Synchronous probes such as Docker checks should also stay hidden."""
        mock_result = MagicMock()
        mock_run.return_value = mock_result

        with patch("src.shared.python.security.secure_subprocess.os.name", "nt"):
            secure_run(["python", "--version"], timeout=5.0)

        kwargs = mock_run.call_args.kwargs
        self.assertEqual(kwargs["creationflags"], 0x08000000)

    def test_secure_run_disallowed_executable(self) -> None:
        """Test secure_run with disallowed executable."""
        with self.assertRaises(SecureSubprocessError):
            secure_run(["rm", "-rf", "/"], suite_root=self.suite_root)

    def test_secure_subprocess_working_directory_validation(self) -> None:
        """Test that working directory is validated."""
        outside_dir = Path(self.temp_dir) / ".." / "outside"
        outside_dir.mkdir(exist_ok=True)

        with self.assertRaises(SecureSubprocessError):
            secure_popen(
                ["python", "--version"],
                cwd=str(outside_dir),
                suite_root=self.suite_root,
            )


if __name__ == "__main__":
    unittest.main()
