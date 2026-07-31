"""Tests for secure subprocess utilities."""

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Note: Import paths for the `src` package are configured at the test runner /
# package level (e.g., via pyproject.toml or conftest.py), so no manual
from src.shared.python.security.secure_subprocess import (
    SecureSubprocessError,
    _is_within_root,
    secure_popen,
    secure_run,
    validate_executable,
    validate_script_path,
)

pytestmark = pytest.mark.unit


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

    def test_is_within_root_rejects_sibling_prefix(self) -> None:
        """A sibling sharing a string prefix must not count as contained.

        Regression for the prefix-collision bypass (mirrors issue #7689): an
        allowed root ``.../models`` must not admit the sibling
        ``.../models-backup`` just because the path strings share a prefix.
        """
        base = Path(self.temp_dir)
        root = base / "models"
        sibling = base / "models-backup"
        root.mkdir()
        sibling.mkdir()

        self.assertTrue(_is_within_root(root.resolve(), root.resolve()))
        self.assertTrue(_is_within_root((root / "src").resolve(), root.resolve()))
        self.assertFalse(_is_within_root(sibling.resolve(), root.resolve()))
        self.assertFalse(
            _is_within_root((sibling / "src" / "x.py").resolve(), root.resolve())
        )

    def test_validate_script_path_rejects_sibling_prefix(self) -> None:
        """validate_script_path must reject a sibling-prefix script directly.

        The script lives in a sibling directory whose path string is prefixed
        by the suite root (``<suite>-evil``). A separator-blind containment
        check would treat it as in-suite; the fix rejects it as outside the
        allowed roots.
        """
        base = Path(self.temp_dir)
        suite = base / "suite"
        evil_root = base / "suite-evil"
        (suite / "src").mkdir(parents=True)
        evil_script = evil_root / "src" / "evil.py"
        evil_script.parent.mkdir(parents=True)
        evil_script.write_text("print('evil')")

        with self.assertRaises(SecureSubprocessError) as ctx:
            validate_script_path(evil_script, suite)
        self.assertIn("outside allowed suite/tools directories", str(ctx.exception))

    def test_validate_script_path_nonexistent(self) -> None:
        """Test that nonexistent scripts are rejected."""
        nonexistent = self.suite_root / "engines" / "nonexistent.py"

        with self.assertRaises(SecureSubprocessError):
            validate_script_path(nonexistent, self.suite_root)

    @patch("src.shared.python.security.secure_subprocess.subprocess.Popen")
    def test_secure_popen_allows_movement_optimizer_sibling(self, mock_popen) -> None:
        """The launcher may execute the trusted Movement-Optimizer sibling entry point."""
        suite_root = self.suite_root / "UpstreamDrift"
        suite_root.mkdir()
        optimizer_root = self.suite_root / "Movement_Optimizer"
        script_path = optimizer_root / "src" / "movement_optimizer" / "__main__.py"
        script_path.parent.mkdir(parents=True)
        script_path.write_text("pass\n", encoding="utf-8")

        secure_popen(
            [sys.executable, str(script_path)],
            cwd=optimizer_root,
            suite_root=suite_root,
        )

        mock_popen.assert_called_once()

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
