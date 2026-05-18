"""Integration tests for Phase 1 security hardening.

This module tests the complete security hardening implementation including:
- Secure subprocess execution
- Path validation and sanitization
- Integration with golf launcher
- Error handling and logging
"""

import contextlib
import os
import subprocess  # nosec B404
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.shared.python.engine_core.engine_availability import (
    PYQT6_AVAILABLE,
    skip_if_unavailable,
)
from src.shared.python.security.secure_subprocess import (
    SecureSubprocessError,
    secure_popen,
    secure_run,
    validate_executable,
    validate_script_path,
)

# Use the current Python executable for cross-platform subprocess tests
PYTHON_EXE = sys.executable

# UpstreamDriftLauncher requires PyQt6, import conditionally
if PYQT6_AVAILABLE:
    from src.launchers.upstream_drift_launcher import UpstreamDriftLauncher
else:
    UpstreamDriftLauncher = None  # type: ignore[misc, assignment]


if __name__ == "__main__":
    unittest.main()
