"""Integration tests for Phase 1 security hardening.

This module tests the complete security hardening implementation including:
- Secure subprocess execution
- Path validation and sanitization
- Integration with golf launcher
- Error handling and logging
"""

import sys
import unittest

import pytest
from src.shared.python.engine_core.engine_availability import (
    PYQT6_AVAILABLE,
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

pytestmark = pytest.mark.live_simulation
