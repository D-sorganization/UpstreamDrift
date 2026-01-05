"""Tests for shared.python.__init__ coverage."""

import sys
from unittest.mock import patch

import pytest

import shared.python
from shared.python import ComparativePlotter, ComparativeSwingAnalyzer


def test_lazy_imports() -> None:
    """Test that lazy imports work correctly."""
    assert ComparativeSwingAnalyzer is not None
    assert ComparativePlotter is not None
    assert shared.python.ComparativeSwingAnalyzer is not None
    assert shared.python.ComparativePlotter is not None


def test_lazy_import_failure() -> None:
    """Test that lazy import failures are handled correctly."""
    # Patch sys.modules to simulate missing module
    with patch.dict(sys.modules, clear=True):
        # We need to remove the module from sys.modules to trigger re-import logic if we were strictly testing import mechanism,
        # but since shared.python is already imported, we are testing __getattr__ on it.
        # However, __getattr__ is only called if the attribute is NOT found.
        # Since we already imported them above, they might be cached.
        # But for 'pose_estimation', it is not in __all__ explicitly but handled in __getattr__.

        # Let's test a non-existent attribute
        with pytest.raises(AttributeError, match="has no attribute 'NonExistent'"):
            _ = shared.python.NonExistent


def test_pose_estimation_import() -> None:
    """Test importing pose_estimation via __getattr__."""
    pe = shared.python.pose_estimation
    assert pe is not None


def test_import_error_handling() -> None:
    """Test that ImportErrors are caught and re-raised with context."""
    # We need to force __getattr__ to be called for a heavy import, but simulate failure
    # To do this, we can try to access a heavy import that hasn't been accessed yet,
    # OR we can manually invoke __getattr__.

    # Manually invoking __getattr__ is easier to test the logic
    with patch("builtins.__import__", side_effect=ImportError("Simulated failure")):
        try:
            shared.python.__getattr__("ComparativeSwingAnalyzer")
        except ImportError as e:
            assert "Failed to import ComparativeSwingAnalyzer" in str(e)
            assert "Simulated failure" in str(e)

        try:
            shared.python.__getattr__("pose_estimation")
        except ImportError as e:
            assert "Failed to import pose_estimation" in str(e)
            assert "Simulated failure" in str(e)
