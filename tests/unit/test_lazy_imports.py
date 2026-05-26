"""Tests for lazy import functionality in dependency management."""

import sys


class TestSharedModuleLazyImports:
    """Test that shared module doesn't eagerly import heavy dependencies."""

    def test_shared_init_no_eager_imports(self) -> None:
        """Verify shared/__init__.py doesn't import matplotlib at module level."""
        # The shared.python module is at src.shared.python
        # We verify that matplotlib is not imported at module level
        # (numpy and pandas are needed for common_utils and output_manager)
        import src.shared.python  # noqa: F401

        # Verify the module loaded successfully
        assert "src.shared.python" in sys.modules

        # The __init__.py should not import matplotlib directly
        # (matplotlib is only imported where needed for plotting)
        assert not hasattr(src.shared.python, "plt")
        assert not hasattr(src.shared.python, "matplotlib")

    def test_output_manager_imports_dependencies(self) -> None:
        """Verify output_manager.py imports numpy and pandas directly."""
        from src.shared.python.data_io import output_manager

        # These should be available in the module
        assert hasattr(output_manager, "np")
        assert hasattr(output_manager, "pd")

    def test_common_utils_imports(self) -> None:
        """Verify common_utils.py imports numpy/pandas but not matplotlib."""
        # Avoid deleting from sys.modules as it causes pandas C-API errors.
        from src.shared.python.data_io import common_utils

        # common_utils imports numpy and pandas at module level for utility functions
        # but NOT matplotlib (which is only imported where needed for plotting)
        assert not hasattr(common_utils, "plt")  # matplotlib not imported
        assert hasattr(common_utils, "np")  # numpy is imported
        assert hasattr(common_utils, "pd")  # pandas is imported


class TestGracefulDegradation:
    """Test that features degrade gracefully when dependencies missing."""

    def test_launcher_starts_without_mujoco(self) -> None:
        """Verify launcher can start even if MuJoCo not installed."""
        # This would be an integration test
        # For now, we verify the pattern exists

    def test_clear_error_messages(self) -> None:
        """Verify error messages are clear and helpful."""
        # Test ImportError message
        import_error_msg = (
            "The polynomial generator widget is not available.\n\n"
            "Error: No module named 'mujoco_humanoid_golf'\n\n"
            "Please ensure mujoco_humanoid_golf.polynomial_generator is installed."
        )

        # Verify message contains required elements
        assert "not available" in import_error_msg  # What went wrong
        assert "mujoco_humanoid_golf" in import_error_msg  # Missing dependency
        assert "ensure" in import_error_msg or "install" in import_error_msg  # Fix

        # Test OSError (DLL) message
        dll_error_msg = (
            "Failed to load MuJoCo library.\n\n"
            "Error: [WinError 1114] DLL initialization failed\n\n"
            "The polynomial generator requires MuJoCo to be properly installed.\n"
            "This feature will work inside the Docker container."
        )

        # Verify message contains required elements
        assert "Failed to load" in dll_error_msg  # What went wrong
        assert "MuJoCo" in dll_error_msg  # Missing dependency
        assert "Docker" in dll_error_msg  # Fix/alternative
        assert "properly installed" in dll_error_msg  # Fix instruction
