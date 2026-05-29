"""Tests for the MuJoCo engine quality check scripts."""

from unittest.mock import patch


def test_scripts_quality_check():
    """Verify that the scripts/quality_check.py delegates to main."""
    with patch("src.engines.physics_engines.mujoco.scripts.quality_check.main"):
        from src.engines.physics_engines.mujoco.scripts import quality_check

        # Just importing the module will not run main because of the if __name__ block
        # But we can verify it imports correctly
        assert hasattr(quality_check, "main")


def test_tools_code_quality_check():
    """Verify that the tools/code_quality_check.py delegates to main."""
    with patch("src.engines.physics_engines.mujoco.tools.code_quality_check.main"):
        from src.engines.physics_engines.mujoco.tools import code_quality_check

        # Just importing the module will not run main because of the if __name__ block
        # But we can verify it imports correctly
        assert hasattr(code_quality_check, "main")
