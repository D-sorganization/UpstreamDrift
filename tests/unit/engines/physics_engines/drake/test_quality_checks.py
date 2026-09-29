"""Tests for drake quality check proxies."""

from unittest.mock import patch


@patch("src.tools.code_quality_check.main")
def test_scripts_quality_check(mock_main):
    """Test scripts/quality_check.py."""
    # We just import it and check if it has main in scope
    import src.engines.physics_engines.drake.scripts.quality_check as qc

    assert hasattr(qc, "main")


@patch("src.tools.code_quality_check.main")
def test_tools_code_quality_check(mock_main):
    """Test tools/code_quality_check.py."""
    import src.engines.physics_engines.drake.tools.code_quality_check as cqc

    assert hasattr(cqc, "main")
