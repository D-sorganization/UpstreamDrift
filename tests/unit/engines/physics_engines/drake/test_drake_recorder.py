"""Tests for drake_recorder.py import proxy."""


def test_drake_recorder_imports() -> None:
    """Test that the proxy module successfully exposes expected classes."""
    from src.engines.physics_engines.drake.python.src.drake_recorder import (
        DrakeInducedAccelerationAnalyzer,
        DrakeRecorder,
        setup_logging,
    )

    assert DrakeInducedAccelerationAnalyzer is not None
    assert DrakeRecorder is not None
    assert setup_logging is not None
