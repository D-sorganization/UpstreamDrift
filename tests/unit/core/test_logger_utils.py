"""Tests for src.shared.python.logging_pkg.logger_utils (Issues #1949, #1744)."""

from __future__ import annotations

import logging
import time

from src.shared.python.logging_pkg.logger_utils import (
    DEFAULT_SEED,
    LOG_FORMAT,
    get_logger,
    log_execution_time,
    set_seeds,
    setup_logging,
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_default_seed_is_int(self) -> None:
        assert isinstance(DEFAULT_SEED, int)

    def test_default_seed_non_negative(self) -> None:
        assert DEFAULT_SEED >= 0

    def test_log_format_is_string(self) -> None:
        assert isinstance(LOG_FORMAT, str)

    def test_log_format_non_empty(self) -> None:
        assert len(LOG_FORMAT) > 0


# ---------------------------------------------------------------------------
# get_logger
# ---------------------------------------------------------------------------


class TestGetLogger:
    def test_returns_logger_like(self) -> None:
        logger = get_logger(__name__)
        assert hasattr(logger, "info")
        assert hasattr(logger, "error")
        assert hasattr(logger, "debug")

    def test_named_logger(self) -> None:
        logger = get_logger("test_module")
        assert logger is not None


# ---------------------------------------------------------------------------
# setup_logging
# ---------------------------------------------------------------------------


class TestSetupLogging:
    def test_does_not_raise(self) -> None:
        setup_logging(level=logging.WARNING)  # should not raise

    def test_accepts_debug_level(self) -> None:
        setup_logging(level=logging.DEBUG)  # should not raise


# ---------------------------------------------------------------------------
# set_seeds
# ---------------------------------------------------------------------------


class TestSetSeeds:
    def test_set_default_seed_does_not_raise(self) -> None:
        set_seeds()  # should not raise

    def test_set_specific_seed(self) -> None:
        set_seeds(seed=0)  # should not raise

    def test_set_large_seed(self) -> None:
        set_seeds(seed=999999)  # should not raise


# ---------------------------------------------------------------------------
# log_execution_time
# ---------------------------------------------------------------------------


class TestLogExecutionTime:
    def test_context_manager_runs_body(self) -> None:
        executed = []
        with log_execution_time("test_op"):
            executed.append(1)
        assert executed == [1]

    def test_context_manager_does_not_swallow_exception(self) -> None:
        import pytest

        with pytest.raises(ValueError), log_execution_time("test_op"):
            raise ValueError("deliberate error")

    def test_timing_is_non_negative(self) -> None:
        start = time.perf_counter()
        with log_execution_time("test_op"):
            pass
        elapsed = time.perf_counter() - start
        assert elapsed >= 0.0
