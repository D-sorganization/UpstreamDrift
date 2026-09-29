"""Tests for src/shared/python/logging_pkg/logger_utils.py.

Covers both the full-implementation path (numpy/data_io available) and the
public surface that does not depend on optional libraries.
"""

from __future__ import annotations

import logging
import random

import pytest

from logging_pkg import logger_utils as lu


class TestModuleConstants:
    def test_default_seed(self) -> None:
        assert isinstance(lu.DEFAULT_SEED, int)
        assert lu.DEFAULT_SEED >= 0

    def test_log_format(self) -> None:
        assert "%(asctime)s" in lu.LOG_FORMAT
        assert "%(name)s" in lu.LOG_FORMAT
        assert "%(levelname)s" in lu.LOG_FORMAT
        assert "%(message)s" in lu.LOG_FORMAT

    def test_log_level(self) -> None:
        assert lu.LOG_LEVEL == logging.INFO

    def test_all_exports_importable(self) -> None:
        for name in lu.__all__:
            assert hasattr(lu, name), f"missing export: {name}"


class TestGetLogger:
    def test_returns_logger_instance(self) -> None:
        log = lu.get_logger("wave7.test")
        assert isinstance(log, logging.Logger)
        assert log.name == "wave7.test"

    def test_default_name(self) -> None:
        log = lu.get_logger()
        assert isinstance(log, logging.Logger)


class TestSetupLogging:
    def test_default_level(self) -> None:
        # Should not raise. We don't assert root level because other tests
        # may have already configured root logging.
        lu.setup_logging()

    def test_custom_level(self) -> None:
        lu.setup_logging(level=logging.DEBUG)


class TestSetSeeds:
    def test_deterministic_python_random(self) -> None:
        lu.set_seeds(123)
        a = random.random()
        lu.set_seeds(123)
        b = random.random()
        assert a == b

    def test_rejects_negative_seed(self) -> None:
        # Full-implementation path raises; fallback path also raises when
        # validate=True (the default).
        with pytest.raises(ValueError, match="non-negative"):
            lu.set_seeds(-1)

    def test_default_seed_runs(self) -> None:
        lu.set_seeds()


class TestLogExecutionTime:
    def test_logs_duration(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.INFO), lu.log_execution_time("widget-build"):
            pass
        joined = " ".join(r.message for r in caplog.records)
        assert "widget-build" in joined
        assert "seconds" in joined or "Telemetry" in joined

    def test_uses_provided_logger(self, caplog: pytest.LogCaptureFixture) -> None:
        custom = logging.getLogger("wave7.custom")
        with (
            caplog.at_level(logging.INFO, logger="wave7.custom"),
            lu.log_execution_time("op", logger_obj=custom),
        ):
            pass
        assert any(r.name == "wave7.custom" for r in caplog.records)

    def test_logs_even_on_exception(self, caplog: pytest.LogCaptureFixture) -> None:
        with (
            caplog.at_level(logging.INFO),
            pytest.raises(RuntimeError),
            lu.log_execution_time("explodes"),
        ):
            raise RuntimeError("boom")
        # Duration message logged even when body raises.
        joined = " ".join(r.message for r in caplog.records)
        assert "explodes" in joined

    def test_rejects_empty_operation_name(self) -> None:
        # Bug-fix coverage for the fallback implementation only: previously
        # the duplicated `not (operation_name is not None)` check let empty
        # strings slip through. Now empty strings raise. The full
        # implementation (data_io.reproducibility) is wave-5 scope and not
        # touched here.
        if lu._USING_FULL_IMPLEMENTATION:
            pytest.skip("fallback path not exercised; full impl is wave-5 scope")
        with (
            pytest.raises(ValueError, match="non-empty"),
            lu.log_execution_time(""),
        ):
            pass
