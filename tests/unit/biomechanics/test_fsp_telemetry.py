"""Unit tests for ``fsp_telemetry`` (Phase 3 of the FSP epic, #5504).

``log_fsp_computed`` is best-effort — when the app-state ring buffer is
unavailable (older branches, headless test runs), the call must not
raise.  When the buffer *is* available, the FSP event must land in the
history store.
"""

from __future__ import annotations

import importlib

import pytest


def test_module_imports() -> None:
    mod = importlib.import_module("src.shared.python.biomechanics.fsp_telemetry")
    assert hasattr(mod, "log_fsp_computed")


def test_log_fsp_computed_does_not_crash_without_state_logger() -> None:
    """Telemetry is best-effort — must not raise even with no state logger."""
    from src.shared.python.biomechanics.fsp_telemetry import log_fsp_computed

    class _Result:
        slope_deg = 12.5
        direction_deg = -3.0

    log_fsp_computed(_Result())  # must not raise


def test_log_fsp_computed_tolerates_missing_attributes() -> None:
    """If ``slope_deg`` or ``direction_deg`` are missing, no exception."""
    from src.shared.python.biomechanics.fsp_telemetry import log_fsp_computed

    class _Empty:
        pass

    log_fsp_computed(_Empty())  # must not raise


def test_log_fsp_computed_records_event_when_state_logger_available() -> None:
    """If the state logger is importable, the event lands in the store."""
    try:
        from src.shared.python.app_state import get_state_logger
    except ImportError:
        pytest.skip("app_state not available on this branch")

    from src.shared.python.biomechanics.fsp_telemetry import log_fsp_computed

    logger = get_state_logger()
    before = len(logger.store)

    class _Result:
        slope_deg = 42.0
        direction_deg = 7.5

    log_fsp_computed(_Result())

    assert len(logger.store) == before + 1
