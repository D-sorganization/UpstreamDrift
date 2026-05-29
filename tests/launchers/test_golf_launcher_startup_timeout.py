"""Tests for the startup-timeout safety net in UpstreamDriftLauncher (issue #5490).

When ``UpstreamDriftLauncher`` is constructed with ``loading=True`` it waits for
``update_startup_results`` to fire and replace the skeleton UI.  Prior to
issue #5490 the launcher silently spun forever if the async worker crashed
before calling that method, leaving the user staring at an empty grid.

These tests pin down the new behavior:

1. Entering the wait state emits a diagnostic log line.
2. A ``QTimer.singleShot(STARTUP_TIMEOUT_SEC * 1000, ...)`` is scheduled to
   recover from a hung startup worker.
3. ``_handle_startup_timeout`` clears ``loading``, surfaces a user-visible
   error toast, and re-enables a manual retry.
4. If ``update_startup_results`` fires before the timeout, the timeout
   handler is a no-op (no spurious error toast after success).
5. Module precondition: ``STARTUP_TIMEOUT_SEC > 0``.
"""

from __future__ import annotations

import contextlib
import logging
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

from src.launchers import upstream_drift_launcher
from src.launchers.upstream_drift_launcher import (
    STARTUP_TIMEOUT_SEC,
    UpstreamDriftLauncher,
)
from src.launchers.ui_components import StartupResults


@contextlib.contextmanager
def _patch_launcher_ui() -> Generator[None, None, None]:
    """Mirror the patches used by ``test_upstream_drift_launcher.py``.

    We deliberately do NOT patch ``QTimer`` at the module level here — the
    tests in this file want to observe ``QTimer.singleShot`` arguments
    directly.
    """
    with patch("src.launchers.upstream_drift_launcher.DockerCheckThread"):
        yield


def _upstream_drift_launcher_importable() -> bool:
    """Return True iff ``UpstreamDriftLauncher()`` can construct under current mocks.

    Some local environments (e.g. Python 3.14 + the ``conftest.py`` Qt
    mocks) lack the ``QApplication.primaryScreen`` shim needed for
    ``UpstreamDriftLauncher.__init__`` to run.  CI uses a real PyQt6, so the tests
    pass there.  We skip locally when the test harness clearly can't
    construct the widget at all.
    """
    try:  # pragma: no cover - environment guard
        from PyQt6.QtWidgets import QApplication

        return hasattr(QApplication, "primaryScreen")
    except (ImportError, AttributeError):  # pragma: no cover
        return False


_REQUIRES_REAL_QT = pytest.mark.skipif(
    not _upstream_drift_launcher_importable(),
    reason="Local Qt mocks lack QApplication.primaryScreen; runs in CI.",
)


def _make_startup_results() -> StartupResults:
    results = StartupResults()
    results.startup_time_ms = 100
    results.docker_available = True
    results.registry = MagicMock()
    results.engine_manager = MagicMock()
    return results


# ---------------------------------------------------------------------------
# Module-level invariants
# ---------------------------------------------------------------------------


def test_startup_timeout_constant_is_positive() -> None:
    """DbC precondition: ``STARTUP_TIMEOUT_SEC`` must be strictly positive."""
    assert isinstance(STARTUP_TIMEOUT_SEC, int | float)
    assert STARTUP_TIMEOUT_SEC > 0


# ---------------------------------------------------------------------------
# Entering the wait state
# ---------------------------------------------------------------------------


@_REQUIRES_REAL_QT
def test_loading_mode_schedules_timeout_via_qtimer(qapp) -> None:
    """``loading=True`` must arm a ``QTimer.singleShot`` for the timeout."""
    with (
        _patch_launcher_ui(),
        patch("src.launchers.upstream_drift_launcher.QTimer") as mock_qtimer,
    ):
        launcher = UpstreamDriftLauncher(loading=True)

        # At least one ``singleShot`` call must be the timeout arming.
        assert mock_qtimer.singleShot.called
        timeout_calls = [
            call_args
            for call_args in mock_qtimer.singleShot.call_args_list
            if call_args.args and call_args.args[0] == int(STARTUP_TIMEOUT_SEC * 1000)
        ]
        assert timeout_calls, (
            "Expected QTimer.singleShot to be armed with "
            f"{int(STARTUP_TIMEOUT_SEC * 1000)}ms timeout; "
            f"got calls={mock_qtimer.singleShot.call_args_list}"
        )
        # The scheduled callback must be ``_handle_startup_timeout``.
        assert any(
            getattr(call_args.args[1], "__name__", "") == "_handle_startup_timeout"
            or call_args.args[1] == launcher._handle_startup_timeout
            for call_args in timeout_calls
        )


@_REQUIRES_REAL_QT
def test_loading_mode_emits_diagnostic_log(qapp, caplog) -> None:
    """Entering the wait state must emit an informational log line."""
    caplog.set_level(logging.INFO, logger=upstream_drift_launcher.logger.name)
    with (
        _patch_launcher_ui(),
        patch("src.launchers.upstream_drift_launcher.QTimer"),
    ):
        UpstreamDriftLauncher(loading=True)

    messages = [r.getMessage().lower() for r in caplog.records]
    assert any("startup" in m and ("wait" in m or "timeout" in m) for m in messages), (
        f"Expected a log mentioning startup wait/timeout; got {messages}"
    )


# ---------------------------------------------------------------------------
# Timeout handler behavior
# ---------------------------------------------------------------------------


@_REQUIRES_REAL_QT
def test_handle_startup_timeout_clears_loading_and_surfaces_error(qapp) -> None:
    """The timeout handler must clear ``loading`` and notify the user."""
    with (
        _patch_launcher_ui(),
        patch("src.launchers.upstream_drift_launcher.QTimer"),
    ):
        launcher = UpstreamDriftLauncher(loading=True)
        assert launcher.loading is True

        with patch.object(launcher, "show_toast") as mock_toast:
            launcher._handle_startup_timeout()

        assert launcher.loading is False
        assert mock_toast.called
        # Toast must be an error and mention "timed out" or "startup".
        toast_call = mock_toast.call_args
        message = toast_call.args[0].lower()
        toast_type = toast_call.kwargs.get("toast_type") or (
            toast_call.args[1] if len(toast_call.args) > 1 else ""
        )
        assert "startup" in message or "timed out" in message
        assert toast_type == "error"


@_REQUIRES_REAL_QT
def test_handle_startup_timeout_is_idempotent_after_success(qapp) -> None:
    """If ``update_startup_results`` already fired, timeout is a no-op."""
    with (
        _patch_launcher_ui(),
        patch(
            "src.launchers.upstream_drift_launcher._lazy_load_model_registry"
        ) as mock_reg,
        patch(
            "src.launchers.upstream_drift_launcher._lazy_load_engine_manager"
        ) as mock_eng,
        patch("src.launchers.upstream_drift_launcher.QTimer"),
    ):
        mock_reg.return_value = MagicMock()
        mock_eng.return_value = (MagicMock(), MagicMock())

        launcher = UpstreamDriftLauncher(loading=True)
        # Simulate successful async startup completing before timeout.
        launcher.update_startup_results(_make_startup_results())
        assert launcher.loading is False

        with patch.object(launcher, "show_toast") as mock_toast:
            launcher._handle_startup_timeout()

        # No toast should be shown — the wait completed successfully.
        assert not mock_toast.called


@_REQUIRES_REAL_QT
def test_startup_clears_loading_before_grid_rebuild(qapp) -> None:
    """Regression #6611: Home tiles must render on launch, not after a sidebar click.

    ``_rebuild_grid`` paints placeholder ``SkeletonCard``s while ``self.loading``
    is True. ``update_startup_results`` must therefore clear ``loading`` *before*
    ``_load_layout`` triggers the grid rebuild, otherwise the Home view is left
    showing skeletons until the next rebuild trigger (a sidebar click).
    """
    with (
        _patch_launcher_ui(),
        patch(
            "src.launchers.upstream_drift_launcher._lazy_load_model_registry"
        ) as mock_reg,
        patch(
            "src.launchers.upstream_drift_launcher._lazy_load_engine_manager"
        ) as mock_eng,
        patch("src.launchers.upstream_drift_launcher.QTimer"),
    ):
        mock_reg.return_value = MagicMock()
        mock_eng.return_value = (MagicMock(), MagicMock())

        launcher = UpstreamDriftLauncher(loading=True)

        observed: dict[str, bool] = {}
        real_load_layout = launcher._load_layout

        def _spy_load_layout() -> None:
            observed["loading_at_load_layout"] = launcher.loading
            real_load_layout()

        launcher._load_layout = _spy_load_layout  # type: ignore[method-assign]
        launcher.update_startup_results(_make_startup_results())

    assert observed.get("loading_at_load_layout") is False, (
        "loading must be cleared before _load_layout rebuilds the grid, "
        "so the startup rebuild renders real cards instead of skeletons"
    )
    assert launcher.loading is False


@_REQUIRES_REAL_QT
def test_handle_startup_timeout_safe_without_toast_manager(qapp) -> None:
    """Timeout handler must not crash if ``toast_manager`` isn't ready."""
    with (
        _patch_launcher_ui(),
        patch("src.launchers.upstream_drift_launcher.QTimer"),
    ):
        launcher = UpstreamDriftLauncher(loading=True)
        launcher.toast_manager = None
        # Must not raise even if the toast subsystem is unavailable.
        launcher._handle_startup_timeout()
        assert launcher.loading is False


# ---------------------------------------------------------------------------
# Dead-code removal: create_model_card
# ---------------------------------------------------------------------------


def test_create_model_card_is_removed() -> None:
    """The dead placeholder ``create_model_card`` must be gone (issue #5490)."""
    assert not hasattr(UpstreamDriftLauncher, "create_model_card"), (
        "UpstreamDriftLauncher.create_model_card was an unused empty placeholder and "
        "should have been removed as part of issue #5490."
    )


# ---------------------------------------------------------------------------
# Pure-Python coverage of the handler (no Qt instantiation needed)
#
# These tests bind ``_handle_startup_timeout`` to a lightweight stand-in so
# the logic is exercised even in environments where the full launcher
# constructor cannot run (e.g. Python 3.14 + the Qt mock in conftest).
# ---------------------------------------------------------------------------


class _FakeLauncher:
    """Minimal stand-in mirroring the attributes the handler touches."""

    def __init__(self, *, loading: bool, toast_manager: object | None) -> None:
        self.loading = loading
        self.toast_manager = toast_manager
        self.toast_calls: list[tuple[str, str]] = []

    def show_toast(self, message: str, toast_type: str = "info") -> None:
        self.toast_calls.append((message, toast_type))


def test_handler_logic_clears_loading_when_still_loading(caplog) -> None:
    fake = _FakeLauncher(loading=True, toast_manager=object())
    caplog.set_level(logging.ERROR, logger=upstream_drift_launcher.logger.name)

    UpstreamDriftLauncher._handle_startup_timeout(fake)  # type: ignore[arg-type]

    assert fake.loading is False
    assert len(fake.toast_calls) == 1
    message, toast_type = fake.toast_calls[0]
    assert toast_type == "error"
    assert "timed out" in message.lower()
    assert any("did not complete" in r.getMessage().lower() for r in caplog.records)


def test_handler_logic_noop_when_already_loaded() -> None:
    fake = _FakeLauncher(loading=False, toast_manager=object())

    UpstreamDriftLauncher._handle_startup_timeout(fake)  # type: ignore[arg-type]

    assert fake.loading is False
    assert fake.toast_calls == []


def test_handler_logic_safe_without_toast_manager() -> None:
    fake = _FakeLauncher(loading=True, toast_manager=None)

    UpstreamDriftLauncher._handle_startup_timeout(fake)  # type: ignore[arg-type]

    assert fake.loading is False
    assert fake.toast_calls == []
