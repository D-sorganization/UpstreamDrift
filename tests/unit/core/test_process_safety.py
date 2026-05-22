"""Tests for src.shared.python.core.process_safety.

Covers the three helpers added by issue #5911 to address the error-handling
findings from the 2026-05-21 adversarial review:

* ``managed_popen``  — context manager that guarantees subprocess cleanup
* ``safe_gather``    — ``asyncio.gather`` wrapper that defaults to
                       ``return_exceptions=True`` and logs partial failures
* ``narrow_catch``   — context manager that catches only listed exception
                       types and re-raises everything else

These helpers eliminate the patterns that caused the 4/10 score:

* bare ``except Exception: pass``
* ``subprocess.Popen`` without ``.wait()`` / ``.terminate()``
* ``asyncio.gather`` without ``return_exceptions=True``
"""

from __future__ import annotations

import logging
import subprocess
import sys

import pytest

from src.shared.python.core.process_safety import (
    AllTasksFailedError,
    managed_popen,
    narrow_catch,
    safe_gather,
)


# ---------------------------------------------------------------------------
# managed_popen
# ---------------------------------------------------------------------------


class TestManagedPopen:
    """Subprocess context manager guarantees cleanup."""

    def test_returns_popen_object(self) -> None:
        with managed_popen([sys.executable, "-c", "pass"]) as proc:
            assert isinstance(proc, subprocess.Popen)
        assert proc.returncode == 0

    def test_waits_on_clean_exit(self) -> None:
        with managed_popen(
            [sys.executable, "-c", "import time; time.sleep(0.05)"]
        ) as proc:
            pass
        assert proc.returncode is not None  # i.e. .wait() ran

    def test_terminates_on_exception_in_body(self) -> None:
        with (
            pytest.raises(RuntimeError, match="boom"),
            managed_popen(
                [sys.executable, "-c", "import time; time.sleep(60)"],
                timeout=2.0,
            ) as proc,
        ):
            assert proc.poll() is None  # still running
            raise RuntimeError("boom")
        # Process must be cleaned up even though body raised
        assert proc.returncode is not None

    def test_timeout_terminates_long_runner(self) -> None:
        with managed_popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            timeout=0.5,
        ) as proc:
            pass
        # Either terminated cleanly or killed; either way returncode is set
        assert proc.returncode is not None

    def test_kill_runs_when_terminate_does_not_stop(self) -> None:
        # Python script ignores SIGTERM by catching KeyboardInterrupt; ensure
        # the helper escalates to kill within kill_timeout
        script = (
            "import signal, time; "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "time.sleep(60)"
        )
        with managed_popen(
            [sys.executable, "-c", script],
            timeout=0.3,
            kill_timeout=0.3,
        ) as proc:
            pass
        assert proc.returncode is not None

    def test_rejects_string_args_to_prevent_shell_injection(self) -> None:
        with (
            pytest.raises(TypeError, match="args must be a sequence"),
            managed_popen("echo hello") as _proc,  # type: ignore[arg-type]
        ):
            pass

    def test_rejects_empty_args(self) -> None:
        with (
            pytest.raises(ValueError, match="args must be a non-empty sequence"),
            managed_popen([]) as _proc,
        ):
            pass

    def test_rejects_shell_true(self) -> None:
        with (
            pytest.raises(ValueError, match="shell=True is forbidden"),
            managed_popen([sys.executable, "-c", "pass"], shell=True) as _proc,
        ):
            pass

    def test_logs_on_nonzero_exit(self, caplog: pytest.LogCaptureFixture) -> None:
        with (
            caplog.at_level(logging.WARNING),
            managed_popen([sys.executable, "-c", "import sys; sys.exit(7)"]),
        ):
            pass
        assert any("exit code 7" in r.getMessage() for r in caplog.records)


# ---------------------------------------------------------------------------
# safe_gather
# ---------------------------------------------------------------------------


class TestSafeGather:
    """asyncio.gather wrapper that defaults to return_exceptions=True."""

    @pytest.mark.asyncio
    async def test_all_success(self) -> None:
        async def ok(n: int) -> int:
            return n * 2

        results = await safe_gather(ok(1), ok(2), ok(3))
        assert results == [2, 4, 6]

    @pytest.mark.asyncio
    async def test_partial_failure_returns_exceptions(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        async def ok() -> str:
            return "ok"

        async def boom() -> str:
            raise ValueError("nope")

        with caplog.at_level(logging.WARNING):
            results = await safe_gather(ok(), boom(), ok())

        assert results[0] == "ok"
        assert isinstance(results[1], ValueError)
        assert results[2] == "ok"
        # Partial failure must be logged
        assert any("1/3 tasks failed" in r.getMessage() for r in caplog.records)

    @pytest.mark.asyncio
    async def test_raise_on_all_failed(self) -> None:
        async def boom(i: int) -> int:
            raise RuntimeError(f"task {i} failed")

        with pytest.raises(AllTasksFailedError) as exc_info:
            await safe_gather(boom(1), boom(2), raise_on_all_failed=True)
        # The wrapper should expose the underlying exceptions
        assert len(exc_info.value.failures) == 2
        assert all(isinstance(f, RuntimeError) for f in exc_info.value.failures)

    @pytest.mark.asyncio
    async def test_empty_input_returns_empty_list(self) -> None:
        assert await safe_gather() == []

    @pytest.mark.asyncio
    async def test_rejects_non_awaitable(self) -> None:
        with pytest.raises(TypeError, match="awaitable"):
            await safe_gather("not an awaitable")  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# narrow_catch
# ---------------------------------------------------------------------------


class TestNarrowCatch:
    """Context manager that catches only listed exception types."""

    def test_catches_listed_type(self, caplog: pytest.LogCaptureFixture) -> None:
        with (
            caplog.at_level(logging.ERROR),
            narrow_catch(ValueError, log_message="checked op"),
        ):
            raise ValueError("nope")
        assert any("checked op" in r.getMessage() for r in caplog.records)

    def test_reraises_unlisted_type(self) -> None:
        with (
            pytest.raises(RuntimeError),
            narrow_catch(ValueError, log_message="checked op"),
        ):
            raise RuntimeError("uncaught")

    def test_no_exception_passes_through(self) -> None:
        side_effect: list[int] = []
        with narrow_catch(ValueError, log_message="op"):
            side_effect.append(1)
        assert side_effect == [1]

    def test_rejects_empty_exception_tuple(self) -> None:
        with (
            pytest.raises(ValueError, match="at least one"),
            narrow_catch(log_message="op"),  # type: ignore[call-overload]
        ):
            pass

    def test_rejects_non_exception_type(self) -> None:
        with (
            pytest.raises(TypeError, match="must be an Exception subclass"),
            narrow_catch(int, log_message="op"),  # type: ignore[arg-type]
        ):
            pass

    def test_rejects_catching_bare_exception(self) -> None:
        # The whole point of this helper is to AVOID bare except Exception.
        # If a caller passes Exception itself, we reject it.
        with (
            pytest.raises(
                ValueError, match="catching bare Exception defeats the purpose"
            ),
            narrow_catch(Exception, log_message="op"),
        ):
            pass

    def test_logs_traceback_via_exception_method(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        with (
            caplog.at_level(logging.ERROR),
            narrow_catch(ValueError, log_message="op X"),
        ):
            raise ValueError("detail")
        # Must use logger.exception(...) so the traceback is captured
        record = next(r for r in caplog.records if "op X" in r.getMessage())
        assert record.exc_info is not None
