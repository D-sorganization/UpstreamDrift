"""Resource-cleanup helpers for subprocesses, asyncio tasks, and narrow exception handling.

This module exists to replace the three error-handling anti-patterns flagged
by the 2026-05-21 adversarial review (issue #5911 / epic #5907):

* bare ``except Exception: pass``  →  :func:`narrow_catch`
* ``subprocess.Popen`` without cleanup  →  :func:`managed_popen`
* ``asyncio.gather`` without ``return_exceptions=True``  →  :func:`safe_gather`

All public APIs validate inputs (Design-by-Contract) and document
invariants. Helpers compose with the existing :mod:`error_decorators` module
(``log_errors``, ``retry_on_error``, ``ErrorContext``); they do not duplicate
its behaviour.

Law-of-Demeter note: these helpers never reach into caller objects. They
wrap, return, and re-raise.
"""

from __future__ import annotations

import asyncio
import logging
import subprocess
from collections.abc import Awaitable, Iterator, Sequence
from contextlib import contextmanager
from typing import Any, TypeVar

logger = logging.getLogger(__name__)

__all__ = [
    "AllTasksFailedError",
    "managed_popen",
    "narrow_catch",
    "safe_gather",
]

T = TypeVar("T")

# Default escalation timeout (seconds) between SIGTERM and SIGKILL when a
# subprocess does not terminate on first request.
_DEFAULT_KILL_TIMEOUT_S = 5.0


class AllTasksFailedError(RuntimeError):
    """Raised by :func:`safe_gather` when every awaitable failed.

    Attributes:
        failures: List of the underlying exceptions in input order.
    """

    def __init__(self, failures: list[BaseException]) -> None:
        if not failures:
            raise ValueError("failures must be a non-empty list")
        self.failures = list(failures)
        types = ", ".join(sorted({type(f).__name__ for f in failures}))
        super().__init__(f"all {len(failures)} tasks failed (types: {types})")


# ---------------------------------------------------------------------------
# Subprocess
# ---------------------------------------------------------------------------


@contextmanager
def managed_popen(
    args: Sequence[str],
    *,
    timeout: float | None = None,
    kill_timeout: float = _DEFAULT_KILL_TIMEOUT_S,
    **popen_kwargs: Any,
) -> Iterator[subprocess.Popen[Any]]:
    """Run a subprocess with guaranteed cleanup on context exit.

    On normal exit, ``.wait(timeout)`` is called and a non-zero return code
    is logged at WARNING level.

    On an exception raised inside the ``with`` block, or if ``.wait`` times
    out, the process is sent ``.terminate()`` (SIGTERM on POSIX, ``CTRL_BREAK``
    on Windows). If the process is still alive after ``kill_timeout`` seconds,
    it is ``.kill()``-ed.

    Args:
        args: Command and arguments. Must be a non-empty sequence (no
            ``shell=True`` string-form invocation allowed; see
            :class:`ValueError` cases below).
        timeout: Maximum seconds to wait for normal completion. ``None``
            waits indefinitely.
        kill_timeout: Seconds to wait after ``.terminate()`` before
            escalating to ``.kill()``. Must be non-negative.
        **popen_kwargs: Forwarded to :class:`subprocess.Popen`. ``shell``
            must not be passed (or must be falsy).

    Yields:
        The live :class:`subprocess.Popen` instance.

    Raises:
        TypeError: If ``args`` is a string or non-sequence.
        ValueError: If ``args`` is empty, or ``shell=True`` is supplied, or
            ``kill_timeout`` is negative.

    Invariant on exit:
        ``proc.returncode is not None`` — the process has been reaped.
    """
    if isinstance(args, str) or not hasattr(args, "__iter__"):
        raise TypeError(
            "args must be a sequence of strings (no shell=True string form)"
        )
    args_list = list(args)
    if not args_list:
        raise ValueError("args must be a non-empty sequence")
    if popen_kwargs.get("shell"):
        raise ValueError("shell=True is forbidden by managed_popen; use the list form")
    if kill_timeout < 0:
        raise ValueError(f"kill_timeout must be non-negative, got {kill_timeout!r}")

    proc = subprocess.Popen(args_list, **popen_kwargs)
    try:
        yield proc
    except BaseException:
        _cleanup_process(proc, kill_timeout=kill_timeout)
        raise
    else:
        _wait_or_terminate(proc, timeout=timeout, kill_timeout=kill_timeout)
    finally:
        # Defensive: even if both paths above somehow returned without
        # reaping, ensure the invariant holds.
        if proc.returncode is None:
            _cleanup_process(proc, kill_timeout=kill_timeout)

    if proc.returncode != 0:
        logger.warning(
            "subprocess %s exited with exit code %s",
            args_list[0],
            proc.returncode,
        )


def _wait_or_terminate(
    proc: subprocess.Popen[Any],
    *,
    timeout: float | None,
    kill_timeout: float,
) -> None:
    """Wait for ``proc`` up to ``timeout`` seconds, else terminate."""
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning(
            "subprocess %s did not exit within %ss; terminating",
            proc.args,
            timeout,
        )
        _cleanup_process(proc, kill_timeout=kill_timeout)


def _cleanup_process(proc: subprocess.Popen[Any], *, kill_timeout: float) -> None:
    """Terminate, then kill, then wait. Idempotent if process already exited."""
    if proc.returncode is not None:
        return
    try:
        proc.terminate()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=kill_timeout)
        return
    except subprocess.TimeoutExpired:
        logger.warning(
            "subprocess %s ignored terminate(); escalating to kill()",
            proc.args,
        )
    try:
        proc.kill()
    except ProcessLookupError:
        return
    try:
        proc.wait(timeout=kill_timeout)
    except subprocess.TimeoutExpired:
        logger.error(
            "subprocess %s did not exit after kill(); zombie possible",
            proc.args,
        )


# ---------------------------------------------------------------------------
# Async gather
# ---------------------------------------------------------------------------


async def safe_gather(
    *coros: Awaitable[T],
    raise_on_all_failed: bool = False,
    log_partial: bool = True,
) -> list[T | BaseException]:
    """Run awaitables concurrently, never propagating partial failures.

    Equivalent to ``asyncio.gather(*coros, return_exceptions=True)`` but with
    structured logging on partial failure and an opt-in fail-loud mode when
    every task fails.

    Args:
        *coros: Awaitables to run concurrently.
        raise_on_all_failed: If every awaitable raised, wrap the exceptions
            in :class:`AllTasksFailedError` and raise. Default ``False`` —
            callers receive the exception list and decide.
        log_partial: When ``True`` and at least one task failed but not all,
            log a single WARNING summarising the failure count. Default
            ``True``.

    Returns:
        List of results in input order. Failed entries are the raised
        exception (not re-raised).

    Raises:
        TypeError: If any input is not awaitable.
        AllTasksFailedError: If ``raise_on_all_failed=True`` and every input
            failed.
    """
    if not coros:
        return []
    for i, coro in enumerate(coros):
        if not hasattr(coro, "__await__"):
            raise TypeError(f"argument {i} ({type(coro).__name__!r}) is not awaitable")

    results: list[T | BaseException] = await asyncio.gather(
        *coros, return_exceptions=True
    )
    failures = [r for r in results if isinstance(r, BaseException)]

    if failures and log_partial and len(failures) < len(results):
        logger.warning("safe_gather: %d/%d tasks failed", len(failures), len(results))

    if failures and len(failures) == len(results) and raise_on_all_failed:
        raise AllTasksFailedError(failures)

    return results


# ---------------------------------------------------------------------------
# Narrow exception handler
# ---------------------------------------------------------------------------


@contextmanager
def narrow_catch(
    *exception_types: type[BaseException],
    log_message: str,
) -> Iterator[None]:
    """Catch only the listed exception types; re-raise everything else.

    Replaces the ``try / except Exception: ...`` anti-pattern with an
    explicit list of what is actually expected and acceptable. Anything not
    in the list propagates.

    Args:
        *exception_types: One or more exception classes to catch. Must NOT
            include :class:`Exception` itself — that defeats the purpose
            and the call is rejected with :class:`ValueError`.
        log_message: Human-readable description of the operation being
            guarded, used as the log message.

    Raises:
        ValueError: If ``exception_types`` is empty or contains
            :class:`Exception`.
        TypeError: If any entry in ``exception_types`` is not an Exception
            subclass.

    Logs:
        ``logger.exception(log_message)`` on a matched exception so the
        full traceback is preserved.
    """
    if not exception_types:
        raise ValueError("narrow_catch requires at least one exception type")
    for exc_type in exception_types:
        if not (isinstance(exc_type, type) and issubclass(exc_type, BaseException)):
            raise TypeError(f"{exc_type!r} must be an Exception subclass")
        if exc_type is Exception:
            raise ValueError(
                "catching bare Exception defeats the purpose of narrow_catch; "
                "list the specific types you expect"
            )
    if not log_message:
        raise ValueError("log_message must be a non-empty string")

    try:
        yield
    except exception_types:
        logger.exception(log_message)
