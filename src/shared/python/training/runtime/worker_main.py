"""Subprocess worker entry point for :class:`SubprocessDriver`.

Run as ``python -m training.runtime.worker_main``. The worker reads a
single ``{"command": "run", "config": <dict>}`` from stdin, imports the
``entry_point`` referenced by the config (``"module.path:callable"``),
and invokes it with a stdout-publishing :class:`ProgressSink` and a
:class:`ThreadingCancelToken` flipped by a stdin-reader thread that
watches for ``{"command": "cancel"}``.

Exactly one ``{"event": "result", "result": <RunResult dict>}`` line is
emitted before the worker exits — even on failure — so the parent
:class:`SubprocessDriver` never has to special-case worker crashes
arriving as silent stream closures.

Worker stdout IS the wire protocol — the no-``print()``-in-``src/``
rule (CLAUDE.md) does not apply here. We use ``sys.stdout.write`` plus
explicit ``flush()`` to keep framing unambiguous.
"""

from __future__ import annotations

import contextlib
import importlib
import sys
import threading
import time
import traceback
from typing import Any

from ..contracts import ProgressSink, ThreadingCancelToken
from ..identifiers import RunId, new_run_id
from ..job import RunResult
from ..metrics import TrainingMetric
from ..persistence import (
    run_result_to_dict,
    training_config_from_dict,
    training_metric_to_dict,
)
from ..status import TERMINAL_STATUSES, TrainingStatus
from .wire_protocol import (
    COMMAND_CANCEL,
    COMMAND_RUN,
    EVENT_METRIC,
    EVENT_RESULT,
    EVENT_STATUS,
    WireProtocolError,
    decode_command,
    encode_event,
)

__all__ = ["main", "run_worker"]


# Mirrors `in_process_driver._RUNNER_FAILURE_TYPES`. The worker is the
# trust boundary that turns a buggy entry-point into a FAILED RunResult
# the parent can surface; without this catch a runner exception would
# kill the worker with no result event and the parent would only see
# stream closure. `KeyboardInterrupt`, `SystemExit`, and the rest of
# `BaseException` are intentionally not caught.
_RUNNER_FAILURE_TYPES: tuple[type[BaseException], ...] = (
    ArithmeticError,
    AssertionError,
    AttributeError,
    ImportError,
    LookupError,
    MemoryError,
    NotImplementedError,
    OSError,
    RecursionError,
    RuntimeError,
    TypeError,
    UnicodeError,
    ValueError,
)


class _StdoutProgressSink:
    """:class:`ProgressSink` that JSON-encodes every emission to stdout.

    Conforms structurally to the :class:`ProgressSink` Protocol; no
    inheritance so the Protocol stays narrow.
    """

    __slots__ = ("_lock", "_stream")

    def __init__(self, stream: Any | None = None) -> None:
        # Stream defaults to live sys.stdout but tests can inject a buffer.
        self._stream = stream if stream is not None else sys.stdout
        self._lock = threading.Lock()

    def emit_metric(self, metric: TrainingMetric) -> None:
        if not isinstance(metric, TrainingMetric):
            raise TypeError("emit_metric expects a TrainingMetric")
        line = encode_event(EVENT_METRIC, {"metric": training_metric_to_dict(metric)})
        self._write_line(line)

    def emit_status(
        self, status: TrainingStatus, *, message: str | None = None
    ) -> None:
        if not isinstance(status, TrainingStatus):
            raise TypeError("emit_status expects a TrainingStatus")
        line = encode_event(
            EVENT_STATUS,
            {"status": status.value, "message": message},
        )
        self._write_line(line)

    def emit_result(self, result_dict: dict[str, Any]) -> None:
        """Worker-only helper — emits the final result event."""

        line = encode_event(EVENT_RESULT, {"result": result_dict})
        self._write_line(line)

    def _write_line(self, line: str) -> None:
        # Worker stdout IS the wire protocol. No `print()` here — write
        # the framed JSON directly and flush so the parent observes the
        # line immediately.
        with self._lock:
            self._stream.write(line)  # noqa: T201 - wire-protocol stdout, not user print
            self._stream.flush()


def _resolve_entry_point(entry_point: str) -> Any:
    """Resolve ``"module.path:callable"`` to a callable.

    Raises :class:`ValueError` for malformed strings,
    :class:`ImportError` when the module cannot be loaded, and
    :class:`AttributeError` when the attribute is missing.
    """

    if not isinstance(entry_point, str) or ":" not in entry_point:
        raise ValueError(
            f"entry_point must be 'module.path:callable' (got {entry_point!r})"
        )
    module_path, _, attr = entry_point.partition(":")
    if not module_path or not attr:
        raise ValueError(
            f"entry_point must be 'module.path:callable' (got {entry_point!r})"
        )
    module = importlib.import_module(module_path)
    return getattr(module, attr)


def _spawn_cancel_watcher(
    stdin: Any,
    cancel_token: ThreadingCancelToken,
) -> threading.Thread:
    """Start a daemon thread that flips ``cancel_token`` on cancel commands.

    The thread exits silently when stdin closes or any line fails to
    parse — the parent's job is to send well-formed commands, and any
    malformed input means the IPC channel is already broken.
    """

    def _watch() -> None:
        try:
            for line in stdin:
                try:
                    command, _ = decode_command(line)
                except WireProtocolError:
                    # Malformed input → IPC is broken; nothing useful to do.
                    return
                if command == COMMAND_CANCEL:
                    cancel_token.request_cancel()
        except (OSError, ValueError):
            # Stream closed or unreadable — runner will observe cancel
            # via timeout or simply finish; nothing to recover here.
            return

    thread = threading.Thread(
        target=_watch,
        name="worker-cancel-watcher",
        daemon=True,
    )
    thread.start()
    return thread


def _failure_result(
    run_id: RunId,
    duration_s: float,
    message: str,
) -> dict[str, Any]:
    return run_result_to_dict(
        RunResult(
            run_id=run_id,
            status=TrainingStatus.FAILED,
            duration_s=max(duration_s, 0.0),
            error=message,
        )
    )


def run_worker(
    stdin: Any,
    stdout: Any,
    *,
    runtime_run_id: RunId | None = None,
) -> int:
    """Programmatic entry point used by tests and :func:`main`.

    Args:
        stdin: Line-iterable read source (typically ``sys.stdin``).
        stdout: Write-capable stream (typically ``sys.stdout``).
        runtime_run_id: If supplied, used as the result's ``run_id``;
            otherwise a fresh :class:`RunId` is minted. Tests inject a
            deterministic value.

    Returns:
        Process exit code: ``0`` on a clean COMPLETED / CANCELLED /
        FAILED RunResult delivery, ``1`` on a protocol-level error
        before a result could be emitted.
    """

    sink = _StdoutProgressSink(stream=stdout)
    run_id = runtime_run_id if runtime_run_id is not None else new_run_id()
    start = time.monotonic()

    first_line = stdin.readline()
    if not first_line:
        sink.emit_result(
            _failure_result(
                run_id,
                time.monotonic() - start,
                "worker: stdin closed before run command arrived",
            )
        )
        return 1
    try:
        command, payload = decode_command(first_line)
    except WireProtocolError as exc:
        sink.emit_result(
            _failure_result(
                run_id,
                time.monotonic() - start,
                f"worker: malformed first command: {exc}",
            )
        )
        return 1
    if command != COMMAND_RUN:
        sink.emit_result(
            _failure_result(
                run_id,
                time.monotonic() - start,
                f"worker: first command must be {COMMAND_RUN!r} (got {command!r})",
            )
        )
        return 1

    config_dict = payload.get("config")
    if not isinstance(config_dict, dict):
        sink.emit_result(
            _failure_result(
                run_id,
                time.monotonic() - start,
                "worker: run command missing 'config' dict",
            )
        )
        return 1

    try:
        config = training_config_from_dict(config_dict)
    except _RUNNER_FAILURE_TYPES as exc:
        sink.emit_result(
            _failure_result(
                run_id,
                time.monotonic() - start,
                f"worker: invalid TrainingConfig: {exc}",
            )
        )
        return 1

    cancel_token = ThreadingCancelToken()
    _spawn_cancel_watcher(stdin, cancel_token)

    try:
        entry = _resolve_entry_point(config.entry_point)
    except _RUNNER_FAILURE_TYPES as exc:
        sink.emit_result(
            _failure_result(
                run_id,
                time.monotonic() - start,
                f"worker: cannot resolve entry_point {config.entry_point!r}: {exc}",
            )
        )
        return 0

    result_dict = _invoke_entry_point(entry, config, sink, cancel_token, run_id, start)
    sink.emit_result(result_dict)
    return 0


def _invoke_entry_point(
    entry: Any,
    config: Any,
    sink: ProgressSink,
    cancel_token: ThreadingCancelToken,
    run_id: RunId,
    start: float,
) -> dict[str, Any]:
    """Call the user entry point and normalise its return / exceptions."""

    try:
        result = entry(config, progress=sink, cancel=cancel_token)
    except _RUNNER_FAILURE_TYPES as exc:
        trace = traceback.format_exc()
        return _failure_result(
            run_id,
            time.monotonic() - start,
            f"entry_point raised: {exc}\n{trace}",
        )
    if not isinstance(result, RunResult):
        return _failure_result(
            run_id,
            time.monotonic() - start,
            f"entry_point returned {type(result).__name__}, expected RunResult",
        )
    if result.status not in TERMINAL_STATUSES:
        return _failure_result(
            run_id,
            time.monotonic() - start,
            f"entry_point returned non-terminal status {result.status.value!r}",
        )
    return run_result_to_dict(result)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point — ``python -m training.runtime.worker_main``."""

    del argv  # currently no CLI options; reserved for future use.
    # Force line-buffered stdout when possible so the parent gets each
    # framed JSON line as soon as the worker writes it. Already
    # configured or a stream type without reconfigure → no-op.
    with contextlib.suppress(AttributeError, OSError):
        sys.stdout.reconfigure(line_buffering=True)  # type: ignore[attr-defined]
    return run_worker(sys.stdin, sys.stdout)


if __name__ == "__main__":  # pragma: no cover - exercised via subprocess tests
    raise SystemExit(main(sys.argv[1:]))
