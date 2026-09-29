"""Wire-format constants and codec helpers for the subprocess driver.

The :class:`SubprocessDriver` and its worker entry point exchange
newline-delimited JSON over the worker's stdin / stdout. Keeping the
constants and the tiny encode/decode helpers in a dedicated module
makes the protocol a single source of truth — tests round-trip every
event type here so a future field rename is impossible to miss.

Wire format (one JSON object per line, ``"\\n"`` terminator):

* **Parent to child (stdin):**
  - ``{"command": "run", "config": <TrainingConfig dict>}`` — exactly
    once, as the first line.
  - ``{"command": "cancel"}`` — zero or more times, on demand.
* **Child to parent (stdout):**
  - ``{"event": "status", "status": <str>, "message": <str|null>}``
  - ``{"event": "metric", "metric": <TrainingMetric dict>}``
  - ``{"event": "result", "result": <RunResult dict>}`` — exactly once,
    as the final line before the worker exits.

The dict payloads are the same ones produced by
:mod:`training.persistence`, so the wire schema and the persistence
schema move together.
"""

from __future__ import annotations

import json
from typing import Any

__all__ = [
    "COMMAND_CANCEL",
    "COMMAND_RUN",
    "EVENT_METRIC",
    "EVENT_RESULT",
    "EVENT_STATUS",
    "WireProtocolError",
    "decode_command",
    "decode_event",
    "encode_command",
    "encode_event",
]


COMMAND_RUN = "run"
"""Parent-to-child command kicking off execution. Carries ``config``."""

COMMAND_CANCEL = "cancel"
"""Parent-to-child command signalling cooperative cancellation."""

EVENT_STATUS = "status"
"""Child-to-parent event announcing a :class:`TrainingStatus` change."""

EVENT_METRIC = "metric"
"""Child-to-parent event carrying a :class:`TrainingMetric` dict."""

EVENT_RESULT = "result"
"""Child-to-parent event carrying the final :class:`RunResult` dict."""


_VALID_COMMANDS: frozenset[str] = frozenset({COMMAND_RUN, COMMAND_CANCEL})
_VALID_EVENTS: frozenset[str] = frozenset({EVENT_STATUS, EVENT_METRIC, EVENT_RESULT})


class WireProtocolError(ValueError):
    """Raised when a wire-format payload fails structural validation."""


def encode_event(event: str, payload: dict[str, Any]) -> str:
    """Serialise a child-to-parent event as a newline-terminated JSON line.

    Args:
        event: One of :data:`EVENT_STATUS`, :data:`EVENT_METRIC`,
            :data:`EVENT_RESULT`.
        payload: Event-specific fields. ``"event"`` is added by this
            helper; it must not appear in ``payload``.

    Returns:
        A JSON object terminated by ``"\\n"``. Suitable to write directly
        to a stream the parent reads line-by-line.

    Raises:
        WireProtocolError: When ``event`` is not a known event name, when
            ``payload`` already contains an ``"event"`` key, or when
            ``payload`` is not a dict.
    """

    if event not in _VALID_EVENTS:
        raise WireProtocolError(
            f"unknown event {event!r}; expected one of {sorted(_VALID_EVENTS)!r}"
        )
    if not isinstance(payload, dict):
        raise WireProtocolError(
            f"payload must be a dict (got {type(payload).__name__})"
        )
    if "event" in payload:
        raise WireProtocolError(
            "payload must not contain an 'event' key; it is added by encode_event"
        )
    record: dict[str, Any] = {"event": event, **payload}
    return json.dumps(record, separators=(",", ":")) + "\n"


def decode_event(line: str) -> tuple[str, dict[str, Any]]:
    """Inverse of :func:`encode_event`.

    Args:
        line: A single line read from the worker's stdout. The trailing
            newline (if any) is stripped before parsing.

    Returns:
        ``(event_name, payload_dict)`` where ``payload_dict`` excludes the
        ``"event"`` key.

    Raises:
        WireProtocolError: When the line is not valid JSON, is not a
            JSON object, lacks an ``"event"`` field, or carries an
            unknown event name.
    """

    if not isinstance(line, str):
        raise WireProtocolError(f"line must be str (got {type(line).__name__})")
    stripped = line.strip()
    if not stripped:
        raise WireProtocolError("line must be a non-empty JSON object")
    try:
        record = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise WireProtocolError(f"line is not valid JSON: {exc.msg}") from exc
    if not isinstance(record, dict):
        raise WireProtocolError(
            f"line must encode a JSON object (got {type(record).__name__})"
        )
    event = record.get("event")
    if event not in _VALID_EVENTS:
        raise WireProtocolError(
            f"unknown or missing event {event!r}; expected one of "
            f"{sorted(_VALID_EVENTS)!r}"
        )
    payload = {k: v for k, v in record.items() if k != "event"}
    return event, payload


def encode_command(command: str, payload: dict[str, Any] | None = None) -> str:
    """Serialise a parent-to-child command as a newline-terminated JSON line.

    Args:
        command: One of :data:`COMMAND_RUN`, :data:`COMMAND_CANCEL`.
        payload: Optional command-specific fields. For ``COMMAND_RUN`` this
            should include ``"config"``; for ``COMMAND_CANCEL`` it is
            typically omitted.

    Returns:
        A JSON object terminated by ``"\\n"``. Suitable to write directly
        to the child's stdin.

    Raises:
        WireProtocolError: When ``command`` is unknown, ``payload`` is not
            a dict, or ``payload`` contains a ``"command"`` key.
    """

    if command not in _VALID_COMMANDS:
        raise WireProtocolError(
            f"unknown command {command!r}; expected one of {sorted(_VALID_COMMANDS)!r}"
        )
    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise WireProtocolError(
            f"payload must be a dict or None (got {type(payload).__name__})"
        )
    if "command" in payload:
        raise WireProtocolError(
            "payload must not contain a 'command' key; it is added by encode_command"
        )
    record: dict[str, Any] = {"command": command, **payload}
    return json.dumps(record, separators=(",", ":")) + "\n"


def decode_command(line: str) -> tuple[str, dict[str, Any]]:
    """Inverse of :func:`encode_command`.

    Args:
        line: A single line read from the worker's stdin.

    Returns:
        ``(command_name, payload_dict)`` where ``payload_dict`` excludes
        the ``"command"`` key.

    Raises:
        WireProtocolError: When the line is not valid JSON, is not a JSON
            object, lacks a ``"command"`` field, or carries an unknown
            command name.
    """

    if not isinstance(line, str):
        raise WireProtocolError(f"line must be str (got {type(line).__name__})")
    stripped = line.strip()
    if not stripped:
        raise WireProtocolError("line must be a non-empty JSON object")
    try:
        record = json.loads(stripped)
    except json.JSONDecodeError as exc:
        raise WireProtocolError(f"line is not valid JSON: {exc.msg}") from exc
    if not isinstance(record, dict):
        raise WireProtocolError(
            f"line must encode a JSON object (got {type(record).__name__})"
        )
    command = record.get("command")
    if command not in _VALID_COMMANDS:
        raise WireProtocolError(
            f"unknown or missing command {command!r}; expected one of "
            f"{sorted(_VALID_COMMANDS)!r}"
        )
    payload = {k: v for k, v in record.items() if k != "command"}
    return command, payload
