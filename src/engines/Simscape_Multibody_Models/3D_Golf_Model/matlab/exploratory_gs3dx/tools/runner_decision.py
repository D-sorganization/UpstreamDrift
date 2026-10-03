"""GS3DX MATLAB batch-runner decision and exit contract.

Evaluates execution outcomes with Design-by-Contract (DbC) invariants, ensuring
truthful exit codes, distinguishing completed scripts from watchdog-killed shutdowns,
sanitizing completion markers, and producing structured execution receipts.
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

EXIT_CODE_SUCCESS: int = 0
EXIT_CODE_SCRIPT_FAILED: int = 1
EXIT_CODE_TIMEOUT: int = 124
EXIT_CODE_SHUTDOWN_UNVERIFIED: int = 125

TERMINATION_NATURAL: str = "natural_exit"
TERMINATION_HUNG_AFTER_DONE: str = "hung_after_done"
TERMINATION_TIMEOUT: str = "timeout"
TERMINATION_UNVERIFIED: str = "unverified_exit"

ALLOWED_TERMINATION_REASONS: frozenset[str] = frozenset(
    {
        TERMINATION_NATURAL,
        TERMINATION_HUNG_AFTER_DONE,
        TERMINATION_TIMEOUT,
        TERMINATION_UNVERIFIED,
    }
)

KNOWN_SUCCESS_STATUSES: frozenset[str] = frozenset({"success", "passed", "ok"})

_DONE_MARKER_REGEX = re.compile(r"^[ \t]*GS3DX_BATCH_DONE[ \t]*\r?$", re.MULTILINE)
_STATUS_REGEX = re.compile(
    r"^[ \t]*STATUS[ \t]+([A-Za-z0-9_-]+)[ \t]*\r?$", re.MULTILINE
)


class DecisionError(Exception):
    """Base class for runner decision errors."""


class PreconditionError(DecisionError):
    """Raised when DbC preconditions are violated."""


class PostconditionError(DecisionError):
    """Raised when DbC postconditions are violated."""


@dataclass(frozen=True)
class RunOutcome:
    """Immutable result of evaluating a runner execution outcome."""

    final_exit_code: int
    actual_process_exit_code: int | None
    termination_reason: str
    done_marker: bool
    script_status: str | None
    command_identity: str
    owned_pids: list[int]
    timestamp: str
    summary_message: str

    def to_receipt(self) -> dict[str, Any]:
        """Convert outcome into a structured receipt dictionary."""
        return {
            "actual_process_exit_code": self.actual_process_exit_code,
            "final_exit_code": self.final_exit_code,
            "termination_reason": self.termination_reason,
            "done_marker": self.done_marker,
            "script_status": self.script_status,
            "command_identity": self.command_identity,
            "owned_pids": list(self.owned_pids),
            "timestamp": self.timestamp,
        }

    def to_receipt_json(self) -> str:
        """Convert outcome receipt to compact JSON string."""
        return json.dumps(self.to_receipt(), separators=(",", ":"))


def has_exact_done_marker(log_content: str) -> bool:
    """Return True if GS3DX_BATCH_DONE appears as an exact complete line.

    Prevents comments (e.g. '% fprintf("GS3DX_BATCH_DONE")') or shell echoes
    from falsely indicating completion.
    """
    if not log_content:
        return False
    return bool(_DONE_MARKER_REGEX.search(log_content))


def parse_script_status(log_content: str) -> str | None:
    """Extract the last explicit STATUS marker (e.g. 'STATUS success' or 'STATUS failed')."""
    if not log_content:
        return None
    matches = _STATUS_REGEX.findall(log_content)
    if not matches:
        return None
    return str(matches[-1]).lower()


def _validate_preconditions(
    actual_process_exit_code: int | None,
    termination_reason: str,
    command_identity: str,
    owned_pids: Sequence[int],
) -> None:
    """Validate DbC preconditions for decide_run_outcome."""
    if isinstance(actual_process_exit_code, bool):
        raise PreconditionError(
            f"actual_process_exit_code cannot be bool, got {actual_process_exit_code!r}."
        )

    if actual_process_exit_code is not None and not isinstance(
        actual_process_exit_code, int
    ):
        raise PreconditionError(
            f"actual_process_exit_code must be int or None, got {actual_process_exit_code!r}."
        )

    if not isinstance(command_identity, str) or not command_identity.strip():
        raise PreconditionError("command_identity must be a non-empty string.")

    if not owned_pids:
        raise PreconditionError("owned_pids must not be empty.")

    for pid in owned_pids:
        if isinstance(pid, bool) or not isinstance(pid, int) or pid <= 0:
            raise PreconditionError(
                f"Each PID in owned_pids must be a positive integer (not bool), got {pid!r}."
            )

    if termination_reason not in ALLOWED_TERMINATION_REASONS:
        raise PreconditionError(
            f"termination_reason {termination_reason!r} not in allowed: {sorted(ALLOWED_TERMINATION_REASONS)}"
        )


def _map_nonzero_exit_code(actual_process_exit_code: int) -> int:
    """Map native exit code sanely to 1..255 for non-zero process exits."""
    if 0 <= actual_process_exit_code <= 255:
        return actual_process_exit_code
    mapped = actual_process_exit_code & 0xFF
    return mapped if mapped != 0 else 1


def _resolve_natural_exit(
    actual_process_exit_code: int | None,
    script_status: str | None,
) -> tuple[int, int | None, str]:
    """Resolve exit code, verified process code, and summary message for natural exits."""
    if actual_process_exit_code is None:
        return (
            EXIT_CODE_SHUTDOWN_UNVERIFIED,
            None,
            "RUNNER: natural exit missing integer exit code; fail closed with returncode 125",
        )

    if actual_process_exit_code != 0:
        final_exit_code = _map_nonzero_exit_code(actual_process_exit_code)
        return (
            final_exit_code,
            actual_process_exit_code,
            f"RUNNER: process exited naturally with error code {actual_process_exit_code}",
        )

    if script_status is not None:
        if script_status in KNOWN_SUCCESS_STATUSES:
            return (
                EXIT_CODE_SUCCESS,
                0,
                f"RUNNER: completed successfully with status '{script_status}' and returncode 0",
            )
        return (
            EXIT_CODE_SCRIPT_FAILED,
            0,
            f"RUNNER: process exited 0 but script recorded non-success STATUS '{script_status}'; "
            f"failing run with returncode {EXIT_CODE_SCRIPT_FAILED}",
        )

    return (
        EXIT_CODE_SUCCESS,
        0,
        "RUNNER: completed successfully with returncode 0",
    )


def _resolve_exit_decision(
    termination_reason: str,
    actual_process_exit_code: int | None,
    script_status: str | None,
) -> tuple[int, int | None, str]:
    """Resolve final exit code, verified process exit code, and summary message."""
    if termination_reason == TERMINATION_TIMEOUT:
        return (
            EXIT_CODE_TIMEOUT,
            None,
            "WATCHDOG: killed after timeout deadline (exit code unverified; returncode 124)",
        )

    if termination_reason == TERMINATION_HUNG_AFTER_DONE:
        return (
            EXIT_CODE_SHUTDOWN_UNVERIFIED,
            None,
            "RUNNER: script finished (GS3DX_BATCH_DONE); MATLAB hung at exit and was killed "
            "(exit code unverified; returncode 125)",
        )

    if termination_reason == TERMINATION_UNVERIFIED:
        return (
            EXIT_CODE_SHUTDOWN_UNVERIFIED,
            None,
            "RUNNER: process exit unverified or null; fail closed with returncode 125",
        )

    if termination_reason == TERMINATION_NATURAL:
        return _resolve_natural_exit(actual_process_exit_code, script_status)

    raise PreconditionError(f"Unhandled termination reason: {termination_reason}")


def _validate_postconditions(outcome: RunOutcome) -> None:
    """Validate DbC postconditions on RunOutcome."""
    if not (0 <= outcome.final_exit_code <= 255):
        raise PostconditionError(
            f"final_exit_code {outcome.final_exit_code} out of bounds [0, 255]."
        )

    if (
        outcome.termination_reason == TERMINATION_HUNG_AFTER_DONE
        and outcome.final_exit_code == 0
    ):
        raise PostconditionError(
            "Postcondition violated: hung_after_done must never yield final exit code 0."
        )

    if (
        outcome.termination_reason == TERMINATION_TIMEOUT
        and outcome.final_exit_code != EXIT_CODE_TIMEOUT
    ):
        raise PostconditionError(
            f"Postcondition violated: timeout must yield {EXIT_CODE_TIMEOUT}, got {outcome.final_exit_code}."
        )

    if (
        outcome.script_status is not None
        and outcome.script_status not in KNOWN_SUCCESS_STATUSES
        and outcome.final_exit_code == 0
    ):
        raise PostconditionError(
            f"Postcondition violated: non-success script_status '{outcome.script_status}' must not yield 0."
        )

    if outcome.actual_process_exit_code is None and outcome.final_exit_code == 0:
        raise PostconditionError(
            "Postcondition violated: missing actual process exit code must never yield exit code 0."
        )


def decide_run_outcome(
    actual_process_exit_code: int | None,
    termination_reason: str,
    log_content: str,
    stderr_content: str,
    command_identity: str,
    owned_pids: Sequence[int],
    timestamp: str | None = None,
) -> RunOutcome:
    """Evaluate run evidence and determine final truthful exit code and receipt.

    Adheres strictly to Design-by-Contract (DbC):
      Preconditions:
        - command_identity must be a non-empty string.
        - owned_pids must be a non-empty sequence of positive integers (booleans rejected).
        - termination_reason must be in ALLOWED_TERMINATION_REASONS.
        - actual_process_exit_code must be int or None (booleans rejected).
      Postconditions:
        - final_exit_code in 0..255.
        - forced watchdog kills never synthesize exit code 0.
        - hung_after_done yields 125.
        - timeout yields 124.
        - script_status not in KNOWN_SUCCESS_STATUSES fails closed (yields 1).
    """
    _validate_preconditions(
        actual_process_exit_code=actual_process_exit_code,
        termination_reason=termination_reason,
        command_identity=command_identity,
        owned_pids=owned_pids,
    )

    ts = timestamp or datetime.now(UTC).isoformat()
    pids_list = [int(p) for p in owned_pids]
    done_marker = has_exact_done_marker(log_content)
    script_status = parse_script_status(log_content)

    final_exit_code, verified_actual_code, summary_message = _resolve_exit_decision(
        termination_reason=termination_reason,
        actual_process_exit_code=actual_process_exit_code,
        script_status=script_status,
    )

    outcome = RunOutcome(
        final_exit_code=final_exit_code,
        actual_process_exit_code=verified_actual_code,
        termination_reason=termination_reason,
        done_marker=done_marker,
        script_status=script_status,
        command_identity=command_identity,
        owned_pids=pids_list,
        timestamp=ts,
        summary_message=summary_message,
    )

    _validate_postconditions(outcome)
    return outcome


def main() -> int:
    """CLI entry point for evaluate_run_outcome."""
    parser = argparse.ArgumentParser(
        description="Evaluate GS3DX runner execution outcome."
    )
    parser.add_argument("--log", required=True, help="Path to primary stdout log file.")
    parser.add_argument("--stderr", default="", help="Path to stderr file (optional).")
    parser.add_argument(
        "--actual-exit", default="null", help="Raw process exit code or 'null'."
    )
    parser.add_argument(
        "--reason",
        required=True,
        choices=sorted(ALLOWED_TERMINATION_REASONS),
        help="Termination reason.",
    )
    parser.add_argument("--command", required=True, help="Command identity.")
    parser.add_argument(
        "--pid", type=int, required=True, action="append", help="Owned PID(s)."
    )
    parser.add_argument(
        "--append-to-log",
        action="store_true",
        help="Append messages and receipt to log.",
    )

    args = parser.parse_args()

    log_path = Path(args.log)
    log_content = (
        log_path.read_text(encoding="utf-8", errors="replace")
        if log_path.is_file()
        else ""
    )

    stderr_path = Path(args.stderr) if args.stderr else None
    stderr_content = (
        stderr_path.read_text(encoding="utf-8", errors="replace")
        if (stderr_path and stderr_path.is_file())
        else ""
    )

    actual_code: int | None
    if args.actual_exit.lower() in ("null", "none", ""):
        actual_code = None
    else:
        try:
            actual_code = int(args.actual_exit)
        except ValueError:
            actual_code = None

    outcome = decide_run_outcome(
        actual_process_exit_code=actual_code,
        termination_reason=args.reason,
        log_content=log_content,
        stderr_content=stderr_content,
        command_identity=args.command,
        owned_pids=args.pid,
    )

    receipt_json = outcome.to_receipt_json()

    # Write receipt json to sidecar
    receipt_file = log_path.parent / f"{log_path.name}.receipt.json"
    try:
        receipt_file.write_text(receipt_json, encoding="utf-8")
    except OSError as exc:
        logger.warning("Could not write receipt file %s: %s", receipt_file, exc)

    if args.append_to_log:
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                if stderr_content.strip():
                    f.write("\n--- STDERR ---\n")
                    f.write(stderr_content.strip())
                    f.write("\n")
                f.write(f"{outcome.summary_message}\n")
                f.write(f"RECEIPT: {receipt_json}\n")
                f.write(f"EXIT {outcome.final_exit_code}\n")
        except OSError as exc:
            logger.error("Could not append outcome to log file %s: %s", log_path, exc)

    sys.stdout.write(receipt_json + "\n")
    return outcome.final_exit_code


if __name__ == "__main__":
    sys.exit(main())
