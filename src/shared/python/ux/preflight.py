"""Pre-flight checklists and confirmations (epic #5968, Phase 4.1).

A :class:`PreflightCheck` is one item evaluated before a high-risk
action (run simulation, overwrite pose, switch engine, delete
results, launch batch).  :func:`run_preflight` aggregates a list of
checks into a :class:`PreflightResult` whose ``can_proceed()``
predicate gates the action.

The model is pure data so it can be reused unchanged across:

* PyQt6 ``PreflightDialog`` (Phase 4.2)
* React ``<PreflightDialog>`` (Phase 4.2)
* The API server's batch-launch endpoint (Phase 4.2 server side)
* CLI tools that want the same gating logic

Severity semantics
------------------
* :attr:`Severity.INFO` — purely informational; a failure never
  blocks proceeding.
* :attr:`Severity.WARN` — non-blocking but surfaced prominently; the
  action proceeds, the user is told.
* :attr:`Severity.BLOCK` — blocking; the action is disabled until the
  check passes or an explicit typed ``override_reason`` is supplied.
"""

from __future__ import annotations

import enum
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from src.shared.python.contracts import require

_ID_PATTERN: re.Pattern[str] = re.compile(r"^[a-z][a-z0-9_]*$")
_MIN_OVERRIDE_REASON_CHARS: int = 8


class PreflightError(ValueError):
    """Raised for any invalid preflight construction or invocation."""


class Severity(enum.Enum):
    """Severity of a single preflight failure."""

    INFO = "info"
    WARN = "warn"
    BLOCK = "block"


@dataclass(frozen=True, slots=True)
class PreflightCheck:
    """One item in a preflight checklist.

    Attributes
    ----------
    id
        Stable snake_case identifier (``r"^[a-z][a-z0-9_]*$"``).
    label
        Human-readable headline ("Engine is ready").
    severity
        :class:`Severity` — controls whether a failure blocks.
    why
        Plain-language explanation of what this check protects
        against.  Always shown to the user, whether the check passes
        or fails (passing checks reassure; failing checks teach).
    fix_action
        Optional one-line action the user can take to make a failing
        check pass.  Surfaced as a button in dialogs.
    passed
        Whether this check passes.
    """

    id: str
    label: str
    severity: Severity
    why: str
    fix_action: str | None
    passed: bool

    def __post_init__(self) -> None:
        if not isinstance(self.id, str) or not _ID_PATTERN.match(self.id):
            raise PreflightError(
                f"PreflightCheck.id must match {_ID_PATTERN.pattern!r}, got {self.id!r}"
            )
        if not self.label:
            raise PreflightError(f"{self.id}: label must be non-empty")
        if not self.why:
            raise PreflightError(f"{self.id}: why must be non-empty")
        if not isinstance(self.severity, Severity):
            raise PreflightError(
                f"{self.id}: severity must be a Severity enum, got "
                f"{type(self.severity).__name__}"
            )
        if self.fix_action is not None and not self.fix_action:
            raise PreflightError(f"{self.id}: fix_action must be None or non-empty")


@dataclass(frozen=True, slots=True)
class PreflightResult:
    """Aggregated outcome of running a list of :class:`PreflightCheck`.

    LOD: callers ask the result yes/no questions rather than reaching
    into ``result.checks[i].severity``.
    """

    checks: tuple[PreflightCheck, ...]
    override_reason: str | None

    @property
    def was_overridden(self) -> bool:
        return self.override_reason is not None

    def blocking_failures(self) -> tuple[PreflightCheck, ...]:
        return tuple(
            c for c in self.checks if not c.passed and c.severity is Severity.BLOCK
        )

    def warning_failures(self) -> tuple[PreflightCheck, ...]:
        return tuple(
            c for c in self.checks if not c.passed and c.severity is Severity.WARN
        )

    def can_proceed(self) -> bool:
        if self.was_overridden:
            return True
        return not self.blocking_failures()

    def summary(self) -> str:
        """Multi-line human-readable summary of failures.

        Returns ``"All preflight checks passed."`` when nothing failed.
        """
        failed = [c for c in self.checks if not c.passed]
        if not failed:
            return "All preflight checks passed."
        lines = []
        for c in failed:
            line = f"[{c.severity.name}] {c.id}: {c.label} — {c.why}"
            if c.fix_action:
                line += f" Fix: {c.fix_action}"
            lines.append(line)
        if self.was_overridden:
            lines.append(f"Overridden: {self.override_reason}")
        return "\n".join(lines)


def run_preflight(
    checks: Iterable[PreflightCheck],
    *,
    override_reason: str | None = None,
) -> PreflightResult:
    """Aggregate ``checks`` and return a :class:`PreflightResult`.

    DbC preconditions
    -----------------
    * Every element must be a :class:`PreflightCheck`.
    * No duplicate ids — duplicates almost always mean the caller
      computed the same check twice and would mis-summarize.
    * If ``override_reason`` is provided, it must be at least
      :data:`_MIN_OVERRIDE_REASON_CHARS` characters of substantive
      explanation (single-character "x" overrides defeat the purpose
      of typed acknowledgement).
    """
    if not isinstance(checks, Sequence):
        checks = tuple(checks)
    require(
        all(isinstance(c, PreflightCheck) for c in checks),
        "run_preflight expects PreflightCheck items",
        checks,
    )
    _ensure_unique_ids(checks)
    cleaned_override = _validate_override(override_reason)
    return PreflightResult(checks=tuple(checks), override_reason=cleaned_override)


def _ensure_unique_ids(checks: Sequence[PreflightCheck]) -> None:
    seen: set[str] = set()
    for c in checks:
        if c.id in seen:
            raise PreflightError(f"duplicate preflight check id: {c.id!r}")
        seen.add(c.id)


def _validate_override(reason: str | None) -> str | None:
    if reason is None:
        return None
    if not isinstance(reason, str):
        raise PreflightError(
            f"override_reason must be str or None, got {type(reason).__name__}"
        )
    stripped = reason.strip()
    if not stripped:
        raise PreflightError("override_reason must be non-empty if provided")
    if len(stripped) < _MIN_OVERRIDE_REASON_CHARS:
        raise PreflightError(
            f"override_reason must be at least {_MIN_OVERRIDE_REASON_CHARS} "
            f"characters of substantive explanation"
        )
    return stripped


__all__ = [
    "PreflightCheck",
    "PreflightError",
    "PreflightResult",
    "Severity",
    "run_preflight",
]
