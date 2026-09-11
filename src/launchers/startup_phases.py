"""Bounded, timestamped startup phases for the launcher splash (issue #8360).

The launcher splash used to wait on an opaque worker: when an optional
provider (the vendored Tools checkout backing the Rate of Closure tile, a
slow engine import, a hung Docker probe) stalled, nothing recorded *which*
phase was responsible and nothing bounded how long the splash stayed up.

This module is pure Python (no Qt) so the state machinery is deterministic
under test:

* :class:`StartupTimeline` runs named phases under an explicit timeout and
  records a :class:`StartupPhaseRecord` (wall-clock start, duration,
  outcome, failure category, detail) for each one.
* :func:`probe_tools_provider` checks the optional Tools/Rate provider
  without importing it, classifying failures as ``missing_checkout``,
  ``missing_dependency`` or ``import_failure``.
* :func:`classify_exception` maps arbitrary provider exceptions onto the
  same closed category set so diagnostics stay comparable.

Design by Contract: public functions validate their arguments and raise
``ValueError``/``TypeError``; phase failures are never suppressed silently
-- every non-``ok`` outcome is recorded with a category and detail that the
splash, the failure dialog and the log all render from one source.
"""

from __future__ import annotations

from importlib.machinery import PathFinder
import logging
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from src.launchers.tools_repo_path import resolve_tools_repo

logger = logging.getLogger(__name__)

__all__ = [
    "CATEGORY_IMPORT_FAILURE",
    "CATEGORY_MISSING_CHECKOUT",
    "CATEGORY_MISSING_DEPENDENCY",
    "CATEGORY_SUBPROCESS_FAILURE",
    "CATEGORY_TIMEOUT",
    "OUTCOME_DEGRADED",
    "OUTCOME_FAILED",
    "OUTCOME_OK",
    "OUTCOME_TIMEOUT",
    "RATE_STANDALONE_ENTRY_POINT",
    "ProviderProbeResult",
    "StartupPhaseError",
    "StartupPhaseRecord",
    "StartupPhaseTimeout",
    "StartupTimeline",
    "classify_exception",
    "format_phase_diagnostics",
    "probe_tools_provider",
    "run_bounded",
]

# Closed outcome set: every phase ends in exactly one of these.
OUTCOME_OK = "ok"
OUTCOME_DEGRADED = "degraded"
OUTCOME_FAILED = "failed"
OUTCOME_TIMEOUT = "timeout"

# Closed failure-category set the issue asks diagnostics to distinguish.
CATEGORY_MISSING_CHECKOUT = "missing_checkout"
CATEGORY_MISSING_DEPENDENCY = "missing_dependency"
CATEGORY_IMPORT_FAILURE = "import_failure"
CATEGORY_SUBPROCESS_FAILURE = "subprocess_failure"
CATEGORY_TIMEOUT = "timeout"

# Direct standalone entry point of the Rate of Closure PyQt6 tool inside a
# Tools checkout. The launcher manifest launches the tile through exactly
# this script (``tools://`` + this path) so Rate never initializes the full
# suite; ``tests/unit/launchers/test_startup_phases.py`` pins the two
# together.
RATE_STANDALONE_ENTRY_POINT = Path("src/rate_of_closure/launch_pyqt6.py")
_RATE_PACKAGE = "rate_of_closure"
_TOOLS_PROVIDER = "tools"


class StartupPhaseTimeout(TimeoutError):
    """A bounded startup phase did not complete within its timeout."""

    def __init__(self, phase: str, timeout_s: float) -> None:
        super().__init__(f"{phase} did not complete within {timeout_s:g}s")
        self.phase = phase
        self.timeout_s = timeout_s


@dataclass(frozen=True)
class StartupPhaseRecord:
    """One structured, timestamped startup phase result.

    Attributes:
        name: Human-readable phase name (also shown on the splash).
        started_at: ISO-8601 local wall-clock start with milliseconds.
        duration_ms: Elapsed milliseconds until the phase settled.
        outcome: One of the ``OUTCOME_*`` constants.
        category: One of the ``CATEGORY_*`` constants, or ``""`` when the
            outcome carries no failure category.
        detail: Free-text detail (exception text, resolved path, reason).
    """

    name: str
    started_at: str
    duration_ms: int
    outcome: str
    category: str = ""
    detail: str = ""

    def format_line(self) -> str:
        """Render the record as one diagnostics line."""
        line = (
            f"{self.started_at} {self.outcome:<8} {self.name} ({self.duration_ms} ms)"
        )
        if self.category:
            line += f" [{self.category}]"
        if self.detail:
            line += f" {self.detail}"
        return line


class StartupPhaseError(RuntimeError):
    """A *required* startup phase failed or timed out.

    Carries the structured record so callers can render the exact phase
    and category instead of a bare message.
    """

    def __init__(self, record: StartupPhaseRecord) -> None:
        if not isinstance(record, StartupPhaseRecord):
            raise TypeError("record must be a StartupPhaseRecord")
        summary = f"{record.name} {record.outcome}"
        if record.category:
            summary += f" [{record.category}]"
        if record.detail:
            summary += f": {record.detail}"
        super().__init__(summary)
        self.record = record


def classify_exception(exc: BaseException) -> str:
    """Map an exception raised by provider/startup code onto a category.

    Returns ``""`` for exceptions that fit none of the closed categories;
    the exception type still travels in the record detail.
    """
    if not isinstance(exc, BaseException):
        raise TypeError("exc must be an exception instance")
    if isinstance(exc, StartupPhaseTimeout):
        return CATEGORY_TIMEOUT
    if isinstance(exc, ModuleNotFoundError):
        return CATEGORY_MISSING_DEPENDENCY
    if isinstance(exc, ImportError | SyntaxError):
        return CATEGORY_IMPORT_FAILURE
    if (
        isinstance(exc, subprocess.SubprocessError)
        or "Subprocess" in type(exc).__name__
    ):
        return CATEGORY_SUBPROCESS_FAILURE
    return ""


def run_bounded(fn: Callable[[], Any], timeout_s: float, *, name: str) -> Any:
    """Run ``fn`` on a daemon thread and return its result within ``timeout_s``.

    Preconditions:
        ``fn`` is callable, ``timeout_s`` is positive, ``name`` is non-empty.

    Postconditions:
        Returns ``fn()``'s value, re-raises ``fn()``'s exception, or raises
        :class:`StartupPhaseTimeout`. A timed-out ``fn`` keeps running on its
        daemon thread but can no longer block the caller.
    """
    if not callable(fn):
        raise TypeError("fn must be callable")
    if timeout_s <= 0:
        raise ValueError("timeout_s must be positive")
    if not name:
        raise ValueError("name must be non-empty")

    outcome: dict[str, Any] = {}

    def _target() -> None:
        try:
            outcome["value"] = fn()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the caller thread
            outcome["error"] = exc

    thread = threading.Thread(target=_target, name=f"startup-phase:{name}", daemon=True)
    thread.start()
    thread.join(timeout_s)
    if thread.is_alive():
        raise StartupPhaseTimeout(name, timeout_s)
    if "error" in outcome:
        raise outcome["error"]
    return outcome.get("value")


def _timestamp(moment: datetime) -> str:
    return moment.isoformat(sep="T", timespec="milliseconds")


class StartupTimeline:
    """Thread-safe recorder of bounded startup phases.

    Records are appended from the worker thread and snapshotted from the
    GUI thread, so every accessor returns an immutable copy.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._records: list[StartupPhaseRecord] = []
        self._origin = time.perf_counter()
        self._current: tuple[str, float] | None = None

    @property
    def records(self) -> tuple[StartupPhaseRecord, ...]:
        """Snapshot of every settled phase in execution order."""
        with self._lock:
            return tuple(self._records)

    @property
    def current_phase(self) -> str | None:
        """Name of the phase currently running, if any."""
        with self._lock:
            return self._current[0] if self._current else None

    def elapsed_ms(self) -> int:
        """Milliseconds since the timeline was created."""
        return int((time.perf_counter() - self._origin) * 1000)

    def blocking_phase(self) -> StartupPhaseRecord | None:
        """First settled phase that did not end ``ok``."""
        return next((r for r in self.records if r.outcome != OUTCOME_OK), None)

    def record(
        self,
        name: str,
        outcome: str,
        *,
        category: str = "",
        detail: str = "",
        started_at: datetime | None = None,
        duration_ms: int = 0,
    ) -> StartupPhaseRecord:
        """Append an explicit record (used for phases not run via ``run_phase``)."""
        if not name:
            raise ValueError("name must be non-empty")
        record = StartupPhaseRecord(
            name=name,
            started_at=_timestamp(started_at or datetime.now()),
            duration_ms=max(0, int(duration_ms)),
            outcome=outcome,
            category=category,
            detail=detail,
        )
        with self._lock:
            self._records.append(record)
            self._current = None
        return record

    def run_phase(
        self,
        name: str,
        fn: Callable[[], Any],
        *,
        timeout_s: float,
        required: bool = False,
        outcome_of: Callable[[Any], tuple[str, str, str]] | None = None,
    ) -> Any:
        """Run one bounded phase and record how it settled.

        Args:
            name: Phase name shown on the splash and in diagnostics.
            fn: Zero-argument callable doing the phase's work.
            timeout_s: Hard bound; longer runs settle as ``timeout``.
            required: When True, a failed/timed-out phase raises
                :class:`StartupPhaseError`; otherwise the phase settles as
                degraded and ``None`` is returned so startup continues.
            outcome_of: Optional mapper from the phase value to
                ``(outcome, category, detail)`` for phases whose *value*
                encodes degradation (for example a provider probe).

        Returns:
            The phase value, or ``None`` when an optional phase failed.
        """
        if not name:
            raise ValueError("name must be non-empty")
        started_wall = datetime.now()
        started = time.perf_counter()
        with self._lock:
            self._current = (name, started)

        def _settle(outcome: str, category: str, detail: str) -> StartupPhaseRecord:
            return self.record(
                name,
                outcome,
                category=category,
                detail=detail,
                started_at=started_wall,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        try:
            value = run_bounded(fn, timeout_s, name=name)
        except StartupPhaseTimeout as exc:
            record = _settle(OUTCOME_TIMEOUT, CATEGORY_TIMEOUT, str(exc))
            if required:
                raise StartupPhaseError(record) from exc
            logger.warning("Optional startup phase timed out: %s", record.format_line())
            return None
        except Exception as exc:  # noqa: BLE001 - provider code may raise anything; the failure is recorded as structured diagnostics, never dropped
            detail = f"{type(exc).__name__}: {exc}"
            record = _settle(OUTCOME_FAILED, classify_exception(exc), detail)
            if required:
                raise StartupPhaseError(record) from exc
            logger.warning("Optional startup phase failed: %s", record.format_line())
            return None

        outcome, category, detail = (
            outcome_of(value) if outcome_of is not None else (OUTCOME_OK, "", "")
        )
        record = _settle(outcome, category, detail)
        if outcome != OUTCOME_OK:
            logger.warning("Startup phase degraded: %s", record.format_line())
        return value

    def format_diagnostics(self) -> str:
        """Render every phase plus any still-running phase as text."""
        with self._lock:
            current = self._current
        running = None
        if current is not None:
            running = (current[0], int((time.perf_counter() - current[1]) * 1000))
        return format_phase_diagnostics(
            self.records, total_ms=self.elapsed_ms(), running=running
        )


def format_phase_diagnostics(
    phases: tuple[StartupPhaseRecord, ...] | list[StartupPhaseRecord],
    *,
    total_ms: int,
    running: tuple[str, int] | None = None,
) -> str:
    """Render phase records as copyable, timestamped diagnostics text."""
    if total_ms < 0:
        raise ValueError("total_ms must be non-negative")
    blocking = next((r for r in phases if r.outcome != OUTCOME_OK), None)
    lines = [
        "UpstreamDrift startup diagnostics",
        f"total {total_ms} ms, {len(phases)} phase(s) settled, "
        f"blocking phase: {blocking.name if blocking else 'none'}",
    ]
    lines.extend(record.format_line() for record in phases)
    if running is not None:
        lines.append(f"still running: {running[0]} ({running[1]} ms so far)")
    return "\n".join(lines)


@dataclass(frozen=True)
class ProviderProbeResult:
    """Outcome of checking an optional provider without importing it.

    Attributes:
        provider: Provider name (``"tools"``).
        available: True when the provider can be launched.
        category: ``""`` when available, else one of the ``CATEGORY_*``.
        detail: Where the provider resolved, or why it did not.
        entry_point: Resolved standalone entry point when available.
        pinned: True only for the vendored gitlink validated against the pin.
    """

    provider: str
    available: bool
    category: str
    detail: str
    entry_point: Path | None = None
    pinned: bool = False

    def phase_outcome(self) -> tuple[str, str, str]:
        """Map the probe onto a phase ``(outcome, category, detail)``."""
        outcome = OUTCOME_OK if self.available else OUTCOME_DEGRADED
        return outcome, self.category, self.detail


def _unavailable(category: str, detail: str) -> ProviderProbeResult:
    return ProviderProbeResult(
        provider=_TOOLS_PROVIDER, available=False, category=category, detail=detail
    )


def probe_tools_provider(
    repo_root: Path,
    env_value: str | None,
    *,
    entry_point: Path = RATE_STANDALONE_ENTRY_POINT,
    package: str = _RATE_PACKAGE,
) -> ProviderProbeResult:
    """Check whether the optional Tools/Rate provider can be launched.

    The probe deliberately never imports the provider or touches
    ``sys.path``: it resolves the checkout through the canonical
    :func:`resolve_tools_repo` facade, confirms the standalone entry point
    exists, and locates the package spec on the checkout's own ``src`` so a
    broken tree surfaces as ``import_failure`` rather than a hang.

    Preconditions:
        ``repo_root`` is a ``Path``; ``entry_point`` is a relative ``Path``;
        ``package`` is a non-empty dotted name.

    Postconditions:
        ``available`` is True iff ``entry_point`` is a file inside a resolved
        checkout whose ``src`` exposes ``package``; otherwise ``category``
        names exactly one failure class and ``detail`` says why.
    """
    if not isinstance(repo_root, Path):
        raise TypeError("repo_root must be a pathlib.Path")
    if not isinstance(entry_point, Path) or entry_point.is_absolute():
        raise ValueError("entry_point must be a relative pathlib.Path")
    if not package:
        raise ValueError("package must be non-empty")

    try:
        resolution = resolve_tools_repo(repo_root, env_value)
    except RuntimeError as exc:
        return _unavailable(CATEGORY_MISSING_CHECKOUT, str(exc))
    if resolution is None:
        return _unavailable(
            CATEGORY_MISSING_CHECKOUT,
            "Tools checkout not found: no TOOLS_REPO_PATH, no initialized "
            "vendor/ud-tools gitlink pin, no sibling Tools checkout",
        )

    location = f"{resolution.source}, {'pinned' if resolution.pinned else 'unpinned'}"
    script = resolution.path / entry_point
    if not script.is_file():
        return _unavailable(
            CATEGORY_MISSING_DEPENDENCY,
            f"Rate standalone entry point missing from Tools ({location}): {script}",
        )

    search_path = str(resolution.path / "src")
    try:
        spec = PathFinder.find_spec(package, [search_path])
    except (ImportError, OSError, ValueError, SyntaxError) as exc:
        return _unavailable(
            CATEGORY_IMPORT_FAILURE,
            f"{package} could not be located under {search_path}: "
            f"{type(exc).__name__}: {exc}",
        )
    if spec is None:
        return _unavailable(
            CATEGORY_MISSING_DEPENDENCY,
            f"optional package {package!r} not found under {search_path}",
        )

    return ProviderProbeResult(
        provider=_TOOLS_PROVIDER,
        available=True,
        category="",
        detail=f"Tools ({location}) at {resolution.path}; Rate entry point {script}",
        entry_point=script,
        pinned=resolution.pinned,
    )
