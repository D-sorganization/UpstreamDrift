"""Skill runner (Tools #2737).

Orchestrates a single skill invocation: precondition check, body execution
with timeout, postcondition check, audit emission. External callers go
through :meth:`SkillRunner.run` — never reach into the registry's private
state (LOD).
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from .base import Skill
from .contracts import SkillInvocation, SkillResult
from .errors import (
    SkillExecutionError,
    SkillNotFoundError,
    SkillPostconditionError,
    SkillPreconditionError,
    SkillTimeoutError,
)
from .registry import SkillRegistry, default_registry

_logger = logging.getLogger(__name__)


def _audit_event(
    kind: str,
    *,
    skill_id: str,
    request_id: str,
    message: str | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a single audit-trail event.

    Keeping construction in one helper enforces DRY for both the runner and
    any future audit sink integration.
    """
    event: dict[str, Any] = {
        "kind": kind,
        "skill_id": skill_id,
        "request_id": request_id,
        "timestamp": time.time(),
    }
    if message is not None:
        event["message"] = message
    if extra:
        event["extra"] = extra
    return event


class SkillRunner:
    """Executes a :class:`SkillInvocation` against a :class:`SkillRegistry`."""

    def __init__(self, *, registry: SkillRegistry | None = None) -> None:
        self._registry = registry if registry is not None else default_registry()

    async def run(self, invocation: SkillInvocation) -> SkillResult:
        """Execute an invocation. Raises :class:`SkillNotFoundError`.

        Contract failures (precondition, postcondition, timeout, body error)
        are returned as ``SkillResult(success=False, error=...)`` rather than
        raised — callers should branch on ``result.success``.
        """
        skill = self._registry.get(invocation.skill_id)  # may raise NotFound
        audit: list[dict[str, Any]] = []
        start = time.perf_counter()
        skill_id = skill.descriptor.id

        audit.append(
            _audit_event(
                "started",
                skill_id=skill_id,
                request_id=invocation.request_id,
            )
        )

        try:
            self._check_preconditions(skill, invocation, audit)
            value, skill_audit = await self._execute_with_timeout(
                skill, invocation, audit
            )
            audit.extend(skill_audit)
            self._check_postconditions(skill, value, invocation, audit)
        except SkillPreconditionError as exc:
            return self._failure(invocation, audit, start, "precondition_failed", exc)
        except SkillPostconditionError as exc:
            return self._failure(invocation, audit, start, "postcondition_failed", exc)
        except SkillTimeoutError as exc:
            return self._failure(invocation, audit, start, "timeout", exc)
        except SkillExecutionError as exc:
            return self._failure(invocation, audit, start, "execution_error", exc)

        elapsed_ms = (time.perf_counter() - start) * 1000.0
        audit.append(
            _audit_event(
                "completed",
                skill_id=skill_id,
                request_id=invocation.request_id,
                extra={"elapsed_ms": elapsed_ms},
            )
        )
        return SkillResult(
            request_id=invocation.request_id,
            success=True,
            value=value,
            error=None,
            elapsed_ms=elapsed_ms,
            audit_trail=audit,
        )

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _check_preconditions(
        self,
        skill: Skill,
        invocation: SkillInvocation,
        audit: list[dict[str, Any]],
    ) -> None:
        try:
            skill.validate_preconditions(invocation.args)
        except ValueError as exc:
            audit.append(
                _audit_event(
                    "precondition_failed",
                    skill_id=skill.descriptor.id,
                    request_id=invocation.request_id,
                    message=str(exc),
                )
            )
            raise SkillPreconditionError(str(exc)) from exc

    async def _execute_with_timeout(
        self,
        skill: Skill,
        invocation: SkillInvocation,
        audit: list[dict[str, Any]],
    ) -> tuple[dict[str, Any], list[dict[str, Any]]]:
        try:
            result = await asyncio.wait_for(
                skill.run(invocation), timeout=invocation.timeout_s
            )
        except TimeoutError as exc:
            audit.append(
                _audit_event(
                    "timeout",
                    skill_id=skill.descriptor.id,
                    request_id=invocation.request_id,
                    message=f"exceeded {invocation.timeout_s}s",
                )
            )
            raise SkillTimeoutError(
                f"skill {skill.descriptor.id!r} timed out after {invocation.timeout_s}s"
            ) from exc
        except (SkillPreconditionError, SkillPostconditionError):
            raise
        except Exception as exc:  # noqa: BLE001 — boundary catch is intentional
            audit.append(
                _audit_event(
                    "execution_error",
                    skill_id=skill.descriptor.id,
                    request_id=invocation.request_id,
                    message=str(exc),
                )
            )
            _logger.exception(
                "Skill %s raised %s", skill.descriptor.id, type(exc).__name__
            )
            raise SkillExecutionError(str(exc)) from exc

        value = result.value if result.value is not None else {}
        return value, list(result.audit_trail)

    def _check_postconditions(
        self,
        skill: Skill,
        value: dict[str, Any],
        invocation: SkillInvocation,
        audit: list[dict[str, Any]],
    ) -> None:
        try:
            skill.validate_postconditions(value)
        except ValueError as exc:
            audit.append(
                _audit_event(
                    "postcondition_failed",
                    skill_id=skill.descriptor.id,
                    request_id=invocation.request_id,
                    message=str(exc),
                )
            )
            raise SkillPostconditionError(str(exc)) from exc

    def _failure(
        self,
        invocation: SkillInvocation,
        audit: list[dict[str, Any]],
        start: float,
        kind: str,
        exc: Exception,
    ) -> SkillResult:
        elapsed_ms = (time.perf_counter() - start) * 1000.0
        audit.append(
            _audit_event(
                "failed",
                skill_id=invocation.skill_id,
                request_id=invocation.request_id,
                message=str(exc),
                extra={"failure_kind": kind, "elapsed_ms": elapsed_ms},
            )
        )
        return SkillResult(
            request_id=invocation.request_id,
            success=False,
            value=None,
            error=str(exc),
            elapsed_ms=elapsed_ms,
            audit_trail=audit,
        )


# Re-export for convenience: callers want one import line.
__all__ = ["SkillRunner", "SkillNotFoundError"]
