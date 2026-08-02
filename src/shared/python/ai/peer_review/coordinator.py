"""ReviewCoordinator (Tools #2738).

The coordinator is the only orchestration surface that callers (chat
integration, REST endpoints, future agent loops) touch. Reviewers and the
registry are deliberately hidden behind it (Law of Demeter).

Lifecycle
---------
1. Validate the request (DbC precondition: non-empty ``criteria_set``).
2. Ask the registry for the panel.
3. Reject empty / undersized panels with the appropriate error.
4. Fan out :meth:`Reviewer.review` calls via ``asyncio.gather`` under a
   single deadline. The first to exhaust the deadline aborts the whole
   batch and surfaces :class:`ReviewerTimeoutError`.
5. Compute consensus, build the audit trail, return a
   :class:`PeerReviewResult`.

Concurrency: each reviewer is awaited in its own task, but the coordinator
itself holds no mutable state across awaits — instances are safe to share
between concurrent ``run_review`` calls.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from ._audit import _audit_event
from .consensus import compute_consensus
from .contracts import (
    PeerReviewResult,
    ReviewRequest,
    ReviewSubject,
    ReviewVerdict,
)
from .errors import (
    InsufficientPanelError,
    NoReviewersError,
    ReviewerTimeoutError,
)
from .registry import ReviewerRegistry

VerdictSink = Callable[[ReviewVerdict], Awaitable[None]]
"""Optional callback that receives each verdict as it arrives (chat stream)."""


class ReviewCoordinator:
    """Orchestrates a single :class:`ReviewRequest` against a panel."""

    def __init__(
        self,
        *,
        registry: ReviewerRegistry,
        min_panel_size: int = 2,
    ) -> None:
        if min_panel_size < 1:
            raise ValueError("min_panel_size must be >= 1")
        self._registry = registry
        self._min_panel_size = min_panel_size

    async def run_review(
        self,
        request: ReviewRequest,
        subject: ReviewSubject,
        *,
        on_verdict: VerdictSink | None = None,
    ) -> PeerReviewResult:
        """Run a peer review and return the result.

        Precondition: ``request.criteria_set`` is non-empty.

        Postcondition: ``len(result.verdicts) >= 1`` on success — the
        coordinator raises :class:`InsufficientPanelError` rather than
        returning an empty verdict list.
        """
        if not request.criteria_set:
            raise ValueError(
                "ReviewCoordinator.run_review: request.criteria_set must "
                "be non-empty (DbC precondition)"
            )

        audit: list[dict[str, Any]] = []
        audit.append(_audit_event("started", request_id=request.request_id))

        panel = self._select_panel(request)
        audit.append(
            _audit_event(
                "panel_selected",
                request_id=request.request_id,
                extra={
                    "size": len(panel),
                    "agent_ids": [r.descriptor.agent_id for r in panel],
                },
            )
        )

        verdicts = await self._gather_verdicts(
            request,
            subject,
            panel,
            on_verdict=on_verdict,
            audit=audit,
        )

        # Postcondition: at least one verdict on success.
        if not verdicts:  # pragma: no cover - defensive; panel was non-empty
            raise InsufficientPanelError(
                "Panel produced no verdicts (postcondition violated)"
            )

        consensus = compute_consensus(verdicts)
        result = PeerReviewResult(
            request_id=request.request_id,
            verdicts=verdicts,
            consensus=consensus,
            final_disposition=consensus,
            audit_trail=audit,
        )
        audit.append(
            _audit_event(
                "completed",
                request_id=request.request_id,
                extra={"consensus": consensus, "verdict_count": len(verdicts)},
            )
        )
        # Re-pack with the final "completed" event included.
        return result.model_copy(update={"audit_trail": list(audit)})

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _select_panel(self, request: ReviewRequest) -> tuple[Any, ...]:
        if not self._registry.list():
            raise NoReviewersError("Registry has no reviewers registered")
        panel = self._registry.panel_for(request.criteria_set)
        if len(panel) < self._min_panel_size:
            raise InsufficientPanelError(
                f"Panel size {len(panel)} below minimum {self._min_panel_size} "
                f"for criteria {sorted(set(request.criteria_set))}"
            )
        return tuple(panel)

    async def _gather_verdicts(
        self,
        request: ReviewRequest,
        subject: ReviewSubject,
        panel: tuple[Any, ...],
        *,
        on_verdict: VerdictSink | None,
        audit: list[dict[str, Any]],
    ) -> list[ReviewVerdict]:
        async def _run_one(reviewer: Any) -> ReviewVerdict:
            verdict = await reviewer.review(request, subject)
            audit.append(
                _audit_event(
                    "verdict_received",
                    request_id=request.request_id,
                    extra={
                        "reviewer_agent_id": verdict.reviewer_agent_id,
                        "verdict": verdict.verdict,
                    },
                )
            )
            if on_verdict is not None:
                await on_verdict(verdict)
            return verdict

        tasks = [asyncio.create_task(_run_one(r)) for r in panel]
        try:
            verdicts = await asyncio.wait_for(
                asyncio.gather(*tasks), timeout=request.deadline_seconds
            )
        except TimeoutError as exc:
            for task in tasks:
                task.cancel()
            audit.append(
                _audit_event(
                    "timeout",
                    request_id=request.request_id,
                    message=f"deadline {request.deadline_seconds}s exceeded",
                )
            )
            raise ReviewerTimeoutError(
                f"Peer review timed out after {request.deadline_seconds}s"
            ) from exc
        return list(verdicts)


__all__ = ["ReviewCoordinator", "VerdictSink"]
