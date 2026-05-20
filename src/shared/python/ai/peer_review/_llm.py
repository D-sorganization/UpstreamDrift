"""Shared LLM client protocol and stub for peer-review reviewers.

The protocol is intentionally narrow: a single ``evaluate`` method that
takes the criteria, content, and reviewer role and returns a JSON-like
mapping with ``verdict``, ``reasoning``, and ``confidence``. Production
adapters implement this on top of provider SDKs (OpenAI, Anthropic,
Ollama, etc); tests inject :class:`StubReviewerLLMClient`.

DRY: defining the protocol once here means all three builtin reviewers
share the same shape (the test files import it directly from each
builtin module via re-export).
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable


@runtime_checkable
class ReviewerLLMClient(Protocol):
    """Minimal contract for an LLM that powers a reviewer.

    Implementations must:

    - Accept ``criteria_set`` (the criteria from the request),
      ``subject_content`` (raw text), and the reviewer's ``role``.
    - Return a mapping with three keys: ``verdict`` (one of
      ``approve``/``request_changes``/``reject``/``abstain``),
      ``reasoning`` (string), and ``confidence`` (float in ``[0, 1]``).
    - Be safe to call concurrently — the coordinator fans out reviews
      with ``asyncio.gather``.
    """

    async def evaluate(
        self,
        *,
        criteria_set: list[str],
        subject_content: str,
        role: str,
    ) -> dict[str, object]: ...


class StubReviewerLLMClient:
    """Deterministic stub used by tests and as a default for offline runs.

    The ``call_count`` field lets tests assert the LLM was (or was not)
    invoked, which matters for precondition coverage on the reviewers.
    """

    def __init__(
        self,
        *,
        canned_verdict: str = "approve",
        canned_reasoning: str = "stub reasoning",
        canned_confidence: float = 0.8,
    ) -> None:
        self._canned_verdict = canned_verdict
        self._canned_reasoning = canned_reasoning
        self._canned_confidence = canned_confidence
        self.call_count = 0

    async def evaluate(
        self,
        *,
        criteria_set: list[str],
        subject_content: str,
        role: str,
    ) -> dict[str, object]:
        self.call_count += 1
        return {
            "verdict": self._canned_verdict,
            "reasoning": self._canned_reasoning,
            "confidence": self._canned_confidence,
        }


__all__ = ["ReviewerLLMClient", "StubReviewerLLMClient"]
