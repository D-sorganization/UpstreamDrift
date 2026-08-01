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

import asyncio
import json
import logging
import re
from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from shared.python.ai.adapters.base import BaseAgentAdapter

_logger = logging.getLogger(__name__)

# Verdicts the reviewers (and consensus) understand. Mirrors
# ``contracts.VerdictKind`` but kept local so this module stays free of a
# Pydantic import on a hot path (Law of Demeter / Orthogonality).
_VALID_VERDICTS = frozenset({"approve", "request_changes", "reject", "abstain"})

# Matches the first ``{ ... }`` JSON object in a possibly chatty CLI reply.
# CLI agents wrap JSON in prose or ```json fences; we extract the object.
_JSON_OBJECT_RE = re.compile(r"\{.*\}", re.DOTALL)


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


def _build_reviewer_prompt(
    *,
    criteria_set: list[str],
    subject_content: str,
    role: str,
) -> str:
    """Build the structured prompt sent to the adapter.

    The prompt instructs the model to answer with a single JSON object so
    the reply can be parsed deterministically. The role and criteria are
    injected so the reviewer's bias (critic/advocate/specialist) is honoured
    by the underlying model.
    """
    criteria_block = "\n".join(f"- {c}" for c in criteria_set) or "- (none provided)"
    return (
        f"You are acting as a peer reviewer with the role: {role}.\n"
        "Evaluate the SUBJECT below against the listed CRITERIA.\n\n"
        f"CRITERIA:\n{criteria_block}\n\n"
        f"SUBJECT:\n{subject_content}\n\n"
        "Respond with ONLY a single JSON object (no prose, no code fences) "
        "of the exact shape:\n"
        '{"verdict": "approve|request_changes|reject|abstain", '
        '"reasoning": "<concise justification>", '
        '"confidence": <float between 0 and 1>}\n'
        'Use "abstain" if you cannot reach a confident verdict.'
    )


def _parse_evaluation(raw_text: str) -> dict[str, object]:
    """Parse an adapter's raw text reply into a verdict mapping.

    Contract:

    - On any parse failure (non-JSON, missing object, wrong types) the
      result degrades to ``verdict="abstain"`` with ``confidence=0.0`` — the
      panel must never crash on a single malformed reviewer reply.
    - ``confidence`` is coerced to ``float`` and clamped into ``[0.0, 1.0]``.
    - An out-of-vocabulary ``verdict`` degrades to ``"abstain"``.
    """
    match = _JSON_OBJECT_RE.search(raw_text or "")
    if match is None:
        _logger.warning("Adapter reply contained no JSON object; abstaining")
        return {
            "verdict": "abstain",
            "reasoning": "no JSON in reply",
            "confidence": 0.0,
        }

    try:
        parsed = json.loads(match.group(0))
    except (json.JSONDecodeError, ValueError):
        _logger.warning("Adapter reply was not valid JSON; abstaining")
        return {
            "verdict": "abstain",
            "reasoning": "malformed JSON in reply",
            "confidence": 0.0,
        }

    if not isinstance(parsed, dict):
        return {
            "verdict": "abstain",
            "reasoning": "JSON was not an object",
            "confidence": 0.0,
        }

    verdict = str(parsed.get("verdict", "abstain")).strip().lower()
    if verdict not in _VALID_VERDICTS:
        verdict = "abstain"

    reasoning = str(parsed.get("reasoning", ""))

    try:
        confidence = float(parsed.get("confidence", 0.0))
    except (TypeError, ValueError):
        confidence = 0.0
    # Clamp into the contractual [0, 1] range rather than abstaining outright,
    # so a model that over/under-shoots still contributes a usable verdict.
    confidence = max(0.0, min(1.0, confidence))

    return {"verdict": verdict, "reasoning": reasoning, "confidence": confidence}


class AdapterReviewerLLMClient:
    """Production :class:`ReviewerLLMClient` backed by a ``BaseAgentAdapter``.

    Drives a real model by building a structured prompt, invoking the
    adapter's synchronous :meth:`~BaseAgentAdapter.send_message` off the event
    loop (the coordinator fans reviews out with ``asyncio.gather``), and
    parsing the JSON reply into ``{verdict, reasoning, confidence}``.

    Robustness contract:

    - Malformed / non-JSON adapter output degrades to ``verdict="abstain"``.
    - ``confidence`` is clamped into ``[0.0, 1.0]``.
    - Adapter exceptions are caught and surfaced as an ``abstain`` verdict so a
      single failing provider never aborts the panel. (The reviewers' shared
      ``evaluate_to_verdict`` helper also guards this, but degrading here keeps
      the client usable in isolation.)
    """

    def __init__(self, adapter: BaseAgentAdapter) -> None:
        """Wrap an adapter.

        Args:
            adapter: A constructed provider adapter. Must expose
                ``send_message``.

        Raises:
            TypeError: If ``adapter`` does not expose a callable
                ``send_message`` (Design-by-Contract precondition).
        """
        if not callable(getattr(adapter, "send_message", None)):
            raise TypeError("adapter must expose a callable send_message method")
        self._adapter = adapter

    async def evaluate(
        self,
        *,
        criteria_set: list[str],
        subject_content: str,
        role: str,
    ) -> dict[str, object]:
        """Run one review via the adapter and return a verdict mapping.

        Never raises on provider failure: any adapter exception or malformed
        reply degrades to ``{"verdict": "abstain", ...}``.
        """
        # Local import keeps the peer_review package importable without the
        # adapters subpackage at module load (Orthogonality / Law of Demeter).
        from shared.python.ai.types import ConversationContext

        prompt = _build_reviewer_prompt(
            criteria_set=list(criteria_set),
            subject_content=subject_content,
            role=role,
        )
        context = ConversationContext()

        def _call() -> str:
            response = self._adapter.send_message(prompt, context, [])
            return getattr(response, "content", "") or ""

        try:
            raw_text = await asyncio.to_thread(_call)
        except Exception as exc:  # noqa: BLE001 — provider boundary catch
            _logger.warning(
                "Adapter %s raised during peer review (%s); abstaining",
                type(self._adapter).__name__,
                type(exc).__name__,
            )
            return {
                "verdict": "abstain",
                "reasoning": f"adapter error: {exc}",
                "confidence": 0.0,
            }

        return _parse_evaluation(raw_text)


__all__ = [
    "AdapterReviewerLLMClient",
    "ReviewerLLMClient",
    "StubReviewerLLMClient",
]
