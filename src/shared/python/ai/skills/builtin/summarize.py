"""Summarize skill — demonstrates an LLM-backed skill with audit trail.

Tools #2737. The LLM client is injectable so tests can pass a stub and
production code can pass a real adapter without changing this file.
"""

from __future__ import annotations

from typing import Any, ClassVar, Protocol

from ..base import Skill
from ..contracts import SkillDescriptor, SkillInvocation, SkillResult


class LLMClient(Protocol):
    """Minimal contract for an LLM client used by :class:`SummarizeSkill`.

    Implementations forward to whatever provider adapter the caller wants
    (OpenAI, Anthropic, Ollama, etc). The skill itself does not know.
    """

    async def summarize(self, text: str) -> str: ...


class StubLLMClient:
    """Deterministic stub used by tests and as a usable default in offline
    environments. ``call_count`` lets tests assert the LLM was (or was not)
    invoked, which matters for precondition coverage.
    """

    def __init__(self, canned_response: str) -> None:
        self._canned_response = canned_response
        self.call_count = 0

    async def summarize(self, text: str) -> str:
        self.call_count += 1
        return self._canned_response


class SummarizeSkill(Skill):
    """Summarises a non-empty text passage using an injected LLM client."""

    descriptor: ClassVar[SkillDescriptor] = SkillDescriptor(
        id="builtin.summarize",
        name="Summarize",
        version="1.0.0",
        description="Summarises text using the injected LLM client.",
        inputs={"text": "string"},
        outputs={"summary": "string"},
        preconditions=["text_is_non_empty_string"],
        postconditions=["summary_is_string"],
    )

    def __init__(self, *, llm_client: LLMClient | None = None) -> None:
        self._llm = llm_client or StubLLMClient(canned_response="(no summary)")
        self._last_audit_extra: dict[str, Any] = {}

    def validate_preconditions(self, args: dict[str, Any]) -> None:
        text = args.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError(
                "text_is_non_empty_string: 'text' must be a non-empty string"
            )

    def validate_postconditions(self, result: dict[str, Any]) -> None:
        summary = result.get("summary")
        if not isinstance(summary, str):
            raise ValueError("summary_is_string: 'summary' must be a string")

    async def run(self, invocation: SkillInvocation) -> SkillResult:
        text: str = invocation.args["text"]
        summary = await self._llm.summarize(text)
        # The skill emits one extra audit event (kind="llm_call") so callers
        # can see the LLM was invoked. The runner appends this to the trail.
        audit_trail = [
            {
                "kind": "llm_call",
                "skill_id": self.descriptor.id,
                "request_id": invocation.request_id,
                "message": "summarize",
                "extra": {"input_length": len(text)},
            }
        ]
        return SkillResult(
            request_id=invocation.request_id,
            success=True,
            value={"summary": summary},
            error=None,
            elapsed_ms=0.0,
            audit_trail=audit_trail,
        )
