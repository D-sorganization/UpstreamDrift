"""Built-in reference skills for ``ai.skills`` (Tools #2737)."""

from __future__ import annotations

from .echo import EchoSkill
from .summarize import StubLLMClient, SummarizeSkill

__all__ = ["EchoSkill", "StubLLMClient", "SummarizeSkill"]
