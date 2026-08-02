"""Shared chat capability and response-style contracts.

This module is intentionally dependency-free with respect to ``chat`` and
``ai`` so provider adapters and chat UI code can share value objects without
importing each other's packages.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

ThinkingLevelName = Literal["none", "low", "medium", "high"]
_VALID_THINKING_NAMES: frozenset[str] = frozenset({"none", "low", "medium", "high"})


@dataclass(frozen=True)
class ThinkingLevel:
    """One reasoning-budget level for a model."""

    name: ThinkingLevelName
    budget_tokens: int
    label: str

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or self.name not in _VALID_THINKING_NAMES:
            raise ValueError(
                "ThinkingLevel.name must be one of "
                f"{sorted(_VALID_THINKING_NAMES)!r}, got {self.name!r}"
            )
        if not isinstance(self.budget_tokens, int) or self.budget_tokens < 0:
            raise ValueError(
                "ThinkingLevel.budget_tokens must be a non-negative int, "
                f"got {self.budget_tokens!r}"
            )
        if not isinstance(self.label, str) or not self.label.strip():
            raise ValueError("ThinkingLevel.label must be a non-empty string")


@dataclass(frozen=True)
class ThinkingCapabilities:
    """Reasoning levels supported by a provider/model combination."""

    provider: str
    levels: tuple[ThinkingLevel, ...]
    default_level_name: ThinkingLevelName

    def __post_init__(self) -> None:
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("ThinkingCapabilities.provider must be non-empty")
        if not self.levels:
            raise ValueError("ThinkingCapabilities.levels must be non-empty")
        names = {level.name for level in self.levels}
        if self.default_level_name not in names:
            raise ValueError(
                "ThinkingCapabilities.default_level_name "
                f"{self.default_level_name!r} not present in "
                f"level names {sorted(names)!r}"
            )

    def level_names(self) -> tuple[str, ...]:
        """Return level names in declared order."""
        return tuple(level.name for level in self.levels)

    def find_level(self, name: str) -> ThinkingLevel | None:
        """Return the :class:`ThinkingLevel` for ``name`` or ``None``."""
        for level in self.levels:
            if level.name == name:
                return level
        return None


def make_none_only_capabilities(provider: str) -> ThinkingCapabilities:
    """Build a ``ThinkingCapabilities`` with just the ``"none"`` level."""
    return ThinkingCapabilities(
        provider=provider,
        levels=(ThinkingLevel(name="none", budget_tokens=0, label="Off"),),
        default_level_name="none",
    )


def make_full_thinking_capabilities(
    provider: str,
    *,
    low_budget: int = 1024,
    medium_budget: int = 4096,
    high_budget: int = 16384,
    default_level_name: ThinkingLevelName = "none",
) -> ThinkingCapabilities:
    """Build a four-level (none/low/medium/high) capability bundle."""
    return ThinkingCapabilities(
        provider=provider,
        levels=(
            ThinkingLevel(name="none", budget_tokens=0, label="Off"),
            ThinkingLevel(name="low", budget_tokens=low_budget, label="Low"),
            ThinkingLevel(name="medium", budget_tokens=medium_budget, label="Medium"),
            ThinkingLevel(name="high", budget_tokens=high_budget, label="High"),
        ),
        default_level_name=default_level_name,
    )


ResponseStyle = Literal["concise", "standard", "detailed"]
DEFAULT_RESPONSE_STYLE: ResponseStyle = "standard"

RESPONSE_STYLE_PROMPTS: dict[ResponseStyle, str] = {
    "concise": (
        "Reply concisely. Prefer code, tables, and short bullet lists over "
        "prose. Skip preamble and recap."
    ),
    "standard": (
        "Reply at a standard level of detail. Briefly explain reasoning "
        "where it helps the user act on the answer."
    ),
    "detailed": (
        "Reply in detail. Walk through reasoning, name relevant trade-offs, "
        "and include worked examples when they clarify the answer."
    ),
}


def style_prompt(style: ResponseStyle | str | None) -> str:
    """Return the system-prompt fragment for a ``response_style`` value."""
    if style in ("concise", "standard", "detailed"):
        return RESPONSE_STYLE_PROMPTS[style]  # type: ignore[index,unused-ignore]
    return RESPONSE_STYLE_PROMPTS[DEFAULT_RESPONSE_STYLE]
