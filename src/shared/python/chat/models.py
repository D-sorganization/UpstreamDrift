"""Chat API contract models.

Generic Pydantic models for the WebSocket chat protocol used by
ChatDockWidget. These are reusable across any application that
integrates the shared chat system.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

# ---------------------------------------------------------------------------
# Reasoning / thinking budget contracts (Tools issue #2871)
#
# These small immutable value objects let every adapter describe which
# "thinking" / reasoning-budget levels its currently configured model
# supports.  They are deliberately decoupled from the adapter so the
# shared ChatDockWidget can populate its Thinking dropdown without
# reaching into adapter internals (Law of Demeter).
# ---------------------------------------------------------------------------

ThinkingLevelName = Literal["none", "low", "medium", "high"]
_VALID_THINKING_NAMES: frozenset[str] = frozenset({"none", "low", "medium", "high"})


@dataclass(frozen=True)
class ThinkingLevel:
    """One reasoning-budget level for a model.

    Attributes:
        name: One of ``"none"``, ``"low"``, ``"medium"``, ``"high"``.
        budget_tokens: Provider-side thinking budget in tokens; must be
            ``>= 0``. The ``"none"`` level always has budget ``0``.
        label: Short human-readable display label (e.g. ``"Low"``).

    Contract:
        Pre: ``name`` is a member of ``_VALID_THINKING_NAMES``.
        Pre: ``budget_tokens >= 0``.
        Pre: ``label`` is a non-empty string.
    """

    name: ThinkingLevelName
    budget_tokens: int
    label: str

    # Enum compatibility
    OFF: ThinkingLevel = None  # type: ignore
    LOW: ThinkingLevel = None  # type: ignore
    MEDIUM: ThinkingLevel = None  # type: ignore
    HIGH: ThinkingLevel = None  # type: ignore

    @property
    def value(self) -> str:
        return self.name

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


# Initialize Enum-compatibility class attributes
ThinkingLevel.OFF = ThinkingLevel(name="none", budget_tokens=0, label="Off")
ThinkingLevel.LOW = ThinkingLevel(name="low", budget_tokens=1024, label="Low")
ThinkingLevel.MEDIUM = ThinkingLevel(name="medium", budget_tokens=4096, label="Medium")
ThinkingLevel.HIGH = ThinkingLevel(name="high", budget_tokens=16384, label="High")


@dataclass(frozen=True)
class ThinkingCapabilities:
    """Reasoning levels supported by a provider/model combination.

    Attributes:
        provider: Provider id string (e.g. ``"openai"``); non-empty.
        levels: Tuple of supported :class:`ThinkingLevel`; non-empty.
        default_level_name: Name of the level to select by default; must
            match one of the levels' ``name`` values.

    Contract:
        Pre: ``provider`` is a non-empty/non-whitespace string.
        Pre: ``levels`` is a non-empty tuple of :class:`ThinkingLevel`.
        Pre: ``default_level_name`` is a member of
             ``{level.name for level in levels}``.
    """

    provider: str = ""
    levels: tuple[ThinkingLevel, ...] = ()
    default_level_name: ThinkingLevelName = "none"

    def __init__(
        self,
        provider: str = "",
        levels: tuple[ThinkingLevel, ...] | list[ThinkingLevel] = (),
        default_level_name: ThinkingLevelName = "none",
        supports_levels: bool | None = None,
        available_levels: list[ThinkingLevel] | tuple[ThinkingLevel, ...] | None = None,
    ) -> None:
        if available_levels is not None and not levels:
            levels = tuple(available_levels)
        if provider == "" and available_levels is not None:
            provider = "mock_provider"

        object.__setattr__(self, "provider", provider)
        object.__setattr__(self, "levels", tuple(levels))
        object.__setattr__(self, "default_level_name", default_level_name)

        if supports_levels is None:
            supports_levels = len(levels) > 1 or (
                len(levels) == 1 and levels[0].name != "none"
            )
        object.__setattr__(self, "_supports_levels", supports_levels)
        object.__setattr__(self, "_available_levels", list(levels))

        # Validation checks
        if not isinstance(self.provider, str) or not self.provider.strip():
            raise ValueError("ThinkingCapabilities.provider must be non-empty")
        if self.levels:
            names = {level.name for level in self.levels}
            if self.default_level_name not in names:
                raise ValueError(
                    "ThinkingCapabilities.default_level_name "
                    f"{self.default_level_name!r} not present in "
                    f"level names {sorted(names)!r}"
                )

    @property
    def supports_levels(self) -> bool:
        return self._supports_levels

    @property
    def available_levels(self) -> list[ThinkingLevel]:
        return self._available_levels

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


# Tools issue #2552 / PR #2568: ``response_style`` is the new contract field
# describing how verbose the AI's reply should be. ``expertise_level`` is
# kept as a deprecated alias and auto-mapped to a ``response_style`` value
# so older clients keep working during the cutover.
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

_EXPERTISE_TO_STYLE: dict[str, ResponseStyle] = {
    "beginner": "detailed",
    "intermediate": "standard",
    "advanced": "concise",
    "expert": "concise",
}


def style_prompt(style: ResponseStyle | str | None) -> str:
    """Return the system-prompt fragment for a ``response_style`` value.

    Unknown / ``None`` values fall back to ``DEFAULT_RESPONSE_STYLE``.
    """
    if style in RESPONSE_STYLE_PROMPTS:
        return RESPONSE_STYLE_PROMPTS[style]
    return RESPONSE_STYLE_PROMPTS[DEFAULT_RESPONSE_STYLE]


class ChatMessageRequest(BaseModel):
    """Request to send a chat message.

    The ``response_style`` field (Tools #2552) describes how verbose the
    AI's response should be. The legacy ``expertise_level`` field is
    retained for backward compatibility and is auto-mapped to a
    ``response_style`` value when ``response_style`` is not set.
    """

    message: str = Field(..., min_length=1, max_length=10000)
    app_context: str | None = Field(
        None, description="Active application context (e.g. 'mujoco', 'gasification')"
    )
    response_style: ResponseStyle = Field(
        DEFAULT_RESPONSE_STYLE,
        description=(
            "How verbose the AI reply should be. One of 'concise', "
            "'standard', or 'detailed' (Tools issue #2552)."
        ),
    )
    expertise_level: str = Field(
        "beginner",
        description="DEPRECATED: use ``response_style`` instead (Tools #2552).",
    )

    @model_validator(mode="after")
    def _back_fill_response_style(self) -> ChatMessageRequest:
        """Map a legacy ``expertise_level`` onto ``response_style``."""
        fields_set = self.model_fields_set
        if "expertise_level" in fields_set and "response_style" not in fields_set:
            mapped = _EXPERTISE_TO_STYLE.get(self.expertise_level.lower())
            if mapped is not None:
                object.__setattr__(self, "response_style", mapped)
        return self


class ChatChunkResponse(BaseModel):
    """A single streaming chunk from the AI."""

    content: str
    is_final: bool = False
    index: int = 0


class ChatSessionInfo(BaseModel):
    """Summary info for a chat session."""

    session_id: str
    message_count: int
    created_at: str
    last_active: str
    app_contexts: list[str] = Field(default_factory=list)


class ChatHistoryResponse(BaseModel):
    """Full message history for a session."""

    session_id: str
    messages: list[dict]


class ChatModelInfo(BaseModel):
    """Single available chat model entry.

    Mirrors the Tools-side ``ChatModelInfo`` contract introduced in
    Tools issue #2547 / PR #2566. Serialised in ``model_list`` payloads
    sent over the chat WebSocket.
    """

    id: str | None = Field(None, description="Provider-specific stable model id")
    name: str = Field(..., description="Provider-specific model name")
    provider: str = Field(
        "unknown", description="Provider id (e.g. 'ollama', 'openai', 'anthropic')"
    )
    display_name: str | None = Field(
        None, description="Optional human-readable label for UI display"
    )
    available: bool = Field(True, description="Whether this model can be selected")

    def __init__(
        self,
        name: str | None = None,
        display_name: str | None = None,
        provider: str = "unknown",
        id: str | None = None,
        available: bool = True,
        **kwargs: Any,
    ) -> None:
        init_kwargs: dict[str, Any] = {
            "name": name,
            "display_name": display_name,
            "provider": provider,
            "id": id or name,
            "available": available,
        }
        init_kwargs.update(kwargs)
        super().__init__(**init_kwargs)

    @property
    def model_id(self) -> str:
        """Alias for id or name."""
        return self.id or self.name or ""


class ChatModelListResponse(BaseModel):
    """Server response to a ``refresh_models`` action.

    Sent in reply to the WebSocket ``{"action": "refresh_models"}`` request,
    carrying the freshly polled list of available models for the configured
    provider plus an ISO-8601 ``refreshed_at`` timestamp.
    """

    models: list[ChatModelInfo] = Field(default_factory=list)
    refreshed_at: str = Field(
        ..., description="ISO-8601 UTC timestamp of when the list was polled"
    )


class ChatIndexStatusResponse(BaseModel):
    """Server progress / completion event for the ``index_codebase`` action.

    Tools issue #2549 / PR #2567 introduced the ``auto_index_on_open`` flag
    on ``ChatDockWidget`` and an ``index_codebase`` WebSocket action that
    rebuilds the in-tree codemap. The server pushes ``index_status``
    messages of this shape so the widget can surface progress and
    completion to the user.
    """

    state: str = Field(..., description="One of 'running', 'complete', or 'error'")
    files_parsed: int = Field(
        0,
        ge=0,
        description="Files indexed so far (or total when state=='complete')",
    )
    symbols_inserted: int = Field(
        0,
        ge=0,
        description="Symbols inserted so far (or total when state=='complete')",
    )
    duration_seconds: float | None = Field(
        None,
        ge=0,
        description="Wall-clock elapsed seconds (set when state=='complete')",
    )
    error: str | None = Field(None, description="Error message when state=='error'")
