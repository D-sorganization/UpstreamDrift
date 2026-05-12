"""Chat API contract models."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, model_validator

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

# Legacy ``expertise_level`` -> ``response_style`` translation table.
# beginner / intermediate map to the more verbose styles; advanced /
# expert map to the more concise styles.
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
        return RESPONSE_STYLE_PROMPTS[style]  # type: ignore[index]
    return RESPONSE_STYLE_PROMPTS[DEFAULT_RESPONSE_STYLE]


class ChatMessageRequest(BaseModel):
    """Request to send a chat message.

    The ``response_style`` field (Tools #2552) describes how verbose the
    AI's response should be. The legacy ``expertise_level`` field is
    retained for backward compatibility and is auto-mapped to a
    ``response_style`` value when ``response_style`` is not set.
    """

    message: str = Field(..., min_length=1, max_length=10000)
    engine_context: str | None = Field(
        None, description="Active engine type (e.g. 'mujoco', 'drake')"
    )
    response_style: ResponseStyle = Field(
        DEFAULT_RESPONSE_STYLE,
        description=(
            "How verbose the AI reply should be. One of 'concise', "
            "'standard', or 'detailed' (Tools issue #2552)."
        ),
    )
    # Deprecated alias kept so older clients (and the stored
    # ``ai/expertise_level`` QSettings key) keep working through the
    # cutover. New code should set ``response_style`` instead.
    expertise_level: str = Field(
        "beginner",
        description="DEPRECATED: use ``response_style`` instead (Tools #2552).",
    )

    @model_validator(mode="after")
    def _back_fill_response_style(self) -> ChatMessageRequest:
        """Map a legacy ``expertise_level`` onto ``response_style``.

        Only applied when the caller did *not* supply ``response_style``,
        so explicit ``response_style`` values always win and pure-default
        requests resolve to ``DEFAULT_RESPONSE_STYLE``.
        """
        fields_set = self.model_fields_set
        if "expertise_level" in fields_set and "response_style" not in fields_set:
            mapped = _EXPERTISE_TO_STYLE.get(self.expertise_level.lower())
            if mapped is not None:
                # ``model_copy(update=...)`` is the supported way to mutate a
                # frozen-by-default Pydantic model in a validator.
                object.__setattr__(self, "response_style", mapped)
        return self


class ChatModelInfo(BaseModel):
    """Single available chat model entry.

    Mirrors the Tools-side ``ChatModelInfo`` contract introduced in
    Tools issue #2547 / PR #2566. Serialised in ``model_list`` payloads
    sent over the chat WebSocket.
    """

    name: str = Field(..., description="Provider-specific model identifier")
    provider: str = Field(
        ..., description="Provider id (e.g. 'ollama', 'openai', 'anthropic')"
    )
    display_name: str | None = Field(
        None, description="Optional human-readable label for UI display"
    )


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
    engine_contexts: list[str] = Field(default_factory=list)


class ChatHistoryResponse(BaseModel):
    """Full message history for a session."""

    session_id: str
    messages: list[dict[str, Any]]
