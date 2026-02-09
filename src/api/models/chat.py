"""Chat API contract models."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessageRequest(BaseModel):
    """Request to send a chat message."""

    message: str = Field(..., min_length=1, max_length=10000)
    engine_context: str | None = Field(
        None, description="Active engine type (e.g. 'mujoco', 'drake')"
    )
    expertise_level: str = Field("beginner")


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
    messages: list[dict]
