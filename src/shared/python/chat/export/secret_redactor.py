"""Secret redaction utility for chat exports (Tools issue #2735).

Single registry of regular expressions used to strip API keys, bearer
tokens, GitHub PATs, AWS access key IDs, JWTs, and generic ``sk-*`` /
``sk_live_*`` style secrets out of message content before it is written
to disk or copied to the clipboard.

Pinned messages (``metadata["pin"] is True``) are returned untouched so
users can intentionally keep example secrets in an audit trail.
"""

from __future__ import annotations

import re
from dataclasses import replace

from chat.service_base import ChatMessage

_REDACTED = "[REDACTED]"

# One regex registry shared across export + copy paths. Order matters — the
# longer/more-specific patterns must come before the catch-all sk- pattern
# so they win when both could match.
_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # JWT (three base64url segments)
    (
        "jwt",
        re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}"),
    ),
    # Bearer / Authorization token
    ("bearer", re.compile(r"Bearer\s+[A-Za-z0-9._\-]+", re.IGNORECASE)),
    # GitHub personal access token / fine-grained / oauth tokens
    ("github_pat", re.compile(r"gh[pousr]_[A-Za-z0-9]{20,}")),
    # AWS access key id
    ("aws_access_key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    # sk_live / sk_test style provider keys (Stripe, etc.)
    ("provider_live_key", re.compile(r"sk_(?:live|test)_[A-Za-z0-9]{16,}")),
    # Generic sk- API keys (OpenAI, Anthropic, etc.). Match last to avoid
    # eating prefixes of more specific tokens.
    ("openai_sk", re.compile(r"sk-[A-Za-z0-9]{16,}")),
)


class SecretRedactor:
    """Apply the secret regex registry to strings or :class:`ChatMessage`."""

    def redact(self, text: str) -> str:
        """Return ``text`` with every known secret replaced by ``[REDACTED]``.

        Pre:
            ``text`` is a string (``TypeError`` otherwise).
        Post:
            For every registered pattern, the returned string contains no
            match of that pattern.
        """
        if not isinstance(text, str):
            raise TypeError("SecretRedactor.redact expects a str")
        out = text
        for _, pattern in _SECRET_PATTERNS:
            out = pattern.sub(_REDACTED, out)
        return out

    def redact_message(self, message: ChatMessage) -> ChatMessage:
        """Return a copy of ``message`` with redacted content.

        Messages with ``metadata["pin"] is True`` are returned untouched so
        the user's intentional anchors survive the round-trip.
        """
        if message.metadata.get("pin") is True:
            return message
        return replace(message, content=self.redact(message.content))


__all__ = ["SecretRedactor"]
