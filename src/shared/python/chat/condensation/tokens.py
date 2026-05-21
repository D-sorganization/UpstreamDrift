"""Token estimation utilities (Tools issue #2736).

Uses ``tiktoken`` when available, otherwise falls back to a simple
``len(text) // 4`` heuristic which is the canonical OpenAI rule-of-thumb
for English prose.
"""

from __future__ import annotations

from functools import lru_cache


def _fallback_estimate(text: str) -> int:
    """Heuristic estimate of tokens (``len // 4``) for any string."""
    return len(text) // 4


@lru_cache(maxsize=1)
def _get_tiktoken_encoder() -> object | None:
    try:
        import tiktoken
    except ImportError:
        return None
    try:
        return tiktoken.get_encoding("cl100k_base")
    except (ValueError, RuntimeError):
        return None


def estimate_tokens(text: str) -> int:
    """Return an approximate token count for ``text``.

    Pre:
        ``text`` is a string. ``TypeError`` for non-string input.
    Post:
        Returned count is ``>= 0``; empty string yields ``0``.
    """
    if not isinstance(text, str):
        raise TypeError("estimate_tokens expects a str")
    if not text:
        return 0
    encoder = _get_tiktoken_encoder()
    if encoder is not None:
        try:
            return len(encoder.encode(text))  # type: ignore[attr-defined]
        except (RuntimeError, ValueError):
            return _fallback_estimate(text)
    return _fallback_estimate(text)


__all__ = ["estimate_tokens", "_fallback_estimate"]
