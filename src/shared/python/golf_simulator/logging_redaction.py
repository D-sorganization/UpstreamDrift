"""Structured logging and secret redaction filter (GS-08, #10197).

Ensures tokens, secrets, passwords, and API keys are redacted from logs and diagnostics.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Mapping

# Regex patterns matching common secret indicators
_BEARER_PATTERN = re.compile(r"Bearer\s+([A-Za-z0-9_\-\.\~]+)", re.IGNORECASE)
_KEY_VALUE_SECRET_PATTERN = re.compile(
    r"""(?i)\b(api[_-]?key|token|auth[_-]?token|secret|password)\s*[:=]\s*(['"]?)([^'"\s,;]+)\2"""
)
_SENSITIVE_KEY_SUBSTRINGS = (
    "token",
    "secret",
    "password",
    "api_key",
    "auth_token",
    "credential",
)


def redact_text(text: str) -> str:
    """Scrub secrets, tokens, and authorization values from string."""
    if not isinstance(text, str):
        return text

    # Mask Bearer tokens
    cleaned = _BEARER_PATTERN.sub("Bearer [REDACTED]", text)
    # Mask key=value and key: value pairs
    cleaned = _KEY_VALUE_SECRET_PATTERN.sub(r"\1=[REDACTED]", cleaned)
    return cleaned


def redact_mapping(data: Mapping[str, Any]) -> dict[str, Any]:
    """Recursively scrub sensitive keys in a dictionary."""
    result: dict[str, Any] = {}
    for k, v in data.items():
        k_str = str(k).lower()
        if any(sub in k_str for sub in _SENSITIVE_KEY_SUBSTRINGS):
            result[k] = "[REDACTED]"
        elif isinstance(v, Mapping):
            result[k] = redact_mapping(v)
        elif isinstance(v, list):
            result[k] = [
                redact_mapping(item) if isinstance(item, Mapping) else item
                for item in v
            ]
        elif isinstance(v, str):
            result[k] = redact_text(v)
        else:
            result[k] = v
    return result


class SecretRedactionFilter(logging.Filter):
    """Logging filter that scrubs secrets from LogRecord messages and arguments."""

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            msg_str = record.getMessage()
        except Exception:
            msg_str = str(record.msg)
        record.msg = redact_text(msg_str)
        record.args = ()
        return True
