"""Unit tests for structured logging and secret/token redaction (GS-08, #10197).

Follows TDD, DbC, Law of Demeter, and DRY.
"""

from __future__ import annotations

import io
import logging
import pytest

from src.shared.python.golf_simulator.logging_redaction import (
    SecretRedactionFilter,
    redact_mapping,
    redact_text,
)

pytestmark = pytest.mark.unit


def test_redact_text_bearer_token() -> None:
    raw = "Authorization: Bearer secret_jwt_token_12345 in request header"
    clean = redact_text(raw)
    assert "secret_jwt_token_12345" not in clean
    assert "Bearer [REDACTED]" in clean


def test_redact_text_various_secrets() -> None:
    raw = (
        "Connecting with token='xyz987', password='mypassword', and api_key=key_abcdef"
    )
    clean = redact_text(raw)
    assert "xyz987" not in clean
    assert "mypassword" not in clean
    assert "key_abcdef" not in clean
    assert "[REDACTED]" in clean


def test_redact_mapping() -> None:
    data = {
        "user": "alice",
        "auth_token": "secret_abc_123",
        "nested": {
            "api_key": "key_456",
            "safe_val": 42,
        },
    }
    redacted = redact_mapping(data)
    assert redacted["user"] == "alice"
    assert redacted["auth_token"] == "[REDACTED]"
    assert redacted["nested"]["api_key"] == "[REDACTED]"
    assert redacted["nested"]["safe_val"] == 42


def test_logging_filter_scrubs_records() -> None:
    logger = logging.getLogger("test.redaction.logger")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(SecretRedactionFilter())
    handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logger.addHandler(handler)

    try:
        logger.info(
            "Remote connection established with Bearer eyJhbGciOiJIUzI1NiIsIn..."
        )
        logger.info("Config loaded with api_key=supersecretkey")
        log_content = stream.getvalue()

        assert "eyJhbGciOiJIUzI1NiIsIn" not in log_content
        assert "supersecretkey" not in log_content
        assert "Bearer [REDACTED]" in log_content
        assert "api_key=[REDACTED]" in log_content
    finally:
        logger.removeHandler(handler)


def test_logging_filter_scrubs_interpolated_args() -> None:
    logger = logging.getLogger("test.redaction.interpolated")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.addFilter(SecretRedactionFilter())
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)

    try:
        # Pass sensitive token as positional formatting argument
        logger.info("Connecting using %s credential", "token=sensitive123")
        content = stream.getvalue()
        assert "sensitive123" not in content
        assert "token=[REDACTED]" in content
    finally:
        logger.removeHandler(handler)
