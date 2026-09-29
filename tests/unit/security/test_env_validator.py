"""Tests for src.shared.python.security.env_validator (Issues #1949, #1744)."""

from __future__ import annotations

import secrets

import pytest
from src.shared.python.core.error_utils import (
    EnvironmentError as EnvironmentValidationError,
)
from src.shared.python.security.env_validator import (
    KNOWN_ADMIN_PASSWORD_PLACEHOLDERS,
    KNOWN_SECRET_KEY_PLACEHOLDERS,
    generate_secure_key_command,
    validate_admin_password_strength,
    validate_secret_key_strength,
)

# ---------------------------------------------------------------------------
# validate_secret_key_strength
# ---------------------------------------------------------------------------


class TestValidateSecretKeyStrength:
    def test_strong_key_passes(self) -> None:
        key = secrets.token_urlsafe(64)
        validate_secret_key_strength(key)  # should not raise

    def test_empty_key_raises(self) -> None:
        with pytest.raises(EnvironmentValidationError):
            validate_secret_key_strength("")

    def test_short_key_raises(self) -> None:
        with pytest.raises(EnvironmentValidationError, match="too short"):
            validate_secret_key_strength("x" * 10)

    def test_custom_min_length(self) -> None:
        # 32-char key should pass with min_length=32
        key = "a" * 32
        validate_secret_key_strength(key, min_length=32)  # should not raise

    def test_unsafe_placeholder_raises(self) -> None:
        placeholder = "UNSAFE-NO-SECRET-KEY-SET-AUTHENTICATION-WILL-FAIL"
        with pytest.raises(EnvironmentValidationError):
            validate_secret_key_strength(placeholder)

    def test_key_exactly_at_min_length_passes(self) -> None:
        key = "a" * 64
        validate_secret_key_strength(key, min_length=64)  # should not raise

    def test_key_one_below_min_raises(self) -> None:
        key = "a" * 63
        with pytest.raises(EnvironmentValidationError):
            validate_secret_key_strength(key, min_length=64)


# ---------------------------------------------------------------------------
# generate_secure_key_command
# ---------------------------------------------------------------------------


class TestGenerateSecureKeyCommand:
    def test_env_validator_returns_string(self) -> None:
        result = generate_secure_key_command()
        assert isinstance(result, str)

    def test_env_validator_non_empty(self) -> None:
        result = generate_secure_key_command()
        assert len(result) > 0

    def test_contains_command_like_text(self) -> None:
        result = generate_secure_key_command()
        # Should contain some reference to generating a key
        lower = result.lower()
        assert any(
            word in lower for word in ["python", "token", "secret", "key", "openssl"]
        )


# ---------------------------------------------------------------------------
# Placeholder rejection (issue #5920)
# ---------------------------------------------------------------------------


class TestSecretKeyPlaceholderRejection:
    """Refuse the literal placeholders shipped in `.env.example`."""

    @pytest.mark.parametrize(
        "placeholder",
        sorted(KNOWN_SECRET_KEY_PLACEHOLDERS),
    )
    def test_known_secret_key_placeholder_raises(self, placeholder: str) -> None:
        with pytest.raises(EnvironmentValidationError):
            validate_secret_key_strength(placeholder)

    def test_dotenv_example_secret_key_placeholder_is_recognised(self) -> None:
        # Sanity-check the literal string from `.env.example` is covered.
        assert "generate_a_random_string_here" in KNOWN_SECRET_KEY_PLACEHOLDERS

    def test_dotenv_example_x_api_key_placeholder_is_recognised(self) -> None:
        assert "generate_another_random_string_here" in KNOWN_SECRET_KEY_PLACEHOLDERS


class TestValidateAdminPasswordStrength:
    def test_strong_password_passes(self) -> None:
        validate_admin_password_strength(secrets.token_urlsafe(16))

    def test_empty_raises(self) -> None:
        with pytest.raises(EnvironmentValidationError):
            validate_admin_password_strength("")

    def test_short_raises(self) -> None:
        with pytest.raises(EnvironmentValidationError, match="too short"):
            validate_admin_password_strength("short")

    @pytest.mark.parametrize(
        "placeholder",
        sorted(KNOWN_ADMIN_PASSWORD_PLACEHOLDERS),
    )
    def test_known_admin_password_placeholder_raises(self, placeholder: str) -> None:
        with pytest.raises(EnvironmentValidationError):
            validate_admin_password_strength(placeholder)

    def test_dotenv_example_admin_password_is_recognised(self) -> None:
        # The literal default shipped in `.env.example` must be rejected.
        assert "change_me_in_production" in KNOWN_ADMIN_PASSWORD_PLACEHOLDERS

    def test_custom_min_length(self) -> None:
        validate_admin_password_strength("a" * 8, min_length=8)
