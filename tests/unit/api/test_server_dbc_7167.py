"""DbC tests for the API server contract gaps in issue #7167.

Covers the remaining sub-defects:

* D2 — CORS origins validated at startup (fail-fast), not mid-middleware-setup.
* D4 — API-key ``name`` length invariant enforced at the persistence boundary
  (defense-in-depth beyond the single Pydantic layer).
"""

import os
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit


# ── D2: CORS validated at startup ──────────────────────────────────────


def test_validate_cors_at_startup_rejects_wildcard() -> None:
    """A wildcard CORS origin with credentials must fail fast at startup."""
    from src.api.server import _validate_cors_at_startup

    with (
        patch.dict(os.environ, {"CORS_ORIGINS": "https://a.com,*"}),
        pytest.raises(ValueError, match=r"CORS_ORIGINS must not contain"),
    ):
        _validate_cors_at_startup()


def test_validate_cors_at_startup_returns_origins_when_valid() -> None:
    """A valid explicit allowlist passes and is returned."""
    from src.api.server import _validate_cors_at_startup

    with patch.dict(os.environ, {"CORS_ORIGINS": "https://a.com, https://b.com"}):
        origins = _validate_cors_at_startup()

    assert origins == ["https://a.com", "https://b.com"]


# ── D4: API-key name length invariant at the persistence boundary ──────


def test_api_key_name_max_length_is_single_source_of_truth() -> None:
    """The model column, DB CHECK, and Pydantic Field share one length bound."""
    from src.api.auth.models import API_KEY_NAME_MAX_LENGTH, APIKey, APIKeyCreate

    # Pydantic Field upper bound matches the shared constant.
    field = APIKeyCreate.model_fields["name"]
    constraints = getattr(field, "metadata", [])
    max_lengths = [
        getattr(c, "max_length", None)
        for c in constraints
        if getattr(c, "max_length", None) is not None
    ]
    assert API_KEY_NAME_MAX_LENGTH in max_lengths

    # SQLAlchemy column type length matches.
    assert APIKey.__table__.c.name.type.length == API_KEY_NAME_MAX_LENGTH

    # A DB-level CHECK constraint on name length exists (defense in depth).
    check_names = {
        c.name
        for c in APIKey.__table__.constraints
        if c.__class__.__name__ == "CheckConstraint"
    }
    assert "api_key_name_length" in check_names


def test_api_key_create_rejects_overlong_name() -> None:
    """The Pydantic layer rejects a name longer than the shared bound."""
    from pydantic import ValidationError

    from src.api.auth.models import API_KEY_NAME_MAX_LENGTH, APIKeyCreate

    with pytest.raises(ValidationError):
        APIKeyCreate(name="x" * (API_KEY_NAME_MAX_LENGTH + 1))


def test_api_key_create_rejects_empty_name() -> None:
    """The Pydantic layer rejects an empty name (min_length=1)."""
    from pydantic import ValidationError

    from src.api.auth.models import APIKeyCreate

    with pytest.raises(ValidationError):
        APIKeyCreate(name="")
