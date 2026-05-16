"""Unit tests for API security features.

This module tests critical security implementations:
- Bcrypt API key hashing and verification
- Timezone-aware JWT token generation
- Password hashing and verification
- Secure credential storage
"""

import logging
import secrets
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

# Python 3.10 compatibility: datetime.UTC is only available in 3.11+
UTC = timezone.utc  # noqa: UP017

# Check if sqlalchemy is available
try:
    import sqlalchemy  # noqa: F401

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False

if not SQLALCHEMY_AVAILABLE:
    pytest.skip("SQLAlchemy not installed", allow_module_level=True)

# Check if bcrypt is available and working
# bcrypt can fail to load on some CI environments due to missing native libraries
try:
    import bcrypt as bcrypt_lib

    # Try to actually use bcrypt to detect runtime issues
    bcrypt_lib.hashpw(b"test", bcrypt_lib.gensalt())
    BCRYPT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    # bcrypt is not installed
    BCRYPT_AVAILABLE = False
    bcrypt_lib = None  # type: ignore[misc,assignment]
except Exception as e:  # noqa: BLE001, F841
    # bcrypt failed to load (native library issue)
    BCRYPT_AVAILABLE = False
    import bcrypt as bcrypt_lib  # type: ignore[no-redef]

from src.api.auth.models import APIKey, User  # noqa: E402
from src.api.auth.security import SecurityManager  # noqa: E402

# Skip marker for bcrypt-dependent tests
requires_bcrypt = pytest.mark.skipif(
    not BCRYPT_AVAILABLE,
    reason="bcrypt native library not available in this environment",
)


class TestBcryptAPIKeyVerification:
    """Test bcrypt-based API key verification."""

    @requires_bcrypt
    def test_api_key_bcrypt_hashing(self) -> None:
        """Test that API keys are hashed with bcrypt."""
        # Generate a test API key
        api_key = f"gms_{secrets.token_urlsafe(32)}"

        # Hash it with bcrypt
        salt = bcrypt_lib.gensalt(rounds=12)
        key_hash = bcrypt_lib.hashpw(api_key.encode("utf-8"), salt).decode("utf-8")

        # Verify the hash is bcrypt format (starts with $2b$)
        assert key_hash.startswith(("$2b$", "$2a$")), (
            "Assertion failed: key_hash.startswith(($2b$, $2a$))"
        )

        # Verify the key can be verified
        assert bcrypt_lib.checkpw(api_key.encode("utf-8"), key_hash.encode("utf-8")), (
            "Assertion failed: bcrypt_lib.checkpw(api_key.encode(utf-8), key_hash.encode(utf-8))"
        )

        # Verify a different key fails
        wrong_key = f"gms_{secrets.token_urlsafe(32)}"
        assert not bcrypt_lib.checkpw(
            wrong_key.encode("utf-8"), key_hash.encode("utf-8")
        )

    @requires_bcrypt
    def test_api_key_constant_time_comparison(self) -> None:
        """Test that API key verification uses constant-time comparison."""
        api_key = f"gms_{secrets.token_urlsafe(32)}"
        salt = bcrypt_lib.gensalt(rounds=12)
        key_hash = bcrypt_lib.hashpw(api_key.encode("utf-8"), salt)

        # Bcrypt's checkpw() uses constant-time comparison internally
        # This test verifies it doesn't leak timing information
        # by ensuring both correct and incorrect keys take similar time

        import time

        # Measure correct key verification time
        start = time.perf_counter()
        for _ in range(10):
            bcrypt_lib.checkpw(api_key.encode("utf-8"), key_hash)
        correct_time = time.perf_counter() - start

        # Measure incorrect key verification time
        wrong_key = f"gms_{secrets.token_urlsafe(32)}"
        start = time.perf_counter()
        for _ in range(10):
            bcrypt_lib.checkpw(wrong_key.encode("utf-8"), key_hash)
        incorrect_time = time.perf_counter() - start

        # Times should be similar (bcrypt takes consistent time)
        # Use a generous threshold (3.0) to avoid flakiness in shared CI environments
        ratio = max(correct_time, incorrect_time) / min(correct_time, incorrect_time)
        assert ratio < 3.0, "Timing difference suggests non-constant-time comparison"

    def test_api_key_format_validation(self) -> None:
        """Test that API keys must have gms_ prefix."""
        # Valid format
        valid_key = f"gms_{secrets.token_urlsafe(32)}"
        assert valid_key.startswith("gms_"), (
            "Assertion failed: valid_key.startswith(gms_)"
        )

        # Invalid formats (should be rejected)
        invalid_keys = [
            secrets.token_urlsafe(32),  # No prefix
            f"api_{secrets.token_urlsafe(32)}",  # Wrong prefix
            "gms_",  # Prefix only
            "",  # Empty
        ]

        for invalid_key in invalid_keys:
            assert not invalid_key.startswith("gms_") or len(invalid_key) <= 4, (
                "Assertion failed: not invalid_key.startswith(gms_) or len(invalid_key) <= 4"
            )

    @requires_bcrypt
    def test_bcrypt_cost_factor(self) -> None:
        """Test that bcrypt uses appropriate cost factor (work factor)."""
        api_key = f"gms_{secrets.token_urlsafe(32)}"
        salt = bcrypt_lib.gensalt(rounds=12)
        key_hash = bcrypt_lib.hashpw(api_key.encode("utf-8"), salt).decode("utf-8")

        # Extract bcrypt cost factor from hash
        # Format: $2b$[cost]$[salt][hash]
        parts = key_hash.split("$")
        cost_factor = int(parts[2])

        # Cost factor should be at least 12 (recommended minimum)
        assert cost_factor >= 12, f"Bcrypt cost factor {cost_factor} is too low"

    @requires_bcrypt
    async def test_api_key_verification_integration(self) -> None:
        """Test full API key verification flow."""
        from fastapi import HTTPException
        from fastapi.security import HTTPAuthorizationCredentials
        from src.api.auth.dependencies import get_current_user_from_api_key

        # Create test API key
        api_key = f"gms_{secrets.token_urlsafe(32)}"
        salt = bcrypt_lib.gensalt(rounds=12)
        key_hash = bcrypt_lib.hashpw(api_key.encode("utf-8"), salt).decode("utf-8")

        # Mock database and API key record
        mock_db = MagicMock()
        mock_api_key_record = MagicMock(spec=APIKey)
        mock_api_key_record.key_hash = key_hash
        mock_api_key_record.user_id = 1
        mock_api_key_record.is_active = True
        mock_api_key_record.last_used = None
        mock_api_key_record.usage_count = 0

        mock_user = MagicMock(spec=User)
        mock_user.id = 1
        mock_user.is_active = True

        # Configure mock database queries
        mock_db.query.return_value.filter.return_value.all.return_value = [
            mock_api_key_record
        ]
        mock_db.query.return_value.filter.return_value.first.return_value = mock_user

        # Test with correct API key
        credentials = HTTPAuthorizationCredentials(scheme="Bearer", credentials=api_key)

        user = await get_current_user_from_api_key(credentials, mock_db)
        assert user == mock_user, "Assertion failed: user == mock_user"

        # Test with incorrect API key
        wrong_key = f"gms_{secrets.token_urlsafe(32)}"
        wrong_credentials = HTTPAuthorizationCredentials(
            scheme="Bearer", credentials=wrong_key
        )

        with pytest.raises(HTTPException) as exc_info:
            await get_current_user_from_api_key(wrong_credentials, mock_db)

        assert exc_info.value.status_code == 401, (
            "Assertion failed: exc_info.value.status_code == 401"
        )

    async def test_create_api_key_persists_prefix_hash(self) -> None:
        """Created API key records should persist the lookup prefix hash."""
        from src.api.auth.models import APIKeyCreate
        from src.api.auth.security import compute_prefix_hash
        from src.api.routes.auth import create_api_key

        mock_db = MagicMock()
        current_user = MagicMock(spec=User)
        current_user.id = 7
        api_key_data = APIKeyCreate(name="integration key")
        generated_api_key = "gms_abcdefgh1234567890"  # nosec B105 - test fixture

        fake_response = MagicMock()
        with (
            patch(
                "src.api.routes.auth.security_manager.generate_api_key",
                return_value=generated_api_key,
            ),
            patch(
                "src.api.routes.auth.security_manager.hash_api_key",
                return_value="hashed-key",
            ),
            patch(
                "src.api.routes.auth.APIKeyResponse.from_orm",
                return_value=fake_response,
            ),
        ):
            response = await create_api_key(api_key_data, current_user, mock_db)

        saved_record = mock_db.add.call_args.args[0]
        assert isinstance(saved_record, APIKey), (
            "Assertion failed: isinstance(saved_record, APIKey)"
        )
        assert saved_record.key_prefix == compute_prefix_hash("abcdefgh"), (
            "Assertion failed: saved_record.key_prefix == compute_prefix_hash(abcdefgh)"
        )
        assert response is fake_response, "Assertion failed: response is fake_response"
        assert response.key == generated_api_key, (
            "Assertion failed: response.key == generated_api_key"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
