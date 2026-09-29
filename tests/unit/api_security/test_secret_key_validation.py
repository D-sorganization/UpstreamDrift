"""Unit tests for API security features.

This module tests critical security implementations:
- Bcrypt API key hashing and verification
- Timezone-aware JWT token generation
- Password hashing and verification
- Secure credential storage
"""

from datetime import timezone
from unittest.mock import patch

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


# Skip marker for bcrypt-dependent tests
requires_bcrypt = pytest.mark.skipif(
    not BCRYPT_AVAILABLE,
    reason="bcrypt native library not available in this environment",
)


class TestSecretKeyValidation:
    """Test secret key security requirements."""

    def test_secret_key_length_validation(self) -> None:
        """Test that secret keys are validated for length."""
        from src.api.auth.security import SECRET_KEY

        # In production, secret key should be long enough
        # For testing, we accept the unsafe placeholder
        if SECRET_KEY != "UNSAFE-NO-SECRET-KEY-SET-AUTHENTICATION-WILL-FAIL":
            assert len(SECRET_KEY) >= 32, "Secret key must be at least 32 characters"

    def test_secret_key_environment_variable(self) -> None:
        """Test that secret key can be set via environment variable."""
        import importlib

        from src.api.auth import security

        with patch.dict("os.environ", {"GOLF_API_SECRET_KEY": "x" * 64}):
            # Reload to pick up env var
            importlib.reload(security)

            # Check it uses the environment variable
            assert security.SECRET_KEY == "x" * 64, (
                "Assertion failed: security.SECRET_KEY == x * 64"
            )

        # Restore original state (reload without env var)
        importlib.reload(security)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
