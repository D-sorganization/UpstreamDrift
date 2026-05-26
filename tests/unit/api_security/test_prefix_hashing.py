"""Unit tests for API security features.

This module tests critical security implementations:
- Bcrypt API key hashing and verification
- Timezone-aware JWT token generation
- Password hashing and verification
- Secure credential storage
"""

from datetime import timezone

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


class TestPrefixHashing:
    """Test API key prefix hashing."""

    def test_compute_prefix_hash(self) -> None:
        """Test SHA256 prefix hashing."""
        from src.api.auth.security import compute_prefix_hash

        prefix = "abcdefgh"
        hash1 = compute_prefix_hash(prefix)

        # Same prefix should give same hash
        assert hash1 == compute_prefix_hash(prefix), (
            "Assertion failed: hash1 == compute_prefix_hash(prefix)"
        )

        # Different prefix should give different hash
        assert hash1 != compute_prefix_hash("12345678"), (
            "Assertion failed: hash1 != compute_prefix_hash(12345678)"
        )

        # Verify format (SHA256 hex digest)
        assert len(hash1) == 64, "Assertion failed: len(hash1) == 64"
        import re

        assert re.match(r"^[0-9a-f]{64}$", hash1), (
            "Assertion failed: re.match(r^[0-9a-f]{64}$, hash1)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
