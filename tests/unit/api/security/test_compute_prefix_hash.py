"""Tests for security - Authentication and authorization utilities.

These tests verify the security module using Design by Contract principles.
"""

import os
from datetime import timedelta
from unittest.mock import MagicMock, patch

import pytest

# Configure async tests to use asyncio backend only
pytestmark = pytest.mark.anyio


@pytest.fixture(scope="module")
def anyio_backend() -> str:
    """Use asyncio backend only (trio not installed)."""
    return "asyncio"


class TestComputePrefixHash:
    """Tests for compute_prefix_hash function."""

    def test_security_returns_string(self) -> None:
        """Test that compute_prefix_hash returns a string."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import compute_prefix_hash

            result = compute_prefix_hash("gms_test")
            assert isinstance(result, str)

    def test_returns_hex_string(self) -> None:
        """Test that result is a valid hex string."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import compute_prefix_hash

            result = compute_prefix_hash("gms_test")
            # SHA256 produces 64 hex characters
            assert len(result) == 64
            int(result, 16)  # Should not raise

    def test_same_input_same_output(self) -> None:
        """Test deterministic output."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import compute_prefix_hash

            hash1 = compute_prefix_hash("gms_abcd")
            hash2 = compute_prefix_hash("gms_abcd")
            assert hash1 == hash2

    def test_different_input_different_output(self) -> None:
        """Test different inputs produce different outputs."""
        with patch.dict(
            os.environ, {"GOLF_API_SECRET_KEY": "test-secret-key-32chars-long!!"}
        ):
            from src.api.auth.security import compute_prefix_hash

            hash1 = compute_prefix_hash("prefix_a")
            hash2 = compute_prefix_hash("prefix_b")
            assert hash1 != hash2
