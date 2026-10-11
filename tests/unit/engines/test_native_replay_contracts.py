"""Shared native replay contract primitives."""

import pytest

from src.engines.native_replay_contracts import require_sha256_digest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("value", ["a" * 64, "0123456789abcdef" * 4])
def test_require_sha256_digest_accepts_lowercase_hex(value: str) -> None:
    require_sha256_digest(value, "fixture digest")


@pytest.mark.parametrize("value", [None, "A" * 64, "g" * 64, "a" * 63])
def test_require_sha256_digest_rejects_noncanonical_values(value: object) -> None:
    with pytest.raises(ValueError, match="lowercase SHA-256 digest"):
        require_sha256_digest(value, "fixture digest")
