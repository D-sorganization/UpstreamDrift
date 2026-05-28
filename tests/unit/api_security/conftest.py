"""Shared fixtures and markers for api_security tests.

The ``requires_bcrypt`` marker is defined here once and imported by each
test module so there is a single canonical definition (issue #6095 DRY).
"""

from __future__ import annotations

import pytest

# Check if bcrypt is available and working.
# bcrypt can fail to load on some CI environments due to missing native
# libraries, so we test it at import time rather than just checking the spec.
try:
    import bcrypt as _bcrypt_lib

    _bcrypt_lib.hashpw(b"test", _bcrypt_lib.gensalt())
    _BCRYPT_AVAILABLE = True
except (ImportError, ModuleNotFoundError):
    _BCRYPT_AVAILABLE = False
except Exception:  # noqa: BLE001
    _BCRYPT_AVAILABLE = False

requires_bcrypt = pytest.mark.skipif(
    not _BCRYPT_AVAILABLE,
    reason="bcrypt native library not available in this environment",
)
