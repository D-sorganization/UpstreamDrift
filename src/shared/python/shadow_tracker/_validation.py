"""Internal validation helpers for Shadow Tracker contracts."""

from __future__ import annotations

import math
from typing import Any
import urllib.parse

SOURCE_SCHEMA_VERSION = "shadow-tracker/source/1.0.0"
FRAME_SCHEMA_VERSION = "shadow-tracker/frame/1.0.0"
MASK_SCHEMA_VERSION = "shadow-tracker/mask/1.0.0"

_HEX_DIGITS = frozenset("0123456789abcdef")
_ALLOWED_URI_SCHEMES = frozenset(("https", "http", "urn"))


def check_str(val: object, field_name: str) -> str:
    """Validate that val is a nonempty trimmed string."""
    if not isinstance(val, str):
        raise TypeError(f"{field_name} must be a str, got {type(val).__name__}")
    if not val:
        raise ValueError(f"{field_name} cannot be empty")
    if val.strip() != val:
        raise ValueError(f"{field_name} must be trimmed, got {val!r}")
    return val


def check_id(val: object, field_name: str) -> str:
    """Validate an identifier field."""
    return check_str(val, field_name)


def check_sha256(val: object, field_name: str) -> str:
    """Validate that val is exactly 64 lowercase hex characters."""
    s = check_str(val, field_name)
    if len(s) != 64 or not all(c in _HEX_DIGITS for c in s):
        raise ValueError(
            f"{field_name} must be exactly 64 lowercase hex characters, got {s!r}"
        )
    return s


def check_int(val: object, field_name: str) -> int:
    """Validate signed int, rejecting booleans and numeric strings."""
    if isinstance(val, bool) or not isinstance(val, int):
        raise TypeError(f"{field_name} must be an int, got {type(val).__name__}")
    return val


def check_pos_int(val: object, field_name: str) -> int:
    """Validate positive int, rejecting booleans and numeric strings."""
    i = check_int(val, field_name)
    if i <= 0:
        raise ValueError(f"{field_name} must be positive, got {i}")
    return i


def check_uri(val: object, field_name: str = "source_uri") -> str:
    """Validate URI locator, rejecting local filesystem paths and unpermitted schemes."""
    uri = check_str(val, field_name)
    # Reject local filesystem paths
    if uri.startswith(("file:", "/", "\\")) or (len(uri) >= 2 and uri[1] == ":"):
        raise ValueError(f"{field_name} cannot be a local filesystem path: {uri!r}")

    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme not in _ALLOWED_URI_SCHEMES:
        raise ValueError(
            f"{field_name} scheme must be one of {sorted(_ALLOWED_URI_SCHEMES)}, got {parsed.scheme!r}"
        )
    return uri


def check_payload_keys(
    payload: dict[str, Any], allowed_keys: set[str] | frozenset[str]
) -> None:
    """Reject unexpected keys during deserialization."""
    extra = set(payload.keys()) - set(allowed_keys)
    if extra:
        raise ValueError(f"Unknown fields rejected: {sorted(extra)}")
