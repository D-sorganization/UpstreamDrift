"""Internal validation helpers for Shadow Tracker contracts."""

from __future__ import annotations

import math
from typing import Any
import urllib.parse

SOURCE_SCHEMA_VERSION = "shadow-tracker/source/1.0.0"
FRAME_SCHEMA_VERSION = "shadow-tracker/frame/1.0.0"
MASK_SCHEMA_VERSION = "shadow-tracker/mask/1.0.0"
SHOT_SCHEMA_VERSION = "shadow-tracker/shot/1.0.0"
FRAME_OBSERVATION_SCHEMA_VERSION = "shadow-tracker/frame-observation/1.0.0"
CAMERA_TRACK_SCHEMA_VERSION = "shadow-tracker/camera-track/1.0.0"
SUBJECT_BINDING_SCHEMA_VERSION = "shadow-tracker/subject-binding/1.0.0"
FIT_REQUEST_SCHEMA_VERSION = "shadow-tracker/fit-request/1.0.0"
REPLAY_AUDIT_SCHEMA_VERSION = "shadow-tracker/replay-audit/1.0.0"
CANDIDATE_RESULT_SCHEMA_VERSION = "shadow-tracker/candidate-result/1.0.0"
RESULT_BUNDLE_SCHEMA_VERSION = "shadow-tracker/result-bundle/1.0.0"

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


def check_float(val: object, field_name: str) -> float:
    """Validate finite float, rejecting booleans, ints, and numeric strings."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise TypeError(f"{field_name} must be a float, got {type(val).__name__}")
    f = float(val)
    if not math.isfinite(f):
        raise ValueError(f"{field_name} must be finite, got {f}")
    return f


def check_strict_float(val: object, field_name: str) -> float:
    """Validate finite float strictly (rejecting int coercion, booleans, and strings)."""
    if isinstance(val, bool) or not isinstance(val, float):
        raise TypeError(f"{field_name} must be a float, got {type(val).__name__}")
    if not math.isfinite(val):
        raise ValueError(f"{field_name} must be finite, got {val}")
    return val


def check_pos_float(val: object, field_name: str) -> float:
    """Validate positive finite float."""
    f = check_float(val, field_name)
    if f <= 0.0:
        raise ValueError(f"{field_name} must be positive, got {f}")
    return f


def check_nonneg_float(val: object, field_name: str) -> float:
    """Validate non-negative finite float."""
    f = check_float(val, field_name)
    if f < 0.0:
        raise ValueError(f"{field_name} must be non-negative, got {f}")
    return f


def check_optional_float(val: object, field_name: str) -> float | None:
    """Validate optional finite float."""
    if val is None:
        return None
    return check_strict_float(val, field_name)


def check_bool(val: object, field_name: str) -> bool:
    """Validate boolean."""
    if not isinstance(val, bool):
        raise TypeError(f"{field_name} must be a bool, got {type(val).__name__}")
    return val


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


def check_schema_version(val: object, expected: str) -> str:
    """Validate that schema_version is a string matching expected exactly."""
    if not isinstance(val, str):
        raise TypeError(f"schema_version must be a str, got {type(val).__name__}")
    if val != expected:
        raise ValueError(f"schema_version must be exactly {expected!r}, got {val!r}")
    return val


def check_payload_keys(
    payload: object, allowed_keys: set[str] | frozenset[str]
) -> dict[str, Any]:
    """Validate payload container type, missing keys, and unexpected keys during deserialization."""
    if not isinstance(payload, dict):
        raise TypeError(f"Payload must be a dict, got {type(payload).__name__}")

    # Check for unknown keys
    extra = set(payload.keys()) - set(allowed_keys)
    if extra:
        raise ValueError(f"Unknown fields rejected: {sorted(extra)}")

    # Check for missing required keys
    missing = set(allowed_keys) - set(payload.keys())
    if missing:
        raise ValueError(f"Missing required fields: {sorted(missing)}")

    return payload
