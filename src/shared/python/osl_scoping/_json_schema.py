"""Minimal JSON Schema validation helpers (OSL scoping)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator


def validate_payload(instance: Any, schema: Mapping[str, Any]) -> None:
    """Raise :exc:`jsonschema.ValidationError` when *instance* violates *schema*."""
    Draft202012Validator(dict(schema)).validate(instance)
