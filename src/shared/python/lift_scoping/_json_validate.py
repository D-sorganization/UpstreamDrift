"""JSON Schema validation (minimal duplicate of other scoping packages)."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from jsonschema import Draft202012Validator


def validate_payload(instance: Any, schema: Mapping[str, Any]) -> None:
    Draft202012Validator(dict(schema)).validate(instance)
