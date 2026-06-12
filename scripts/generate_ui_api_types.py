#!/usr/bin/env python3
"""Generate TypeScript API types from the FastAPI OpenAPI contract.

The web UI must never hand-write a payload interface that already exists as a
Pydantic model in ``src/api/models/`` (issue #7447). This script dumps the
OpenAPI schema from the FastAPI app (``src/api/server.py``) and emits a
deterministic TypeScript declaration file at ``ui/src/api/generated/types.ts``
(checked in). A pytest freshness gate
(``tests/api/test_generated_ui_api_types.py``) regenerates the file and diffs
it against the committed copy so it can never go stale.

Why a self-contained emitter instead of the ``openapi-typescript`` npm tool:
the freshness gate runs inside the *Python* CI test job, which has no
``node_modules``. Emitting TypeScript here keeps generation and verification
in one toolchain with zero new dependencies. The schema *source* is still the
OpenAPI contract, so route response models are covered, not just bare models.

Determinism contract:
    - Schema names are emitted in sorted order.
    - Property order follows the Pydantic field definition order (stable).
    - The banner contains no timestamps; output depends only on the contract.

Usage:
    python scripts/generate_ui_api_types.py            # rewrite the file
    python scripts/generate_ui_api_types.py --check    # exit 1 if stale
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_PATH = REPO_ROOT / "ui" / "src" / "api" / "generated" / "types.ts"

_BANNER = """\
/**
 * AUTO-GENERATED FILE - DO NOT EDIT BY HAND.
 *
 * TypeScript mirror of the API contract defined by the Pydantic models in
 * src/api/models/ and the FastAPI route response models (src/api/server.py).
 *
 * Regenerate with:
 *     python scripts/generate_ui_api_types.py
 *
 * Freshness is enforced by tests/api/test_generated_ui_api_types.py.
 * See issue #7447.
 */
"""

# TypeScript reserved words that cannot be used as bare interface names.
_TS_RESERVED = frozenset({"string", "number", "boolean", "object", "any"})


def _bootstrap_sys_path() -> None:
    """Make ``src.api.server`` and its sibling packages importable.

    Mirrors the layout CI uses (editable install): the repo root for
    ``src.*`` imports, plus ``src`` and ``src/shared/python`` for packages
    that import themselves by their top-level name (e.g. ``bunkershot3d``,
    ``humanoid_character_builder``).
    """
    for path in (
        REPO_ROOT,
        REPO_ROOT / "src",
        REPO_ROOT / "src" / "shared" / "python",
    ):
        str_path = str(path)
        if str_path not in sys.path:
            sys.path.insert(0, str_path)


def load_openapi_schema() -> dict[str, Any]:
    """Import the FastAPI app and return its OpenAPI schema.

    Returns:
        The OpenAPI schema dictionary from ``app.openapi()``.

    Raises:
        ImportError: If the server app (or one of its mandatory route
            modules) cannot be imported in this environment.
    """
    _bootstrap_sys_path()
    from src.api.server import app

    schema: dict[str, Any] = app.openapi()
    return schema


def _ref_name(ref: str) -> str:
    """Resolve a ``#/components/schemas/X`` reference to its TS type name."""
    prefix = "#/components/schemas/"
    if not ref.startswith(prefix):
        raise ValueError(f"Unsupported $ref target: {ref}")
    return _sanitize_name(ref[len(prefix) :])


def _sanitize_name(name: str) -> str:
    """Convert an OpenAPI schema name into a valid TS identifier."""
    cleaned = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in name)
    if not cleaned or cleaned[0].isdigit() or cleaned in _TS_RESERVED:
        cleaned = f"Schema_{cleaned}"
    return cleaned


def _union(parts: list[str]) -> str:
    """Join type expressions into a deduplicated TS union."""
    seen: list[str] = []
    for part in parts:
        if part not in seen:
            seen.append(part)
    return " | ".join(seen) if seen else "unknown"


def _wrap_for_suffix(expr: str) -> str:
    """Parenthesize a type expression when a ``[]`` suffix needs it."""
    needs_parens = ("|" in expr) or ("&" in expr) or expr.startswith("{")
    return f"({expr})" if needs_parens else expr


def ts_type(schema: Any) -> str:
    """Convert a JSON Schema fragment (OpenAPI 3.1) to a TS type expression.

    Handles the subset that Pydantic v2 / FastAPI emit: ``$ref``, ``enum``,
    ``const``, ``anyOf``/``oneOf``/``allOf``, nullable unions, arrays
    (including ``prefixItems`` tuples), objects with ``properties`` and/or
    ``additionalProperties``, and the JSON primitive types.
    """
    if schema is True or schema == {}:
        return "unknown"
    if not isinstance(schema, dict):
        return "unknown"

    if "$ref" in schema:
        return _ref_name(schema["$ref"])

    if "const" in schema:
        return json.dumps(schema["const"])

    if "enum" in schema:
        return _union([json.dumps(v) for v in schema["enum"]])

    for combinator in ("anyOf", "oneOf"):
        if combinator in schema:
            return _union([ts_type(s) for s in schema[combinator]])

    if "allOf" in schema:
        parts = [_wrap_for_suffix(ts_type(s)) for s in schema["allOf"]]
        return " & ".join(parts) if parts else "unknown"

    schema_type = schema.get("type")
    if isinstance(schema_type, list):
        return _union([ts_type({**schema, "type": t}) for t in schema_type])

    if schema_type == "null":
        return "null"
    if schema_type == "string":
        return "string"
    if schema_type in ("integer", "number"):
        return "number"
    if schema_type == "boolean":
        return "boolean"
    if schema_type == "array":
        if "prefixItems" in schema:
            elements = ", ".join(ts_type(s) for s in schema["prefixItems"])
            return f"[{elements}]"
        return f"{_wrap_for_suffix(ts_type(schema.get('items', True)))}[]"
    if schema_type == "object" or "properties" in schema:
        return _object_literal(schema)

    return "unknown"


def _is_required(name: str, prop_schema: Any, required: set[str]) -> bool:
    """Decide whether a property is required in the emitted TS.

    Fields with a ``default`` are emitted as required: the server always
    serializes them in responses (FastAPI does not exclude unset fields by
    default), matching the ``defaultNonNullable`` behavior of
    ``openapi-typescript``.
    """
    if name in required:
        return True
    return isinstance(prop_schema, dict) and "default" in prop_schema


def _property_lines(schema: dict[str, Any], indent: str) -> list[str]:
    """Emit the property declarations for an object schema."""
    properties: dict[str, Any] = schema.get("properties", {})
    required = set(schema.get("required", []))
    lines: list[str] = []
    for prop_name, prop_schema in properties.items():
        description = (
            prop_schema.get("description") if isinstance(prop_schema, dict) else None
        )
        if description:
            lines.append(f"{indent}/** {_jsdoc_safe(description)} */")
        optional = "" if _is_required(prop_name, prop_schema, required) else "?"
        key = json.dumps(prop_name) if not prop_name.isidentifier() else prop_name
        lines.append(f"{indent}{key}{optional}: {ts_type(prop_schema)};")
    additional = schema.get("additionalProperties")
    if additional is not None and additional is not False and not properties:
        # Pure map type is rendered by _object_literal as Record<...>;
        # reaching here means properties+additionalProperties coexist.
        pass
    if additional is not None and additional is not False and properties:
        lines.append(f"{indent}[key: string]: unknown;")
    return lines


def _object_literal(schema: dict[str, Any]) -> str:
    """Render an inline object type (used for nested anonymous objects)."""
    properties = schema.get("properties", {})
    additional = schema.get("additionalProperties")
    if not properties:
        if additional is None or additional is True or additional == {}:
            return "Record<string, unknown>"
        if additional is False:
            return "Record<string, never>"
        return f"Record<string, {ts_type(additional)}>"
    lines = _property_lines(schema, indent="  ")
    body = "\n".join(lines)
    return "{\n" + body + "\n}"


def _jsdoc_safe(text: str) -> str:
    """Collapse a description to one JSDoc-safe line."""
    return " ".join(text.split()).replace("*/", "*\\/")


def _emit_declaration(name: str, schema: dict[str, Any]) -> str:
    """Emit one top-level ``export interface``/``export type`` declaration."""
    ts_name = _sanitize_name(name)
    doc_lines: list[str] = []
    description = schema.get("description")
    if description:
        doc_lines.append("/**")
        doc_lines.append(f" * {_jsdoc_safe(description)}")
        doc_lines.append(" */")
    doc = "\n".join(doc_lines)

    is_plain_object = (
        schema.get("type") == "object" or "properties" in schema
    ) and not any(k in schema for k in ("anyOf", "oneOf", "allOf", "enum", "$ref"))
    if is_plain_object and schema.get("properties"):
        body = "\n".join(_property_lines(schema, indent="  "))
        declaration = f"export interface {ts_name} {{\n{body}\n}}"
    else:
        declaration = f"export type {ts_name} = {ts_type(schema)};"
    return f"{doc}\n{declaration}" if doc else declaration


def generate_types_source(schema: dict[str, Any] | None = None) -> str:
    """Generate the full TypeScript source for the generated types file.

    Args:
        schema: Optional pre-loaded OpenAPI schema (for tests). When omitted,
            the FastAPI app is imported and queried.

    Returns:
        Complete file contents, LF-terminated, deterministic for a given
        contract (no timestamps, sorted schema names).
    """
    if schema is None:
        schema = load_openapi_schema()
    components: dict[str, Any] = schema.get("components", {}).get("schemas", {})
    declarations = [
        _emit_declaration(name, components[name]) for name in sorted(components)
    ]
    return _BANNER + "\n" + "\n\n".join(declarations) + "\n"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point. Returns process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check",
        action="store_true",
        help="Do not write; exit 1 if the committed file is stale.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_PATH,
        help=f"Output path (default: {OUTPUT_PATH})",
    )
    args = parser.parse_args(argv)

    source = generate_types_source()

    if args.check:
        committed = args.out.read_text(encoding="utf-8") if args.out.exists() else ""
        if committed != source:
            sys.stderr.write(
                f"STALE: {args.out} does not match the API contract.\n"
                "Run: python scripts/generate_ui_api_types.py\n"
            )
            return 1
        sys.stdout.write(f"OK: {args.out} is up to date.\n")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(source, encoding="utf-8", newline="\n")
    sys.stdout.write(f"Wrote {args.out} ({len(source.splitlines())} lines).\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
