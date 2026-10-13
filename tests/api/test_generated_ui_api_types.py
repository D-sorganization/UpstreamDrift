"""Freshness gate for the generated TypeScript API types (issue #7447).

``ui/src/api/generated/types.ts`` is generated from the FastAPI OpenAPI
contract by ``scripts/generate_ui_api_types.py`` and checked in. These tests
regenerate the file content in-memory and diff it against the committed copy
so the generated contract can never silently go stale, and verify the
generator is deterministic (no timestamps, stable ordering).
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
GENERATOR_PATH = REPO_ROOT / "scripts" / "generate_ui_api_types.py"
GENERATED_PATH = REPO_ROOT / "ui" / "src" / "api" / "generated" / "types.ts"

pytestmark = pytest.mark.unit


def _load_generator() -> ModuleType:
    """Import the generator script as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(
        "generate_ui_api_types", GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("generate_ui_api_types", module)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def generator() -> ModuleType:
    return _load_generator()


@pytest.fixture(scope="module")
def openapi_schema(generator: ModuleType) -> dict:
    """Load the OpenAPI schema once; skip when the app cannot be imported."""
    try:
        return generator.load_openapi_schema()
    except ImportError as exc:  # pragma: no cover - slim local envs only
        pytest.skip(f"FastAPI app not importable in this environment: {exc}")


def test_generated_types_file_exists() -> None:
    assert GENERATED_PATH.exists(), (
        f"{GENERATED_PATH} is missing - run: python scripts/generate_ui_api_types.py"
    )


def test_generated_types_are_fresh(generator: ModuleType, openapi_schema: dict) -> None:
    """The committed types.ts must match the current API contract exactly."""
    expected = generator.generate_types_source(openapi_schema)
    committed = GENERATED_PATH.read_text(encoding="utf-8")
    assert committed == expected, (
        "ui/src/api/generated/types.ts is stale: the API contract "
        "(src/api/models/ or route response models) changed without "
        "regenerating the TypeScript types.\n"
        "Run: python scripts/generate_ui_api_types.py\n"
        "and commit the updated file."
    )


def test_generation_is_deterministic(
    generator: ModuleType, openapi_schema: dict
) -> None:
    """Two generations from the same contract must be byte-identical."""
    first = generator.generate_types_source(openapi_schema)
    second = generator.generate_types_source(openapi_schema)
    assert first == second


def test_banner_has_no_timestamp(generator: ModuleType) -> None:
    """The banner must stay stable across runs (no dates, no times)."""
    import re

    banner = generator._BANNER
    assert not re.search(r"\d{4}-\d{2}-\d{2}|\d{2}:\d{2}", banner)
    assert "DO NOT EDIT" in banner


def test_emitter_handles_core_schema_shapes(generator: ModuleType) -> None:
    """Unit coverage for the JSON Schema -> TS conversion rules."""
    ts = generator.ts_type
    assert ts({"type": "string"}) == "string"
    assert ts({"type": "integer"}) == "number"
    assert ts({"type": "null"}) == "null"
    assert ts({"$ref": "#/components/schemas/Foo"}) == "Foo"
    assert ts({"enum": ["full", "partial", "none"]}) == '"full" | "partial" | "none"'
    assert ts({"const": "v1"}) == '"v1"'
    assert ts({"anyOf": [{"type": "string"}, {"type": "null"}]}) == "string | null"
    assert ts({"type": "array", "items": {"type": "number"}}) == "number[]"
    assert (
        ts(
            {
                "type": "array",
                "items": {"anyOf": [{"type": "string"}, {"type": "null"}]},
            }
        )
        == "(string | null)[]"
    )
    assert (
        ts({"type": "object", "additionalProperties": {"type": "integer"}})
        == "Record<string, number>"
    )
    assert ts({"type": "object"}) == "Record<string, unknown>"
    assert ts(True) == "unknown"


def test_fields_with_defaults_are_required(generator: ModuleType) -> None:
    """Fields with a serialized default are emitted as required (responses
    always include them), mirroring openapi-typescript defaultNonNullable."""
    schema = {
        "type": "object",
        "properties": {
            "explicit": {"type": "string"},
            "defaulted": {"type": "boolean", "default": False},
            "optional": {"type": "string"},
        },
        "required": ["explicit"],
    }
    declaration = generator._emit_declaration("Sample", schema)
    assert "explicit: string;" in declaration
    assert "defaulted: boolean;" in declaration
    assert "optional?: string;" in declaration


def _generate_in_subprocess(hash_seed: str, out_path: Path) -> str:
    """Run the generator CLI in a fresh interpreter with a fixed hash seed."""
    import os
    import subprocess

    env = {**os.environ, "PYTHONHASHSEED": hash_seed}
    result = subprocess.run(
        [sys.executable, str(GENERATOR_PATH), "--out", str(out_path)],
        cwd=REPO_ROOT,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return out_path.read_text(encoding="utf-8")


@pytest.mark.timeout(300)
def test_generation_is_independent_of_hash_seed(
    generator: ModuleType, openapi_schema: dict, tmp_path: Path
) -> None:
    """Literal/enum member order must not depend on PYTHONHASHSEED (#12422)."""
    del openapi_schema  # skip this test when the app cannot be imported
    outputs = {
        seed: _generate_in_subprocess(seed, tmp_path / f"types_{seed}.ts")
        for seed in ("1", "2")
    }
    baseline = outputs["1"]
    for seed, text in outputs.items():
        assert text == baseline, f"types.ts differs under PYTHONHASHSEED={seed}"
    assert baseline == GENERATED_PATH.read_text(encoding="utf-8")


def _collect_enums(node: object, found: list[list[object]]) -> None:
    """Gather every ``enum`` member list found anywhere in a schema tree."""
    if isinstance(node, dict):
        members = node.get("enum")
        if isinstance(members, list):
            found.append(members)
        for value in node.values():
            _collect_enums(value, found)
    elif isinstance(node, list):
        for value in node:
            _collect_enums(value, found)


def test_equal_member_enums_share_one_order(openapi_schema: dict) -> None:
    """Enums with the same members must list them in one order (#12422).

    ``typing.Literal`` equality ignores member order, so two Literal
    declarations with the same members in different orders collapse to
    whichever was created first - an import-order dependent result.
    """
    enums: list[list[object]] = []
    _collect_enums(openapi_schema, enums)
    orders: dict[frozenset[str], set[tuple[str, ...]]] = {}
    for members in enums:
        key = frozenset(str(m) for m in members)
        orders.setdefault(key, set()).add(tuple(str(m) for m in members))
    conflicts = {tuple(sorted(k)): v for k, v in orders.items() if len(v) > 1}
    assert not conflicts, f"enum member order is ambiguous: {conflicts}"
