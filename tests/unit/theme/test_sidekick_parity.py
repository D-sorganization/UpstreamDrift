"""Cross-shell parity tests for Sidekick design tokens.

Asserts that the Python token contract in
``src.shared.python.theme.sidekick_tokens`` matches the TypeScript contract in
``ui/src/api/themeClient.ts``. This guards against silent drift when one shell
adds, removes, or rewrites a token without updating the other.

The TypeScript side is parsed as plain text using ``re`` so the test stays
stdlib-only and does not require Node, npm, or a JavaScript runtime in CI.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from src.shared.python.theme.sidekick_tokens import (
    COLOR_TOKEN_MAP,
    DEFAULT_SIDEKICK_TOKENS,
)

pytestmark = pytest.mark.unit


REPO_ROOT = Path(__file__).resolve().parents[3]
THEME_CLIENT_PATH = REPO_ROOT / "ui" / "src" / "api" / "themeClient.ts"


def _extract_ts_object(source: str, identifier: str) -> dict[str, str]:
    """Extract a flat ``key: 'value'`` TypeScript object literal as a dict.

    Preconditions:
        ``source`` must contain a top-level ``export const <identifier> = {...}``
        whose entries are plain ``'<key>': '<value>'`` pairs. Trailing
        ``as const`` or ``satisfies`` clauses are tolerated.

    Postconditions:
        Returns a ``dict[str, str]`` where keys and values exactly match the
        TypeScript literal contents (no surrounding quotes).
    """
    if not source:
        raise ValueError("source must be a non-empty string")
    if not identifier:
        raise ValueError("identifier must be a non-empty string")

    pattern = re.compile(
        r"export\s+const\s+" + re.escape(identifier) + r"\s*=\s*\{(?P<body>.*?)\}",
        re.DOTALL,
    )
    match = pattern.search(source)
    if match is None:
        raise AssertionError(
            f"Could not find `export const {identifier} = {{...}}` in themeClient.ts",
        )

    body = match.group("body")
    entry_pattern = re.compile(
        r"['\"](?P<key>[^'\"]+)['\"]\s*:\s*['\"](?P<value>[^'\"]+)['\"]\s*,?",
    )
    extracted: dict[str, str] = {}
    for entry in entry_pattern.finditer(body):
        extracted[entry.group("key")] = entry.group("value")

    if not extracted:
        raise AssertionError(
            f"Parsed empty object literal for `{identifier}` in themeClient.ts",
        )
    return extracted


@pytest.fixture(scope="module")
def theme_client_source() -> str:
    """Read the TypeScript theme client as text."""
    assert THEME_CLIENT_PATH.exists(), (
        f"themeClient.ts not found at {THEME_CLIENT_PATH}"
    )
    return THEME_CLIENT_PATH.read_text(encoding="utf-8")


def test_color_token_map_parity_between_python_and_typescript(
    theme_client_source: str,
) -> None:
    """Python COLOR_TOKEN_MAP must match TS SIDEKICK_COLOR_TOKEN_MAP exactly."""
    ts_map = _extract_ts_object(theme_client_source, "SIDEKICK_COLOR_TOKEN_MAP")
    py_map = dict(COLOR_TOKEN_MAP)

    assert set(py_map.keys()) == set(ts_map.keys()), (
        "Sidekick color token names differ between Python and TypeScript: "
        f"py-only={set(py_map) - set(ts_map)}, ts-only={set(ts_map) - set(py_map)}"
    )
    assert py_map == ts_map, (
        "Sidekick color token theme-key mappings differ between Python and TS: "
        f"{[(k, py_map[k], ts_map[k]) for k in py_map if py_map[k] != ts_map[k]]}"
    )


def test_fallback_color_tokens_parity_between_python_and_typescript(
    theme_client_source: str,
) -> None:
    """TS SIDEKICK_FALLBACK_COLOR_TOKENS must be a color-key subset of Python."""
    ts_fallback = _extract_ts_object(
        theme_client_source, "SIDEKICK_FALLBACK_COLOR_TOKENS"
    )
    py_defaults = dict(DEFAULT_SIDEKICK_TOKENS)

    ts_keys = set(ts_fallback.keys())
    py_keys = set(py_defaults.keys())
    assert ts_keys.issubset(py_keys), (
        "TS fallback contains tokens missing from Python DEFAULT_SIDEKICK_TOKENS: "
        f"{sorted(ts_keys - py_keys)}"
    )

    drift = {
        key: (py_defaults[key], ts_fallback[key])
        for key in ts_keys
        if py_defaults[key] != ts_fallback[key]
    }
    assert not drift, (
        f"Sidekick fallback color values drift between Python and TypeScript: {drift}"
    )


def test_python_color_keys_match_color_token_map(theme_client_source: str) -> None:
    """Color tokens declared in COLOR_TOKEN_MAP must all have a default value."""
    ts_fallback = _extract_ts_object(
        theme_client_source, "SIDEKICK_FALLBACK_COLOR_TOKENS"
    )
    py_color_tokens = set(COLOR_TOKEN_MAP.keys())

    missing_py = py_color_tokens - set(DEFAULT_SIDEKICK_TOKENS.keys())
    assert not missing_py, (
        f"COLOR_TOKEN_MAP tokens missing from DEFAULT_SIDEKICK_TOKENS: {missing_py}"
    )

    missing_ts = py_color_tokens - set(ts_fallback.keys())
    assert not missing_ts, (
        "Color tokens missing from TS SIDEKICK_FALLBACK_COLOR_TOKENS: "
        f"{sorted(missing_ts)}"
    )
