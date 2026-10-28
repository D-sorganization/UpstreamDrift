"""Published agent hook commands must be valid bash (bash -n)."""

from __future__ import annotations

import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]

HOOK_CONFIG_PATHS = (
    ROOT / ".gemini/settings.json",
    ROOT / ".agents/hooks.json",
    ROOT / ".codex/hooks.json",
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def _iter_hook_commands(node: object) -> Iterator[str]:
    if isinstance(node, dict):
        if node.get("type") == "command" and isinstance(node.get("command"), str):
            yield node["command"]
        for value in node.values():
            yield from _iter_hook_commands(value)
    elif isinstance(node, list):
        for item in node:
            yield from _iter_hook_commands(item)


def _commands_in(path: Path) -> list[str]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return list(_iter_hook_commands(data))


@pytest.mark.parametrize(
    "path",
    HOOK_CONFIG_PATHS,
    ids=[p.relative_to(ROOT).as_posix() for p in HOOK_CONFIG_PATHS],
)
def test_hook_commands_pass_bash_syntax_check(path: Path) -> None:
    commands = _commands_in(path)
    assert commands, f"no command hooks found in {path.relative_to(ROOT)}"
    for command in commands:
        result = subprocess.run(
            ["bash", "-n", "-c", command],
            capture_output=True,
            text=True,
            check=False,
        )
        assert result.returncode == 0, (
            f"{path.relative_to(ROOT)}: bash -n failed for {command!r}: "
            f"{result.stderr.strip()}"
        )
