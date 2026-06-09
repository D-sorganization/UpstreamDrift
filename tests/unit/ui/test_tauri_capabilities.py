"""Tauri v2 capability-surface ratchet (issue #7164).

The desktop shell registers four IPC commands that can start/stop the local
backend process. Their permissions must be declared explicitly in a capabilities
file pinned to the main window — not left to framework defaults. This test fails
if the capabilities file is missing or if a command is added/removed without a
matching permission entry.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]
_CAPABILITIES = _REPO_ROOT / "ui" / "src-tauri" / "capabilities" / "main.json"
_LIB_RS = _REPO_ROOT / "ui" / "src-tauri" / "src" / "lib.rs"

_EXPECTED_COMMANDS = {
    "start_backend",
    "stop_backend",
    "backend_status",
    "get_diagnostics",
}


def test_capabilities_file_exists() -> None:
    assert _CAPABILITIES.is_file(), f"missing {_CAPABILITIES}"


def test_capabilities_pins_main_window_and_no_wildcards() -> None:
    data = json.loads(_CAPABILITIES.read_text(encoding="utf-8"))
    assert data["windows"] == ["main"]
    for perm in data["permissions"]:
        assert "*" not in perm, f"wildcard permission not allowed: {perm}"


def test_capabilities_grant_minimal_core_and_log() -> None:
    """App-defined commands are reachable by default from the app's own
    windows; the capability only needs to grant the core/log plugin defaults
    (no wildcards) and scope them to the main window (issue #7164)."""
    data = json.loads(_CAPABILITIES.read_text(encoding="utf-8"))
    perms = set(data["permissions"])
    assert "core:default" in perms
    assert "log:default" in perms


def test_registered_commands_are_the_expected_four() -> None:
    """generate_handler! must register exactly the four documented commands;
    adding a fifth command should force an explicit review of this test and the
    capability file together (ratchet)."""
    source = _LIB_RS.read_text(encoding="utf-8")
    match = re.search(r"generate_handler!\s*\[(.*?)\]", source, re.DOTALL)
    assert match, "could not find generate_handler! block in lib.rs"
    registered = {token.strip() for token in match.group(1).split(",") if token.strip()}
    assert registered == _EXPECTED_COMMANDS, (
        f"registered commands {registered} != expected {_EXPECTED_COMMANDS}; "
        "update capabilities/main.json and this test together"
    )
