"""Dependency declaration checks for the Python realtime WebSocket backend."""

from __future__ import annotations

import tomllib
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_WEBSOCKETS_SPEC = "websockets>=12.0,<18.0"


def test_pyproject_declares_websockets_for_realtime_extra() -> None:
    """ws_pubsub.subscribe imports websockets lazily; it must be an declared extra."""
    pyproject = tomllib.loads(
        (_REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    extras = pyproject["project"]["optional-dependencies"]

    assert "realtime" in extras
    assert _WEBSOCKETS_SPEC in extras["realtime"]
