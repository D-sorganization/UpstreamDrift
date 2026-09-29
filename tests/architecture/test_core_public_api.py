"""Snapshot test for the core public API surface."""

from __future__ import annotations

import importlib
import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.core_only

SNAPSHOT_PATH = Path(__file__).with_name("snapshots") / "core_public_api.json"


def _public_names(module_name: str) -> set[str]:
    module = importlib.import_module(module_name)
    exported_names = getattr(module, "__all__", None)
    if exported_names is not None:
        return {name for name in exported_names if isinstance(name, str)}
    return {name for name in vars(module) if not name.startswith("_")}


def test_core_public_api_has_no_removals() -> None:
    """Public names in the frozen core snapshot may not disappear silently."""
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    for module_name, expected_names in snapshot.items():
        removed_names = set(expected_names) - _public_names(module_name)
        assert not removed_names, f"{module_name} removed: {sorted(removed_names)}"


def test_core_public_api_snapshot_is_sorted() -> None:
    """Keep the snapshot deterministic for reviewable API changes."""
    snapshot = json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))

    for expected_names in snapshot.values():
        assert expected_names == sorted(expected_names)
