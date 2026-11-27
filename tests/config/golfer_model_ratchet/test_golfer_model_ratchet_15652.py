"""Committed golfer models may only move toward the current model (#15652).

``src/config/golfer_model_legacy_assets.json`` lists every committed spec and
bundle that still resolves to a part ``golfer_model_versions.json`` deprecates.
A new spec or bundle on a deprecated part fails here, and so does a listed file
that was migrated or removed without its entry, so the list only shrinks.
"""

from __future__ import annotations

import json
import subprocess
from collections.abc import Mapping
from functools import cache
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.model_appearance.golfer_model_versions import deprecated_parts

pytestmark = pytest.mark.unit
REPO = Path(__file__).resolve().parents[3]
BASELINE = REPO / "src/config/golfer_model_legacy_assets.json"


def _tracked(pattern: str) -> list[str]:
    listing = subprocess.run(
        ["git", "ls-files", "-z", pattern],
        cwd=REPO,
        capture_output=True,
        check=True,
        text=True,
    )
    return sorted(path for path in listing.stdout.split("\0") if path)


def _spec_document(path: str) -> Mapping[str, Any] | None:
    try:
        document = json.loads((REPO / path).read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    is_spec = isinstance(document, dict) and {"joints", "subject"} <= set(document)
    return document if is_spec else None


def _bundle_spec(path: str) -> Mapping[str, Any] | None:
    # InputBundle stores the spec as JSON bytes under "spec" (same_input.bundle).
    with np.load(REPO / path, allow_pickle=False) as data:
        if "spec" not in data.files:
            return None
        return json.loads(data["spec"].tobytes())  # type: ignore[no-any-return]


@cache
def _legacy(kind: str) -> tuple[str, ...]:
    pattern, read = {
        "specs": ("*.json", _spec_document),
        "bundles": ("*.npz", _bundle_spec),
    }[kind]
    found = []
    for path in _tracked(pattern):
        document = read(path)
        if document is not None and deprecated_parts(document):
            found.append(path)
    return tuple(found)


def _baseline(kind: str) -> list[str]:
    raw = json.loads(BASELINE.read_text(encoding="utf-8"))
    return list(raw[f"legacy_{kind}"])


@pytest.mark.parametrize("kind", ["specs", "bundles"])
def test_no_new_spec_or_bundle_uses_a_deprecated_part(kind: str) -> None:
    new = sorted(set(_legacy(kind)) - set(_baseline(kind)))
    assert not new, (
        f"{new} resolve to a deprecated golfer model part; build them on the "
        "current model (docs/agents/golfer_model_versions.md)"
    )


@pytest.mark.parametrize("kind", ["specs", "bundles"])
def test_migrated_or_removed_files_leave_the_baseline(kind: str) -> None:
    stale = sorted(set(_baseline(kind)) - set(_legacy(kind)))
    assert not stale, (
        f"{stale} no longer resolve to a deprecated part (or are gone); delete "
        f"them from {BASELINE.name}"
    )


@pytest.mark.parametrize("kind", ["specs", "bundles"])
def test_the_baseline_is_sorted_and_unique(kind: str) -> None:
    listed = _baseline(kind)
    assert listed == sorted(set(listed))
