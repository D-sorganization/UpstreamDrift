"""Vendored studio assets match their recorded provenance (#14141 step 8b)."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_VENDOR = Path(__file__).resolve().parents[3] / "src/apps/putting_studio/vendor"
_ROW = re.compile(r"^\| `(?P<name>[^`]+)` +\|.*\| `(?P<sha>[0-9a-f]{64})` +\|$")


def _recorded_hashes() -> dict[str, str]:
    lines = (_VENDOR / "README.md").read_text(encoding="utf-8").splitlines()
    return {m["name"]: m["sha"] for m in map(_ROW.match, lines) if m}


def test_every_vendored_asset_is_recorded_with_its_hash() -> None:
    recorded = _recorded_hashes()
    assert set(recorded) == {"three.min.js", "tailwind.min.css"}
    for name, sha in recorded.items():
        digest = hashlib.sha256((_VENDOR / name).read_bytes()).hexdigest()
        assert digest == sha, f"{name} differs from its recorded SHA-256"


def test_licenses_ship_beside_the_assets() -> None:
    for license_name in ("three.LICENSE", "tailwind.LICENSE"):
        assert "MIT" in (_VENDOR / license_name).read_text(encoding="utf-8")


def test_vendored_css_fetches_nothing_remote() -> None:
    css = (_VENDOR / "tailwind.min.css").read_text(encoding="utf-8")
    # The license banner names tailwindcss.com; only loads would fetch.
    assert "@import" not in css
    assert not re.search(r"url\(\s*['\"]?(https?:)?//", css)
