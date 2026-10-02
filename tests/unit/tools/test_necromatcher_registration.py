"""Necromatcher must be reachable from each host's canonical launcher."""

import json
from pathlib import Path
import tomllib
import pytest
import yaml

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]


def test_necromatcher_has_native_and_web_registration():
    manifest = json.loads(
        (ROOT / "src/config/launcher_manifest.json").read_text(encoding="utf-8")
    )
    tile = next(x for x in manifest["tiles"] if x["id"] == "necromatcher")
    assert tile["web"] == {"mode": "route", "route": "/tools/necromatcher"}
    assert (ROOT / tile["path"]).is_file()
    models = yaml.safe_load(
        (ROOT / "src/config/models.yaml").read_text(encoding="utf-8")
    )
    entries = models["models"]
    assert any(x["id"] == "necromatcher" for x in entries)
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert (
        project["project"]["entry-points"]["upstream_drift.embeddable_tools"][
            "necromatcher"
        ]
        == "src.tools.necromatcher._embed_adapter"
    )
    assert "/tools/necromatcher" in (ROOT / "ui/src/App.tsx").read_text(
        encoding="utf-8"
    )
