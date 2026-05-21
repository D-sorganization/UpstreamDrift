"""Regression coverage for launcher tile icon completeness."""

from __future__ import annotations

from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = REPO_ROOT / "src" / "config" / "models.yaml"
ICON_ROOTS = (
    REPO_ROOT / "src" / "launchers" / "assets",
    REPO_ROOT / "assets" / "logos",
    REPO_ROOT / "assets",
)


def _configured_models() -> list[dict]:
    config = yaml.safe_load(CONFIG_PATH.read_text(encoding="utf-8"))
    return list(config.get("models", []))


def _icon_exists(logo: str) -> bool:
    name = Path(logo).name
    return any((root / name).is_file() for root in ICON_ROOTS)


def test_every_launcher_tile_declares_an_icon() -> None:
    missing = [
        model.get("id", "<unknown>")
        for model in _configured_models()
        if model.get("launcher") and not model["launcher"].get("logo")
    ]

    assert missing == []


def test_every_launcher_tile_icon_resolves_to_an_asset() -> None:
    missing = [
        (model.get("id", "<unknown>"), model["launcher"]["logo"])
        for model in _configured_models()
        if model.get("launcher") and not _icon_exists(str(model["launcher"]["logo"]))
    ]

    assert missing == []
