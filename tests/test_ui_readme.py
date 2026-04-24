from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
UI_README = ROOT / "ui" / "README.md"


def test_ui_readme_is_upstreamdrift_specific() -> None:
    content = UI_README.read_text(encoding="utf-8")

    assert "# UpstreamDrift UI" in content
    assert (
        "This template provides a minimal setup to get React working in Vite"
        not in content
    )
    assert "npm run dev" in content
    assert "npm run tauri:build" in content
