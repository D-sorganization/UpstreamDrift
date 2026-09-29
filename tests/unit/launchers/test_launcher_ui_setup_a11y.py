"""Accessibility-focused unit tests for launcher UI setup helpers."""

from src.launchers import launcher_ui_setup


def test_zoom_accessible_description_matches_current_scale_constants() -> None:
    """The default accessibility text should reflect the configured tile bounds."""
    assert launcher_ui_setup._build_zoom_accessible_description() == (
        "Adjust tile size from 25% to 200%. Use arrow keys or drag to adjust."
    )


def test_zoom_accessible_description_updates_when_scale_constants_change(
    monkeypatch,
) -> None:
    """The description must be derived from live constants, not hardcoded prose."""
    monkeypatch.setattr(launcher_ui_setup, "TILE_SCALE_MIN", 0.5)
    monkeypatch.setattr(launcher_ui_setup, "TILE_SCALE_MAX", 1.5)

    assert launcher_ui_setup._build_zoom_accessible_description() == (
        "Adjust tile size from 50% to 150%. Use arrow keys or drag to adjust."
    )
