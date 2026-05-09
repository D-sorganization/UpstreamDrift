"""Regression tests for theme system backward compatibility.

Ensures ThemePreset enum and apply_golf_suite_style() remain importable
and functional after theme module refactors.
"""

import enum

from src.shared.python.theme import ThemePreset, apply_golf_suite_style


class TestThemePresetCompat:
    """Verify ThemePreset enum shim works correctly."""

    def test_theme_compat_importable(self) -> None:
        assert ThemePreset is not None

    def test_has_dark(self) -> None:
        assert ThemePreset.DARK.value == "Dark"

    def test_has_light(self) -> None:
        assert ThemePreset.LIGHT.value == "Light"

    def test_has_high_contrast(self) -> None:
        assert ThemePreset.HIGH_CONTRAST.value == "High Contrast"

    def test_is_enum(self) -> None:
        assert issubclass(ThemePreset, enum.Enum)

    def test_all_presets_are_strings(self) -> None:
        for preset in ThemePreset:
            assert isinstance(preset.value, str)


class TestApplyGolfSuiteStyle:
    """Verify apply_golf_suite_style() backward-compat shim."""

    def test_theme_compat_importable(self) -> None:
        assert callable(apply_golf_suite_style)

    def test_callable_without_error(self) -> None:
        # Should not raise even if matplotlib is missing
        apply_golf_suite_style()

    def test_in_all(self) -> None:
        import src.shared.python.theme as theme_mod

        assert "ThemePreset" in theme_mod.__all__
        assert "apply_golf_suite_style" in theme_mod.__all__
