"""Fleet-wide shared theme management system.

This module provides a unified color theme system for all PyQt6 GUI applications
across the D-sorganization repository fleet.

Features:
- 12+ built-in themes (Light, Dark, Neon Warm Dark, Vampire Dark, Frost Dark, etc.)
- Custom theme support with persistence
- Theme inheritance for docked applications
- Qt stylesheet generation
- Matplotlib integration for consistent plotting colors
- Signal-based theme change notifications

Usage:
    from shared.python.theme import ThemeManager, get_theme_manager

    # Get singleton instance
    manager = get_theme_manager()

    # Get available themes
    themes = manager.get_available_themes()

    # Change theme
    manager.change_theme("Dark")

    # Apply to a window
    manager.apply_theme_to_window(my_window)

    # Connect to theme changes
    manager.themeChanged.connect(self.on_theme_changed)

    # Access current colors for custom styling
    colors = manager.get_current_colors()
    bg_color = colors["bg"]
"""

from typing import Any
from types import SimpleNamespace as _NS

from .colors import (
    BUILTIN_THEMES,
    CHART_COLORS,
    Colors,
    SEMANTIC_COLOR_KEYS,
    THEME_COLOR_KEYS,
    get_matplotlib_colors,
    get_rgba,
    is_dark_theme,
    is_valid_hex_color,
    normalise_hex_color,
)
from .protocols import StylesheetGenerator, ThemeProvider, ThemeSwitcher
from .sidekick_tokens import (
    REQUIRED_SIDEKICK_TOKENS,
    get_current_sidekick_tokens,
    sidekick_tokens_from_theme,
)
from .stylesheets import generate_minimal_stylesheet, generate_stylesheet
from .typography import (
    Sizes,
    Weights,
    get_display_font,
    get_mono_font,
    get_qfont,
)

# Convenience fallback: a ThemeColors-compatible dark theme.
# Used by launcher code as a safe default when PyQt6 ThemeManager is unavailable.
# Built via fleet_to_theme_colors to get all proper semantic attributes.
try:
    from .fleet_adapter import fleet_to_theme_colors as _f2tc

    DARK_THEME = _f2tc("Dark")
except Exception as e:  # noqa: BLE001, F841
    # Ultimate fallback: minimal SimpleNamespace if fleet adapter is unavailable
    _dark = BUILTIN_THEMES.get("Dark", {})
    DARK_THEME = _NS(  # type: ignore[assignment]
        bg=_dark.get("bg", "#1a1d23"),
        bg_elevated=_dark.get("table_header", "#2a2d35"),
        bg_highlight=_dark.get("title_bg", "#2e3340"),
        border_default=_dark.get("border", "#3a3d45"),
        border_strong=_dark.get("border", "#4a4d55"),
        primary=_dark.get("accent", "#0A84FF"),
        accent=_dark.get("accent", "#0A84FF"),
        success=_dark.get("success", "#30D158"),
        success_hover="#38e066",
        error=_dark.get("error", "#FF375F"),
        text_primary=_dark.get("text", "#FFFFFF"),
        text_secondary="#AAAAAA",
        text_tertiary="#888888",
        text_quaternary="#666666",
        chart_cyan="#00BCD4",
        chart_purple="#9C27B0",
    )

# PyQt6-dependent imports - only available when PyQt6 is installed
try:
    from .colors import get_qcolor
    from .dialogs import (
        ColorFieldEditor,
        ColorPickerButton,
        CustomThemeDialog,
        CustomThemeEditor,
        ThemeListItem,
        ThemeManagerDialog,
        ThemePreviewWidget,
    )
    from .integration import (
        ThemedWindowMixin,
        apply_theme_to_window,
        create_theme_menu,
        setup_themed_app,
    )
    from .font_manager import FontManager, get_font_manager
    from .responsive import (
        TextWidthSpec,
        configure_form_layout_for_readability,
        derive_text_candidates,
        readable_text_width,
        set_text_minimum_width,
        wrap_in_scroll_area,
    )
    from .theme_manager import ThemeManager, get_theme_manager
    from .zoom import (
        ApplicationZoomController,
        ZoomConfig,
        ZoomTokenSet,
        install_application_zoom,
        scale_px,
    )

    _PYQT6_AVAILABLE = True
except ImportError:
    _PYQT6_AVAILABLE = False
    ThemeManager = None  # type: ignore[assignment, misc]
    get_theme_manager = None  # type: ignore[assignment]
    FontManager = None  # type: ignore[assignment, misc]
    get_font_manager = None  # type: ignore[assignment]
    get_qcolor = None  # type: ignore[assignment]
    ThemedWindowMixin = None  # type: ignore[assignment, misc]
    apply_theme_to_window = None  # type: ignore[assignment]
    create_theme_menu = None  # type: ignore[assignment]
    setup_themed_app = None  # type: ignore[assignment]
    ColorFieldEditor = None  # type: ignore[assignment, misc]
    ColorPickerButton = None  # type: ignore[assignment, misc]
    CustomThemeDialog = None  # type: ignore[assignment, misc]
    CustomThemeEditor = None  # type: ignore[assignment, misc]
    ThemeListItem = None  # type: ignore[assignment, misc]
    ThemeManagerDialog = None  # type: ignore[assignment, misc]
    ThemePreviewWidget = None  # type: ignore[assignment, misc]
    ApplicationZoomController = None  # type: ignore[assignment, misc]
    ZoomConfig = None  # type: ignore[assignment, misc]
    ZoomTokenSet = None  # type: ignore[assignment, misc]
    TextWidthSpec = None  # type: ignore[assignment, misc]
    configure_form_layout_for_readability = None  # type: ignore[assignment]
    derive_text_candidates = None  # type: ignore[assignment]
    install_application_zoom = None  # type: ignore[assignment]
    readable_text_width = None  # type: ignore[assignment]
    scale_px = None  # type: ignore[assignment]
    set_text_minimum_width = None  # type: ignore[assignment]
    wrap_in_scroll_area = None  # type: ignore[assignment]

# ---------------------------------------------------------------------------
# Backward-compatible aliases for launcher code that references the old API.
# ThemePreset was replaced by string-based theme names; apply_golf_suite_style
# was inlined into the theme manager.  These shims keep existing callers working.
# ---------------------------------------------------------------------------
import enum as _enum


class ThemePreset(_enum.Enum):
    """Legacy enum mapping to string theme names."""

    DARK = "Dark"
    LIGHT = "Light"
    HIGH_CONTRAST = "High Contrast"
    MONOKAI = "Neon Warm Dark"
    DRACULA = "Vampire Dark"
    ONE_DARK = "Frost Dark"


def apply_golf_suite_style() -> None:
    """Apply matplotlib styling consistent with the current theme.

    Safe no-op if matplotlib is unavailable.
    """
    try:
        import matplotlib as _mpl

        _mpl.rcParams.update(
            {
                "figure.facecolor": "#1a1d23",
                "axes.facecolor": "#22252d",
                "axes.edgecolor": "#3a3d45",
                "text.color": "#cccccc",
                "xtick.color": "#aaaaaa",
                "ytick.color": "#aaaaaa",
                "grid.color": "#333333",
                "grid.alpha": 0.3,
            }
        )
    except ImportError:
        pass


__all__ = [
    # Protocols (no PyQt6 dependency)
    "StylesheetGenerator",
    "ThemeProvider",
    "ThemeSwitcher",
    # Theme manager (requires PyQt6)
    "ThemeManager",
    "get_theme_manager",
    # Font manager (requires PyQt6)
    "FontManager",
    "get_font_manager",
    # Integration helpers (requires PyQt6)
    "ThemedWindowMixin",
    "apply_theme_to_window",
    "create_theme_menu",
    "setup_themed_app",
    # Dialogs (requires PyQt6)
    "ColorFieldEditor",
    "ColorPickerButton",
    "CustomThemeDialog",
    "CustomThemeEditor",
    "ThemeListItem",
    "ThemeManagerDialog",
    "ThemePreviewWidget",
    "ApplicationZoomController",
    "ZoomConfig",
    "ZoomTokenSet",
    "TextWidthSpec",
    # Color utilities
    "BUILTIN_THEMES",
    "CHART_COLORS",
    "DARK_THEME",
    "SEMANTIC_COLOR_KEYS",
    "THEME_COLOR_KEYS",
    "get_matplotlib_colors",
    "get_qcolor",
    "get_rgba",
    "is_dark_theme",
    "is_valid_hex_color",
    "normalise_hex_color",
    "Colors",
    "configure_form_layout_for_readability",
    "derive_text_candidates",
    "install_application_zoom",
    "readable_text_width",
    "scale_px",
    "set_text_minimum_width",
    "wrap_in_scroll_area",
    # Typography utilities
    "Sizes",
    "Weights",
    "get_display_font",
    "get_mono_font",
    "get_qfont",
    # Sidekick design-token adapter
    "REQUIRED_SIDEKICK_TOKENS",
    "get_current_sidekick_tokens",
    "sidekick_tokens_from_theme",
    # Stylesheet generation
    "generate_minimal_stylesheet",
    "generate_stylesheet",
    # Legacy compatibility
    "ThemePreset",
    "apply_golf_suite_style",
    "get_current_colors",
]


class ThemeColorsCompat(dict):
    """Dictionary subclass supporting attribute-style access for theme compatibility."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as err:
            raise AttributeError(name) from err

    def __setattr__(self, name: str, value: Any) -> None:
        self[name] = value


def get_current_colors() -> ThemeColorsCompat:
    """Return the currently active theme colors in a compatibility-wrapper format.

    Supports both dictionary-like lookup (e.g. colors.get("border")) and
    attribute-style lookup (e.g. colors.bg_elevated).
    """
    try:
        from src.shared.python.theme.theme_manager import get_theme_manager

        mgr = get_theme_manager()
        if mgr and mgr.current_theme:
            try:
                from src.shared.python.theme.fleet_adapter import (
                    fleet_to_theme_colors_dict,
                )

                colors_dict = fleet_to_theme_colors_dict(mgr.current_theme)
                return ThemeColorsCompat(colors_dict)
            except Exception:  # noqa: BLE001
                colors_dict = dict(mgr.get_current_colors())
                return ThemeColorsCompat(colors_dict)
    except Exception:  # noqa: BLE001
        pass

    # Fallback to Dark theme using the full fleet dict
    try:
        from src.shared.python.theme.fleet_adapter import fleet_to_theme_colors_dict

        return ThemeColorsCompat(fleet_to_theme_colors_dict("Dark"))
    except Exception:  # noqa: BLE001
        pass

    # Ultimate fallback to whatever is in DARK_THEME
    try:
        from src.shared.python.theme import DARK_THEME

        d = {}
        if hasattr(DARK_THEME, "__dict__"):
            d.update(
                {k: v for k, v in DARK_THEME.__dict__.items() if not k.startswith("_")}
            )
        elif hasattr(DARK_THEME, "dict"):
            d.update(DARK_THEME.dict())
        return ThemeColorsCompat(d)
    except Exception:  # noqa: BLE001
        return ThemeColorsCompat()
