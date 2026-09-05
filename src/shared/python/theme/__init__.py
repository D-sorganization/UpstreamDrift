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

from .colors import (
    BUILTIN_THEMES,
    CHART_COLORS,
    SEMANTIC_COLOR_KEYS,
    THEME_COLOR_KEYS,
    get_matplotlib_colors,
    get_rgba,
    is_dark_theme,
    is_valid_hex_color,
    normalise_hex_color,
)
from .palette import (
    DARK_THEME,
    SEMANTIC_ALIASES,
    Colors,
    ThemePalette,
    get_current_colors,
)
from .protocols import StylesheetGenerator, ThemeProvider, ThemeSwitcher
from .stylesheets import generate_minimal_stylesheet, generate_stylesheet
from .typography import (
    CSS_FONT_DISPLAY,
    CSS_FONT_MONO,
    CSS_FONT_UI,
    FONT_STACK_DISPLAY,
    FONT_STACK_MONO,
    FONT_STACK_UI,
    FontSizes,
    FontWeights,
    Sizes,
    Weights,
    get_display_font,
    get_mono_font,
    get_qfont,
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
    from .font_manager import FontManager, get_font_manager
    from .integration import (
        ThemedWindowMixin,
        apply_theme_to_window,
        create_theme_menu,
        setup_themed_app,
    )
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
    TextWidthSpec = None  # type: ignore[assignment, misc]
    configure_form_layout_for_readability = None  # type: ignore[assignment]
    derive_text_candidates = None  # type: ignore[assignment]
    readable_text_width = None  # type: ignore[assignment]
    set_text_minimum_width = None  # type: ignore[assignment]
    wrap_in_scroll_area = None  # type: ignore[assignment]
    ApplicationZoomController = None  # type: ignore[assignment, misc]
    ZoomConfig = None  # type: ignore[assignment, misc]
    ZoomTokenSet = None  # type: ignore[assignment, misc]
    install_application_zoom = None  # type: ignore[assignment]
    scale_px = None  # type: ignore[assignment]

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
    # Responsive sizing and zoom helpers (require PyQt6)
    "TextWidthSpec",
    "configure_form_layout_for_readability",
    "derive_text_candidates",
    "readable_text_width",
    "set_text_minimum_width",
    "wrap_in_scroll_area",
    "ApplicationZoomController",
    "ZoomConfig",
    "ZoomTokenSet",
    "install_application_zoom",
    "scale_px",
    # Dialogs (requires PyQt6)
    "ColorFieldEditor",
    "ColorPickerButton",
    "CustomThemeDialog",
    "CustomThemeEditor",
    "ThemeListItem",
    "ThemeManagerDialog",
    "ThemePreviewWidget",
    # Color utilities and Palette
    "BUILTIN_THEMES",
    "CHART_COLORS",
    "Colors",
    "DARK_THEME",
    "SEMANTIC_ALIASES",
    "SEMANTIC_COLOR_KEYS",
    "THEME_COLOR_KEYS",
    "ThemePalette",
    "get_current_colors",
    "get_matplotlib_colors",
    "get_qcolor",
    "get_rgba",
    "is_dark_theme",
    "is_valid_hex_color",
    "normalise_hex_color",
    # Typography
    "CSS_FONT_DISPLAY",
    "CSS_FONT_MONO",
    "CSS_FONT_UI",
    "FONT_STACK_DISPLAY",
    "FONT_STACK_MONO",
    "FONT_STACK_UI",
    "FontSizes",
    "FontWeights",
    "Sizes",
    "Weights",
    "get_display_font",
    "get_mono_font",
    "get_qfont",
    # Stylesheet generation
    "generate_minimal_stylesheet",
    "generate_stylesheet",
]
