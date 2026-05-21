"""Design tokens and stylesheet bridge for the Sidekick sidebar."""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

SIDEKICK_SIDEBAR_OBJECT_NAME = "SidekickToolsSidebar"
SIDEKICK_DOCK_OBJECT_NAME = "SidekickToolsSidebarDock"
SIDEKICK_TABS_OBJECT_NAME = "SidekickTabs"
SIDEKICK_TAB_BAR_OBJECT_NAME = "SidekickTabBar"
SIDEKICK_TOOLBAR_OBJECT_NAME = "SidekickToolbar"
SIDEKICK_PROJECT_EXPLORER_OBJECT_NAME = "SidekickProjectExplorer"
SIDEKICK_PROJECT_TREE_OBJECT_NAME = "SidekickProjectTree"
SIDEKICK_WORKSPACE_TAB_OBJECT_NAME = "SidekickWorkspaceTab"
SIDEKICK_WORKSPACE_LIST_OBJECT_NAME = "SidekickWorkspaceList"
SIDEKICK_PLACEHOLDER_OBJECT_NAME = "SidekickPlaceholder"
SIDEKICK_PLACEHOLDER_LABEL_OBJECT_NAME = "SidekickPlaceholderLabel"
SIDEKICK_FUNCTION_GENERATOR_OBJECT_NAME = "SidekickFunctionGeneratorTab"
SIDEKICK_ROTATION_CONVERTER_OBJECT_NAME = "SidekickRotationConverterTab"

_DEFAULT_TOKEN_VALUES: dict[str, str] = {
    "color.background": "#f7f9fc",
    "color.surface": "#ffffff",
    "color.surface.raised": "#eef3f8",
    "color.border": "#d8e1ec",
    "color.border.strong": "#b8c6d8",
    "color.text": "#1c2430",
    "color.text.muted": "#5d6b7c",
    "color.accent": "#2563eb",
    "color.accent.hover": "#1d4ed8",
    "color.accent.soft": "#dbeafe",
    "color.focus": "#3b82f6",
    "color.selection": "#c7dfff",
    "color.danger": "#dc2626",
    "color.warning": "#d97706",
    "color.success": "#16a34a",
    "radius.panel": "8px",
    "radius.control": "6px",
    "shadow.panel": "0 12px 28px rgba(15, 23, 42, 0.14)",
    "space.1": "4px",
    "space.2": "8px",
    "space.3": "12px",
    "space.4": "16px",
    "size.control.height": "28px",
    "font.family": "Arial",
    "font.size": "12px",
    "font.size.small": "11px",
}

_DEFAULT_TERMINAL_PALETTE: dict[str, str] = {
    "foreground": "color.text",
    "background": "color.surface",
    "cursor": "color.accent",
    "selection": "color.selection",
    "ansi.black": "#1c2430",
    "ansi.red": "color.danger",
    "ansi.green": "color.success",
    "ansi.yellow": "color.warning",
    "ansi.blue": "color.accent",
    "ansi.magenta": "#7c3aed",
    "ansi.cyan": "#0891b2",
    "ansi.white": "#f8fafc",
}

_HEX_COLOR_RE = re.compile(r"^#[0-9a-fA-F]{6}$")

SIDEKICK_TOKEN_NAMES: tuple[str, ...] = tuple(_DEFAULT_TOKEN_VALUES)

_SHARED_THEME_TOKEN_MAP: dict[str, str] = {
    "bg": "color.background",
    "group_bg": "color.surface",
    "table_alt": "color.surface.raised",
    "border": "color.border",
    "title_border": "color.border.strong",
    "text": "color.text",
    "text_secondary": "color.text.muted",
    "accent": "color.accent",
    "button_hover": "color.accent.hover",
    "focus": "color.focus",
    "selection_bg": "color.selection",
    "error": "color.danger",
    "warning": "color.warning",
    "success": "color.success",
}

_SHARED_SPACING_TOKEN_MAP: dict[str, str] = {
    "xs": "space.1",
    "sm": "space.2",
    "md": "space.4",
}

_SHARED_RADII_TOKEN_MAP: dict[str, str] = {
    "md": "radius.control",
}

_SIDEKICK_TOKEN_ALIASES: dict[str, str] = {
    "sidekick.color.canvas": "color.background",
    "sidekick.color.background": "color.background",
    "sidekick.color.surface": "color.surface",
    "sidekick.color.surface.elevated": "color.surface.raised",
    "sidekick.color.surface.raised": "color.surface.raised",
    "sidekick.color.surface.muted": "color.surface.raised",
    "sidekick.color.border": "color.border",
    "sidekick.color.border.strong": "color.border.strong",
    "sidekick.color.text": "color.text",
    "sidekick.color.text.muted": "color.text.muted",
    "sidekick.color.text.subtle": "color.text.muted",
    "sidekick.color.accent": "color.accent",
    "sidekick.color.accent.hover": "color.accent.hover",
    "sidekick.color.accent.soft": "color.accent.soft",
    "sidekick.color.focus": "color.focus",
    "sidekick.color.selection": "color.selection",
    "sidekick.color.error": "color.danger",
    "sidekick.color.danger": "color.danger",
    "sidekick.color.warning": "color.warning",
    "sidekick.color.success": "color.success",
    "sidekick.radius.lg": "radius.panel",
    "sidekick.radius.chat": "radius.panel",
    "sidekick.radius.md": "radius.control",
    "sidekick.radius.control": "radius.control",
    "sidekick.shadow.md": "shadow.panel",
    "sidekick.shadow.panel": "shadow.panel",
    "sidekick.control.height": "size.control.height",
    "sidekick.size.control.height": "size.control.height",
}


def _token_slug(name: str) -> str:
    return name.replace(".", "-").replace("_", "-")


def _normalize_token_values(values: Mapping[str, str]) -> dict[str, str]:
    normalized: dict[str, str] = {}
    for name, value in values.items():
        token_name = _SIDEKICK_TOKEN_ALIASES.get(name, name)
        if token_name in _DEFAULT_TOKEN_VALUES:
            normalized[token_name] = value
    return normalized


def _capped_radius(value: str, maximum_px: int = 8) -> str:
    if value.endswith("px"):
        try:
            return f"{min(int(value[:-2]), maximum_px)}px"
        except ValueError:
            return value
    return value


def _shared_theme_values(theme_name: str = "light") -> dict[str, str]:
    try:
        from ..design_tokens import load_design_tokens
    except Exception:  # noqa: BLE001 - Sidekick must import without full install data
        return {}

    try:
        shared_tokens = load_design_tokens()
        theme = shared_tokens.get("themes", {}).get(theme_name, {})
        spacing = shared_tokens.get("spacing", {})
        radii = shared_tokens.get("radii", {})
    except Exception:  # noqa: BLE001 - fall back to local defaults
        return {}

    values: dict[str, str] = {
        token_name: theme[source_name]
        for source_name, token_name in _SHARED_THEME_TOKEN_MAP.items()
        if source_name in theme
    }
    values.update(
        {
            token_name: spacing[source_name]
            for source_name, token_name in _SHARED_SPACING_TOKEN_MAP.items()
            if source_name in spacing
        }
    )
    values.update(
        {
            token_name: radii[source_name]
            for source_name, token_name in _SHARED_RADII_TOKEN_MAP.items()
            if source_name in radii
        }
    )
    if "lg" in radii:
        values["radius.panel"] = _capped_radius(str(radii["lg"]))
    return values


@dataclass(frozen=True)
class SidekickDesignTokens:
    """Reusable Sidekick token set that can emit CSS and QSS mappings."""

    values: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        merged = {**_DEFAULT_TOKEN_VALUES, **_normalize_token_values(self.values)}
        missing = [name for name in SIDEKICK_TOKEN_NAMES if not merged.get(name)]
        if missing:
            missing_names = ", ".join(missing)
            raise ValueError(f"Missing Sidekick design tokens: {missing_names}")
        object.__setattr__(self, "values", MappingProxyType(merged))

    def __getitem__(self, name: str) -> str:
        return self.values[name]

    def css_variables(self, prefix: str = "sidekick") -> dict[str, str]:
        """Return browser CSS custom properties for the token set."""
        return {
            f"--{prefix}-{_token_slug(name)}": value
            for name, value in self.values.items()
        }

    def qss_variables(self, prefix: str = "sidekick") -> dict[str, str]:
        """Return QSS-friendly token names for stylesheet templating."""
        return {
            f"{prefix}-{_token_slug(name)}": value
            for name, value in self.values.items()
        }

    def with_overrides(self, **overrides: str) -> SidekickDesignTokens:
        """Return a copy with selected token values replaced."""
        return SidekickDesignTokens({**self.values, **overrides})

    @classmethod
    def from_sidekick_tokens(cls, tokens: Mapping[str, str]) -> SidekickDesignTokens:
        """Create tokens from host maps using canonical ``sidekick.*`` names."""
        return cls(tokens)

    @classmethod
    def from_shared_theme(cls, theme_name: str = "light") -> SidekickDesignTokens:
        """Create Sidekick tokens from the fleet shared design-token schema."""
        return cls(_shared_theme_values(theme_name))


@dataclass(frozen=True)
class SidekickTerminalTheme:
    """Terminal-scoped colors with inherited Sidekick defaults."""

    mode: str = "inherit"
    values: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.mode not in {"inherit", "custom"}:
            raise ValueError("Terminal theme mode must be 'inherit' or 'custom'")
        normalized = _normalize_terminal_palette(self.values)
        object.__setattr__(self, "values", MappingProxyType(normalized))

    @classmethod
    def inherited(
        cls,
        tokens: SidekickDesignTokens | None = None,
    ) -> SidekickTerminalTheme:
        """Build terminal colors from resolved Sidekick design tokens."""
        token_set = tokens or SIDEKICK_DESIGN_TOKENS
        return cls(
            values={
                name: _resolve_terminal_color(value, token_set)
                for name, value in _DEFAULT_TERMINAL_PALETTE.items()
            },
        )

    @classmethod
    def custom(
        cls,
        *,
        foreground: str,
        background: str,
        cursor: str | None = None,
        selection: str | None = None,
        ansi: Mapping[str, str] | None = None,
    ) -> SidekickTerminalTheme:
        """Build a validated custom terminal palette."""
        palette = {
            "foreground": foreground,
            "background": background,
            "cursor": cursor or foreground,
            "selection": selection or background,
        }
        if ansi:
            palette.update({f"ansi.{name}": value for name, value in ansi.items()})
        return cls(mode="custom", values=palette)

    def __getitem__(self, name: str) -> str:
        return self.values[name]

    def qss(self, object_name: str) -> str:
        """Return QSS scoped to one terminal widget object name."""
        foreground = self.values["foreground"]
        background = self.values["background"]
        cursor = self.values["cursor"]
        selection = self.values["selection"]
        return f"""
QWidget#{object_name} QPlainTextEdit {{
    color: {foreground};
    background: {background};
    selection-background-color: {selection};
    selection-color: {foreground};
}}

QWidget#{object_name} QPlainTextEdit:focus {{
    border: 1px solid {cursor};
}}
""".strip()


SIDEKICK_DESIGN_TOKENS = SidekickDesignTokens.from_shared_theme("light")


def sidekick_qss(tokens: SidekickDesignTokens | None = None) -> str:
    """Build the canonical Qt stylesheet for Sidekick widgets."""
    token_set = tokens or SIDEKICK_DESIGN_TOKENS
    value = token_set.__getitem__
    return f"""
QWidget#{SIDEKICK_SIDEBAR_OBJECT_NAME} {{
    background: {value("color.background")};
    color: {value("color.text")};
    font-family: {value("font.family")};
    font-size: {value("font.size")};
}}

QDockWidget#{SIDEKICK_DOCK_OBJECT_NAME} {{
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
    border: 1px solid {value("color.border")};
}}

QToolBar#{SIDEKICK_TOOLBAR_OBJECT_NAME} {{
    background: {value("color.surface")};
    border: 0;
    border-bottom: 1px solid {value("color.border")};
    spacing: {value("space.1")};
    padding: {value("space.2")};
}}

QToolBar#{SIDEKICK_TOOLBAR_OBJECT_NAME} QToolButton {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: {value("radius.control")};
    color: {value("color.text.muted")};
    min-height: {value("size.control.height")};
    padding: {value("space.1")} {value("space.2")};
}}

QToolBar#{SIDEKICK_TOOLBAR_OBJECT_NAME} QToolButton:hover {{
    background: {value("color.accent.soft")};
    border-color: {value("color.border.strong")};
    color: {value("color.text")};
}}

QToolBar#{SIDEKICK_TOOLBAR_OBJECT_NAME} QToolButton:focus,
QTreeView#{SIDEKICK_PROJECT_TREE_OBJECT_NAME}:focus,
QListWidget#{SIDEKICK_WORKSPACE_LIST_OBJECT_NAME}:focus {{
    border: 1px solid {value("color.focus")};
}}

QTabWidget#{SIDEKICK_TABS_OBJECT_NAME}::pane {{
    border: 0;
    border-top: 1px solid {value("color.border.strong")};
    background: {value("color.surface")};
}}

QTabBar#{SIDEKICK_TAB_BAR_OBJECT_NAME}::tab {{
    background: {value("color.surface.raised")};
    border: 1px solid {value("color.border")};
    border-bottom: 1px solid {value("color.border.strong")};
    border-top-left-radius: 2px;
    border-top-right-radius: 2px;
    color: {value("color.text.muted")};
    margin-right: 0px;
    padding: {value("space.2")} {value("space.3")};
}}

QTabBar#{SIDEKICK_TAB_BAR_OBJECT_NAME}::tab:selected {{
    background: {value("color.surface")};
    border-color: {value("color.border.strong")};
    border-bottom-color: {value("color.surface")};
    margin-bottom: -1px;
    padding-bottom: {value("space.2")}; /* Keep height consistent */
    color: {value("color.text")};
}}

QTreeView#{SIDEKICK_PROJECT_TREE_OBJECT_NAME},
QListWidget#{SIDEKICK_WORKSPACE_LIST_OBJECT_NAME} {{
    background: {value("color.surface")};
    border: 0;
    color: {value("color.text")};
    selection-background-color: {value("color.selection")};
    selection-color: {value("color.text")};
}}

QLabel#{SIDEKICK_PLACEHOLDER_LABEL_OBJECT_NAME} {{
    color: {value("color.text.muted")};
    font-size: {value("font.size.small")};
    padding: {value("space.4")};
}}
""".strip()


def _normalize_terminal_palette(values: Mapping[str, str]) -> dict[str, str]:
    merged = {**_DEFAULT_TERMINAL_PALETTE, **dict(values)}
    color_keys = {"foreground", "background", "cursor", "selection"}
    for name, value in merged.items():
        if not (name in color_keys or name.startswith("ansi.")):
            continue
        color_value = str(value)
        if color_value in SIDEKICK_TOKEN_NAMES:
            continue
        if not _HEX_COLOR_RE.match(color_value):
            raise ValueError(f"Invalid terminal color for {name}: {color_value}")
    return merged


def _resolve_terminal_color(value: str, tokens: SidekickDesignTokens) -> str:
    if value in SIDEKICK_TOKEN_NAMES:
        return tokens[value]
    return value
