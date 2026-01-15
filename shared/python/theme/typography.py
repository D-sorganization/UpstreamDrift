"""Typography system for Golf Modeling Suite.

This module provides cross-platform font stacks and typography utilities
that work consistently across Windows, macOS, and Linux.

Design principles:
- Graceful fallbacks for all platforms
- Clear typographic hierarchy
- Consistent sizing and weights
- Professional, readable fonts
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from PyQt6.QtGui import QFont

# Cross-platform font stacks
# Each stack provides platform-specific fonts with sensible fallbacks

# Primary UI font - clean sans-serif for interface elements
FONT_STACK_UI = (
    '"Inter", '  # Modern, highly legible (if installed)
    '"SF Pro Display", '  # macOS system font
    '"Segoe UI", '  # Windows system font
    '"Roboto", '  # Android/Linux common
    '"Helvetica Neue", '  # macOS fallback
    "system-ui, "  # Browser system font
    "sans-serif"  # Final fallback
)

# Monospace font - for code, data displays, and technical content
FONT_STACK_MONO = (
    '"JetBrains Mono", '  # Modern coding font
    '"SF Mono", '  # macOS monospace
    '"Cascadia Code", '  # Windows Terminal font
    '"Consolas", '  # Windows fallback
    '"Monaco", '  # macOS fallback
    '"Liberation Mono", '  # Linux
    "monospace"  # Final fallback
)

# Display font - for headings and titles
FONT_STACK_DISPLAY = (
    '"SF Pro Display", '  # macOS
    '"Segoe UI", '  # Windows
    '"Inter", '  # Cross-platform
    '"Roboto", '  # Android/Linux
    "sans-serif"
)


@dataclass(frozen=True)
class FontSizes:
    """Standard font sizes in points (pt).

    These follow a modular scale for visual harmony.
    """

    # Extra small - metadata, timestamps
    XS: ClassVar[int] = 8

    # Small - captions, help text
    SM: ClassVar[int] = 9

    # Base - body text, labels
    BASE: ClassVar[int] = 10

    # Medium - emphasized body, subheadings
    MD: ClassVar[int] = 11

    # Large - section headings
    LG: ClassVar[int] = 13

    # Extra large - page titles
    XL: ClassVar[int] = 16

    # 2X large - splash screens, hero text
    XXL: ClassVar[int] = 24

    # 3X large - major splash titles
    XXXL: ClassVar[int] = 32


@dataclass(frozen=True)
class FontWeights:
    """Standard font weights.

    Maps to Qt Font.Weight values and CSS font-weight.
    """

    # Thin weight
    THIN: ClassVar[int] = 100

    # Light weight
    LIGHT: ClassVar[int] = 300

    # Normal/regular weight
    NORMAL: ClassVar[int] = 400

    # Medium weight
    MEDIUM: ClassVar[int] = 500

    # Semi-bold weight
    SEMIBOLD: ClassVar[int] = 600

    # Bold weight
    BOLD: ClassVar[int] = 700

    # Extra bold weight
    EXTRABOLD: ClassVar[int] = 800


# Convenience instances
Sizes = FontSizes()
Weights = FontWeights()


def get_qfont(
    size: int = Sizes.BASE,
    weight: int = Weights.NORMAL,
    family: str | None = None,
    italic: bool = False,
) -> "QFont":
    """Create a QFont with the standard font stack.

    Args:
        size: Font size in points
        weight: Font weight (use FontWeights constants)
        family: Optional specific font family (uses UI stack if None)
        italic: Whether to use italic style

    Returns:
        Configured QFont instance

    Note:
        This is a lazy import to avoid Qt dependency in non-GUI contexts.
    """
    from PyQt6.QtGui import QFont

    font = QFont()

    # Set font families - Qt will try each in order
    if family:
        font.setFamily(family)
    else:
        # Parse the font stack and set families
        families = [f.strip().strip('"') for f in FONT_STACK_UI.split(",")]
        font.setFamilies(families)

    font.setPointSize(size)
    font.setWeight(QFont.Weight(weight))
    font.setItalic(italic)

    return font


def get_display_font(size: int = Sizes.XL, weight: int = Weights.BOLD) -> "QFont":
    """Get a display/heading font.

    Args:
        size: Font size in points
        weight: Font weight

    Returns:
        Configured QFont for headings
    """
    from PyQt6.QtGui import QFont

    font = QFont()
    families = [f.strip().strip('"') for f in FONT_STACK_DISPLAY.split(",")]
    font.setFamilies(families)
    font.setPointSize(size)
    font.setWeight(QFont.Weight(weight))

    return font


def get_mono_font(size: int = Sizes.BASE, weight: int = Weights.NORMAL) -> "QFont":
    """Get a monospace font for code/data display.

    Args:
        size: Font size in points
        weight: Font weight

    Returns:
        Configured monospace QFont
    """
    from PyQt6.QtGui import QFont

    font = QFont()
    families = [f.strip().strip('"') for f in FONT_STACK_MONO.split(",")]
    font.setFamilies(families)
    font.setPointSize(size)
    font.setWeight(QFont.Weight(weight))

    return font


# CSS-compatible font stack strings for stylesheets
CSS_FONT_UI = f"font-family: {FONT_STACK_UI};"
CSS_FONT_MONO = f"font-family: {FONT_STACK_MONO};"
CSS_FONT_DISPLAY = f"font-family: {FONT_STACK_DISPLAY};"


__all__ = [
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
]
