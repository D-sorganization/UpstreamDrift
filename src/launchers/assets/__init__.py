"""Launcher asset utilities: tile image generation and asset optimization."""

from .generate_tile_images import (
    TILE_CONFIGS,
    create_png,
    draw_letter,
    draw_rounded_rect_with_text,
    generate_all_tiles,
    hex_to_rgb,
)
from .optimize_assets import (
    ICON_SIZES,
    MAX_ICON_SIZE_KB,
    MAX_IMAGE_SIZE_KB,
    analyze_assets,
    check_dependencies,
    get_image_info,
    main,
    optimize_png,
)

__all__: list[str] = [
    "ICON_SIZES",
    "MAX_ICON_SIZE_KB",
    "MAX_IMAGE_SIZE_KB",
    "TILE_CONFIGS",
    "analyze_assets",
    "check_dependencies",
    "create_png",
    "draw_letter",
    "draw_rounded_rect_with_text",
    "generate_all_tiles",
    "get_image_info",
    "hex_to_rgb",
    "main",
    "optimize_png",
]
