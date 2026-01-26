#!/usr/bin/env python3
"""
Create Windows-optimized icons that address common blurriness issues.
Uses shared image_utils with 'extreme' sharpening mode.
"""

from pathlib import Path

from scripts.script_utils import run_main, setup_script_logging
from src.shared.python.image_utils import (
    enhance_icon_source,
    load_icon_source,
    save_ico,
    save_png_icons,
)

logger = setup_script_logging(__name__)


def create_windows_optimized_icons() -> int:
    """Create Windows-optimized icons."""
    source_image = Path("GolfingRobot.png")
    assets_dir = Path("src/launchers/assets")

    try:
        img = load_icon_source(source_image)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    # Windows-specific enhancements
    img = enhance_icon_source(img, contrast=1.1)

    # Standard Windows sizes
    windows_sizes = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]

    # Save multiple PNGs for testing/individual use
    save_png_icons(img, assets_dir, "windows_optimized", windows_sizes, mode="extreme")

    # Create the main optimized ICO
    save_ico(
        img,
        assets_dir / "golf_robot_windows_optimized.ico",
        sizes=windows_sizes,
        mode="extreme",
    )

    # Create main large PNG
    save_png_icons(
        img, assets_dir, "golf_robot_windows_optimized", [256], mode="extreme"
    )

    logger.info("Windows-optimized icons created successfully!")
    return 0


if __name__ == "__main__":
    run_main(create_windows_optimized_icons, logger)
