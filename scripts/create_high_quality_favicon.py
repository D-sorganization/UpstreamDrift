#!/usr/bin/env python3
"""
Create high-quality favicon and icon files from GolfingRobot.png.
Uses shared image_utils for advanced sharpening and enhancement.
"""

from pathlib import Path

from scripts.script_utils import run_main, setup_script_logging
from src.shared.python.image_utils import (
    Image,
    enhance_icon_source,
    save_ico,
    save_png_icons,
)

logger = setup_script_logging(__name__)


def create_high_quality_favicon() -> int:
    """Create high-quality favicon and icon files."""
    source_image = Path("GolfingRobot.png")
    assets_dir = Path("src/launchers/assets")

    if not source_image.exists():
        logger.error(f"Source image not found: {source_image}")
        return 1

    img = Image.open(source_image)
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # Use aggressive enhancements for "high quality" variant
    img = enhance_icon_source(img, contrast=1.15, sharpness=1.2)

    # Multi-size ICO
    save_ico(img, assets_dir / "golf_robot_icon.ico", 
             sizes=[16, 20, 24, 32, 40, 48, 64, 128, 256])
    
    # Sharp web favicon
    save_ico(img, assets_dir / "favicon.ico", sizes=[32])

    # PNG variants
    save_png_icons(img, assets_dir, "golf_robot_icon", [48, 64, 128, 256])

    logger.info("Successfully created high-quality icon files!")
    return 0


if __name__ == "__main__":
    run_main(create_high_quality_favicon, logger)
