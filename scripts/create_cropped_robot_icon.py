#!/usr/bin/env python3
"""
Create a cropped and optimized robot icon focusing on the robot character.
Uses shared image_utils for auto-cropping and enhancement.
"""

from pathlib import Path

from scripts.script_utils import run_main, setup_script_logging
from src.shared.python.image_utils import (
    Image,
    auto_crop_to_content,
    enhance_icon_source,
    save_ico,
    save_png_icons,
)

logger = setup_script_logging(__name__)


def create_cropped_robot_icon() -> int:
    """Create cropped robot icon focusing on the robot character."""
    source_image = Path("GolfingRobot.png")
    assets_dir = Path("src/launchers/assets")

    if not source_image.exists():
        logger.error(f"Source image not found: {source_image}")
        return 1

    img = Image.open(source_image)
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    # 1. Auto-crop to focus on robot
    img = auto_crop_to_content(img, padding=50)

    # 2. Enhance for icon use
    img = enhance_icon_source(img, contrast=1.2, sharpness=1.2, color=1.1)

    # 3. Save outputs
    save_ico(
        img,
        assets_dir / "golf_robot_cropped_icon.ico",
        sizes=[16, 20, 24, 32, 40, 48, 64, 128, 256],
    )

    save_ico(img, assets_dir / "favicon_cropped.ico", sizes=[32])

    save_png_icons(img, assets_dir, "golf_robot_cropped_icon", [256])

    # Save cropped source for reference
    img.save(assets_dir / "golf_robot_cropped.png", "PNG", optimize=True)

    logger.info("Successfully created cropped robot icons!")
    return 0


if __name__ == "__main__":
    run_main(create_cropped_robot_icon, logger)
