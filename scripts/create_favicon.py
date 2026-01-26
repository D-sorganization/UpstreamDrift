#!/usr/bin/env python3
"""
Create favicon and icon files from GolfingRobot.png using shared utilities.
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


def create_favicon() -> int:
    """Create favicon and icon files from GolfingRobot.png."""
    source_image = Path("GolfingRobot.png")
    assets_dir = Path("src/launchers/assets")

    if not assets_dir.exists():
        logger.info(f"Creating assets directory: {assets_dir}")
        assets_dir.mkdir(parents=True, exist_ok=True)

    try:
        img = load_icon_source(source_image)
    except FileNotFoundError as e:
        logger.error(str(e))
        return 1

    # Enhance
    img = enhance_icon_source(img)

    # Save outputs
    save_ico(img, assets_dir / "golf_robot_icon.ico")
    save_ico(img, assets_dir / "favicon.ico", sizes=[32])
    save_png_icons(img, assets_dir, "golf_robot_icon", [256])

    logger.info("Successfully created icon files!")
    return 0


if __name__ == "__main__":
    run_main(create_favicon, logger)
