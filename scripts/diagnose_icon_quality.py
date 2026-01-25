#!/usr/bin/env python3
"""
Diagnose icon quality issues and create ultra-sharp icons using shared utilities.
"""

from pathlib import Path

from scripts.script_utils import run_main, setup_script_logging
from src.shared.python.image_utils import (
    Image,
    analyze_image_quality,
    save_ico,
    save_png_icons,
)

logger = setup_script_logging(__name__)


def diagnose_and_fix_icons() -> int:
    """Diagnose current icon quality and create ultra-sharp versions."""
    source_image = Path("GolfingRobot.png")
    assets_dir = Path("src/launchers/assets")

    if not source_image.exists():
        logger.error(f"Source image not found: {source_image}")
        return 1

    # 1. Analyze source image
    logger.info("=== SOURCE IMAGE ANALYSIS ===")
    source_analysis = analyze_image_quality(source_image)
    for key, value in source_analysis.items():
        logger.info(f"{key}: {value}")

    # 2. Analyze current icons
    logger.info("\n=== CURRENT ICON ANALYSIS ===")
    current_icons = [
        "golf_robot_icon.png",
        "golf_robot_cropped_icon.png",
        "golf_icon.png",
    ]

    for icon_name in current_icons:
        icon_path = assets_dir / icon_name
        if icon_path.exists():
            logger.info(f"\n--- {icon_name} ---")
            analysis = analyze_image_quality(icon_path)
            for key, value in analysis.items():
                logger.info(f"{key}: {value}")

    # 3. Create ultra-sharp icons
    logger.info("\n=== CREATING ULTRA-SHARP ICONS ===")
    img = Image.open(source_image)
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    ultra_sharp_sizes = [16, 20, 24, 32, 48, 64, 128, 256]

    # Save individual PNGs for testing
    save_png_icons(img, assets_dir, "ultra_sharp", ultra_sharp_sizes, mode="ultra")

    # Create the ultra-sharp ICO
    save_ico(
        img,
        assets_dir / "golf_robot_ultra_sharp.ico",
        sizes=ultra_sharp_sizes,
        mode="ultra",
    )

    # Save the main 256x256 ultra-sharp PNG
    save_png_icons(img, assets_dir, "golf_robot_ultra_sharp", [256], mode="ultra")

    # 4. Analyze the new ultra-sharp icon
    logger.info("\n=== ULTRA-SHARP ICON ANALYSIS ===")
    main_path = assets_dir / "golf_robot_ultra_sharp_256.png"
    analysis = analyze_image_quality(main_path)
    for key, value in analysis.items():
        logger.info(f"{key}: {value}")

    logger.info("\n=== DIAGNOSIS COMPLETE ===")
    return 0


if __name__ == "__main__":
    run_main(diagnose_and_fix_icons, logger)
