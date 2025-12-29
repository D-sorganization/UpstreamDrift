#!/usr/bin/env python3
"""
Create favicon and icon files from GolfingRobot.png
"""
import logging
from pathlib import Path

try:
    from PIL import Image
except ImportError:
    logging.error("PIL (Pillow) not installed. Install with: pip install Pillow")
    exit(1)


def create_favicon() -> None:
    """Create favicon and icon files from GolfingRobot.png"""

    # Paths
    source_image = Path("GolfingRobot.png")
    assets_dir = Path("launchers/assets")

    if not source_image.exists():
        logging.error(f"Source image not found: {source_image}")
        return False

    if not assets_dir.exists():
        logging.error(f"Assets directory not found: {assets_dir}")
        return False

    try:
        # Load the source image
        img = Image.open(source_image)
        logging.info(f"Loaded image: {source_image} ({img.size})")

        # Convert to RGBA if needed
        if img.mode != "RGBA":
            img = img.convert("RGBA")  # type: ignore[assignment]

        # Create high-quality PNG icon (256x256)
        png_icon = img.resize((256, 256), Image.Resampling.LANCZOS)
        png_path = assets_dir / "golf_robot_icon.png"
        png_icon.save(png_path, "PNG")
        logging.info(f"Created PNG icon: {png_path}")

        # Create ICO file with multiple sizes
        ico_sizes = [16, 32, 48, 64, 128, 256]
        ico_images = []

        for size in ico_sizes:
            resized = img.resize((size, size), Image.Resampling.LANCZOS)
            ico_images.append(resized)

        ico_path = assets_dir / "golf_robot_icon.ico"
        ico_images[0].save(
            ico_path,
            format="ICO",
            sizes=[(s, s) for s in ico_sizes],
            append_images=ico_images[1:],
        )
        logging.info(f"Created ICO file: {ico_path}")

        # Create web favicon (32x32)
        favicon = img.resize((32, 32), Image.Resampling.LANCZOS)
        favicon_path = assets_dir / "favicon.ico"
        favicon.save(favicon_path, "ICO")
        logging.info(f"Created web favicon: {favicon_path}")

        return True

    except Exception as e:
        logging.error(f"Error creating icons: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if create_favicon():
        logging.info("Successfully created all icon files!")
    else:
        logging.error("Failed to create icon files")
        exit(1)
