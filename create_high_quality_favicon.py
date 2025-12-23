#!/usr/bin/env python3
"""
Create high-quality favicon and icon files from GolfingRobot.png
Uses advanced image processing techniques to maintain sharpness at all sizes
"""
import logging
from pathlib import Path

try:
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    logging.error("PIL (Pillow) not installed. Install with: pip install Pillow")
    exit(1)


def sharpen_small_icon(img: Image.Image, target_size: int) -> Image.Image:
    """Apply size-specific sharpening and enhancement for small icons"""

    # First resize with high-quality resampling
    resized = img.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # For very small sizes, apply aggressive sharpening
    if target_size <= 32:
        # Enhance contrast slightly for small icons
        enhancer = ImageEnhance.Contrast(resized)
        resized = enhancer.enhance(1.1)

        # Apply unsharp mask for better edge definition
        resized = resized.filter(
            ImageFilter.UnsharpMask(radius=0.5, percent=150, threshold=2)
        )

        # For tiny icons, apply additional sharpening
        if target_size <= 24:
            resized = resized.filter(ImageFilter.SHARPEN)

    elif target_size <= 64:
        # Medium sizes get moderate sharpening
        resized = resized.filter(
            ImageFilter.UnsharpMask(radius=1.0, percent=100, threshold=1)
        )

    return resized


def create_high_quality_favicon():
    """Create high-quality favicon and icon files from GolfingRobot.png"""

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
            img = img.convert("RGBA")

        # Pre-process the image for better icon quality
        # Enhance contrast slightly for better icon visibility
        contrast_enhancer = ImageEnhance.Contrast(img)
        img = contrast_enhancer.enhance(1.05)

        # Enhance sharpness of the source
        sharpness_enhancer = ImageEnhance.Sharpness(img)
        img = sharpness_enhancer.enhance(1.1)

        # If source is very large, resize to optimal intermediate size first
        if max(img.size) > 1024:
            # Calculate aspect ratio preserving resize
            ratio = min(1024 / img.width, 1024 / img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            logging.info(f"Pre-resized to intermediate size: {img.size}")

        # Create high-quality PNG icon (256x256)
        png_icon = sharpen_small_icon(img, 256)
        png_path = assets_dir / "golf_robot_icon.png"
        png_icon.save(png_path, "PNG", optimize=True, compress_level=6)
        logging.info(f"Created high-quality PNG icon: {png_path}")

        # Create ICO file with multiple sizes using optimized processing
        ico_sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256]
        ico_images = []

        for size in ico_sizes:
            processed_icon = sharpen_small_icon(img, size)
            ico_images.append(processed_icon)
            logging.info(f"Processed {size}x{size} icon with quality enhancement")

        ico_path = assets_dir / "golf_robot_icon.ico"
        ico_images[0].save(
            ico_path,
            format="ICO",
            sizes=[(s, s) for s in ico_sizes],
            append_images=ico_images[1:],
        )
        logging.info(f"Created high-quality ICO file: {ico_path}")

        # Create web favicon (32x32) with maximum sharpness
        favicon = sharpen_small_icon(img, 32)
        favicon_path = assets_dir / "favicon.ico"
        favicon.save(favicon_path, "ICO")
        logging.info(f"Created sharp web favicon: {favicon_path}")

        # Create additional PNG sizes for different use cases
        for size in [48, 64, 128]:
            size_icon = sharpen_small_icon(img, size)
            size_path = assets_dir / f"golf_robot_icon_{size}.png"
            size_icon.save(size_path, "PNG", optimize=True)
            logging.info(f"Created {size}x{size} PNG: {size_path}")

        return True

    except Exception as e:
        logging.error(f"Error creating high-quality icons: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if create_high_quality_favicon():
        logging.info("Successfully created all high-quality icon files!")
    else:
        logging.error("Failed to create high-quality icon files")
        exit(1)
