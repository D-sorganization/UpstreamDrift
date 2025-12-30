#!/usr/bin/env python3
"""
Create a cropped and optimized robot icon focusing on the robot character
This version crops the image to focus on the robot for better icon visibility
"""

import logging
from pathlib import Path

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
except ImportError:
    logging.error("PIL (Pillow) not installed. Install with: pip install Pillow")
    exit(1)


def auto_crop_robot(img: Image.Image) -> Image.Image:
    """Automatically crop the image to focus on the robot character"""

    # Convert to grayscale for edge detection
    gray = img.convert("L")

    # Find the bounding box of non-transparent content
    if img.mode == "RGBA":
        # Use alpha channel to find content bounds
        alpha = img.split()[-1]
        bbox = alpha.getbbox()
    else:
        # Use automatic cropping based on content
        bbox = ImageOps.invert(gray).getbbox()

    if bbox:
        # Add some padding around the detected content
        padding = 50
        left, top, right, bottom = bbox

        # Expand bbox with padding, but keep within image bounds
        left = max(0, left - padding)
        top = max(0, top - padding)
        right = min(img.width, right + padding)
        bottom = min(img.height, bottom + padding)

        # Make it square by expanding the smaller dimension
        width = right - left
        height = bottom - top

        if width > height:
            # Expand height
            diff = width - height
            top = max(0, top - diff // 2)
            bottom = min(img.height, bottom + diff // 2)
        elif height > width:
            # Expand width
            diff = height - width
            left = max(0, left - diff // 2)
            right = min(img.width, right + diff // 2)

        cropped = img.crop((left, top, right, bottom))
        logging.info(f"Auto-cropped from {img.size} to {cropped.size}")
        return cropped

    return img


def enhance_for_icon(img: Image.Image) -> Image.Image:
    """Apply enhancements specifically for icon use"""

    # Enhance contrast for better visibility at small sizes
    contrast_enhancer = ImageEnhance.Contrast(img)
    img = contrast_enhancer.enhance(1.15)

    # Enhance color saturation slightly
    color_enhancer = ImageEnhance.Color(img)
    img = color_enhancer.enhance(1.1)

    # Enhance sharpness
    sharpness_enhancer = ImageEnhance.Sharpness(img)
    img = sharpness_enhancer.enhance(1.2)

    return img


def create_sharp_icon(img: Image.Image, size: int) -> Image.Image:
    """Create a sharp icon at the specified size"""

    # Resize with high-quality resampling
    resized = img.resize((size, size), Image.Resampling.LANCZOS)

    # Apply size-specific enhancements
    if size <= 32:
        # For very small icons, apply aggressive sharpening
        resized = resized.filter(
            ImageFilter.UnsharpMask(radius=0.3, percent=200, threshold=1)
        )
        resized = resized.filter(ImageFilter.SHARPEN)

        # Boost contrast for tiny icons
        enhancer = ImageEnhance.Contrast(resized)
        resized = enhancer.enhance(1.2)

    elif size <= 64:
        # Medium icons get moderate enhancement
        resized = resized.filter(
            ImageFilter.UnsharpMask(radius=0.5, percent=150, threshold=2)
        )

    elif size <= 128:
        # Larger icons get subtle enhancement
        resized = resized.filter(
            ImageFilter.UnsharpMask(radius=1.0, percent=100, threshold=3)
        )

    return resized


def create_cropped_robot_icon() -> bool:
    """Create cropped robot icon focusing on the robot character"""

    source_image = Path("GolfingRobot.png")
    assets_dir = Path("launchers/assets")

    if not source_image.exists():
        logging.error(f"Source image not found: {source_image}")
        return False

    if not assets_dir.exists():
        logging.error(f"Assets directory not found: {assets_dir}")
        return False

    try:
        # Load and process the source image
        img = Image.open(source_image)
        logging.info(f"Loaded source image: {img.size}")

        # Convert to RGBA
        if img.mode != "RGBA":
            img = img.convert("RGBA")  # type: ignore[assignment]

        # Auto-crop to focus on the robot
        cropped_img = auto_crop_robot(img)

        # Enhance for icon use
        enhanced_img = enhance_for_icon(cropped_img)

        # Save the cropped source for reference
        cropped_path = assets_dir / "golf_robot_cropped.png"
        enhanced_img.save(cropped_path, "PNG", optimize=True)
        logging.info(f"Saved cropped robot image: {cropped_path}")

        # Create high-quality icons
        icon_sizes = [16, 20, 24, 32, 40, 48, 64, 128, 256]
        ico_images = []

        for size in icon_sizes:
            sharp_icon = create_sharp_icon(enhanced_img, size)
            ico_images.append(sharp_icon)
            logging.info(f"Created sharp {size}x{size} icon")

        # Save the main PNG icon (256x256)
        main_icon_path = assets_dir / "golf_robot_cropped_icon.png"
        ico_images[-1].save(main_icon_path, "PNG", optimize=True)
        logging.info(f"Saved main icon: {main_icon_path}")

        # Create ICO file with all sizes
        ico_path = assets_dir / "golf_robot_cropped_icon.ico"
        ico_images[0].save(
            ico_path,
            format="ICO",
            sizes=[(s, s) for s in icon_sizes],
            append_images=ico_images[1:],
        )
        logging.info(f"Created ICO file: {ico_path}")

        # Create a super sharp favicon
        favicon = create_sharp_icon(enhanced_img, 32)
        favicon_path = assets_dir / "favicon_cropped.ico"
        favicon.save(favicon_path, "ICO")
        logging.info(f"Created sharp favicon: {favicon_path}")

        return True

    except Exception as e:
        logging.error(f"Error creating cropped robot icon: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if create_cropped_robot_icon():
        logging.info("Successfully created cropped robot icons!")
        logging.info("Check the assets folder for:")
        logging.info("- golf_robot_cropped_icon.png (main icon)")
        logging.info("- golf_robot_cropped_icon.ico (Windows icon)")
        logging.info("- favicon_cropped.ico (web favicon)")
        logging.info("- golf_robot_cropped.png (cropped source)")
    else:
        logging.error("Failed to create cropped robot icons")
        exit(1)
