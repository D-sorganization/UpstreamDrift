#!/usr/bin/env python3
"""
Create Windows-optimized icons that address common blurriness issues
"""
import logging
from pathlib import Path

try:
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError:
    logging.error("PIL (Pillow) not installed. Install with: pip install Pillow")
    exit(1)


def create_windows_optimized_icon(source_img: Image.Image, size: int) -> Image.Image:
    """Create Windows-optimized icon with pixel-perfect clarity"""

    # For Windows, we need to be very aggressive with small sizes
    working_img = source_img.copy()

    # Pre-process: enhance the source image
    if size <= 32:
        # For very small Windows icons, apply extreme processing

        # Step 1: Resize to 2x target size first for better quality
        intermediate = working_img.resize(
            (size * 2, size * 2), Image.Resampling.LANCZOS
        )

        # Step 2: Apply heavy contrast enhancement
        contrast_enhancer = ImageEnhance.Contrast(intermediate)
        intermediate = contrast_enhancer.enhance(1.5)

        # Step 3: Apply aggressive sharpening
        intermediate = intermediate.filter(
            ImageFilter.UnsharpMask(radius=0.5, percent=400, threshold=0)
        )

        # Step 4: Final resize to target size
        final = intermediate.resize((size, size), Image.Resampling.LANCZOS)

        # Step 5: Post-process sharpening
        final = final.filter(ImageFilter.SHARPEN)
        final = final.filter(ImageFilter.SHARPEN)  # Double sharpen

        # Step 6: Final contrast boost
        final_enhancer = ImageEnhance.Contrast(final)
        final = final_enhancer.enhance(1.2)

    else:
        # For larger icons, use standard high-quality processing
        final = working_img.resize((size, size), Image.Resampling.LANCZOS)
        final = final.filter(
            ImageFilter.UnsharpMask(radius=1.0, percent=150, threshold=2)
        )

        # Moderate contrast enhancement
        enhancer = ImageEnhance.Contrast(final)
        final = enhancer.enhance(1.1)

    return final


def create_windows_optimized_icons() -> bool:
    """Create Windows-optimized icons"""

    source_image = Path("GolfingRobot.png")
    assets_dir = Path("launchers/assets")

    if not source_image.exists():
        logging.error(f"Source image not found: {source_image}")
        return False

    try:
        # Load source image
        source_img = Image.open(source_image)
        if source_img.mode != "RGBA":
            source_img = source_img.convert("RGBA")  # type: ignore[assignment]

        logging.info(f"Creating Windows-optimized icons from {source_img.size} source")

        # Windows standard icon sizes (including odd sizes that Windows uses)
        windows_sizes = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
        windows_icons = []

        for size in windows_sizes:
            optimized_icon = create_windows_optimized_icon(source_img, size)
            windows_icons.append(optimized_icon)

            # Save individual test files
            test_path = assets_dir / f"windows_optimized_{size}.png"
            optimized_icon.save(test_path, "PNG", optimize=True)
            logging.info(f"Created Windows-optimized {size}x{size}: {test_path}")

        # Create the Windows-optimized ICO file
        ico_path = assets_dir / "golf_robot_windows_optimized.ico"
        windows_icons[0].save(
            ico_path,
            format="ICO",
            sizes=[(s, s) for s in windows_sizes],
            append_images=windows_icons[1:],
        )
        logging.info(f"Created Windows-optimized ICO: {ico_path}")

        # Create main PNG (256x256)
        main_path = assets_dir / "golf_robot_windows_optimized.png"
        windows_icons[-1].save(main_path, "PNG", optimize=True)
        logging.info(f"Created main Windows-optimized PNG: {main_path}")

        return True

    except Exception as e:
        logging.error(f"Error creating Windows-optimized icons: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if create_windows_optimized_icons():
        logging.info("Windows-optimized icons created successfully!")
        logging.info("These icons are specifically tuned for Windows display clarity.")
    else:
        logging.error("Failed to create Windows-optimized icons")
        exit(1)
