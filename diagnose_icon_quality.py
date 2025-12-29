#!/usr/bin/env python3
"""
Diagnose icon quality issues and create ultra-sharp icons
"""
import logging
from pathlib import Path

try:
    import numpy as np
    from PIL import Image, ImageEnhance, ImageFilter
except ImportError as e:
    logging.error(f"Required packages not installed: {e}")
    logging.error("Install with: pip install Pillow numpy")
    exit(1)


def analyze_image_quality(image_path: Path) -> dict:
    """Analyze image quality metrics"""
    if not image_path.exists():
        return {"error": f"File not found: {image_path}"}

    try:
        img = Image.open(image_path)

        # Convert to grayscale for analysis
        gray = img.convert("L")
        img_array = np.array(gray)

        # Calculate sharpness metrics
        # Laplacian variance (higher = sharper)
        laplacian_var = np.var(
            np.array(
                gray.filter(
                    ImageFilter.Kernel((3, 3), [-1, -1, -1, -1, 8, -1, -1, -1, -1])
                )
            )
        )

        # Edge density
        edges = gray.filter(ImageFilter.FIND_EDGES)
        edge_density = np.mean(np.array(edges))

        # Contrast
        contrast = np.std(img_array)

        return {
            "size": img.size,
            "mode": img.mode,
            "format": img.format,
            "laplacian_variance": float(laplacian_var),
            "edge_density": float(edge_density),
            "contrast": float(contrast),
            "file_size_kb": image_path.stat().st_size / 1024,
        }
    except Exception as e:
        return {"error": str(e)}


def create_ultra_sharp_icon(source_img: Image.Image, target_size: int) -> Image.Image:
    """Create an ultra-sharp icon using advanced techniques"""

    # Start with the source image
    working_img = source_img.copy()

    # If source is much larger than target, do intermediate resize first
    if min(working_img.size) > target_size * 4:
        # Resize to 4x target size first
        intermediate_size = target_size * 4
        working_img = working_img.resize(
            (intermediate_size, intermediate_size), Image.Resampling.LANCZOS
        )

    # Apply pre-sharpening to source
    working_img = working_img.filter(
        ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=1)
    )

    # Enhance contrast before resizing
    enhancer = ImageEnhance.Contrast(working_img)
    working_img = enhancer.enhance(1.1)

    # Final resize with highest quality
    final_img = working_img.resize((target_size, target_size), Image.Resampling.LANCZOS)

    # Apply size-specific post-processing
    if target_size <= 16:
        # Extreme sharpening for tiny icons
        final_img = final_img.filter(
            ImageFilter.UnsharpMask(radius=0.2, percent=300, threshold=0)
        )
        final_img = final_img.filter(ImageFilter.SHARPEN)
        final_img = final_img.filter(ImageFilter.SHARPEN)  # Double sharpen

        # Boost contrast heavily for tiny icons
        enhancer = ImageEnhance.Contrast(final_img)
        final_img = enhancer.enhance(1.4)

    elif target_size <= 32:
        # Heavy sharpening for small icons
        final_img = final_img.filter(
            ImageFilter.UnsharpMask(radius=0.3, percent=250, threshold=1)
        )
        final_img = final_img.filter(ImageFilter.SHARPEN)

        # Boost contrast for small icons
        enhancer = ImageEnhance.Contrast(final_img)
        final_img = enhancer.enhance(1.3)

    elif target_size <= 64:
        # Moderate sharpening for medium icons
        final_img = final_img.filter(
            ImageFilter.UnsharpMask(radius=0.5, percent=180, threshold=2)
        )

        # Slight contrast boost
        enhancer = ImageEnhance.Contrast(final_img)
        final_img = enhancer.enhance(1.15)

    else:
        # Subtle sharpening for large icons
        final_img = final_img.filter(
            ImageFilter.UnsharpMask(radius=1.0, percent=120, threshold=3)
        )

    return final_img


def diagnose_and_fix_icons() -> None:
    """Diagnose current icon quality and create ultra-sharp versions"""

    source_image = Path("GolfingRobot.png")
    assets_dir = Path("launchers/assets")

    if not source_image.exists():
        logging.error(f"Source image not found: {source_image}")
        return False

    # Analyze source image
    logging.info("=== SOURCE IMAGE ANALYSIS ===")
    source_analysis = analyze_image_quality(source_image)
    for key, value in source_analysis.items():
        logging.info(f"{key}: {value}")

    # Analyze current icons
    logging.info("\n=== CURRENT ICON ANALYSIS ===")
    current_icons = [
        "golf_robot_icon.png",
        "golf_robot_cropped_icon.png",
        "golf_icon.png",
    ]

    for icon_name in current_icons:
        icon_path = assets_dir / icon_name
        if icon_path.exists():
            logging.info(f"\n--- {icon_name} ---")
            analysis = analyze_image_quality(icon_path)
            for key, value in analysis.items():
                logging.info(f"{key}: {value}")

    # Load source and create ultra-sharp icons
    logging.info("\n=== CREATING ULTRA-SHARP ICONS ===")

    try:
        source_img = Image.open(source_image)
        if source_img.mode != "RGBA":
            source_img = source_img.convert("RGBA")  # type: ignore[assignment]

        # Create ultra-sharp versions
        ultra_sharp_sizes = [16, 20, 24, 32, 48, 64, 128, 256]
        ultra_sharp_images = []

        for size in ultra_sharp_sizes:
            sharp_icon = create_ultra_sharp_icon(source_img, size)
            ultra_sharp_images.append(sharp_icon)

            # Save individual PNG for testing
            test_path = assets_dir / f"ultra_sharp_{size}.png"
            sharp_icon.save(test_path, "PNG", optimize=True)
            logging.info(f"Created ultra-sharp {size}x{size}: {test_path}")

        # Create ultra-sharp ICO file
        ico_path = assets_dir / "golf_robot_ultra_sharp.ico"
        ultra_sharp_images[0].save(
            ico_path,
            format="ICO",
            sizes=[(s, s) for s in ultra_sharp_sizes],
            append_images=ultra_sharp_images[1:],
        )
        logging.info(f"Created ultra-sharp ICO: {ico_path}")

        # Create main ultra-sharp PNG (256x256)
        main_path = assets_dir / "golf_robot_ultra_sharp.png"
        ultra_sharp_images[-1].save(main_path, "PNG", optimize=True)
        logging.info(f"Created main ultra-sharp PNG: {main_path}")

        # Analyze the new ultra-sharp icon
        logging.info("\n=== ULTRA-SHARP ICON ANALYSIS ===")
        analysis = analyze_image_quality(main_path)
        for key, value in analysis.items():
            logging.info(f"{key}: {value}")

        return True

    except Exception as e:
        logging.error(f"Error creating ultra-sharp icons: {e}")
        return False


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    if diagnose_and_fix_icons():
        logging.info("\n=== DIAGNOSIS COMPLETE ===")
        logging.info("Ultra-sharp icons created!")
        logging.info("Next steps:")
        logging.info("1. Test the ultra-sharp icons in the application")
        logging.info("2. Update launcher to use golf_robot_ultra_sharp.png")
        logging.info("3. Update shortcut to use golf_robot_ultra_sharp.ico")
    else:
        logging.error("Diagnosis failed")
        exit(1)
