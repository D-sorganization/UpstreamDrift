"""Image processing utilities for the Golf Modeling Suite.

This module provides high-quality image enhancement, sharpening, and
auto-cropping capabilities specifically optimized for icon generation.

Principles:
- Orthogonality: Decouples image processing logic from script entry points.
- DRY: Centralizes complex Pillow-based enhancement routines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.shared.python.logging_config import get_logger

try:
    from PIL import Image, ImageEnhance, ImageFilter, ImageOps
    PILLOW_AVAILABLE = True
except ImportError:
    PILLOW_AVAILABLE = False

logger = get_logger(__name__)


def ensure_pillow() -> None:
    """Ensure Pillow is installed."""
    if not PILLOW_AVAILABLE:
        raise ImportError(
            "Pillow (PIL) is not installed. Install with: pip install Pillow"
        )


def auto_crop_to_content(img: Image.Image, padding: int = 50) -> Image.Image:
    """Automatically crop the image to focus on the non-transparent content.

    Args:
        img: Source PIL Image.
        padding: Padding in pixels to add around the detected content.

    Returns:
        Cropped and squared PIL Image.
    """
    ensure_pillow()

    # Find bounding box
    if img.mode == "RGBA":
        alpha = img.split()[-1]
        bbox = alpha.getbbox()
    else:
        gray = img.convert("L")
        bbox = ImageOps.invert(gray).getbbox()

    if not bbox:
        return img

    left, top, right, bottom = bbox

    # Apply padding
    left = max(0, left - padding)
    top = max(0, top - padding)
    right = min(img.width, right + padding)
    bottom = min(img.height, bottom + padding)

    # Force square
    width = right - left
    height = bottom - top

    if width > height:
        diff = width - height
        top = max(0, top - diff // 2)
        bottom = min(img.height, bottom + diff // 2)
    elif height > width:
        diff = height - width
        left = max(0, left - diff // 2)
        right = min(img.width, right + diff // 2)

    return img.crop((left, top, right, bottom))


def enhance_icon_source(
    img: Image.Image,
    contrast: float = 1.1,
    sharpness: float = 1.1,
    color: float = 1.05
) -> Image.Image:
    """Apply global enhancements to an image intended for icon use.

    Args:
        img: Source PIL Image.
        contrast: Contrast enhancement factor.
        sharpness: Sharpness enhancement factor.
        color: Color saturation enhancement factor.

    Returns:
        Enhanced PIL Image.
    """
    ensure_pillow()

    if contrast != 1.0:
        img = ImageEnhance.Contrast(img).enhance(contrast)
    if color != 1.0:
        img = ImageEnhance.Color(img).enhance(color)
    if sharpness != 1.0:
        img = ImageEnhance.Sharpness(img).enhance(sharpness)

    return img


def create_optimized_icon(
    img: Image.Image,
    size: int,
    mode: str = "standard"
) -> Image.Image:
    """Create a sharp icon at the specified size with adaptive sharpening.

    Args:
        img: Source PIL Image.
        size: Target side length for the square icon.
        mode: Clarify mode ('standard', 'extreme', or 'ultra').

    Returns:
        Resized and sharpened PIL Image.
    """
    ensure_pillow()

    if mode == "ultra":
        # Multi-step aggressive sharpening with intermediate resize and pre-sharpening
        working_img = img.copy()
        if min(working_img.size) > size * 4:
            intermediate_size = size * 4
            working_img = working_img.resize(
                (intermediate_size, intermediate_size), Image.Resampling.LANCZOS
            )
        working_img = working_img.filter(
            ImageFilter.UnsharpMask(radius=1.5, percent=120, threshold=1)
        )
        working_img = ImageEnhance.Contrast(working_img).enhance(1.1)
        final = working_img.resize((size, size), Image.Resampling.LANCZOS)

        if size <= 16:
            final = final.filter(
                ImageFilter.UnsharpMask(radius=0.2, percent=300, threshold=0)
            )
            final = final.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
            return ImageEnhance.Contrast(final).enhance(1.4)
        elif size <= 32:
            final = final.filter(
                ImageFilter.UnsharpMask(radius=0.3, percent=250, threshold=1)
            ).filter(ImageFilter.SHARPEN)
            return ImageEnhance.Contrast(final).enhance(1.3)
        return final

    if mode == "extreme" and size <= 32:
        # Multi-step aggressive sharpening for Windows small icons
        intermediate = img.resize((size * 2, size * 2), Image.Resampling.LANCZOS)
        intermediate = ImageEnhance.Contrast(intermediate).enhance(1.5)
        intermediate = intermediate.filter(
            ImageFilter.UnsharpMask(radius=0.5, percent=400, threshold=0)
        )
        final = intermediate.resize((size, size), Image.Resampling.LANCZOS)
        final_sharpened = final.filter(ImageFilter.SHARPEN).filter(ImageFilter.SHARPEN)
        return ImageEnhance.Contrast(final_sharpened).enhance(1.2)

    # Standard Adaptive Sharpening
    resized = img.resize((size, size), Image.Resampling.LANCZOS)
    if size <= 32:
        resized = resized.filter(
            ImageFilter.UnsharpMask(radius=0.3, percent=200, threshold=1)
        ).filter(ImageFilter.SHARPEN)
        resized = ImageEnhance.Contrast(resized).enhance(1.2)
    elif size <= 64:
        resized = resized.filter(
            ImageFilter.UnsharpMask(radius=0.5, percent=150, threshold=2)
        )
    elif size <= 256:
        resized = resized.filter(
            ImageFilter.UnsharpMask(radius=1.0, percent=100, threshold=3)
        )
    return resized


def save_ico(
    img: Image.Image,
    output_path: Path,
    sizes: list[int] | None = None,
    mode: str = "standard"
) -> None:
    """Save an image as an ICO file with multiple sizes.

    Args:
        img: Source PIL Image.
        output_path: Path to save the ICO file.
        sizes: List of square sizes to include.
        mode: Sharpening mode ('standard' or 'extreme').
    """
    ensure_pillow()
    if sizes is None:
        sizes = [16, 32, 48, 64, 128, 256]

    ico_images = [create_optimized_icon(img, s, mode=mode) for s in sorted(sizes)]

    # Save first image with others appended
    ico_images[0].save(
        output_path,
        format="ICO",
        sizes=[(s, s) for s in sorted(sizes)],
        append_images=ico_images[1:],
    )


def save_png_icons(
    img: Image.Image,
    assets_dir: Path,
    base_name: str,
    sizes: list[int],
    mode: str = "standard"
) -> None:
    """Save multiple PNG icons at specified sizes."""
    ensure_pillow()
    for size in sizes:
        icon = create_optimized_icon(img, size, mode=mode)
        icon.save(assets_dir / f"{base_name}_{size}.png", "PNG", optimize=True)


def analyze_image_quality(image_path: Path) -> dict[str, Any]:
    """Analyze image quality metrics (Sharpness, Contrast, Density)."""
    ensure_pillow()
    import numpy as np

    if not image_path.exists():
        return {"error": f"File not found: {image_path}"}

    try:
        img = Image.open(image_path)
        gray = img.convert("L")
        img_array = np.array(gray)

        # Laplacian variance (higher = sharper)
        lap_filter = ImageFilter.Kernel((3, 3), [-1, -1, -1, -1, 8, -1, -1, -1, -1])
        laplacian_var = np.var(np.array(gray.filter(lap_filter)))

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
