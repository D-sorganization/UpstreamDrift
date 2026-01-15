#!/usr/bin/env python3
"""Asset optimization script for Golf Modeling Suite.

This script optimizes PNG and other image assets for faster loading
and reduced memory usage.

Requirements:
    pip install Pillow

Usage:
    python optimize_assets.py [--dry-run] [--aggressive]

Options:
    --dry-run     Show what would be optimized without making changes
    --aggressive  Use more aggressive compression (may reduce quality)
"""

import argparse
import sys
from pathlib import Path

# Target sizes for different asset categories
ICON_SIZES = [16, 20, 24, 32, 40, 48, 64, 96, 128, 256]
MAX_ICON_SIZE_KB = 30  # Icons should be under 30KB
MAX_IMAGE_SIZE_KB = 100  # Regular images under 100KB


def check_dependencies() -> bool:
    """Check if required dependencies are available."""
    try:
        from PIL import Image  # noqa: F401

        return True
    except ImportError:
        print("Error: Pillow is required. Install with: pip install Pillow")
        return False


def get_image_info(path: Path) -> dict:
    """Get information about an image file."""
    from PIL import Image

    size_kb = path.stat().st_size / 1024
    with Image.open(path) as img:
        return {
            "path": path,
            "size_kb": size_kb,
            "dimensions": img.size,
            "mode": img.mode,
            "format": img.format,
        }


def optimize_png(path: Path, quality: int = 85, aggressive: bool = False) -> int:
    """Optimize a PNG file.

    Args:
        path: Path to the PNG file
        quality: Quality level (1-100)
        aggressive: Use more aggressive optimization

    Returns:
        Bytes saved
    """
    from PIL import Image

    original_size = path.stat().st_size

    with Image.open(path) as img:
        # Convert to RGB if RGBA with no transparency
        if img.mode == "RGBA":
            # Check if image actually uses transparency
            alpha = img.getchannel("A")
            if alpha.getextrema() == (255, 255):
                img = img.convert("RGB")

        # Optimize and save
        if aggressive:
            # Use palette mode for smaller file size
            if img.mode == "RGBA":
                img.save(
                    path,
                    optimize=True,
                    compress_level=9,
                )
            else:
                img = img.quantize(colors=256, method=2)
                img.save(path, optimize=True)
        else:
            img.save(
                path,
                optimize=True,
                compress_level=6,
            )

    new_size = path.stat().st_size
    return original_size - new_size


def analyze_assets(assets_dir: Path) -> list[dict]:
    """Analyze all image assets and report optimization opportunities."""
    results = []

    for ext in ["*.png", "*.jpg", "*.jpeg"]:
        for path in assets_dir.glob(ext):
            info = get_image_info(path)

            # Determine if optimization is needed
            needs_optimization = False
            reason = []

            if "icon" in path.name.lower() and info["size_kb"] > MAX_ICON_SIZE_KB:
                needs_optimization = True
                reason.append(f"Icon over {MAX_ICON_SIZE_KB}KB")

            if info["size_kb"] > MAX_IMAGE_SIZE_KB:
                needs_optimization = True
                reason.append(f"Image over {MAX_IMAGE_SIZE_KB}KB")

            info["needs_optimization"] = needs_optimization
            info["reason"] = reason
            results.append(info)

    return results


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Optimize image assets")
    parser.add_argument(
        "--dry-run", action="store_true", help="Show what would be done"
    )
    parser.add_argument(
        "--aggressive", action="store_true", help="Use aggressive compression"
    )
    args = parser.parse_args()

    if not check_dependencies():
        return 1

    assets_dir = Path(__file__).parent

    print("Analyzing assets...")
    results = analyze_assets(assets_dir)

    # Report findings
    total_size = sum(r["size_kb"] for r in results)
    needs_opt = [r for r in results if r["needs_optimization"]]

    print(f"\nTotal assets: {len(results)}")
    print(f"Total size: {total_size:.1f} KB")
    print(f"Need optimization: {len(needs_opt)}")

    if needs_opt:
        print("\nFiles needing optimization:")
        for info in sorted(needs_opt, key=lambda x: -x["size_kb"]):
            print(f"  {info['path'].name}: {info['size_kb']:.1f}KB - {info['reason']}")

    if args.dry_run:
        print("\nDry run - no changes made.")
        return 0

    if not needs_opt:
        print("\nAll assets are optimized!")
        return 0

    # Optimize files
    print("\nOptimizing...")
    total_saved = 0
    for info in needs_opt:
        if info["path"].suffix.lower() == ".png":
            saved = optimize_png(info["path"], aggressive=args.aggressive)
            total_saved += saved
            print(f"  {info['path'].name}: saved {saved / 1024:.1f} KB")

    print(f"\nTotal saved: {total_saved / 1024:.1f} KB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
