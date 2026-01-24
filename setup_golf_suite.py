#!/usr/bin/env python3
"""Golf Modeling Suite - Unified Setup Script.

Syncs repository state, generates optimized icons, and creates desktop shortcuts.

Refactored to address DRY and Orthogonality violations identified in the
Pragmatic Programmer assessment (2026-01-23).
"""

from __future__ import annotations

import platform
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

# Configure logging
from src.shared.python.launcher_utils import (
    check_python_dependencies,
    get_repo_root,
    git_sync_repository,
)
from src.shared.python.logging_config import get_logger, setup_logging

setup_logging()
logger = get_logger(__name__)


def _apply_icon_optimizations(img: PILImage, size: int) -> PILImage:
    """Apply size-specific optimizations to an icon image.

    Args:
        img: Resized image.
        size: Target size in pixels.

    Returns:
        Optimized PIL Image.
    """
    from PIL import ImageEnhance, ImageFilter

    if size <= 32:
        # Small icons need aggressive sharpening and contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.2)
        img = img.filter(ImageFilter.UnsharpMask(radius=0.5, percent=200, threshold=0))
        img = img.filter(ImageFilter.SHARPEN)
    elif size <= 64:
        # Medium icons need moderate sharpening
        img = img.filter(ImageFilter.UnsharpMask(radius=0.8, percent=125, threshold=2))
    return img


def create_optimized_icon(source_path: Path, output_path: Path) -> bool:
    """Generate a Windows-optimized .ico file with correct mipmaps and sharpening.

    Orthogonality: Decouples image processing from file system management.
    """
    if not source_path.exists():
        logger.error(f"Source image not found: {source_path}")
        return False

    try:
        from PIL import Image

        img = Image.open(source_path)
        if img.mode != "RGBA":
            # Type ignore because PIL types can be complex
            img = img.convert("RGBA")  # type: ignore[assignment]

        sizes = [256, 128, 64, 48, 32, 24, 16]
        icon_images = []

        for size in sizes:
            resized = img.resize((size, size), Image.Resampling.LANCZOS)
            optimized = _apply_icon_optimizations(resized, size)
            icon_images.append(optimized)

        # Sort by size descending (standard practice for ICO)
        icon_images.sort(key=lambda x: x.width, reverse=True)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        icon_images[0].save(
            output_path,
            format="ICO",
            sizes=[(i.width, i.height) for i in icon_images],
            append_images=icon_images[1:],
        )

        logger.info(f"Generated optimized icon: {output_path}")
        return True

    except Exception as e:
        logger.exception(f"Failed to generate icon: {e}")
        return False


def create_shortcut_windows(
    target_script: str, working_dir: Path, icon_path: Path, description: str
) -> bool:
    """Create a desktop shortcut using PowerShell interaction."""
    python_exe = sys.executable

    # PowerShell script to create shortcut
    ps_script = f"""
    $WshShell = New-Object -comObject WScript.Shell
    $Desktop = [Environment]::GetFolderPath("Desktop")
    $ShortcutPath = Join-Path $Desktop "Golf Modeling Suite.lnk"
    $Shortcut = $WshShell.CreateShortcut($ShortcutPath)
    $Shortcut.TargetPath = '{python_exe}'
    $Shortcut.Arguments = '"{target_script}"'
    $Shortcut.WorkingDirectory = '{working_dir}'
    $Shortcut.IconLocation = '{icon_path}'
    $Shortcut.Description = '{description}'
    $Shortcut.Save()
    """

    try:
        subprocess.run(
            ["powershell", "-Command", ps_script],
            check=True,
            capture_output=True,
        )
        logger.info("Desktop shortcut created successfully.")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = (
            e.stderr.decode("utf-8", errors="replace") if e.stderr else "Unknown error"
        )
        logger.error(f"Failed to create shortcut: {e}\n{error_msg}")
        return False


def _find_source_image(repo_root: Path) -> Path | None:
    """Find the best available source image for icon generation."""
    potential_sources = [
        repo_root / "GolfingRobot.png",
        repo_root / "launchers" / "assets" / "golf_robot_cropped.png",
        repo_root / "launchers" / "assets" / "golf_icon.png",
        repo_root / "launchers" / "assets" / "golf_robot_icon.png",
    ]
    for src in potential_sources:
        if src.exists():
            logger.info(f"Using source image: {src.name}")
            return src
    return None


def main() -> int:
    """Main setup procedure.

    Broken down to address God Function violation (Orthogonality).
    """
    logger.info("Initializing Golf Modeling Suite setup...")

    # 1. Sync repository
    git_sync_repository()

    # 2. Check dependencies
    if not check_python_dependencies(["PIL"], install_missing=True):
        return 1

    # 3. Resolve Paths
    repo_root = get_repo_root()
    source_icon = _find_source_image(repo_root)
    output_icon = repo_root / "launchers" / "assets" / "golf_suite_unified.ico"
    target_launch_script = repo_root / "launch_golf_suite.py"

    # 4. Generate Icon
    if source_icon:
        create_optimized_icon(source_icon, output_icon)
    elif not output_icon.exists():
        # Fallback to existing icon if generation impossible
        fallback = repo_root / "launchers" / "assets" / "golf_robot_icon.ico"
        if fallback.exists():
            output_icon = fallback
            logger.info("Using fallback existing icon.")

    # 5. Create Desktop Shortcut (Windows only)
    if platform.system() == "Windows":
        try:
            script_path = str(target_launch_script.relative_to(repo_root))
        except ValueError:
            script_path = str(target_launch_script)

        create_shortcut_windows(
            target_script=script_path,
            working_dir=repo_root,
            icon_path=output_icon if output_icon.exists() else Path(""),
            description="Launch the Unified Golf Modeling Suite",
        )
    else:
        logger.info(f"Desktop integration for {platform.system()} not yet implemented.")

    logger.info("Setup complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
