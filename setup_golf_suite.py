#!/usr/bin/env python3
"""
Golf Modeling Suite - Unified Setup Script
Syncs repository state, generates optimized icons, and creates desktop shortcuts.

This script combines the best practices from various utility scripts into a single
robust setup procedure.
"""

import logging
import subprocess
import sys
import os
import platform
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def check_dependencies():
    """Ensure required dependencies are installed."""
    try:
        import PIL
        from PIL import Image, ImageEnhance, ImageFilter
        return True
    except ImportError:
        logger.error("Pillow is not installed.")
        logger.info("Attempting to install Pillow...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "Pillow"])
            logger.info("Pillow installed successfully.")
            return True
        except subprocess.CalledProcessError:
            logger.error("Failed to install Pillow. Please run: pip install Pillow")
            return False


def git_sync():
    """Sync the repository with remote."""
    logger.info("Syncing repository with remote...")
    try:
        # Fetch all changes
        subprocess.check_call(["git", "fetch", "--all"], cwd=Path(__file__).parent)
        
        # Check if we are behind
        # Note: We won't strictly enforce a pull if it causes conflicts, but we will try.
        # Simple fast-forward pull
        subprocess.check_call(["git", "pull"], cwd=Path(__file__).parent)
        logger.info("Repository synced successfully.")
        return True
    except subprocess.CalledProcessError as e:
        logger.warning(f"Git sync failed (might be offline or have conflicts): {e}")
        # We proceed anyway as we might be setting up a local-only dev environment
        return False
    except FileNotFoundError:
        logger.warning("Git command not found. Skipping sync.")
        return False


def create_optimized_icon(source_path: Path, output_path: Path) -> bool:
    """
    Generate a Windows-optimized .ico file with correct mipmaps and sharpening.
    
    Implements 'fundamentally correct' icon generation logic:
    - High-quality downsampling (Lanczos)
    - Specific unsharp masking for small sizes (16px, 32px)
    - Contrast enhancement for visibility
    """
    from PIL import Image, ImageEnhance, ImageFilter

    if not source_path.exists():
        logger.error(f"Source image not found: {source_path}")
        return False

    try:
        img = Image.open(source_path)
        if img.mode != "RGBA":
            img = img.convert("RGBA")
        
        # Define standard Windows icon sizes
        sizes = [256, 128, 64, 48, 32, 24, 16]
        icon_images = []

        for size in sizes:
            # High-quality resize
            resized = img.resize((size, size), Image.Resampling.LANCZOS)

            # Apply specific optimizations based on size
            if size <= 32:
                # Small icons need aggressive sharpening and contrast
                # 1. Contrast boost
                enhancer = ImageEnhance.Contrast(resized)
                resized = enhancer.enhance(1.2)
                
                # 2. Unsharp mask (radius 0.5 for fine details)
                resized = resized.filter(
                    ImageFilter.UnsharpMask(radius=0.5, percent=200, threshold=0)
                )
                
                # 3. Final sharpen
                resized = resized.filter(ImageFilter.SHARPEN)
            
            elif size <= 64:
                # Medium icons need moderate sharpening
                resized = resized.filter(
                    ImageFilter.UnsharpMask(radius=0.8, percent=125, threshold=2)
                )

            icon_images.append(resized)

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Save as ICO (first image is largest, but PIL handles 'sizes' arg)
        # We append the rest. Note: PIL saves the first image in the list as one entry,
        # then appends others. It's best to pass the largest as 'img' and duplicates of others if needed,
        # or just pass the first and append the rest.
        
        # Sort by size descending (standard practice, ensuring 256 is first)
        icon_images.sort(key=lambda x: x.width, reverse=True)
        
        icon_images[0].save(
            output_path,
            format="ICO",
            sizes=[(i.width, i.height) for i in icon_images],
            append_images=icon_images[1:]
        )
        
        logger.info(f"Generated optimized icon: {output_path}")
        return True

    except Exception as e:
        logger.exception(f"Failed to generate icon: {e}")
        return False


def create_shortcut_windows(target_script: str, working_dir: Path, icon_path: Path, description: str):
    """Create a desktop shortcut using PowerShell interaction."""
    desktop_path = Path(os.environ["USERPROFILE"]) / "Desktop"
    shortcut_path = desktop_path / "Golf Modeling Suite.lnk"
    
    python_exe = sys.executable
    
    # PowerShell script to create shortcut
    # Use single quotes for strings to avoid escaping issues
    # Let PowerShell resolve the Desktop path dynamically to handle moved folders/OneDrive
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
        logger.info(f"Shortcut created successfully at: {shortcut_path}")
        return True
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.decode('utf-8', errors='replace') if e.stderr else "No error output"
        logger.error(f"Failed to create shortcut: {e}")
        logger.error(f"PowerShell Error Output: {error_msg}")
        return False


def main():
    logger.info("Initializing Golf Modeling Suite setup...")

    # 1. Sync
    git_sync()

    # 2. Check Dependencies
    if not check_dependencies():
        return 1

    # 3. Resolve Paths
    repo_root = Path(__file__).parent.absolute()
    
    # Try multiple potential source images, prioritizing high-res ones
    potential_sources = [
        repo_root / "GolfingRobot.png",
        repo_root / "launchers" / "assets" / "golf_robot_cropped.png",
        repo_root / "launchers" / "assets" / "golf_icon.png",
        repo_root / "launchers" / "assets" / "golf_robot_icon.png"
    ]
    
    source_icon = None
    for src in potential_sources:
        if src.exists():
            source_icon = src
            logger.info(f"Using source image: {src.name}")
            break
            
    output_icon = repo_root / "launchers" / "assets" / "golf_suite_unified.ico"
    target_script = repo_root / "launch_golf_suite.py"

    # 4. Generate Icon
    if source_icon:
        if not create_optimized_icon(source_icon, output_icon):
            logger.warning("Icon generation failed. Shortcut will use default python icon or fail.")
    else:
        logger.error("No suitable source image found for icon generation.")
        # Try to use an existing ICO if generation fails but one exists
        if output_icon.exists():
             logger.info("Using existing unified icon.")
        else:
             fallback = repo_root / "launchers" / "assets" / "golf_robot_icon.ico"
             if fallback.exists():
                 output_icon = fallback
                 logger.info("Using fallback existing icon.")

    # 5. Create Shortcut
    if platform.system() == "Windows":
        # Check if target_script is in repo_root to use relative path
        try:
             rel_script = target_script.relative_to(repo_root)
             # Use simple quoted string for filename
             script_arg = f'{rel_script}' 
        except ValueError:
             # Fallback to absolute if not relative
             script_arg = str(target_script)

        # Simplify - use raw strings for python to avoid escaping issues in f-string
        create_shortcut_windows(
            target_script=script_arg, 
            working_dir=repo_root,
            icon_path=output_icon if output_icon.exists() else Path(""),
            description="Launch the Unified Golf Modeling Suite"
        )
    else:
        logger.warning(f"Shortcut creation for {platform.system()} is not fully implemented in this unified script yet.")
        logger.info("Please verify the existing 'launch_golf_suite.py' works manually.")

    logger.info("Setup complete!")
    return 0


if __name__ == "__main__":
    sys.exit(main())
