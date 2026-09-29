# UpstreamDrift Branding Guide

## Logo Assets

This directory contains the official UpstreamDrift branding assets.

### Primary Logo

- **File**: `logo.png` - Primary logo for README and documentation
- **Description**: Golfer silhouette with stylized "UD" design in navy blue with blue-to-green gradient swoosh

### Icon Sizes

Generate icons at these sizes for different use cases:

| Size    | Use Case           | Filename       |
| ------- | ------------------ | -------------- |
| 16x16   | Favicon (small)    | `icon-16.png`  |
| 32x32   | Favicon (standard) | `icon-32.png`  |
| 48x48   | Desktop taskbar    | `icon-48.png`  |
| 64x64   | Desktop shortcuts  | `icon-64.png`  |
| 128x128 | Application icon   | `icon-128.png` |
| 256x256 | High-DPI displays  | `icon-256.png` |
| 512x512 | App stores, splash | `icon-512.png` |

### ICO File (Windows)

For Windows applications, create a multi-resolution ICO file:

```bash
# Using ImageMagick to create ICO with multiple sizes
convert icon-16.png icon-32.png icon-48.png icon-64.png icon-128.png icon-256.png icon.ico
```

### Favicon Generation

```bash
# Create favicon.ico for web use
convert logo.png -resize 32x32 favicon.ico

# Create multiple PNG sizes for modern browsers
convert logo.png -resize 16x16 favicon-16x16.png
convert logo.png -resize 32x32 favicon-32x32.png
convert logo.png -resize 180x180 apple-touch-icon.png
convert logo.png -resize 192x192 android-chrome-192x192.png
convert logo.png -resize 512x512 android-chrome-512x512.png
```

## Color Palette

| Color          | Hex       | Use                   |
| -------------- | --------- | --------------------- |
| Navy Blue      | `#1a365d` | Primary brand color   |
| Gradient Start | `#2563eb` | Swoosh gradient start |
| Gradient End   | `#10b981` | Swoosh gradient end   |
| White          | `#ffffff` | Text, backgrounds     |

## Usage Guidelines

### Do

- Use the logo on a white or light background
- Maintain minimum clear space around the logo
- Use official color palette for brand materials

### Don't

- Distort or stretch the logo
- Change the logo colors
- Place the logo on busy backgrounds
- Use the logo at sizes smaller than 16x16px

## Integration Points

### PyQt6 Application

```python
from PyQt6.QtGui import QIcon
from pathlib import Path

ASSETS_DIR = Path(__file__).parent / "assets" / "branding"

def get_app_icon():
    return QIcon(str(ASSETS_DIR / "icon-256.png"))

# In main window
app.setWindowIcon(get_app_icon())
```

### Splash Screen

```python
from PyQt6.QtWidgets import QSplashScreen
from PyQt6.QtGui import QPixmap

splash_pix = QPixmap(str(ASSETS_DIR / "icon-512.png"))
splash = QSplashScreen(splash_pix)
splash.show()
```

### Desktop Entry (Linux)

```ini
[Desktop Entry]
Name=UpstreamDrift
Icon=/path/to/UpstreamDrift/assets/branding/icon-256.png
Exec=python /path/to/UpstreamDrift/launch_golf_suite.py
Type=Application
Categories=Science;Education;
```

### Windows Shortcut

The `icon.ico` file should be used for Windows shortcuts and executables.

## File Checklist

Required files in this directory:

- [ ] `logo.png` - Primary logo (provided by user)
- [ ] `icon-16.png` through `icon-512.png` - Generated icon sizes
- [ ] `icon.ico` - Windows multi-resolution icon
- [ ] `favicon.ico` - Web favicon
- [ ] `BRANDING_GUIDE.md` - This file
