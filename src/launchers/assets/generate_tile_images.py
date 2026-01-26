#!/usr/bin/env python3
"""Generate placeholder tile images for the Golf Modeling Suite Launcher.

This script creates visually appealing tile images for each model/engine
in the launcher grid. Images use a modern dark theme with colored accents.
"""

import struct
import zlib
from pathlib import Path

# Output directory
ASSETS_DIR = Path(__file__).parent

# Tile configurations: name -> (color_hex, icon_char)
TILE_CONFIGS = {
    "mujoco_humanoid": ("#00897B", "MJ"),  # Teal
    "mujoco_hand": ("#00897B", "MH"),  # Teal
    "drake": ("#5C6BC0", "DK"),  # Indigo
    "pinocchio": ("#7E57C2", "PN"),  # Deep Purple
    "openpose": (
        "#EF6C00",
        "OP",
    ),  # Orange (note: file is openpose.jpg in MODEL_IMAGES)
    "opensim": ("#43A047", "OS"),  # Green
    "myosim": ("#E53935", "MS"),  # Red
    "simscape_multibody": ("#1E88E5", "SM"),  # Blue
    "urdf_icon": ("#78909C", "UR"),  # Blue Grey
    "c3d_icon": ("#26A69A", "C3"),  # Teal accent
}


def hex_to_rgb(hex_color: str) -> tuple[int, int, int]:
    """Convert hex color to RGB tuple."""
    hex_color = hex_color.lstrip("#")
    return tuple(int(hex_color[i : i + 2], 16) for i in (0, 2, 4))


def create_png(
    width: int, height: int, pixels: list[tuple[int, int, int, int]]
) -> bytes:
    """Create a PNG file from raw RGBA pixel data.

    This is a minimal PNG encoder that doesn't require PIL.
    """

    def crc32(data: bytes) -> int:
        return zlib.crc32(data) & 0xFFFFFFFF

    def make_chunk(chunk_type: bytes, data: bytes) -> bytes:
        chunk = chunk_type + data
        return struct.pack(">I", len(data)) + chunk + struct.pack(">I", crc32(chunk))

    # PNG signature
    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR chunk
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 6, 0, 0, 0)  # 8-bit RGBA
    ihdr = make_chunk(b"IHDR", ihdr_data)

    # IDAT chunk (image data)
    raw_data = b""
    for y in range(height):
        raw_data += b"\x00"  # filter type: none
        for x in range(width):
            r, g, b, a = pixels[y * width + x]
            raw_data += struct.pack("BBBB", r, g, b, a)

    compressed = zlib.compress(raw_data, 9)
    idat = make_chunk(b"IDAT", compressed)

    # IEND chunk
    iend = make_chunk(b"IEND", b"")

    return signature + ihdr + idat + iend


def draw_rounded_rect_with_text(
    width: int, height: int, bg_color: tuple[int, int, int], text: str, radius: int = 20
) -> list[tuple[int, int, int, int]]:
    """Create a rounded rectangle with centered text."""
    pixels = []

    for y in range(height):
        for x in range(width):
            # Check if inside rounded rectangle
            inside = True

            # Corner checks
            corners = [
                (radius, radius),  # top-left
                (width - radius - 1, radius),  # top-right
                (radius, height - radius - 1),  # bottom-left
                (width - radius - 1, height - radius - 1),  # bottom-right
            ]

            in_corner_region = False
            for cx, cy in corners:
                if (
                    (x < radius and y < radius)
                    or (x >= width - radius and y < radius)
                    or (x < radius and y >= height - radius)
                    or (x >= width - radius and y >= height - radius)
                ):
                    # Check distance from corner center
                    dx = abs(x - cx)
                    dy = abs(y - cy)
                    if dx * dx + dy * dy > radius * radius:
                        inside = False
                    in_corner_region = True
                    break

            if inside:
                # Gradient from top to bottom
                factor = 0.7 + 0.3 * (1 - y / height)
                r = int(bg_color[0] * factor)
                g = int(bg_color[1] * factor)
                b = int(bg_color[2] * factor)

                # Add a subtle border
                if x < 2 or x >= width - 2 or y < 2 or y >= height - 2:
                    r = min(255, r + 40)
                    g = min(255, g + 40)
                    b = min(255, b + 40)

                pixels.append((r, g, b, 255))
            else:
                pixels.append((0, 0, 0, 0))  # Transparent

    # Draw simple text (letter shapes using rectangles)
    text_size = min(width, height) // 3
    text_y = (height - text_size) // 2

    # For 2-char text, center them
    char_width = text_size // 2 + 5
    start_x = (width - len(text) * char_width) // 2

    for i, char in enumerate(text):
        char_x = start_x + i * char_width
        # Draw a simple letter representation
        draw_letter(pixels, width, char_x, text_y, text_size, char)

    return pixels


def draw_letter(pixels: list, width: int, x: int, y: int, size: int, char: str):
    """Draw a simple blocky letter representation."""
    white = (255, 255, 255, 255)
    thickness = max(3, size // 6)

    # Simple letter patterns (blocky style)
    patterns = {
        "M": [
            (0, 0, thickness, size),
            (size - thickness, 0, thickness, size),
            (0, 0, size // 2, thickness),
            (size // 2, 0, size // 2, thickness),
            (size // 2 - thickness // 2, 0, thickness, size // 2),
        ],
        "J": [
            (size // 2, 0, thickness, size),
            (0, size - thickness, size // 2 + thickness, thickness),
            (0, size * 2 // 3, thickness, size // 3),
        ],
        "H": [
            (0, 0, thickness, size),
            (size - thickness, 0, thickness, size),
            (0, size // 2, size, thickness),
        ],
        "D": [
            (0, 0, thickness, size),
            (0, 0, size * 2 // 3, thickness),
            (0, size - thickness, size * 2 // 3, thickness),
            (size * 2 // 3, thickness, thickness, size - thickness * 2),
        ],
        "K": [
            (0, 0, thickness, size),
            (thickness, size // 2, size - thickness, thickness),
            (size // 2, 0, thickness, size // 2),
            (size // 2, size // 2, thickness, size // 2),
        ],
        "P": [
            (0, 0, thickness, size),
            (0, 0, size * 2 // 3, thickness),
            (0, size // 2, size * 2 // 3, thickness),
            (size * 2 // 3, 0, thickness, size // 2),
        ],
        "N": [
            (0, 0, thickness, size),
            (size - thickness, 0, thickness, size),
            (0, 0, size, thickness),
        ],
        "O": [
            (0, 0, thickness, size),
            (size - thickness, 0, thickness, size),
            (0, 0, size, thickness),
            (0, size - thickness, size, thickness),
        ],
        "S": [
            (0, 0, size, thickness),
            (0, 0, thickness, size // 2 + thickness),
            (0, size // 2, size, thickness),
            (size - thickness, size // 2, thickness, size // 2),
            (0, size - thickness, size, thickness),
        ],
        "U": [
            (0, 0, thickness, size),
            (size - thickness, 0, thickness, size),
            (0, size - thickness, size, thickness),
        ],
        "R": [
            (0, 0, thickness, size),
            (0, 0, size * 2 // 3, thickness),
            (0, size // 2, size * 2 // 3, thickness),
            (size * 2 // 3, 0, thickness, size // 2),
            (size // 2, size // 2, thickness, size // 2),
        ],
        "C": [
            (0, 0, thickness, size),
            (0, 0, size, thickness),
            (0, size - thickness, size, thickness),
        ],
        "3": [
            (0, 0, size, thickness),
            (size - thickness, 0, thickness, size),
            (0, size // 2, size, thickness),
            (0, size - thickness, size, thickness),
        ],
    }

    letter_pattern = patterns.get(char, [(0, 0, size, size)])  # Default: filled square

    for rect in letter_pattern:
        rx, ry, rw, rh = rect
        for py in range(ry, min(ry + rh, size)):
            for px in range(rx, min(rx + rw, size)):
                pixel_x = x + px
                pixel_y = y + py
                if 0 <= pixel_x < width and 0 <= pixel_y < len(pixels) // width:
                    idx = pixel_y * width + pixel_x
                    if idx < len(pixels):
                        pixels[idx] = white


def generate_all_tiles():
    """Generate all tile images."""
    size = 200  # 200x200 pixels

    print("Generating launcher tile images...")

    for name, (color, text) in TILE_CONFIGS.items():
        filename = f"{name}.png"
        filepath = ASSETS_DIR / filename

        rgb = hex_to_rgb(color)
        pixels = draw_rounded_rect_with_text(size, size, rgb, text)
        png_data = create_png(size, size, pixels)

        with open(filepath, "wb") as f:
            f.write(png_data)

        print(f"  Created: {filename}")

    # Create openpose.jpg as a copy of openpose.png
    # (MODEL_IMAGES references openpose.jpg)
    openpose_png = ASSETS_DIR / "openpose.png"
    if openpose_png.exists():
        # Just create it as PNG - the loader should handle both
        pass

    print("\nDone! All tile images generated.")
    print(f"Output directory: {ASSETS_DIR}")


if __name__ == "__main__":
    generate_all_tiles()
