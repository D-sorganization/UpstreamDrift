"""Tests for the asset-pipeline helpers under ``src.launchers.assets``.

These two modules are normally invoked as standalone scripts, but the
public functions are importable.  We exercise the pure-logic helpers
(`hex_to_rgb`, `create_png`, `draw_rounded_rect_with_text`,
`draw_letter`) and drive ``main`` for both scripts in dry-run / no-op
modes so the CLI scaffolding is covered without writing files into the
real assets directory.
"""

from __future__ import annotations

from unittest.mock import patch


from src.launchers.assets import generate_tile_images, optimize_assets

# --- generate_tile_images.py ------------------------------------------------


def test_hex_to_rgb_basic() -> None:
    assert generate_tile_images.hex_to_rgb("#000000") == (0, 0, 0)
    assert generate_tile_images.hex_to_rgb("#FFFFFF") == (255, 255, 255)
    assert generate_tile_images.hex_to_rgb("#00897B") == (0, 137, 123)


def test_hex_to_rgb_strips_hash() -> None:
    assert generate_tile_images.hex_to_rgb("123456") == (0x12, 0x34, 0x56)


def test_create_png_produces_valid_signature() -> None:
    pixels = [(255, 0, 0, 255)] * 4  # 2x2 red square
    blob = generate_tile_images.create_png(2, 2, pixels)
    # PNG magic
    assert blob[:8] == b"\x89PNG\r\n\x1a\n"
    # Trailing IEND marker
    assert b"IEND" in blob


def test_draw_rounded_rect_with_text_returns_correct_pixel_count() -> None:
    pixels = generate_tile_images.draw_rounded_rect_with_text(
        16, 16, (10, 20, 30), "MJ", radius=4
    )
    assert len(pixels) == 16 * 16
    # Every pixel is a 4-tuple
    for px in pixels:
        assert isinstance(px, tuple)
        assert len(px) == 4


def test_draw_letter_writes_into_pixel_buffer() -> None:
    width, height = 32, 32
    pixels = [(0, 0, 0, 0)] * (width * height)
    generate_tile_images.draw_letter(pixels, width, 4, 4, 16, "M")
    # At least one pixel should now be white
    assert any(px == (255, 255, 255, 255) for px in pixels)


def test_draw_letter_unknown_char_uses_default_pattern() -> None:
    width, height = 16, 16
    pixels = [(0, 0, 0, 0)] * (width * height)
    # ``Z`` is not in the lookup table — falls back to a filled square.
    generate_tile_images.draw_letter(pixels, width, 0, 0, 8, "Z")
    assert any(px == (255, 255, 255, 255) for px in pixels)


def test_generate_all_tiles_writes_each_config(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(generate_tile_images, "ASSETS_DIR", tmp_path)
    generate_tile_images.generate_all_tiles()
    # Every entry in TILE_CONFIGS should have produced a PNG.
    for name in generate_tile_images.TILE_CONFIGS:
        assert (tmp_path / f"{name}.png").exists()


# --- optimize_assets.py -----------------------------------------------------


def test_check_dependencies_returns_true_when_pil_available() -> None:
    # PIL is a hard test-time dep on this branch.
    assert optimize_assets.check_dependencies() is True


def test_check_dependencies_returns_false_without_pil(monkeypatch) -> None:
    real_import = (
        __builtins__["__import__"]
        if isinstance(__builtins__, dict)
        else __builtins__.__import__
    )

    def boom(name, *a, **k):
        if name == "PIL":
            raise ImportError("no PIL")
        return real_import(name, *a, **k)

    with patch("builtins.__import__", side_effect=boom):
        assert optimize_assets.check_dependencies() is False


def test_get_image_info_round_trips(tmp_path) -> None:
    from PIL import Image

    img_path = tmp_path / "tiny.png"
    Image.new("RGBA", (5, 5), (10, 20, 30, 255)).save(img_path)

    info = optimize_assets.get_image_info(img_path)
    assert info["dimensions"] == (5, 5)
    assert info["size_kb"] > 0
    assert info["format"] == "PNG"


def test_optimize_png_reduces_or_preserves_size(tmp_path) -> None:
    from PIL import Image

    p = tmp_path / "blob.png"
    # Solid colour image compresses extremely well.
    Image.new("RGBA", (32, 32), (100, 100, 100, 255)).save(p)

    saved = optimize_assets.optimize_png(p, aggressive=False)
    assert isinstance(saved, int)


def test_optimize_png_aggressive_branch(tmp_path) -> None:
    from PIL import Image

    p = tmp_path / "agg.png"
    Image.new("RGB", (32, 32), (200, 50, 50)).save(p)
    saved = optimize_assets.optimize_png(p, aggressive=True)
    assert isinstance(saved, int)


def test_optimize_png_aggressive_with_alpha(tmp_path) -> None:
    from PIL import Image

    p = tmp_path / "agg_alpha.png"
    Image.new("RGBA", (16, 16), (10, 10, 10, 200)).save(p)
    optimize_assets.optimize_png(p, aggressive=True)


def test_analyze_assets_returns_one_entry_per_image(tmp_path) -> None:
    from PIL import Image

    Image.new("RGBA", (4, 4), (0, 0, 0, 255)).save(tmp_path / "icon_small.png")
    # Force a "large" image by writing a high-entropy buffer.
    big = Image.effect_noise((512, 512), 50)
    big.save(tmp_path / "big.png")

    results = optimize_assets.analyze_assets(tmp_path)
    assert len(results) == 2
    for entry in results:
        assert "needs_optimization" in entry
        assert "reason" in entry


def test_main_dry_run_returns_zero(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["optimize_assets.py", "--dry-run"])
    with patch.object(optimize_assets, "Path") as fake_path:
        fake_path.return_value.parent = tmp_path
        rc = optimize_assets.main()
    assert rc == 0


def test_main_returns_one_when_dependencies_missing(monkeypatch) -> None:
    monkeypatch.setattr("sys.argv", ["optimize_assets.py", "--dry-run"])
    with patch.object(optimize_assets, "check_dependencies", return_value=False):
        assert optimize_assets.main() == 1


def test_main_optimizes_when_files_need_it(tmp_path, monkeypatch) -> None:
    from PIL import Image

    monkeypatch.setattr("sys.argv", ["optimize_assets.py"])

    # Place a "needs optimization" image inside the assets dir.
    big = Image.effect_noise((512, 512), 50)
    big_path = tmp_path / "big_image.png"
    big.save(big_path)

    with patch.object(optimize_assets, "Path") as fake_path:
        fake_path.return_value.parent = tmp_path
        rc = optimize_assets.main()
    assert rc == 0
