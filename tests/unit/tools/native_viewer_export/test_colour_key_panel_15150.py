"""Drawn colour key panel for the ZTCF/ZVCF renders (#15150 Track Z, Z3)."""

from __future__ import annotations

import numpy as np
import pytest

from src.tools.native_viewer_export.compositor import (
    COLOUR_KEY_TITLE,
    colour_key_box,
    draw_colour_key,
)

pytestmark = pytest.mark.unit

ENTRIES = (
    ("ZTCF drift: gravity, momentum, passive", (0, 114, 178)),
    ("ZVCF control: applied torques alone", (213, 94, 0)),
    ("Total", (242, 242, 242)),
)


def test_the_key_sits_on_the_left_inside_the_frame() -> None:
    shape = (720, 1280, 3)
    x0, y0, x1, y1 = colour_key_box(shape, ENTRIES)
    assert 0 <= x0 < x1 <= 640
    assert 0 <= y0 < y1 <= shape[0]
    assert COLOUR_KEY_TITLE == "Colour Key"


def test_only_pixels_in_the_box_change_and_each_colour_is_drawn() -> None:
    img = np.full((720, 1280, 3), 40, dtype=np.uint8)
    out = draw_colour_key(img, ENTRIES)
    x0, y0, x1, y1 = colour_key_box(img.shape, ENTRIES)
    mask = np.zeros(img.shape[:2], dtype=bool)
    mask[y0:y1, x0:x1] = True
    assert np.array_equal(out[~mask], img[~mask])
    region = out[y0:y1, x0:x1].reshape(-1, 3)
    for _, rgb in ENTRIES:
        assert np.any(np.all(region == np.array(rgb), axis=1)), rgb


def test_an_empty_key_draws_nothing() -> None:
    img = np.zeros((90, 160, 3), dtype=np.uint8)
    assert np.array_equal(draw_colour_key(img, ()), img)


def test_a_bad_colour_is_refused() -> None:
    img = np.zeros((90, 160, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="colour"):
        draw_colour_key(img, (("x", (0, 0, 300)),))


def test_key_stays_clear_of_the_right_hand_impact_card() -> None:
    shape = (720, 1280, 3)
    key = colour_key_box(shape, ENTRIES)
    assert key[2] <= shape[1] // 2


def test_each_row_has_a_solid_swatch_and_light_text() -> None:
    """Review of the whole-swing renders: dark blue or red text on the dark
    panel was unreadable, so each colour is a solid swatch block and the
    text is white."""
    img = np.full((720, 1280, 3), 40, dtype=np.uint8)
    out = draw_colour_key(img, ENTRIES)
    x0, y0, x1, y1 = colour_key_box(img.shape, ENTRIES)
    region = out[y0:y1, x0:x1]
    for _, rgb in ENTRIES:
        hit = np.all(region == np.array(rgb), axis=2)
        rows, cols = np.nonzero(hit)
        # a filled block: at least 8 x 8 pixels of the exact colour
        assert hit.sum() >= 64, rgb
        assert cols.max() - cols.min() >= 7 and rows.max() - rows.min() >= 7, rgb
    white = np.all(region == 255, axis=2)
    assert white.sum() > 200  # the row text
