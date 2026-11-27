"""Side-by-side null-space comparison frames (#15494, #15725)."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageFont

from src.hand_load_video.compare import (
    COMPONENTS,
    OBJECTIVE,
    STRATEGIES,
    UNAVAILABLE,
    GridSpec,
    Provenance,
    align,
    compose,
    load_source,
    panel_from_arrays,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
_RED, _GREEN, _BLUE, _CYAN = (180, 0, 0), (0, 140, 0), (0, 0, 160), (0, 140, 140)
_FONT = ImageFont.load_default()


def _panel(strategy, component, times, colors, *, badge, units="N"):
    frames = [
        None if color is None else np.full((6, 6, 3), color, np.uint8)
        for color in colors
    ]
    return panel_from_arrays(
        strategy, component, times, frames, badge=badge, units=units
    )


def _inked(image, origin, text, rgb) -> bool:
    mask = _FONT.getmask(text, mode="1")
    bitmap = np.asarray(Image.frombytes("L", mask.size, bytes(mask)))
    ys, xs = np.nonzero(bitmap)
    ys, xs = ys + origin[1], xs + origin[0]
    inside = (
        len(xs) and int(xs.max()) < image.shape[1] and int(ys.max()) < image.shape[0]
    )
    return bool(inside and np.all(image[ys, xs] == rgb))


def test_panels_align_by_time_not_by_frame_index() -> None:
    baseline = _panel(
        "baseline",
        "ztcf",
        (0.0, 0.1, 0.4),
        (_RED, _GREEN, _BLUE),
        badge="allocation-only",
    )
    least = _panel(
        "least-force",
        "ztcf",
        (0.1, 0.4, 0.9),
        (_CYAN, _BLUE, _RED),
        badge="forward-replayed",
    )
    synced = align((baseline, least), (0.0, 0.1, 0.4), tolerance_s=0.04)
    assert [frame.time_s for frame in synced] == [0.0, 0.1, 0.4]
    at_start, at_tenth, at_late = synced
    assert at_start.cell("least-force", "ztcf").missing
    assert at_start.cell("baseline", "ztcf").pixels is not None
    assert np.all(at_tenth.cell("baseline", "ztcf").pixels[0, 0] == _GREEN)
    assert np.all(at_tenth.cell("least-force", "ztcf").pixels[0, 0] == _CYAN)
    for frame in synced:
        for cell in frame.cells:
            if not cell.missing:
                assert cell.time_s is not None
                assert abs(cell.time_s - frame.time_s) <= 0.04
    assert at_late.cell("least-force", "zvcf").missing
    assert at_late.cell("least-force", "zvcf").units == UNAVAILABLE


def test_grid_places_strategies_against_components() -> None:
    spec = GridSpec(cell_w=10, cell_h=8, gutter=3, banner_h=5, footer_h=6, label_w=20)
    assert spec.rows == STRATEGIES == ("baseline", "least-force")
    assert spec.columns == COMPONENTS == ("ztcf", "zvcf", "total")
    assert spec.width == 20 + 3 * 10 + 4 * 3
    assert spec.height == 2 * 5 + 2 * (3 + 5 + 8) + 3 + 6
    boxes = [spec.cell_box(row, col) for row in range(2) for col in range(3)]
    for index, box in enumerate(boxes):
        for other in boxes[index + 1 :]:
            assert (
                box[0] + box[2] <= other[0]
                or other[0] + other[2] <= box[0]
                or (box[1] + box[3] <= other[1] or other[1] + other[3] <= box[1])
            )
    left, right = spec.cell_box(0, 0), spec.cell_box(0, 1)
    assert right[0] == left[0] + left[2] + spec.gutter
    assert spec.cell_box(1, 2)[1] > spec.cell_box(0, 2)[1]


def test_every_frame_carries_the_objective_badge_and_footer() -> None:
    spec, provenance = GridSpec(), Provenance("fixture-swing", "toy-model", "abc123def")
    panel = _panel(
        "baseline", "ztcf", (0.0,), (_RED,), badge="allocation-only", units="N"
    )
    other = _panel(
        "least-force", "zvcf", (0.0,), (_GREEN,), badge="forward-replayed", units="N.m"
    )
    image, stamp = compose(
        align((panel, other), (0.0,), tolerance_s=0.0)[0], provenance, spec
    )
    assert stamp.objective == OBJECTIVE
    assert "least squared force" in stamp.objective
    assert "opposing-force cancellation" in stamp.objective
    lowered = stamp.objective.lower()
    assert "wasted energy" not in lowered and "biological optimality" not in lowered
    assert stamp.footer == "source=fixture-swing model=toy-model sha=abc123def"
    assert stamp.badges == ("allocation-only", "forward-replayed")
    assert _inked(image, spec.text_origin("objective"), OBJECTIVE, spec.yellow)
    assert _inked(image, spec.text_origin("footer"), stamp.footer, spec.white)
    assert _inked(image, spec.badge_text_origin(0), "allocation-only", spec.black)
    assert _inked(image, spec.badge_text_origin(1), "forward-replayed", spec.black)
    assert tuple(image[spec.header_sample(0)]) == spec.component_rgb["ztcf"]
    assert tuple(image[spec.header_sample(1)]) == spec.component_rgb["zvcf"]
    assert tuple(image[spec.header_sample(2)]) == spec.component_rgb["total"]
    assert tuple(image[spec.badge_sample(0)]) == spec.badge_rgb["allocation-only"]
    assert tuple(image[spec.badge_sample(1)]) == spec.badge_rgb["forward-replayed"]


def test_unavailable_pixels_stay_visible_and_zero_force_does_not() -> None:
    spec = GridSpec()
    present = _panel(
        "baseline", "total", (0.0, 0.2), ((0, 0, 0), None), badge="allocation-only"
    )
    synced = align((present,), (0.0, 0.2), tolerance_s=0.0)
    shown, stamp = compose(synced[0], Provenance("src", "model", "sha"), spec)
    missing, missing_stamp = compose(synced[1], Provenance("src", "model", "sha"), spec)
    y, x = spec.content_sample(0, 2)
    assert (
        tuple(shown[y, x]) == (0, 0, 0)
        and not stamp.missing[2]
        and missing_stamp.missing[2]
    )
    cell = spec.cell_box(0, 2)
    region = missing[cell[1] : cell[1] + cell[3], cell[0] : cell[0] + cell[2]]
    assert np.any(np.all(region == spec.purple, axis=-1))
    assert _inked(missing, spec.unavailable_origin(0, 2), UNAVAILABLE, spec.yellow)
    assert missing_stamp.units[2] == "N"
    assert synced[0].cell("least-force", "ztcf").units == UNAVAILABLE


def test_local_source_is_the_catalog_seam() -> None:
    panel = _panel("baseline", "total", (0.0,), (_RED,), badge="forward-replayed")

    class _Source:
        def sequences(self):
            return (panel,)

    assert load_source(_Source())[0] is panel
    root = Path(__file__).resolve().parents[3] / "src" / "hand_load_video"
    for path in root.glob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "swing_report_package" not in text
        assert "native_viewer_export" not in text
