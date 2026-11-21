"""The flying ball is drawn as a lit sphere, not a flat disc (#15150)."""

from __future__ import annotations

from dataclasses import replace
import json

from PIL import Image
import numpy as np
import pytest

from src.tools.native_viewer_export import flight_depth as fd, flight_tracer as tr

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

SIZE, CENTRE, RADIUS = 41, (20.0, 20.0), 12.0
WHITE = (255, 255, 255)
SHADING = tr.load_tracer_settings().ball_shading
FLAT = replace(SHADING, shaded=False)
SPHERE = (5.0, 0.02)


def _draw(shading, radius=RADIUS, occlusion=None, colour=WHITE) -> np.ndarray:
    image = Image.new("RGB", (SIZE, SIZE), (40, 90, 40))
    fd.draw_ball(image, CENTRE, radius, colour, SPHERE, occlusion, shading)
    return np.asarray(image).astype(int)


def _flat_reference(radius=RADIUS, occlusion=None) -> np.ndarray:
    image = Image.new("RGB", (SIZE, SIZE), (40, 90, 40))
    fd.draw_ball(image, CENTRE, radius, WHITE, SPHERE, occlusion)
    return np.asarray(image).astype(int)


def _covered(img: np.ndarray) -> np.ndarray:
    return (img != np.array((40, 90, 40))).any(axis=2)


def test_shading_off_is_pixel_identical_to_the_flat_disc() -> None:
    assert (_draw(FLAT) == _flat_reference()).all()
    assert (_draw(None) == _flat_reference()).all()
    assert (_flat_reference() == WHITE).all(axis=2).sum() > 100


def test_shading_on_lights_the_side_toward_the_light() -> None:
    img, lx, ly = _draw(SHADING), *SHADING.light_dir[:2]
    lum = img.sum(axis=2)
    covered = _covered(img)
    assert covered.sum() == _covered(_flat_reference()).sum()
    assert img.min() >= 0 and img.max() <= 255
    v, u = np.unravel_index(np.where(covered, lum, -1).argmax(), lum.shape)
    assert (u - CENTRE[0]) * lx + (v - CENTRE[1]) * ly > 0
    toward = np.array(CENTRE) + RADIUS * 0.9 * np.array([lx, ly]) / np.hypot(lx, ly)
    away = np.array(CENTRE) - RADIUS * 0.9 * np.array([lx, ly]) / np.hypot(lx, ly)
    dark = lum[int(round(away[1])), int(round(away[0]))]
    assert dark < lum[CENTRE_I, CENTRE_I] < lum.max() + 1
    assert dark < lum[int(round(toward[1])), int(round(toward[0]))]


CENTRE_I = int(CENTRE[0])


def test_ambient_floor_keeps_the_dark_limb_above_black() -> None:
    img = _draw(SHADING)
    floor = SHADING.ambient * 255
    assert img[_covered(img)].min() >= int(floor) - 1


def test_occluded_pixels_stay_untouched_when_shaded() -> None:
    depth = np.full((SIZE, SIZE), 100.0, np.float32)
    depth[:, :CENTRE_I] = 1.0  # a scene surface nearer than the ball, left half
    occ = fd.Occlusion(depth, 0.0, 0.0, 3.0, 0.05)
    shaded, flat = _draw(SHADING, occlusion=occ), _flat_reference(occlusion=occ)
    assert (_covered(shaded) == _covered(flat)).all()
    assert not _covered(shaded)[:, : CENTRE_I - 1].any()
    assert _covered(shaded)[:, CENTRE_I + 1 :].any()


@pytest.mark.parametrize("radius", [1.0, 2.0, 3.0])
def test_small_balls_stay_visible_and_bright(radius: float) -> None:
    img = _draw(SHADING, radius=radius)
    covered = _covered(img)
    assert covered.sum() >= 3
    assert img[covered].max() > 200


def test_shading_settings_are_validated_on_load(tmp_path) -> None:
    raw = json.loads(tr.SETTINGS_PATH.read_text(encoding="utf-8"))
    bad = {
        "light_dir": [0, 0, 0],
        "ambient": 1.5,
        "highlight_strength": -0.1,
        "highlight_size": 0.0,
    }
    for key, value in bad.items():
        doc = json.loads(json.dumps(raw))
        doc["ball_shading"][key] = value
        path = tmp_path / "flight_tracer_settings.json"
        path.write_text(json.dumps(doc), encoding="utf-8")
        with pytest.raises(ValueError, match=key):
            tr.load_tracer_settings(path)
    doc = json.loads(json.dumps(raw))
    doc["ball_shading"]["shaded"] = "yes"
    path = tmp_path / "flight_tracer_settings.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    with pytest.raises(TypeError, match="shaded"):
        tr.load_tracer_settings(path)
