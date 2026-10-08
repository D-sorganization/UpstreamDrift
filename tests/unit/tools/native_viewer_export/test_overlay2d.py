"""Projected overlays for viewers without a 3D glyph API (NV-4, #11677)."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("cv2")

from src.shared.python.force_overlay import (  # noqa: E402
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
    build_glyphs,
)
from src.shared.python.golf_view_presets import VIEW_ORDER  # noqa: E402
from src.tools.native_viewer_export.core import default_glyph_style  # noqa: E402
from src.tools.native_viewer_export.overlay2d import (  # noqa: E402
    draw_glyphs_rgb,
    pinhole_for_view,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]
LOOKAT = (1.0, 0.0, 0.9)


@pytest.mark.parametrize("view", VIEW_ORDER)
def test_lookat_projects_to_image_centre(view: str) -> None:
    cam = pinhole_for_view(view, LOOKAT, 3.0, 0.7, (640, 480))
    px, valid = cam.project(np.array([LOOKAT]))
    assert valid.all()
    np.testing.assert_allclose(px[0], (320.0, 240.0), atol=1e-6)


def test_face_on_target_side_is_image_right() -> None:
    cam = pinhole_for_view("face_on", LOOKAT, 3.0, 0.7, (640, 480))
    # target line is -Y; face-on shows the target to the right of the image
    px, _ = cam.project(np.array([[1.0, -0.5, 0.9]]))
    assert px[0, 0] > 320.0
    up_px, _ = cam.project(np.array([[1.0, 0.0, 1.4]]))
    assert up_px[0, 1] < 240.0, "higher world Z is higher in the image (smaller v)"


def test_field_of_view_sets_focal_length() -> None:
    cam = pinhole_for_view("overhead", LOOKAT, None, 0.7, (640, 480))
    assert cam.matrix[1, 1] == pytest.approx(240.0 / np.tan(0.35))


def test_invalid_arguments() -> None:
    with pytest.raises(ValueError):
        pinhole_for_view("face_on", LOOKAT, 3.0, 0.0, (640, 480))
    with pytest.raises(ValueError):
        pinhole_for_view("face_on", LOOKAT, 3.0, 0.7, (0, 480))


def test_glyphs_are_drawn_and_input_is_not_mutated() -> None:
    frame = ForceTorqueFrame(
        time_s=0.0,
        engine="x",
        wrenches=(
            OverlayWrench(
                WrenchKind.CONTACT,
                "contact:grf_r",
                "calcn_r",
                (1.0, 0.0, 0.2),
                force_n=(0.0, 0.0, 900.0),
                source="x",
            ),
        ),
    )
    glyphs = build_glyphs(frame, default_glyph_style())
    assert len(glyphs.arrows) == 1
    cam = pinhole_for_view("face_on", LOOKAT, 3.0, 0.7, (320, 240))
    img = np.zeros((240, 320, 3), np.uint8)
    out = draw_glyphs_rgb(img, glyphs, cam)
    assert img.max() == 0
    assert out.max() > 0 and out.shape == img.shape
