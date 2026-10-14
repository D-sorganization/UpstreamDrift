"""OpenSim decorative ball hidden after impact (GCV-13, #11719)."""

import math
from pathlib import Path

import pytest

from src.shared.python.model_appearance.ball import BALL_RADIUS_M
from src.tools.native_viewer_export.ball import BallPlacement
from tests.helpers.opensim_support import require_opensim

pytestmark = pytest.mark.unit
SPECS = Path(__file__).resolve().parents[4] / "docs/development/full_body_models"


def _worker():
    from src.tools.native_viewer_export.backends import opensim_worker

    return opensim_worker


def test_slide_is_zero_while_visible_and_far_once_hidden() -> None:
    worker = _worker()
    assert worker.ball_slide_m([True, False], 0) == 0.0
    assert worker.ball_slide_m([True, False], 1) == worker.BALL_HIDE_OFFSET_M
    assert worker.BALL_HIDE_OFFSET_M >= 100.0


def test_slide_without_a_visibility_list_keeps_the_ball_in_place() -> None:
    assert _worker().ball_slide_m(None, 3) == 0.0


def test_slide_rejects_a_position_outside_the_visibility_list() -> None:
    with pytest.raises(ValueError, match="position"):
        _worker().ball_slide_m([True], 1)


def test_opensim_backend_no_longer_records_a_ball_limitation() -> None:
    from src.tools.native_viewer_export.backends.opensim_simbody import (
        OpenSimSimbodyBackend,
    )

    assert getattr(OpenSimSimbodyBackend, "ball_limitation", None) is None


def test_ball_body_sits_at_address_and_slides_out_of_view() -> None:
    osim = require_opensim()
    worker = _worker()
    raw = (SPECS / "full_body_spec_anthro_driver.json").read_bytes()
    model, _ = worker.build_model(osim, raw)
    before = model.getCoordinateSet().getSize()
    placement = BallPlacement((0.5, 0.1, 0.02), BALL_RADIUS_M, "test", 1.0)
    name = worker.attach_ball(osim, model, placement.to_dict())
    state = model.initSystem()
    coords = model.getCoordinateSet()
    assert coords.getSize() == before + 1  # only the render-only slide is added
    body = model.getBodySet().get(worker.BALL_BODY)
    for slide, expect_far in ((0.0, False), (worker.BALL_HIDE_OFFSET_M, True)):
        coords.get(name).setValue(state, slide, False)
        model.realizePosition(state)
        p = body.getPositionInGround(state)
        offset = math.dist([p.get(i) for i in range(3)], placement.position_m)
        assert (offset >= 100.0) if expect_far else (offset < 1e-9)
