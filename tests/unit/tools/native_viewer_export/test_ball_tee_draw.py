"""Tee drawn under the lifted native-export ball (#15150)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from src.shared.python.model_appearance.ball import BALL_RADIUS_M
from src.shared.python.model_appearance.ball_tee import (
    default_ball_tee_settings,
    tee_for_ball,
)
from src.tools.native_viewer_export.backends import _ball
from src.tools.native_viewer_export.backends._worker_job import WorkerJob
from src.tools.native_viewer_export.ball import BallPlacement, attach_ball
from src.tools.native_viewer_export.core import ExportSettings
from tests.unit.tools.native_viewer_export.test_export_ball import _fixture_swing

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

LIFTED = BallPlacement((0.5, 0.1, 0.059 + BALL_RADIUS_M), BALL_RADIUS_M, "t", 1.0)
ON_TURF = BallPlacement((0.5, 0.1, BALL_RADIUS_M), BALL_RADIUS_M, "t", 1.0)


def _job(ball: BallPlacement, visible: list[bool]) -> WorkerJob:
    return WorkerJob(
        "b", "q", [0, 1], ["v"], 8, 8, [0, 0, 0], None, "out",
        ball=ball.to_dict(), ball_visible=visible,
    )  # fmt: skip


def _scene() -> tuple[Any, Any, Any]:
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    return mujoco, model, mujoco.MjvScene(model, maxgeom=20)


def _cylinders(mujoco: Any, scene: Any) -> list[Any]:
    kind = int(mujoco.mjtGeom.mjGEOM_CYLINDER)
    return [g for g in scene.geoms[: scene.ngeom] if int(g.type) == kind]


def test_ground_height_round_trips_and_is_omitted_at_zero() -> None:
    assert "ground_height_m" not in LIFTED.to_dict()
    raised = BallPlacement((0.0, 0.0, 0.3), 0.02, "t", None, 0.25)
    assert BallPlacement.from_dict(raised.to_dict()) == raised
    with pytest.raises(ValueError):
        BallPlacement((0.0, 0.0, 0.3), 0.02, "t", None, float("nan"))


def test_attached_ball_carries_its_ground_height() -> None:
    swing = _fixture_swing("driver", returning=True)
    attached, record = attach_ball(swing, ExportSettings())
    assert attached.ball is not None and record is not None
    # the fixture's ground plane is 0.26 mm up; the ball rests on it
    assert attached.ball.ground_height_m == pytest.approx(0.0, abs=1e-3)
    assert attached.ball.position_m[2] - attached.ball.radius_m >= (
        attached.ball.ground_height_m - 1e-9
    )


def test_lifted_ball_gets_a_tee_that_meets_it() -> None:
    mujoco, _, scene = _scene()
    assert _ball.draw_job_ball(scene, _job(LIFTED, [True, True]), 0)
    cylinders = _cylinders(mujoco, scene)
    assert len(cylinders) == 2 and scene.ngeom == 3
    expected = tee_for_ball(
        LIFTED.position_m, LIFTED.radius_m, default_ball_tee_settings()
    )
    assert expected is not None
    tops = [g.pos[2] + g.size[2] for g in cylinders]
    bottoms = [g.pos[2] - g.size[2] for g in cylinders]
    assert min(bottoms) == pytest.approx(LIFTED.ground_height_m, abs=1e-6)
    assert max(tops) == pytest.approx(expected.cup.top_z_m, abs=1e-6)
    assert max(tops) <= LIFTED.position_m[2] - BALL_RADIUS_M + 0.004
    np.testing.assert_allclose(cylinders[0].pos[:2], LIFTED.position_m[:2])


def test_ball_on_turf_gets_no_tee() -> None:
    mujoco, _, scene = _scene()
    assert _ball.draw_job_ball(scene, _job(ON_TURF, [True, True]), 0)
    assert scene.ngeom == 1 and not _cylinders(mujoco, scene)


def test_tee_has_no_collision_and_leaves_the_model_alone() -> None:
    mujoco, model, scene = _scene()
    mass, nbody = float(model.body_subtreemass[0]), model.nbody
    _ball.draw_job_ball(scene, _job(LIFTED, [True, True]), 0)
    assert model.ngeom == 0 and model.nbody == nbody
    assert float(model.body_subtreemass[0]) == mass
    # render geoms name no model object: nothing the contact engine can see
    unknown = int(mujoco.mjtObj.mjOBJ_UNKNOWN)
    assert all(int(g.objtype) == unknown for g in _cylinders(mujoco, scene))


def test_tee_stays_when_ball_leaves() -> None:
    mujoco, _, shown = _scene()
    _, _, gone = _scene()
    job = _job(LIFTED, [True, False])
    assert _ball.draw_job_ball(shown, job, 0)
    assert not _ball.draw_job_ball(gone, job, 1)
    assert gone.ngeom == 2  # tee only, ball hidden
    before = [(tuple(g.pos), tuple(g.size)) for g in _cylinders(mujoco, shown)]
    after = [(tuple(g.pos), tuple(g.size)) for g in _cylinders(mujoco, gone)]
    assert before == after
