"""MuJoCo-backed surface-event regression on a synthetic finite face.

This is a public, self-contained mechanics fixture. It does not load any
captured swing or user-specific model data.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.model_appearance.ball import BALL_RADIUS_M
from src.shared.python.motion_matching.impact_force import (
    ImpactForce,
    step_through_impact,
)
from src.shared.python.motion_matching.same_input.integrator import zoh_rk4_step

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]


def test_surface_event_converges_under_step_halving_on_mujoco() -> None:
    mujoco = pytest.importorskip(
        "mujoco", reason="MuJoCo is required for this regression"
    )
    xml = """<mujoco>
      <option gravity="0 0 0" timestep="0.001"/>
      <worldbody>
        <body name="club">
          <joint name="slide" type="slide" axis="1 0 0"/>
          <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
        </body>
      </worldbody>
    </mujoco>"""
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    def pose(q: np.ndarray) -> dict[str, np.ndarray]:
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        matrix = np.eye(4)
        matrix[:3, :3] = data.xmat[1].reshape(3, 3)
        matrix[:3, 3] = data.xpos[1]
        return {"club": matrix}

    def accel(q: np.ndarray, v: np.ndarray, force: np.ndarray) -> np.ndarray:
        data.qpos[:], data.qvel[:], data.qfrc_applied[:] = q, v, force
        mujoco.mj_forward(model, data)
        return data.qacc.copy()

    class Plant:
        def acceleration(
            self, q: np.ndarray, v: np.ndarray, force: np.ndarray
        ) -> np.ndarray:
            return accel(q, v, force)

    plant = Plant()

    def run(dt_s: float) -> ImpactForce:
        q0, v0 = np.array([0.0]), np.array([1.0])
        root_time = 0.00063
        centre = np.array([root_time + BALL_RADIUS_M, 0.0, 0.0])
        # Two triangles form a finite square in the club frame, wound toward +x.
        face = np.array(
            [
                [[0.0, -0.05, -0.05], [0.0, 0.05, -0.05], [0.0, 0.05, 0.05]],
                [[0.0, -0.05, -0.05], [0.0, 0.05, 0.05], [0.0, -0.05, 0.05]],
            ]
        )
        impact = ImpactForce(
            frame="club",
            point_in_frame=np.zeros(3),
            axis_in_frame=np.array([1.0, 0.0, 0.0]),
            t_start_s=0.0,
            duration_s=0.0002,
            swing_span_s=(0.0, 0.01),
            trigger="surface",
            ball_centre_m=centre,
            face_triangles_in_frame=face,
            armed_at_s=0.0,
        )

        def step(
            t: float,
            h: float,
            q: np.ndarray,
            v: np.ndarray,
            external,
        ) -> tuple[np.ndarray, np.ndarray, None]:
            class ForcedPlant:
                def acceleration(
                    self, q_stage: np.ndarray, v_stage: np.ndarray, force: np.ndarray
                ) -> np.ndarray:
                    applied = force if external is None else force + external(q_stage)
                    return plant.acceleration(q_stage, v_stage, applied)

            qn, vn = zoh_rk4_step(ForcedPlant(), q, v, np.zeros(1), h)
            return qn, vn, None

        q, v, t, result = q0, v0, 0.0, impact
        while t < 0.001 - 1e-15:
            q, v, result, _ = step_through_impact(
                result,
                t,
                dt_s,
                q,
                v,
                step,
                poses=pose,
                accel=accel,
            )
            t += dt_s
        assert result.contact_event is not None
        return result

    coarse = run(0.001)
    fine = run(0.0005)
    coarse_event, fine_event = coarse.contact_event, fine.contact_event
    assert coarse_event is not None and fine_event is not None
    assert coarse_event.time_s == pytest.approx(0.00063, abs=1e-9)
    assert fine_event.time_s == pytest.approx(coarse_event.time_s, abs=1e-9)
    np.testing.assert_allclose(coarse_event.q, fine_event.q, atol=1e-9)
    np.testing.assert_allclose(coarse_event.qd, fine_event.qd, atol=1e-9)
    np.testing.assert_allclose(coarse_event.q, [0.00063], atol=1e-9)
    np.testing.assert_allclose(coarse_event.qd, [1.0], atol=1e-9)
    assert np.linalg.norm(coarse_event.contact_normal - [1.0, 0.0, 0.0]) < 1e-9
    point = pose(coarse_event.q)["club"][:3, 3]
    assert np.linalg.norm(coarse_event.contact_point_m - point) < 1e-9
    residual = (
        np.linalg.norm(
            np.array([0.00063 + BALL_RADIUS_M, 0.0, 0.0]) - coarse_event.contact_point_m
        )
        - BALL_RADIUS_M
    )
    assert abs(residual) < 1e-9
