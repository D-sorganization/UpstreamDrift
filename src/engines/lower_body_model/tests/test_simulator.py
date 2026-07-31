import inspect
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit

pytest.importorskip("mujoco")

from src.engines.lower_body_model import simulator as simulator_module  # noqa: E402


def test_step_history_uses_bounded_deque_without_linear_front_pop(monkeypatch):
    model = SimpleNamespace(
        njnt=0,
        nu=0,
        jnt_qposadr=np.array([], dtype=int),
        jnt_dofadr=np.array([], dtype=int),
    )
    data = SimpleNamespace(
        time=0.0,
        qpos=np.array([0.0, 10.0], dtype=float),
        qvel=np.array([100.0], dtype=float),
        ctrl=np.array([1000.0], dtype=float),
    )
    body_ids = {
        "pelvis": 0,
        "r_calf": 1,
        "l_calf": 2,
        "r_foot": 3,
        "l_foot": 4,
    }
    site_ids = {"r_foot_center": 10, "l_foot_center": 11}
    geom_ids = {"floor": 20, "r_foot_geom": 21, "l_foot_geom": 22}

    def fake_name_to_id(_model, obj_type, name):
        if obj_type == simulator_module.mujoco.mjtObj.mjOBJ_BODY:
            return body_ids[name]
        if obj_type == simulator_module.mujoco.mjtObj.mjOBJ_SITE:
            return site_ids[name]
        if obj_type == simulator_module.mujoco.mjtObj.mjOBJ_GEOM:
            return geom_ids[name]
        raise AssertionError(f"unexpected lookup for {obj_type!r}:{name}")

    def fake_mj_step(_model, mj_data):
        mj_data.time += 0.25
        mj_data.qpos[:] += 1.0
        mj_data.qvel[:] += 10.0
        mj_data.ctrl[:] += 100.0

    monkeypatch.setattr(
        simulator_module.mujoco,
        "MjModel",
        SimpleNamespace(from_xml_string=lambda _xml: model),
    )
    monkeypatch.setattr(simulator_module.mujoco, "MjData", lambda _model: data)
    monkeypatch.setattr(simulator_module.mujoco, "mj_id2name", lambda *_: None)
    monkeypatch.setattr(simulator_module.mujoco, "mj_name2id", fake_name_to_id)
    monkeypatch.setattr(simulator_module.mujoco, "mj_forward", lambda *_: None)
    monkeypatch.setattr(simulator_module.mujoco, "mj_step", fake_mj_step)

    sim = simulator_module.LowerBodySimulator("<mujoco/>")

    assert isinstance(sim.history, deque)
    assert sim.history.maxlen == sim.max_history_length
    assert ".pop(0)" not in inspect.getsource(simulator_module.LowerBodySimulator.step)

    sim.max_history_length = 3
    sim.history = deque(maxlen=sim.max_history_length)

    for _ in range(4):
        sim.step()

    assert len(sim.history) == 3
    assert [frame["time"] for frame in sim.history] == [0.5, 0.75, 1.0]
    np.testing.assert_allclose([frame["qpos"][0] for frame in sim.history], [2, 3, 4])
    np.testing.assert_allclose(
        [frame["qvel"][0] for frame in sim.history],
        [120, 130, 140],
    )
    np.testing.assert_allclose(
        [frame["ctrl"][0] for frame in sim.history],
        [1200, 1300, 1400],
    )


def test_inverse_kinematics_uses_solve_for_damped_least_squares(monkeypatch):
    sim = simulator_module.LowerBodySimulator.__new__(
        simulator_module.LowerBodySimulator
    )
    sim.model = SimpleNamespace(nv=8)
    sim.data = SimpleNamespace(
        qpos=np.zeros(9),
        qvel=np.zeros(8),
        site_xpos=np.array(
            [
                [0.00, -0.10, -0.01],
                [0.02, 0.10, -0.02],
            ],
            dtype=float,
        ),
    )
    sim.site_ids = {"r_foot_center": 0, "l_foot_center": 1}
    sim.qpos_target = None

    solve_calls = []
    original_solve = np.linalg.solve

    def solve_spy(lhs, rhs):
        solve_calls.append((lhs.copy(), rhs.copy()))
        return original_solve(lhs, rhs)

    def inv_forbidden(_lhs):
        raise AssertionError("IK DLS update must use solve(), not inv()")

    def fake_mj_jac_site(_model, _data, jacp, _jacr, site_id):
        jacp[:] = 0.0
        if site_id == sim.site_ids["r_foot_center"]:
            jacp[0, 6] = 1.0
            jacp[1, 7] = 0.25
        else:
            jacp[0, 6] = -0.5
            jacp[1, 7] = 1.0

    def fake_mj_integrate_pos(_model, qpos, dq, _dt):
        qpos[: dq.size] += dq
        sim.data.site_xpos[:] = np.array(
            [
                [0.05, -0.15, -0.03],
                [0.05, 0.15, -0.03],
            ],
            dtype=float,
        )

    monkeypatch.setattr(simulator_module.np.linalg, "solve", solve_spy)
    monkeypatch.setattr(simulator_module.np.linalg, "inv", inv_forbidden)
    monkeypatch.setattr(simulator_module.mujoco, "mj_kinematics", lambda *_: None)
    monkeypatch.setattr(simulator_module.mujoco, "mj_comPos", lambda *_: None)
    monkeypatch.setattr(simulator_module.mujoco, "mj_jacSite", fake_mj_jac_site)
    monkeypatch.setattr(
        simulator_module.mujoco, "mj_integratePos", fake_mj_integrate_pos
    )

    assert sim.inverse_kinematics(
        np.array([0.0, 0.0, 1.0]), np.array([1.0, 0.0, 0.0, 0.0]), max_iters=2
    )
    assert len(solve_calls) == 1
    lhs, rhs = solve_calls[0]
    np.testing.assert_allclose(lhs, lhs.T)
    np.testing.assert_allclose(rhs, np.array([0.05, -0.05, -0.02, 0.03, 0.05, -0.01]))
    assert sim.qpos_target is not None
