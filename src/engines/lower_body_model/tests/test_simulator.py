from types import SimpleNamespace

import numpy as np
import pytest


pytest.importorskip("mujoco")

from src.engines.lower_body_model import simulator as simulator_module  # noqa: E402


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
