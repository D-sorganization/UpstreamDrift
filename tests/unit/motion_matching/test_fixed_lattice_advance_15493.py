"""Fixed-step numerical prefix is not a continuous-physics certificate."""

import numpy as np
import pytest

from src.shared.python.motion_matching.fixed_lattice_advance import FixedLatticeAdvance
from src.shared.python.motion_matching.same_input.integrator import zoh_rk4_step

pytestmark = pytest.mark.unit


class Oscillator:
    def acceleration(self, q, v, tau):
        return -q + tau


def test_prefix_uses_actual_fixed_step_endpoints():
    q, v, tau = np.array([1.0]), np.array([0.0]), np.array([0.1])
    lattice = FixedLatticeAdvance(Oscillator(), q, v, tau, 1.0, 4)
    q_step, v_step = q.copy(), v.copy()
    for i in range(1, 5):
        q_step, v_step = zoh_rk4_step(Oscillator(), q_step, v_step, tau, 0.25)
        actual_q, actual_v = lattice(i * 0.25)
        np.testing.assert_array_equal(actual_q, q_step)
        np.testing.assert_array_equal(actual_v, v_step)
    assert lattice.projection_status == "unchecked"
    assert lattice.coverage == "uncertified"
    left, _ = lattice(np.nextafter(0.5, 0.0))
    right, _ = lattice(np.nextafter(0.5, 1.0))
    np.testing.assert_allclose(left, right, atol=1e-13, rtol=0)


def test_projection_discontinuity_and_invalid_queries_fail_closed():
    q, v, tau = np.array([1.0]), np.array([0.0]), np.array([0.0])
    first = FixedLatticeAdvance(Oscillator(), q, v, tau, 1.0, 4)
    projected = first(1.0)
    identical = FixedLatticeAdvance(
        Oscillator(), q, v, tau, 1.0, 4, projected_end=projected
    )
    assert identical.projection_status == "identical"
    changed = (projected[0] + 1e-8, projected[1])
    with pytest.raises(ValueError, match="projection"):
        FixedLatticeAdvance(Oscillator(), q, v, tau, 1.0, 4, projected_end=changed)
    changed_velocity = (projected[0], projected[1] + 1e-8)
    with pytest.raises(ValueError, match="projection"):
        FixedLatticeAdvance(
            Oscillator(), q, v, tau, 1.0, 4, projected_end=changed_velocity
        )
    for t in (-0.01, 1.01, np.nan):
        with pytest.raises(ValueError):
            first(t)
    returned, _ = first(0.25)
    returned[:] = 99
    assert first(0.25)[0][0] != 99


def test_hold_change_requires_separate_lattices():
    lattice = FixedLatticeAdvance(Oscillator(), [1.0], [0.0], [0.1], 0.5, 2)
    end_q, end_v = lattice(0.5)
    next_lattice = FixedLatticeAdvance(Oscillator(), end_q, end_v, [-0.1], 0.5, 2)
    np.testing.assert_array_equal(next_lattice(0)[0], end_q)


@pytest.mark.requires_mujoco
def test_mujoco_lattice_drives_unqualified_finite_face_diagnostic():
    mujoco = pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.face_trajectory_diagnostic import diagnose

    model = mujoco.MjModel.from_xml_string("""<mujoco>
      <option gravity="0 0 0"/>
      <worldbody><body name="club"><joint type="slide" axis="1 0 0"/>
        <inertial pos="0 0 0" mass="1" diaginertia="1 1 1"/>
      </body></worldbody></mujoco>""")
    data = mujoco.MjData(model)

    class Plant:
        def acceleration(self, q, v, tau):
            data.qpos[:], data.qvel[:], data.qfrc_applied[:] = q, v, tau
            mujoco.mj_forward(model, data)
            return data.qacc.copy()

    lattice = FixedLatticeAdvance(Plant(), [0], [1], [0], 0.8, 8)
    face = np.array([[[0.0, -1.0, -1.0], [0.0, 1.0, -1.0], [0.0, 0.0, 1.0]]])

    def triangles(q):
        return face + [q[0], 0, 0]

    def point_velocity(q, qd, point):
        return np.array([qd[0], 0.0, 0.0])

    report = diagnose(lattice, triangles, point_velocity, [0.7, 0, 0], 0.2, 0.8)
    assert report.candidate is not None
    assert report.candidate.time_s == pytest.approx(0.5, abs=1e-8)
    assert report.status == "unqualified" and report.coverage == "uncertified"
    assert lattice.projection_status == "unchecked"
