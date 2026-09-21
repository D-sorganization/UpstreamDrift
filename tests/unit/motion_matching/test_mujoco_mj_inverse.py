"""MS-16 / MS-72: ``mj_inverse`` computed-torque tracking backend (#10366).

Torques from ``mj_inverse`` on an IK trajectory, replayed forward for 50 steps,
must reproduce the reference within a predeclared discretization error budget
verified by timestep refinement, with and without contact.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)
from src.engines.physics_engines.mujoco.python.inverse_dynamics import (
    DISCRETIZATION_REFINEMENT_FACTOR,
    KKT_EQUIVALENCE_RTOL,
    REPLAY_DISCRETIZATION_RMS_RAD,
    REPLAY_DT_S,
    REPLAY_STEPS,
    inverse_dynamics_mj_inverse,
    tracking_controller_mj_inverse,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(
        importlib.util.find_spec("mujoco") is None, reason="mujoco not installed"
    ),
]

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = REPO_ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


def _full_body_simulator():
    if not SPEC_PATH.is_file():
        pytest.skip("Full-body spec missing")
    adapter = NativeMujocoFullBodyModel(SPEC_PATH.read_bytes())
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs

    return fs.FullBodySimulator(adapter)


def _replay_rms(
    simulator,
    q_ref: np.ndarray,
    *,
    dt_s: float,
    steps: int,
    use_mj_inverse: bool,
) -> float:
    duration_s = dt_s * (steps - 1)
    times = np.linspace(0.0, duration_s, steps)
    if q_ref.shape[0] != steps:
        source_times = np.linspace(0.0, duration_s, q_ref.shape[0], dtype=float)
        reference = np.array(
            [
                [np.interp(t, source_times, q_ref[:, k]) for k in range(simulator.nv)]
                for t in times
            ]
        )
    else:
        reference = q_ref
    if use_mj_inverse:
        controller = tracking_controller_mj_inverse(
            simulator,
            times,
            reference,
            omega_rad_s=8.0,
            zeta=1.0,
        )
    else:
        from src.shared.python.motion_matching import full_body_forward_dynamics as fs

        controller = fs.tracking_controller(
            simulator, times, reference, omega_rad_s=8.0, zeta=1.0
        )
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs

    q0 = fs.preload_feet(simulator, reference[0])
    v0 = np.zeros(simulator.nv)
    record = simulator.run(
        q0,
        v0,
        controller,
        duration_s=float(times[-1]),
        dt_s=dt_s,
        record_every=1,
    )
    sim_q = np.array(
        [
            [np.interp(t, record.time_s, record.q[:, k]) for k in range(simulator.nv)]
            for t in times
        ]
    )
    return float(np.sqrt(np.mean((sim_q - reference) ** 2)))


def test_mj_inverse_matches_kkt_without_contact() -> None:
    sim = _full_body_simulator()
    q = np.zeros(sim.nv)
    q[2] = 2.0
    v = np.zeros(sim.nv)
    rng = np.random.default_rng(11)
    qacc = np.zeros(sim.nv)
    qacc[6:] = rng.uniform(-0.5, 0.5, size=sim.nv - 6)
    kkt = sim.inverse_dynamics(q, v, qacc[sim.actuated])
    mj = inverse_dynamics_mj_inverse(sim, q, v, qacc)
    np.testing.assert_allclose(
        mj[sim.actuated],
        kkt[sim.actuated],
        rtol=KKT_EQUIVALENCE_RTOL,
        atol=5.0,
    )


def _stable_replay_reference(simulator, *, use_contact: bool) -> np.ndarray:
    """Build a short smooth hold trajectory stable under computed-torque replay."""
    q_base = np.zeros(simulator.nv)
    q_base[2] = 1.0 if use_contact else 1.5
    q_base[6:] = 0.02
    q_ref = np.repeat(q_base[None, :], REPLAY_STEPS, axis=0)
    phase = np.linspace(0.0, np.pi, REPLAY_STEPS)
    q_ref[:, 6] += 0.03 * np.sin(phase)
    q_ref[:, 7] += 0.02 * np.sin(phase + 0.4)
    return q_ref


@pytest.mark.parametrize("use_contact", [False, True])
def test_mj_inverse_tracking_replay_within_discretization_budget(
    use_contact: bool,
) -> None:
    simulator = _full_body_simulator()
    q_ref = _stable_replay_reference(simulator, use_contact=use_contact)

    coarse = _replay_rms(
        simulator, q_ref, dt_s=REPLAY_DT_S, steps=REPLAY_STEPS, use_mj_inverse=True
    )
    fine_dt = REPLAY_DT_S / DISCRETIZATION_REFINEMENT_FACTOR
    fine_steps = REPLAY_STEPS * DISCRETIZATION_REFINEMENT_FACTOR
    fine = _replay_rms(
        simulator, q_ref, dt_s=fine_dt, steps=fine_steps, use_mj_inverse=True
    )
    assert coarse <= REPLAY_DISCRETIZATION_RMS_RAD
    assert fine <= coarse + REPLAY_DISCRETIZATION_RMS_RAD
