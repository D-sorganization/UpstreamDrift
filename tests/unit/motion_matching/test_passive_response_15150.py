"""Passive coordinates in the kinetic match (#15150 Track Z, Z6b).

A flexible shaft has coordinates that no golfer actuates and that the hand
closure does not reach. Their rows of ``M a + b = S^T tau + J^T lambda``
carry no unknown, so the reference must satisfy them exactly: the shaft has
to be the passive response to the golfer's prescribed motion, not a
trajectory copied from another simulation. These tests pin that contract.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.integrate import solve_ivp

from src.shared.python.motion_matching.passive_response import (
    passive_indices,
    passive_response,
)

pytestmark = pytest.mark.unit

ROOT = 6
M_ACTIVE, M_PASSIVE, COUPLING = 2.0, 0.05, 0.02
STIFFNESS, DAMPING = 40.0, 0.03
AMPLITUDE, FREQ_HZ = 0.4, 1.5
SUBSTEP_S = 1e-4


class _Linear:
    """Root (6 still rows), one actuated row, one passive spring row."""

    nv = 8
    actuated_indices = (6,)

    def compute_mass_and_bias(self, q, v):
        mass = np.eye(self.nv)
        mass[6, 6], mass[7, 7] = M_ACTIVE, M_PASSIVE
        mass[6, 7] = mass[7, 6] = COUPLING
        bias = np.zeros(self.nv)
        bias[7] = STIFFNESS * q[7] + DAMPING * v[7]
        return mass, bias


def _active(times):
    w = 2.0 * np.pi * FREQ_HZ
    return (
        AMPLITUDE * np.sin(w * times),
        AMPLITUDE * w * np.cos(w * times),
        -AMPLITUDE * w**2 * np.sin(w * times),
    )


def _arrays(times, passive_q=0.0):
    q, v, a = (np.zeros((times.size, _Linear.nv)) for _ in range(3))
    q[:, 6], v[:, 6], a[:, 6] = _active(times)
    q[:, 7] = passive_q
    return q, v, a


def _passive_residual(adapter, q, v, a, idx):
    worst = 0.0
    for k in range(q.shape[0]):
        mass, bias = adapter.compute_mass_and_bias(q[k], v[k])
        worst = max(worst, float(np.abs((mass @ a[k] + bias)[idx]).max()))
    return worst


def test_passive_indices_are_the_unactuated_non_root_rows() -> None:
    np.testing.assert_array_equal(passive_indices(_Linear()), [7])


def test_passive_rows_balance_exactly_at_every_frame() -> None:
    times = np.arange(120) * (1.0 / 360.0)
    q, v, a = _arrays(times, passive_q=0.3)
    passive_response(_Linear(), times, q, v, a, SUBSTEP_S)
    assert _passive_residual(_Linear(), q, v, a, [7]) < 1e-12


def test_active_rows_are_left_alone() -> None:
    times = np.arange(60) * (1.0 / 360.0)
    q, v, a = _arrays(times, passive_q=0.3)
    before = (q[:, :7].copy(), v[:, :7].copy(), a[:, :7].copy())
    passive_response(_Linear(), times, q, v, a, SUBSTEP_S)
    for after, ref in zip((q[:, :7], v[:, :7], a[:, :7]), before, strict=True):
        np.testing.assert_array_equal(after, ref)


def test_response_matches_an_independent_integration() -> None:
    """The passive row is ``m q'' + c v + k q = -C a_active``, from rest."""
    times = np.arange(360) * (1.0 / 360.0)
    q, v, a = _arrays(times)
    passive_response(_Linear(), times, q, v, a, SUBSTEP_S)

    def rhs(t, y):
        accel = _active(np.array([t]))[2][0]
        return [
            y[1],
            -(COUPLING * accel + DAMPING * y[1] + STIFFNESS * y[0]) / M_PASSIVE,
        ]

    # The response starts from quasi-static equilibrium with the frame-0 load.
    start = -COUPLING * a[0, 6] / STIFFNESS
    ref = solve_ivp(
        rhs, (0.0, times[-1]), [start, 0.0], t_eval=times, rtol=1e-10, atol=1e-12
    )
    np.testing.assert_allclose(q[:, 7], ref.y[0], atol=2e-4 * AMPLITUDE)


def test_no_passive_rows_leaves_everything_unchanged() -> None:
    class _AllActuated(_Linear):
        actuated_indices = (6, 7)

    times = np.arange(20) * (1.0 / 360.0)
    q, v, a = _arrays(times, passive_q=0.3)
    copies = (q.copy(), v.copy(), a.copy())
    assert passive_response(_AllActuated(), times, q, v, a, SUBSTEP_S) == 0
    for after, ref in zip((q, v, a), copies, strict=True):
        np.testing.assert_array_equal(after, ref)


@pytest.mark.parametrize("substep", [0.0, -1e-4, float("nan")])
def test_substep_must_be_positive(substep) -> None:
    times = np.arange(10) * 0.01
    with pytest.raises(ValueError, match="substep"):
        passive_response(_Linear(), times, *_arrays(times), substep)


def test_shapes_must_agree() -> None:
    times = np.arange(10) * 0.01
    q, v, a = _arrays(times)
    with pytest.raises(ValueError, match="shape"):
        passive_response(_Linear(), times, q, v[:-1], a, SUBSTEP_S)


SEEDS = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds"
)


@pytest.mark.requires_mujoco
def test_held_address_pose_leaves_the_shaft_at_its_static_sag() -> None:
    """Golfer still: the chain shaft sits at gravity sag and stays there."""
    pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.execution.spec_builder import (
        with_shaft_model,
    )
    from src.shared.python.motion_matching.multi_engine_torque_allocator import (
        MujocoForceAdapter,
    )

    doc = json.loads((SEEDS / "full_body_spec_hipcal_scaled.json").read_bytes())
    adapter = MujocoForceAdapter(
        json.dumps(with_shaft_model(doc, "driver", "chain")).encode()
    )
    idx = passive_indices(adapter)
    assert idx.size == 8
    native = list(adapter.coordinate_order)
    with np.load(SEEDS / "dynamics_record.npz") as rec:
        pose = np.zeros(adapter.nv)
        for i, name in enumerate(doc["coordinate_order"]):
            pose[native.index(name)] = rec["q"][0, i]
    times = np.arange(30) * (1.0 / 360.0)
    q = np.tile(pose, (times.size, 1))
    v, a = np.zeros_like(q), np.zeros_like(q)
    passive_response(adapter, times, q, v, a, SUBSTEP_S)
    assert _passive_residual(adapter, q, v, a, idx) < 1e-8
    assert np.abs(q[:, idx]).max() > 1e-5  # gravity bends the shaft
    np.testing.assert_allclose(q[-1, idx], q[0, idx], atol=1e-9)
    assert np.abs(v[:, idx]).max() < 1e-6
