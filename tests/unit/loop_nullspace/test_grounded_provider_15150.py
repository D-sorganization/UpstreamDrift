"""Planted feet in the ZTCF/ZVCF split (#15150 Track Z, Z5).

ZTCF and ZVCF re-solve every reaction for each part. The hand-club closure
was the only constraint, so with ``tau = 0`` the whole golfer fell freely:
gravity cancelled inside the body, the hands no longer held the club's
weight, and leg torques threw a floating body about instead of pushing on
the ground. The feet are reactions too, so the bodies that carry the contact
spheres are welded to the ground and re-solved in each part like the hands.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.loop_nullspace.grounded_provider import GroundedProvider

pytestmark = pytest.mark.unit


class _Fake:
    nv = 4

    def mass_matrix(self, q):
        return np.eye(4)

    def bias_forces(self, q, v):
        return np.arange(4.0)

    def closure_jacobian(self, q):
        return np.ones((2, 4))

    def closure_drift(self, q, v):
        return np.full(2, 7.0)


def _rows(q, v):
    return np.eye(4)[:3], np.asarray(v, dtype=float)[:3]


def test_ground_rows_follow_the_closure_rows() -> None:
    grounded = GroundedProvider(_Fake(), _rows)
    q, v = np.zeros(4), np.array([1.0, 2.0, 3.0, 4.0])
    jac = grounded.closure_jacobian(q)
    assert jac.shape == (5, 4)
    np.testing.assert_array_equal(jac[:2], 1.0)
    np.testing.assert_array_equal(jac[2:], np.eye(4)[:3])
    np.testing.assert_array_equal(grounded.closure_drift(q, v), [7, 7, 1, 2, 3])
    assert grounded.closure_rows(q) == 2
    assert grounded.nv == 4
    np.testing.assert_array_equal(grounded.bias_forces(q, v), np.arange(4.0))


def test_mismatched_rows_are_refused() -> None:
    def bad(q, v):
        return np.eye(4)[:3], np.zeros(2)

    with pytest.raises(ValueError, match="rows"):
        GroundedProvider(_Fake(), bad).closure_drift(np.zeros(4), np.zeros(4))


SEEDS = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds"
)


@pytest.fixture(scope="module")
def adapter():
    pytest.importorskip("pinocchio")
    from src.engines.physics_engines.pinocchio.python.force_adapter import (
        PinocchioForceAdapter,
    )

    return PinocchioForceAdapter(
        (SEEDS / "full_body_spec_hipcal_scaled.json").read_bytes()
    )


def _pose(adapter, row: int) -> tuple[np.ndarray, np.ndarray]:
    with np.load(SEEDS / "dynamics_record.npz") as rec:
        return rec["q"][row].copy(), rec["v"][row].copy()


@pytest.mark.requires_pinocchio
def test_one_weld_per_contact_body_and_its_drift(adapter) -> None:
    """Rows are the feet's world-aligned twists; drift is their rate at a = 0."""
    q, v = _pose(adapter, 300)
    jac, drift = adapter.ground_weld_rows(q, v)
    spec = json.loads((SEEDS / "full_body_spec_hipcal_scaled.json").read_bytes())
    bodies = tuple(dict.fromkeys(s["body"] for s in spec["contact"]["spheres"]))
    assert adapter.planted_bodies() == bodies == ("calcn_r", "calcn_l")
    assert jac.shape == (6 * len(bodies), adapter.nv)
    step = 1e-6
    ahead, _ = adapter.ground_weld_rows(q + step * v, v)
    behind, _ = adapter.ground_weld_rows(q - step * v, v)
    rate = (ahead @ v - behind @ v) / (2 * step)
    # Classical (point) acceleration of the body origin: d/dt of J v at a = 0.
    np.testing.assert_allclose(drift, rate, rtol=1e-4, atol=1e-6)


def _holding_torques(adapter, q, provider):
    """Actuated torques and reactions that hold pose ``q`` still."""
    zero = np.zeros(adapter.nv)
    gravity = provider.bias_forces(q, zero)
    jac = provider.closure_jacobian(q)
    select = np.zeros((adapter.nv, len(adapter.actuated_indices)))
    for col, i in enumerate(adapter.actuated_indices):
        select[i, col] = 1.0
    solution, *_ = np.linalg.lstsq(np.hstack([select, jac.T]), gravity, rcond=None)
    tau = select @ solution[: select.shape[1]]
    return tau


@pytest.mark.requires_pinocchio
def test_a_held_golfer_stands_on_the_ground_and_holds_the_club(adapter) -> None:
    """Still pose, holding torques: no motion, the ground carries the body.

    Closure-only, the same torques leave the root unsupported and the body
    accelerating; planted, the ground's reaction (its rows of drift plus
    control) is the whole body weight and the hands hold the club's weight.
    """
    from src.shared.python.loop_nullspace.provider import AdapterConstrainedProvider
    from src.shared.python.simulation_backends.constrained_ztcf_zvcf import (
        drift_and_control_states,
    )
    from src.shared.python.simulation_backends.hand_force_split import (
        CLOSURE_ROWS,
        HANDS,
        PARTS,
        hand_wrench_split,
    )

    q, _ = _pose(adapter, 5)
    zero = np.zeros(adapter.nv)
    closure_only = AdapterConstrainedProvider(adapter)
    grounded = GroundedProvider(closure_only, adapter.ground_weld_rows)
    tau = _holding_torques(adapter, q, grounded)
    gravity = np.asarray(adapter.gravity_m_s2())

    (a_d, lam_d), (a_c, lam_c) = drift_and_control_states(grounded, q, zero, tau)
    np.testing.assert_allclose(a_d + a_c, 0.0, atol=1e-8)
    ground = (lam_d + lam_c)[CLOSURE_ROWS:].reshape(-1, 6)[:, :3].sum(axis=0)
    body_mass = sum(float(i.mass) for i in adapter._model.inertias)
    np.testing.assert_allclose(ground, -body_mass * gravity, rtol=1e-9, atol=1e-8)

    (f_d, _), (f_c, _) = drift_and_control_states(closure_only, q, zero, tau)
    assert np.abs((f_d + f_c)[:6]).max() > 1.0  # the root falls without ground

    split = hand_wrench_split(
        grounded,
        adapter.closure_wrench_frame,
        adapter.club_subtree_load,
        adapter.lead_wrist_point,
        q,
        zero,
        tau,
    )
    hands = sum(split.part(h, p).force_n[0] for h in HANDS for p in PARTS)
    weight = -adapter.club_subtree_mass_kg() * gravity
    np.testing.assert_allclose(hands, weight, atol=1e-6)
