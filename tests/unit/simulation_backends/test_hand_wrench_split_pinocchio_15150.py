"""Two-hand ZTCF/ZVCF split on the live Pinocchio adapter (#15150 Track Z, Z2).

``club_subtree_load`` (the bare club plus its flexible-shaft links) is checked
against Pinocchio's own recursive Newton-Euler pass on a model whose club
body holds the bare club only: the hand solids welded into the club body in
the specification belong to the hands, so the load the two hands supply
together is the momentum rate minus gravity of head, shaft and grip.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]

SEEDS = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver_seeds"
)
RTOL = 1e-10
FRAME = 400


@pytest.fixture(scope="module")
def pin():
    return pytest.importorskip("pinocchio")


def _adapter(shaft: str):
    from src.engines.physics_engines.pinocchio.python.force_adapter import (
        PinocchioForceAdapter,
    )
    from src.shared.python.motion_matching.execution.spec_builder import (
        with_shaft_model,
    )

    doc = json.loads((SEEDS / "full_body_spec_hipcal_scaled.json").read_bytes())
    return PinocchioForceAdapter(
        json.dumps(with_shaft_model(doc, "driver", shaft)).encode()
    )


@pytest.fixture(scope="module", params=["rigid", "chain"])
def adapter(request, pin):
    return _adapter(request.param)


def _state(adapter, seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(SEEDS / "dynamics_record.npz") as rec:
        q = np.zeros(adapter.nv)
        q[: rec["q"].shape[1]] = rec["q"][FRAME]
    rng = np.random.default_rng(seed)
    return q, rng.normal(size=adapter.nv), rng.normal(size=adapter.nv)


def _bare_club_model(adapter, pin):
    """A model copy whose club body holds head, shaft and grip only."""
    from src.shared.python.biomechanics.club_inertia_partition import (
        club_solids_partition,
    )

    model = adapter._model.copy()
    _, pose = adapter._plant.body_placement(adapter._spec["closure"]["body_b"])
    bare = club_solids_partition(adapter._spec).bare_club
    bare = bare.expressed_in(pose.homogeneous)
    model.inertias[adapter.club_joint_index()] = pin.Inertia(
        bare.mass_kg, np.asarray(bare.com_m), np.asarray(bare.inertia_com_kg_m2)
    )
    return model


def _welded_hand_mass_kg(adapter) -> float:
    from src.shared.python.biomechanics.club_inertia_partition import (
        club_solids_partition,
    )

    hands = club_solids_partition(adapter._spec).welded_hands
    return float(sum(h.mass_kg for h in hands.values()))


def _rnea_subtree_wrench(adapter, pin, q, v, a, gravity: bool):
    """World wrench (about the origin) the club joint passes to its subtree."""
    model = _bare_club_model(adapter, pin)
    if not gravity:
        model.gravity = pin.Motion.Zero()
    data = model.createData()
    pin.rnea(model, data, q, v, a)
    pin.forwardKinematics(model, data, q)  # rnea leaves data.oMi stale
    joint = adapter.club_joint_index()
    world = data.oMi[joint].act(data.f[joint])
    return world.linear.copy(), world.angular.copy()


@pytest.mark.parametrize("gravity", [True, False])
def test_club_subtree_load_matches_rnea(adapter, pin, gravity: bool) -> None:
    q, v, a = _state(adapter, 15150)
    force, moment = adapter.club_subtree_load(q, v, a, gravity=gravity)
    ref_f, ref_m = _rnea_subtree_wrench(adapter, pin, q, v, a, gravity)
    scale = max(1.0, float(np.max(np.abs(np.r_[ref_f, ref_m]))))
    assert np.max(np.abs(force - ref_f)) <= RTOL * scale
    assert np.max(np.abs(moment - ref_m)) <= RTOL * scale
    assert np.linalg.norm(ref_f) > 1e-6


def test_at_rest_the_hands_hold_the_club_weight_including_shaft_links(
    adapter,
) -> None:
    q, _, _ = _state(adapter, 1)
    zero = np.zeros(adapter.nv)
    force, _ = adapter.club_subtree_load(q, zero, zero, gravity=True)
    mass = adapter.club_subtree_mass_kg()
    gravity = np.asarray(adapter._model.gravity.linear)
    assert np.allclose(force, -mass * gravity, rtol=0.0, atol=1e-12 * mass * 10)
    still, _ = adapter.club_subtree_load(q, zero, zero, gravity=False)
    assert np.allclose(still, 0.0, atol=1e-12)


def test_the_welded_hand_solids_are_not_in_the_club_load(adapter, pin) -> None:
    """Review of #15708: the hands' own mass is the hands', not the club's."""
    hands = _welded_hand_mass_kg(adapter)
    assert hands > 0.0
    full = sum(float(adapter._model.inertias[j].mass) for j in adapter._club_subtree())
    assert adapter.club_subtree_mass_kg() == pytest.approx(full - hands, rel=1e-12)


def test_chain_subtree_is_heavier_than_the_rigid_club_body(pin) -> None:
    """The chain moves shaft and head mass onto links the rigid body owned."""
    rigid, chain = _adapter("rigid"), _adapter("chain")
    joint = chain.club_joint_index()
    body = float(chain._model.inertias[joint].mass) - _welded_hand_mass_kg(chain)
    assert chain.club_subtree_mass_kg() > body
    assert np.isclose(
        chain.club_subtree_mass_kg(), rigid.club_subtree_mass_kg(), rtol=1e-9
    )


def test_lead_wrist_point_is_the_club_joint_origin(adapter, pin) -> None:
    q, _, _ = _state(adapter, 2)
    data = adapter._model.createData()
    pin.forwardKinematics(adapter._model, data, q)
    expected = data.oMi[adapter.club_joint_index()].translation
    assert np.allclose(adapter.lead_wrist_point(q), expected, atol=1e-12)


def test_two_hand_split_on_the_live_closure(adapter) -> None:
    """Each hand's parts sum to its total; the hands sum to the club load."""
    from src.shared.python.loop_nullspace.provider import AdapterConstrainedProvider
    from src.shared.python.simulation_backends.hand_force_split import (
        HANDS,
        hand_wrench_split,
    )

    q, v, _ = _state(adapter, 3)
    tau = np.random.default_rng(4).normal(scale=5.0, size=adapter.nv)
    split = hand_wrench_split(
        AdapterConstrainedProvider(adapter),
        adapter.closure_wrench_frame,
        adapter.club_subtree_load,
        adapter.lead_wrist_point,
        q,
        v,
        tau,
    )
    for part in ("ztcf", "zvcf"):
        lead, trail = (split.part(h, part) for h in HANDS)
        total_f = lead.force_n[0] + trail.force_n[0]
        assert np.all(np.isfinite(total_f))
        assert np.linalg.norm(total_f) > 0.0
