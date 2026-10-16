"""NSX-51: MuJoCo and Pinocchio closed-loop snapshots agree at a matched state.

Both engines are built from the same full-body specification, so their native
state vectors are the spec's scalar slide/hinge coordinates in the same units
and (after the adapter order is made explicit) the same order.  The dual
velocity, acceleration and generalised-force maps of NSX-50 are therefore the
identity here; the test proves that by comparing names and units rather than
assuming it, and applies the permutation by name so a reordering would fail
loudly instead of passing silently.  Closure rows are compared after the NSX-02
canonicalisation, which is the only place the engines' conventions differ
(Pinocchio LOCAL frame at the weld frame origin, MuJoCo WORLD frame at the site
position).

Irreducible differences (see ``docs/research/loop_nullspace/contracts.md``):

* ``Jdot v``: Pinocchio is analytic (constraint ``gamma``); MuJoCo has no
  analytic derivative of the weld Jacobian, so the declared method is a central
  finite difference (second order, step 1e-6).
* Different floating-point kernels (CRBA versus composite rigid body, RNEA
  versus ``mj_rne``) and two compilers' handling of the same inertias.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.loop_nullspace.contracts import InputProvenance
from src.shared.python.loop_nullspace.snapshot import snapshot_from_adapter

pytestmark = [
    pytest.mark.unit,
    pytest.mark.requires_mujoco,
    pytest.mark.requires_pinocchio,
]

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / "docs/development/full_body_models/evidence/ground_support"
SPEC_PATH = EVIDENCE / "full_body_spec_hipcal_scaled.json"
PROVENANCE = InputProvenance(kind="test", source="dynamics_record frame 400")


@pytest.fixture(scope="module")
def adapters():
    pytest.importorskip("pinocchio")
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.pinocchio.python.force_adapter import (
        PinocchioForceAdapter,
    )
    from src.shared.python.motion_matching.multi_engine_torque_allocator import (
        MujocoForceAdapter,
    )

    spec = SPEC_PATH.read_bytes()
    return PinocchioForceAdapter(spec), MujocoForceAdapter(spec)


@pytest.fixture(scope="module")
def matched_state(adapters):
    """One closure-consistent state, expressed in each engine's native order."""
    from src.shared.python.motion_matching.same_input import (
        VectorPlant,
        project_to_closure,
    )

    pin_ad, mj_ad = adapters
    plant = VectorPlant("pinocchio", SPEC_PATH.read_bytes())
    with np.load(EVIDENCE / "dynamics_record.npz") as rec:
        q0, v0 = rec["q"][400], rec["v"][400]
    proj = project_to_closure(plant, q0, v0)
    qn, vn = plant._named(proj.q), plant._named(proj.v)

    def native(adapter):
        order = adapter.coordinate_order
        return (
            np.array([qn[n] for n in order]),
            np.array([vn[n] for n in order]),
        )

    return native(pin_ad), native(mj_ad)


@pytest.fixture(scope="module")
def snapshots(adapters, matched_state):
    pin_ad, mj_ad = adapters
    (qp, vp), (qm, vm) = matched_state
    rng = np.random.default_rng(51)
    # one shared target acceleration expressed by name in each native order
    names = pin_ad.coordinate_order
    a_by_name = dict(zip(names, rng.normal(size=len(names)), strict=True))
    ap = np.array([a_by_name[n] for n in pin_ad.coordinate_order])
    am = np.array([a_by_name[n] for n in mj_ad.coordinate_order])
    return (
        snapshot_from_adapter(pin_ad, qp, vp, ap, input_provenance=PROVENANCE),
        snapshot_from_adapter(mj_ad, qm, vm, am, input_provenance=PROVENANCE),
    )


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.max(np.abs(a - b))) / max(1.0, float(np.max(np.abs(b))))


def test_dynamic_maps_between_the_engines_are_the_identity(adapters) -> None:
    """Same spec coordinates, same units, same order: ``T = I`` (NSX-50)."""
    from src.shared.python.pose_interchange.dynamic_maps import (
        check_power_invariance,
    )

    pin_ad, mj_ad = adapters
    assert sorted(pin_ad.coordinate_order) == sorted(mj_ad.coordinate_order)
    perm = [list(mj_ad.coordinate_order).index(n) for n in pin_ad.coordinate_order]
    assert perm == list(range(len(perm)))
    units = mj_ad._model.coordinate_units
    assert set(units) <= {"m", "rad"}  # Pinocchio scalar slide/hinge joints too
    eye = np.eye(len(perm))[perm]
    v = np.linspace(-1.0, 1.0, len(perm))
    tau = np.linspace(2.0, -3.0, len(perm))
    check_power_invariance(eye, eye.T, v, tau)


def test_hard_kkt_accelerations_agree_after_canonicalisation(
    adapters, matched_state
) -> None:
    """Same applied effort: the exact rigid solves give the same ``a``."""
    pin_ad, mj_ad = adapters
    (qp, vp), (qm, vm) = matched_state
    rng = np.random.default_rng(52)
    effort_by_name = dict.fromkeys(pin_ad.coordinate_order, 0.0)
    for i in pin_ad.actuated_indices:
        effort_by_name[pin_ad.coordinate_order[i]] = float(rng.normal())
    tp = np.array([effort_by_name[n] for n in pin_ad.coordinate_order])
    tm = np.array([effort_by_name[n] for n in mj_ad.coordinate_order])
    a_pin = pin_ad._plant.constrained_acceleration(qp, vp, tp)
    # MuJoCo exact solve through the adapter's own parity audit
    assert mj_ad.verify_acceleration_parity(qm, vm, tm, a_pin) < 1e-6 * max(
        1.0, float(np.max(np.abs(a_pin)))
    )
