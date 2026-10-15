"""NSX-53: native Drake force adapter and closed-loop snapshot (#12581).

Every expectation is built from raw ``MultibodyPlant`` calls on a second
context, never through the adapter methods under test.  Cross-engine equality
with Pinocchio lives in ``tests/integration/cross_engine``.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.biomechanics.club_inertia_partition import (
    club_solids_partition,
)
from src.shared.python.engine_core.capabilities import (
    Capability,
    CapabilityLevel,
    CapabilityUnavailable,
)
from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    ClosureKind,
    InputProvenance,
)
from src.shared.python.loop_nullspace.snapshot import snapshot_from_adapter
from src.shared.python.motion_matching.multi_engine_torque_allocator import (
    BaseEngineForceAdapter,
    ClosedLoopSnapshotAdapter,
    EngineType,
    create_engine_force_adapter,
)
from src.shared.python.motion_matching.polynomial_actuation import ROOT_COORDINATES

pytestmark = [pytest.mark.unit, pytest.mark.requires_drake]

EVIDENCE = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/evidence/ground_support"
)
SPEC_PATH = EVIDENCE / "full_body_spec_hipcal_scaled.json"
PROVENANCE = InputProvenance(kind="test", source="dynamics_record frame 400")
ATOL = 1e-9


@pytest.fixture(scope="module")
def adapter():
    pytest.importorskip("pydrake")
    from src.engines.physics_engines.drake.python.force_adapter import (
        DrakeForceAdapter,
    )

    return DrakeForceAdapter(SPEC_PATH.read_bytes())


@pytest.fixture(scope="module")
def state(adapter):
    """A closure-consistent ``(q, v)`` in the adapter's coordinate order."""
    from src.shared.python.motion_matching.same_input import (
        VectorPlant,
        project_to_closure,
    )

    plant = VectorPlant("drake", SPEC_PATH.read_bytes())
    with np.load(EVIDENCE / "dynamics_record.npz") as rec:
        q0, v0 = rec["q"][400], rec["v"][400]
    proj = project_to_closure(plant, q0, v0)
    named_q, named_v = plant._named(proj.q), plant._named(proj.v)
    order = adapter.coordinate_order
    return (
        np.array([named_q[n] for n in order]),
        np.array([named_v[n] for n in order]),
    )


@pytest.fixture(scope="module")
def reference(adapter, state):
    """Raw Drake evaluation on a separate context (plant order -> adapter)."""
    model = adapter._model
    plant = model.plant
    ctx = plant.CreateDefaultContext()
    q, v = state
    plant.SetPositions(ctx, model.to_plant_positions(q))
    plant.SetVelocities(ctx, model.to_plant_velocities(v))
    order = np.array(model.plant_position_indices)  # plant slot per coordinate
    mass = np.asarray(plant.CalcMassMatrix(ctx))[np.ix_(order, order)]
    bias = (
        np.asarray(plant.CalcBiasTerm(ctx))
        - np.asarray(plant.CalcGravityGeneralizedForces(ctx))
    )[order]
    return {"M": mass, "h": bias}


def test_is_a_base_and_closed_loop_adapter(adapter) -> None:
    assert isinstance(adapter, BaseEngineForceAdapter)
    assert isinstance(adapter, ClosedLoopSnapshotAdapter)
    assert adapter.engine_type is EngineType.DRAKE
    assert adapter.capability(Capability.CLOSED_LOOP_SNAPSHOT) is CapabilityLevel.FULL


def test_coordinate_inventory(adapter) -> None:
    spec = json.loads(SPEC_PATH.read_bytes())
    assert adapter.coordinate_order == tuple(spec["coordinate_order"])
    assert adapter.nv == len(spec["coordinate_order"])
    assert set(ROOT_COORDINATES) <= set(adapter.coordinate_order)
    names = adapter.coordinate_order
    assert [names[i] for i in adapter.actuated_indices] == [
        n for n in names if n not in ROOT_COORDINATES
    ]
    assert len(adapter.contact_names) == adapter.n_contact_spheres > 0
    assert len(adapter.model_hash) == 64


def test_mass_and_bias_match_raw_plant_calls(adapter, state, reference) -> None:
    mass, bias = adapter.compute_mass_and_bias(*state)
    np.testing.assert_allclose(mass, reference["M"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(bias, reference["h"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(mass, mass.T, rtol=0, atol=1e-10)


def test_inverse_dynamics_is_mass_times_a_plus_bias(adapter, state) -> None:
    q, v = state
    a = np.random.default_rng(1).normal(size=adapter.nv)
    mass, bias = adapter.compute_mass_and_bias(q, v)
    np.testing.assert_allclose(
        adapter.compute_inverse_dynamics(q, v, a), mass @ a + bias, atol=1e-9
    )


def test_gravity_acts_on_the_root_height(adapter, state) -> None:
    """Static bias (v = 0) carries the weight on the vertical root force."""
    q, _ = state
    _, bias = adapter.compute_mass_and_bias(q, np.zeros(adapter.nv))
    assert abs(bias[adapter.coordinate_order.index("TranslationInputZ")]) > 400.0


def test_contact_jacobian_shape_and_rigid_root_translation(adapter, state) -> None:
    """The root translation block is one rotation, identical for every sphere."""
    q, _ = state
    jac = adapter.compute_contact_jacobian(q)
    assert jac.shape == (3 * adapter.n_contact_spheres, adapter.nv)
    cols = [
        adapter.coordinate_order.index(n)
        for n in ("TranslationInputX", "TranslationInputY", "TranslationInputZ")
    ]
    first = jac[:3][:, cols]
    np.testing.assert_allclose(first.T @ first, np.eye(3), atol=1e-12)
    for k in range(adapter.n_contact_spheres):
        np.testing.assert_allclose(jac[3 * k : 3 * k + 3][:, cols], first, atol=1e-12)


def test_grip_jacobian_is_zero_off_the_loop(adapter, state) -> None:
    q, _ = state
    jac = adapter.compute_grip_jacobian(q)
    assert jac.shape == (6, adapter.nv)
    support = set(adapter.closure_topology().loop_support_indices)
    assert support
    nonzero = {i for i in range(adapter.nv) if np.abs(jac[:, i]).max() > 0}
    assert nonzero <= support


def test_drift_matches_finite_difference_of_the_jacobian(adapter, state) -> None:
    q, _ = state
    v = np.random.default_rng(3).normal(scale=0.3, size=adapter.nv)
    h = 1e-6
    fd = (
        (
            adapter.compute_grip_jacobian(q + h * v)
            - adapter.compute_grip_jacobian(q - h * v)
        )
        / (2 * h)
    ) @ v
    drift = adapter.compute_closure_drift(q, v)
    assert drift.shape == (6,)
    assert np.max(np.abs(drift - fd)) / max(1.0, np.max(np.abs(fd))) < 1e-6


def test_snapshot_equals_raw_plant_reference(adapter, state, reference) -> None:
    q, v = state
    a = np.zeros(adapter.nv)
    snap = snapshot_from_adapter(adapter, q, v, a, input_provenance=PROVENANCE)
    assert isinstance(snap, ClosedLoopSnapshot)
    assert snap.topology.closure_kind is ClosureKind.WELD6
    assert snap.topology.coordinate_order == adapter.coordinate_order
    np.testing.assert_allclose(snap.M, reference["M"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(snap.h, reference["h"], rtol=0, atol=1e-10)
    assert snap.model_provenance.content_hash == adapter.model_hash
    assert snap.model_provenance.name == "drake"
    assert snap.J_g is None


def test_closed_loop_holds_at_the_projected_state(adapter, state) -> None:
    q, v = state
    snap = snapshot_from_adapter(
        adapter, q, v, np.zeros(adapter.nv), input_provenance=PROVENANCE
    )
    assert np.max(np.abs(snap.J_c @ snap.v)) < 1e-9 * max(1.0, np.max(np.abs(v)))


def test_wrench_frame_and_convention(adapter, state) -> None:
    from src.engines.physics_engines.drake.python.full_body_model import (
        CLOSURE_CONVENTION,
    )

    q, _ = state
    rot, origin = adapter.closure_wrench_frame(q)
    assert rot.shape == (3, 3) and origin.shape == (3,)
    assert adapter.native_closure_convention == CLOSURE_CONVENTION
    assert adapter.native_closure_convention.row_order == "ang_lin"
    assert adapter.native_closure_convention.relative_sign == -1


def test_club_inertia_split_is_bare_and_welded(adapter) -> None:
    bare, hands = adapter.club_inertia_split()
    expected = club_solids_partition(json.loads(SPEC_PATH.read_bytes())).bare_club
    assert bare.mass == pytest.approx(expected.mass_kg, abs=1e-12)
    assert len(hands) >= 1 and all(h.mass > 0.0 for h in hands)


def test_parity_detects_wrong_target_and_accepts_own_acceleration(
    adapter, state
) -> None:
    q, v = state
    tau = np.zeros(adapter.nv)
    tau[list(adapter.actuated_indices)] = np.random.default_rng(2).normal(
        size=len(adapter.actuated_indices)
    )
    model = adapter._model
    names = adapter.coordinate_order
    accel = model.accelerations(
        dict(zip(names, q, strict=True)),
        dict(zip(names, v, strict=True)),
        dict(zip(names, tau, strict=True)),
    )
    # The model's own solve includes ground contact; with the feet off the
    # ground this is the pure loop solve.  Perturbing it must be detected.
    target = np.array([accel[n] for n in names])
    err_wrong = adapter.verify_acceleration_parity(q, v, tau, target + 0.5)
    assert err_wrong > 0.1


def test_state_shape_is_validated(adapter, state) -> None:
    q, v = state
    with pytest.raises(ValueError):
        adapter.compute_mass_and_bias(q[:-1], v)
    with pytest.raises(ValueError):
        adapter.compute_grip_jacobian(np.full(adapter.nv, np.nan))


def test_factory_returns_native_adapter_and_no_drake_quarantine() -> None:
    pytest.importorskip("pydrake")
    from src.engines.physics_engines.drake.python.force_adapter import (
        DrakeForceAdapter,
    )

    native = create_engine_force_adapter(EngineType.DRAKE, spec_path=SPEC_PATH)
    assert isinstance(native, DrakeForceAdapter)
    # production route no longer raises the "not yet registered" quarantine error
    default = create_engine_force_adapter(EngineType.DRAKE)
    assert isinstance(default, DrakeForceAdapter)


def test_missing_pydrake_fails_closed_without_polluting_sys_modules() -> None:
    """A mock in ``sys.modules`` is refused (patch.dict cleans up afterwards)."""
    from unittest.mock import MagicMock, patch

    pytest.importorskip("pydrake")
    from src.engines.physics_engines.drake.python.force_adapter import (
        DrakeForceAdapter,
    )

    with patch.dict("sys.modules", {"pydrake.all": MagicMock()}):
        with pytest.raises(ImportError):
            DrakeForceAdapter(SPEC_PATH.read_bytes())


def test_synthetic_drake_declares_none_and_refuses() -> None:
    synth = create_engine_force_adapter("drake", allow_synthetic=True, nv=44)
    assert synth.capability(Capability.CLOSED_LOOP_SNAPSHOT) is CapabilityLevel.NONE
    q = np.zeros(synth.nv)
    with pytest.raises(CapabilityUnavailable, match="closed_loop_snapshot"):
        snapshot_from_adapter(synth, q, q, q, input_provenance=PROVENANCE)
