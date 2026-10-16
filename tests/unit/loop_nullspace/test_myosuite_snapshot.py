"""NSX-52: MyoSuite closed-loop snapshot reusing the MuJoCo path (#12580).

Gating, provenance and DRY tests run without MyoSuite by substituting the
plain MuJoCo model for the MyoSuite-held one (the adapter logic under test is
identical).  Tests that need the real MyoSuite runtime skip when it is absent.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.myosuite.python.force_adapter import (
    MyoSuiteForceAdapter,
)
from src.shared.python.engine_core.capabilities import Capability, CapabilityLevel
from src.shared.python.loop_nullspace.contracts import InputProvenance
from src.shared.python.loop_nullspace.snapshot import (
    has_closed_loop_protocol,
    snapshot_from_adapter,
)
from src.shared.python.motion_matching.multi_engine_torque_allocator import (
    MujocoForceAdapter,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / "docs/development/full_body_models/evidence/ground_support"
SPEC_PATH = EVIDENCE / "full_body_spec_hipcal_scaled.json"
PROVENANCE = InputProvenance(kind="test", source="dynamics_record frame 400")


class _PlainMujocoBacked(MyoSuiteForceAdapter):
    """Same adapter logic with the plain MuJoCo model (no MyoSuite needed)."""

    def _make_model(self, spec_bytes: bytes):
        from src.engines.physics_engines.mujoco.python.full_body_model import (
            NativeMujocoFullBodyModel,
        )

        model = NativeMujocoFullBodyModel(spec_bytes)
        model.kkt_regularization = 0.0
        return model


@pytest.fixture(scope="module")
def adapter():
    pytest.importorskip("mujoco")
    return _PlainMujocoBacked(SPEC_PATH.read_bytes())


@pytest.fixture(scope="module")
def mujoco_adapter():
    pytest.importorskip("mujoco")
    return MujocoForceAdapter(SPEC_PATH.read_bytes())


@pytest.fixture(scope="module")
def state(mujoco_adapter):
    from src.shared.python.motion_matching.same_input import (
        VectorPlant,
        project_to_closure,
    )

    plant = VectorPlant("mujoco", SPEC_PATH.read_bytes())
    with np.load(EVIDENCE / "dynamics_record.npz") as rec:
        q0, v0 = rec["q"][400], rec["v"][400]
    proj = project_to_closure(plant, q0, v0)
    qn, vn = plant._named(proj.q), plant._named(proj.v)
    order = mujoco_adapter.coordinate_order
    return np.array([qn[n] for n in order]), np.array([vn[n] for n in order])


def test_reuses_the_mujoco_adapter_without_a_second_implementation() -> None:
    assert issubclass(MyoSuiteForceAdapter, MujocoForceAdapter)
    for name in (
        "compute_closure_drift",
        "compute_grip_jacobian",
        "compute_mass_and_bias",
        "verify_acceleration_parity",
        "closure_topology",
        "club_inertia_split",
    ):
        assert name not in vars(MyoSuiteForceAdapter), name


def test_capability_is_full_only_when_regularisation_is_zero(adapter) -> None:
    assert adapter.kkt_regularization == 0.0
    assert adapter.capability(Capability.CLOSED_LOOP_SNAPSHOT) is CapabilityLevel.FULL
    assert has_closed_loop_protocol(adapter)
    try:
        for value in (1e-12, 1e-6):
            adapter.kkt_regularization = value
            level = adapter.capability(Capability.CLOSED_LOOP_SNAPSHOT)
            assert level is CapabilityLevel.NONE, value
    finally:
        adapter.kkt_regularization = 0.0


def test_regularised_adapter_refuses_a_snapshot(adapter, state) -> None:
    from src.shared.python.engine_core.capabilities import CapabilityUnavailable

    q, v = state
    adapter.kkt_regularization = 1e-6
    try:
        with pytest.raises(CapabilityUnavailable, match="closed_loop_snapshot"):
            snapshot_from_adapter(adapter, q, v, v, input_provenance=PROVENANCE)
    finally:
        adapter.kkt_regularization = 0.0


def test_regularisation_must_be_a_finite_nonnegative_number(adapter) -> None:
    for bad in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="kkt_regularization"):
            adapter.kkt_regularization = bad
    assert adapter.kkt_regularization == 0.0


def test_provenance_records_regularisation_and_drift_method(adapter, state) -> None:
    q, v = state
    snap = snapshot_from_adapter(adapter, q, v, v, input_provenance=PROVENANCE)
    version = snap.model_provenance.version
    assert "kkt_regularization=0" in version
    assert "drift=central_finite_difference" in version
    assert snap.model_provenance.name == "myosuite"


def test_other_capabilities_stay_none(adapter) -> None:
    kept = {Capability.CLOSED_LOOP_SNAPSHOT, Capability.CLOSED_LOOP_DERIVATIVES}
    others = [c for c in Capability if c not in kept]
    assert all(adapter.capability(c) is CapabilityLevel.NONE for c in others)


def test_snapshot_equals_mujoco_on_the_shared_model(
    adapter, mujoco_adapter, state
) -> None:
    q, v = state
    mass, bias = mujoco_adapter.compute_mass_and_bias(q, v)
    a = np.linalg.solve(mass, -bias)  # any finite vector; equality is the point
    mine = snapshot_from_adapter(adapter, q, v, a, input_provenance=PROVENANCE)
    ref = snapshot_from_adapter(mujoco_adapter, q, v, a, input_provenance=PROVENANCE)
    for field in ("M", "h", "J_c", "Jdot_v", "S"):
        np.testing.assert_array_equal(getattr(mine, field), getattr(ref, field))
    assert mine.model_provenance.content_hash == ref.model_provenance.content_hash


def test_factory_builds_myosuite_adapter_for_the_myosuite_engine(
    monkeypatch,
) -> None:
    from src.shared.python.motion_matching import multi_engine_torque_allocator as mod

    EngineType = mod.EngineType

    monkeypatch.setattr(
        MyoSuiteForceAdapter, "_make_model", _PlainMujocoBacked._make_model
    )
    built = mod.create_engine_force_adapter(
        EngineType.MYOSUITE, spec_path=SPEC_PATH, allow_synthetic=False
    )
    assert isinstance(built, MyoSuiteForceAdapter)


def test_real_myosuite_runtime_snapshot_equals_mujoco(mujoco_adapter, state) -> None:
    """Needs the MyoSuite package; skipped where it is not installed."""
    pytest.importorskip("myosuite")
    real = MyoSuiteForceAdapter(SPEC_PATH.read_bytes())
    assert real.kkt_regularization == 0.0
    assert real.capability(Capability.CLOSED_LOOP_SNAPSHOT) is CapabilityLevel.FULL
    q, v = state
    mass, bias = mujoco_adapter.compute_mass_and_bias(q, v)
    a = np.linalg.solve(mass, -bias)
    mine = snapshot_from_adapter(real, q, v, a, input_provenance=PROVENANCE)
    ref = snapshot_from_adapter(mujoco_adapter, q, v, a, input_provenance=PROVENANCE)
    np.testing.assert_allclose(mine.M, ref.M, rtol=0, atol=1e-10)
    np.testing.assert_allclose(mine.J_c, ref.J_c, rtol=0, atol=1e-10)
    np.testing.assert_allclose(mine.Jdot_v, ref.Jdot_v, rtol=0, atol=1e-6)
