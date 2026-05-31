"""Tests for the MyoSuite canonical-core adapter slice (#6803)."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.myosuite.python.canonical_adapter import (
    CANONICAL_CONVENTION,
    MyoSuiteCanonicalAdapter,
    MyoSuiteMuscleOutputs,
    NativeMyoSuiteState,
)
from src.engines.physics_engines.myosuite.python.myosuite_physics_engine import (
    MyoSuitePhysicsEngine,
)
from src.shared.python.engine_core.capabilities import CapabilityLevel

pytestmark = pytest.mark.unit


def _muscle_outputs(rows: int = 2) -> MyoSuiteMuscleOutputs:
    activations = np.tile(np.array([[0.2, 0.8]], dtype=float), (rows, 1))
    forces = np.tile(np.array([[10.0, 20.0]], dtype=float), (rows, 1))
    lengths = np.tile(np.array([[0.10, 0.12]], dtype=float), (rows, 1))
    velocities = np.tile(np.array([[-0.01, 0.02]], dtype=float), (rows, 1))
    return MyoSuiteMuscleOutputs(
        muscle_names=("biceps", "triceps"),
        activations=activations,
        forces=forces,
        lengths=lengths,
        velocities=velocities,
    )


def test_myosuite_declares_activation_driven_capabilities() -> None:
    engine = MyoSuitePhysicsEngine()

    caps = engine.get_capabilities()

    assert caps.muscles == CapabilityLevel.FULL
    assert caps.forward_sim == CapabilityLevel.FULL
    assert caps.contact_forces == CapabilityLevel.PARTIAL
    assert caps.inverse_dynamics == CapabilityLevel.NONE
    assert caps.to_dict()["muscles"] == "full"
    assert caps.extra["supported_capabilities"] == (
        "MUSCLES",
        "FORWARD_DYN",
        "CONTACT",
    )
    assert caps.extra["unsupported_capabilities"] == ("JOINT_TORQUE_INVERSE_DYN",)


def test_myosuite_canonical_state_round_trips_and_normalizes_quaternion() -> None:
    adapter = MyoSuiteCanonicalAdapter()
    native = NativeMyoSuiteState(
        qpos=np.array([1.0, 2.0, 3.0, 2.0, 0.0, 0.0, 0.0, 0.25]),
        qvel=np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.4]),
        qacc=np.array([0.5, 0.6, 0.7, 0.04, 0.05, 0.06, 0.8]),
        time=0.125,
    )

    canonical = adapter.to_canonical_state(native)
    restored = adapter.from_canonical_state(canonical)

    assert canonical.convention == CANONICAL_CONVENTION
    np.testing.assert_allclose(canonical.q[3:7], [1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(restored.qpos, canonical.q)
    np.testing.assert_allclose(restored.qvel, native.qvel)
    np.testing.assert_allclose(restored.qacc, native.qacc)
    assert restored.time == pytest.approx(native.time)


def test_myosuite_native_state_rejects_bad_qpos_qvel_shape() -> None:
    with pytest.raises(ValueError, match="qpos/qvel sizes"):
        NativeMyoSuiteState(qpos=np.zeros(5), qvel=np.zeros(2))


def test_activation_advance_routes_through_upstream_muscle(monkeypatch) -> None:
    adapter = MyoSuiteCanonicalAdapter()
    called = {}

    def fake_activation_step_batch(u, a, dt, **_kwargs):
        called["args"] = (u.copy(), a.copy(), dt)
        return np.full_like(a, 0.42)

    monkeypatch.setattr(
        "src.engines.physics_engines.myosuite.python.canonical_adapter."
        "rust_muscle.activation_step_batch",
        fake_activation_step_batch,
    )

    result = adapter.advance_activations([1.0, 0.0], [0.1, 0.9], 0.005)

    np.testing.assert_allclose(result, [0.42, 0.42])
    np.testing.assert_allclose(called["args"][0], [1.0, 0.0])
    np.testing.assert_allclose(called["args"][1], [0.1, 0.9])
    assert called["args"][2] == pytest.approx(0.005)


def test_build_trace_carries_muscle_outputs() -> None:
    adapter = MyoSuiteCanonicalAdapter()
    outputs = _muscle_outputs(rows=2)

    trace = adapter.build_trace(
        t=np.array([0.0, 0.01]),
        q=np.zeros((2, 8)),
        v=np.zeros((2, 7)),
        muscle_outputs=outputs,
        dt=0.01,
    )

    assert trace.backend == "myosuite"
    assert trace.meta["activation_source"] == "upstream-muscle"
    assert trace.muscle_names == ("biceps", "triceps")
    np.testing.assert_allclose(trace.muscle_activations, outputs.activations)
    np.testing.assert_allclose(trace.muscle_forces, outputs.forces)
    np.testing.assert_allclose(trace.muscle_lengths, outputs.lengths)
    np.testing.assert_allclose(trace.muscle_velocities, outputs.velocities)


def test_muscle_outputs_validate_name_columns() -> None:
    with pytest.raises(ValueError, match="muscle_names"):
        MyoSuiteMuscleOutputs(
            muscle_names=("only_one",),
            activations=np.zeros((2, 2)),
            forces=np.zeros((2, 2)),
            lengths=np.zeros((2, 2)),
            velocities=np.zeros((2, 2)),
        )
