"""Unit tests for PF-09: Model-conformant native force adapters and bridge quarantine (#10439).

Verifies:
1. Quarantine enforcement: create_engine_force_adapter fails closed with RuntimeError
   for unbridged native engines (Drake, OpenSim, Simscape) unless allow_synthetic=True.
2. Protocol conformance: BaseEngineForceAdapter protocol requirements including
   model_hash, coordinate_order, contact_names, and compute_mass_and_bias.
3. MuJoCo adapter conformance: Native dynamic equations, non-passive raw inverse dynamics,
   and mass/bias extraction.
4. Pinocchio adapter interface conformance: Protocol adherence and method contracts.
5. CLI synthetic gate: allocate_swing_torques CLI rejects synthetic fixtures unless flagged.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from scripts.allocate_swing_torques import main as cli_main
from src.engines.physics_engines.pinocchio.python.force_adapter import (
    PinocchioForceAdapter,
)
from src.shared.python.motion_matching.multi_engine_torque_allocator import (
    BaseEngineForceAdapter,
    DrakeForceAdapter,
    EngineType,
    MujocoForceAdapter,
    OpenSimForceAdapter,
    SimscapeForceAdapter,
    SyntheticMultibodyFixture,
    create_engine_force_adapter,
)

pytestmark = pytest.mark.unit


def test_quarantine_fails_closed_without_allow_synthetic() -> None:
    """Unregistered/unbridged native engines must raise RuntimeError when allow_synthetic=False."""
    with pytest.raises(
        RuntimeError, match="Drake MultibodyPlant force adapter is not yet registered"
    ):
        create_engine_force_adapter(EngineType.DRAKE, allow_synthetic=False)

    with pytest.raises(
        RuntimeError, match="OpenSim Simbody force adapter is not yet registered"
    ):
        create_engine_force_adapter(EngineType.OPENSIM, allow_synthetic=False)

    with pytest.raises(
        RuntimeError, match="Simscape force adapter is not yet registered"
    ):
        create_engine_force_adapter(EngineType.SIMSCAPE, allow_synthetic=False)


def test_quarantine_permits_synthetic_when_explicitly_flagged() -> None:
    """When allow_synthetic=True, synthetic fixtures instantiate cleanly."""
    drake = create_engine_force_adapter(EngineType.DRAKE, allow_synthetic=True)
    assert isinstance(drake, DrakeForceAdapter)
    assert isinstance(drake, SyntheticMultibodyFixture)
    assert isinstance(drake, BaseEngineForceAdapter)

    opensim = create_engine_force_adapter(EngineType.OPENSIM, allow_synthetic=True)
    assert isinstance(opensim, OpenSimForceAdapter)
    assert isinstance(opensim, SyntheticMultibodyFixture)
    assert isinstance(opensim, BaseEngineForceAdapter)

    simscape = create_engine_force_adapter(EngineType.SIMSCAPE, allow_synthetic=True)
    assert isinstance(simscape, SimscapeForceAdapter)
    assert isinstance(simscape, SyntheticMultibodyFixture)
    assert isinstance(simscape, BaseEngineForceAdapter)


def test_synthetic_fixture_protocol_properties() -> None:
    """SyntheticMultibodyFixture satisfies BaseEngineForceAdapter protocol properties."""
    fixture = SyntheticMultibodyFixture(nv=44, n_spheres=6)
    assert isinstance(fixture, BaseEngineForceAdapter)
    assert fixture.model_hash == "synthetic-fixture-v1"
    assert len(fixture.coordinate_order) == 44
    assert len(fixture.contact_names) == 6

    q = np.zeros(44)
    v = np.zeros(44)
    mass, bias = fixture.compute_mass_and_bias(q, v)
    assert mass.shape == (44, 44)
    assert bias.shape == (44,)
    assert bias[2] > 500.0  # Gravity bias


def test_mujoco_adapter_protocol_conformance() -> None:
    """MujocoForceAdapter satisfies the extended BaseEngineForceAdapter protocol."""
    repo_root = Path(__file__).resolve().parents[3]
    spec_path = (
        repo_root
        / "docs"
        / "development"
        / "full_body_models"
        / "full_body_spec_v1.json"
    )
    if not spec_path.is_file():
        pytest.skip("full_body_spec_v1.json not found")

    adapter = create_engine_force_adapter(EngineType.MUJOCO, spec_path=spec_path)
    assert isinstance(adapter, BaseEngineForceAdapter)
    assert adapter.engine_type == EngineType.MUJOCO
    assert adapter.nv == 41
    assert len(adapter.coordinate_order) == 41
    assert len(adapter.contact_names) == adapter.n_contact_spheres
    assert isinstance(adapter.model_hash, str)

    q = np.zeros(41)
    q[2] = 0.85
    v = np.zeros(41)
    mass, bias = adapter.compute_mass_and_bias(q, v)
    assert mass.shape == (41, 41)
    assert bias.shape == (41,)
    # At zero acceleration, unconstrained generalized force must equal bias
    a_zero = np.zeros(41)
    tau_zero = adapter.compute_inverse_dynamics(q, v, a_zero)
    np.testing.assert_allclose(tau_zero, bias, atol=1e-6)


def test_pinocchio_adapter_protocol_structure() -> None:
    """PinocchioForceAdapter protocol conformance with mocked native backend."""
    mock_plant = MagicMock()
    mock_plant.model_sha256 = "test-pinocchio-sha256"
    mock_plant._velocity_indices = {"pelvis_tx": 0, "pelvis_ty": 1, "pelvis_tz": 2}
    mock_plant.actuated_velocity_indices = []
    mock_contact = MagicMock()
    mock_contact.name = "heel_l"
    mock_plant.contact_spheres = [mock_contact]
    mock_plant.contact_frame_ids = [10]
    mock_plant.model.nq = 3
    mock_plant.model.nv = 3
    mock_plant.data = MagicMock()

    mock_pin = MagicMock()
    mock_pin.crba.return_value = np.eye(3)
    mock_pin.nonLinearEffects.return_value = np.array([0.0, 0.0, 9.81])
    mock_pin.rnea.return_value = np.array([1.0, 2.0, 3.0])

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(
            "src.engines.physics_engines.pinocchio.python.force_adapter.pin",
            mock_pin,
            raising=False,
        )
        adapter = PinocchioForceAdapter.__new__(PinocchioForceAdapter)
        adapter._plant = mock_plant
        adapter._model = mock_plant.model
        adapter._data = mock_plant.data
        adapter._pin = mock_pin
        adapter._names = ("pelvis_tx", "pelvis_ty", "pelvis_tz")
        adapter._actuated = ()
        adapter._contact_names = ("heel_l",)
        adapter._contact_ids = (10,)

        assert isinstance(adapter, BaseEngineForceAdapter)
        assert adapter.engine_type == EngineType.PINOCCHIO
        assert adapter.nv == 3
        assert adapter.coordinate_order == ("pelvis_tx", "pelvis_ty", "pelvis_tz")
        assert adapter.contact_names == ("heel_l",)
        assert adapter.model_hash == "test-pinocchio-sha256"

        q = np.zeros(3)
        v = np.zeros(3)
        mass, bias = adapter.compute_mass_and_bias(q, v)
        assert mass.shape == (3, 3)
        assert bias.shape == (3,)


def test_cli_rejects_synthetic_without_flag(tmp_path: Path) -> None:
    """CLI fails with error if synthetic engine requested without --allow-synthetic."""
    candidate_file = tmp_path / "candidate_test.npz"
    np.savez_compressed(
        candidate_file,
        time_s=np.array([0.0, 0.1]),
        q=np.zeros((2, 44)),
        v=np.zeros((2, 44)),
        a=np.zeros((2, 44)),
    )
    out_file = tmp_path / "out.npz"

    with pytest.raises(RuntimeError, match="synthetic fixture is quarantined"):
        cli_main(
            [
                "--engine",
                "drake",
                "--candidate",
                str(candidate_file),
                "--out",
                str(out_file),
            ]
        )
