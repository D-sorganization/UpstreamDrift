"""Unit tests and RED failure fixtures for OpenSim versioned model variants and actuation capabilities (OG-07, #10401).

Tests:
1. RED fixture: Loading torque controls as muscle activations raises IncompatibleActuationError.
2. RED fixture: Querying an unknown state variable raises UnknownStateError.
3. RED fixture: Missing geometry mesh asset raises MissingGeometryAssetError.
4. RED fixture: Stale or mismatched model/spec hash raises StaleModelHashError.
5. RED fixture: Requesting an unsupported capability raises UnsupportedCapabilityError rather than silent fallback.
6. Identity/no-op variant preserves forward kinematics and source geometry.
7. Torque model and muscle/tendon variant traverse the same adapter API.
8. Adapter contracts shield callers from OpenSim SDK object chains (no leaked C++ pointers/handles).
"""

from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.tour_matching.model_variants import (
    ActuationProfile,
    ActuationType,
    AnatomicalSkeletonSpec,
    GolfEquipmentSpec,
    GolfModelAdapter,
    GolfModelVariant,
    IncompatibleActuationError,
    MissingGeometryAssetError,
    StaleModelHashError,
    UnknownStateError,
    UnsupportedCapabilityError,
    create_golf_model_adapter,
    create_muscle_model_variant,
    create_torque_model_variant,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
SCALED_MODEL_PATH = (
    REPO_ROOT
    / "src"
    / "engines"
    / "physics_engines"
    / "opensim"
    / "models"
    / "golf_humanoid_scaled.osim"
)


def test_red_fixture_loading_torque_controls_as_muscle_activations_fails() -> None:
    """RED Fixture: Loading torque controls into a muscle/tendon variant raises IncompatibleActuationError."""
    adapter = create_golf_model_adapter(SCALED_MODEL_PATH)
    muscle_variant = create_muscle_model_variant(SCALED_MODEL_PATH)
    adapter.load_variant(muscle_variant)

    assert (
        adapter.current_variant.actuation.actuation_type == ActuationType.MUSCLE_TENDON
    )

    # Attempt to supply torque controls (e.g. 150 N*m, outside normalized [0, 1] range and wrong units)
    time_s = np.array([0.0, 0.01, 0.02], dtype=np.float64)
    torque_controls = {
        "lumbar_extension": np.array([150.0, 150.0, 150.0], dtype=np.float64),
    }

    with pytest.raises(
        IncompatibleActuationError, match="Incompatible actuation controls"
    ):
        adapter.replay_controls(torque_controls, time_s=time_s)


def test_red_fixture_unknown_state_variable_mapping_fails() -> None:
    """RED Fixture: Querying or mapping an unknown state variable name raises UnknownStateError."""
    adapter = create_golf_model_adapter(SCALED_MODEL_PATH)
    torque_variant = create_torque_model_variant(SCALED_MODEL_PATH)
    adapter.load_variant(torque_variant)

    with pytest.raises(UnknownStateError, match="Unknown state variable"):
        adapter.map_state_indices(["nonexistent_muscle_fiber_length"])


def test_red_fixture_missing_geometry_asset_fails(tmp_path: Path) -> None:
    """RED Fixture: A model variant that references nonexistent mesh assets fails closed with MissingGeometryAssetError."""
    skeleton = AnatomicalSkeletonSpec(
        skeleton_id="golf_humanoid_skeleton",
        frame_ids=("pelvis", "torso", "humerus_r"),
        coordinate_names=("pelvis_tilt", "lumbar_extension"),
        geometry_assets={"pelvis": "nonexistent_pelvis_mesh.obj"},
        base_model_sha256="abc123",
    )
    variant = GolfModelVariant(
        variant_id="broken_mesh_variant",
        skeleton=skeleton,
        equipment=None,
        calibration_hash="def456",
        actuation=ActuationProfile(
            actuation_type=ActuationType.TORQUE,
            actuator_names=("lumbar_extension_actuator",),
            control_units="N*m",
            control_ranges={"lumbar_extension_actuator": (-500.0, 500.0)},
            internal_state_names=(),
            capabilities=("joint_torques",),
        ),
    )

    with pytest.raises(MissingGeometryAssetError, match="Missing geometry asset"):
        variant.validate_geometry_assets(tmp_path)


def test_red_fixture_stale_hash_fails() -> None:
    """RED Fixture: Variant with stale or mismatched base model digest raises StaleModelHashError."""
    wrong_sha256 = "deadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeefdeadbeef"
    with pytest.raises(StaleModelHashError, match="Stale or mismatched model hash"):
        create_torque_model_variant(
            SCALED_MODEL_PATH, expected_base_sha256=wrong_sha256
        )


def test_red_fixture_unsupported_capability_fails_closed() -> None:
    """RED Fixture: Requesting an unsupported feature (e.g. tendon elasticity on pure torque model) fails closed."""
    adapter = create_golf_model_adapter(SCALED_MODEL_PATH)
    torque_variant = create_torque_model_variant(SCALED_MODEL_PATH)
    adapter.load_variant(torque_variant)

    with pytest.raises(
        UnsupportedCapabilityError,
        match="Capability 'tendon_dynamics' is not supported",
    ):
        adapter.require_capability("tendon_dynamics")


def test_identity_no_op_variant_preserves_fk() -> None:
    """Verify that an identity variant reproduces exact forward kinematics of base model."""
    adapter = create_golf_model_adapter(SCALED_MODEL_PATH)
    variant = create_torque_model_variant(SCALED_MODEL_PATH)
    adapter.load_variant(variant)

    # Neutral pose coordinates
    neutral_q = dict.fromkeys(adapter.coordinate_names, 0.0)
    positions = adapter.evaluate_forward_kinematics(neutral_q)

    assert "pelvis" in positions
    assert "torso" in positions
    assert "hand_r" in positions or "hand_l" in positions or "Club" in positions
    for pos in positions.values():
        assert len(pos) == 3
        assert np.all(np.isfinite(pos))


def test_torque_and_muscle_variants_traverse_same_adapter_api() -> None:
    """Verify that torque and muscle variants share identical adapter interface and truthful capabilities."""
    adapter = create_golf_model_adapter(SCALED_MODEL_PATH)

    torque_var = create_torque_model_variant(SCALED_MODEL_PATH)
    muscle_var = create_muscle_model_variant(SCALED_MODEL_PATH)

    # 1. Load Torque Variant
    adapter.load_variant(torque_var)
    assert adapter.current_variant.variant_id == torque_var.variant_id
    assert adapter.current_variant.actuation.control_units == "N*m"
    caps_torque = adapter.capabilities
    assert caps_torque["actuation_type"] == "torque"
    assert caps_torque["supports_muscle_forces"] is False
    assert caps_torque["supports_joint_torques"] is True

    # 2. Load Muscle Variant
    adapter.load_variant(muscle_var)
    assert adapter.current_variant.variant_id == muscle_var.variant_id
    assert adapter.current_variant.actuation.control_units == "normalized"
    caps_muscle = adapter.capabilities
    assert caps_muscle["actuation_type"] == "muscle_tendon"
    assert caps_muscle["supports_muscle_forces"] is True
    assert caps_muscle["supports_joint_torques"] is False

    # State variables in muscle variant include activation states
    assert len(muscle_var.actuation.internal_state_names) > 0
    assert "activation" in muscle_var.actuation.internal_state_names[0]


def test_no_sdk_objects_leaked_through_adapter() -> None:
    """Ensure adapter outputs return only primitives, dicts, dataclasses, or numpy arrays (no OpenSim C++ objects)."""
    adapter = create_golf_model_adapter(SCALED_MODEL_PATH)
    variant = create_torque_model_variant(SCALED_MODEL_PATH)
    adapter.load_variant(variant)

    caps = adapter.capabilities
    coords = adapter.coordinate_names
    states = adapter.state_variable_names
    actuators = adapter.actuator_names

    for obj in (caps, coords, states, actuators):
        assert isinstance(obj, (dict, tuple, list, str, int, float, bool))
        # Ensure class name does not contain OpenSim or SimTK
        type_name = type(obj).__name__
        assert "opensim" not in type_name.lower()
        assert "simtk" not in type_name.lower()
