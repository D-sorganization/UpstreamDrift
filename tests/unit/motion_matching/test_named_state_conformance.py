"""Named-state and torque-transfer conformance tests (MMR-09-I, #11109).

Verifies:
1. Permuted named state storage preserves invariant forward kinematics and rejects unknown/missing names.
2. Quaternion analytical vs. finite-difference velocity mapping with SO(3) double-cover sign invariance.
3. Capture attachment conformance: iron capture rejects driver attachments and solve/replay tolerance divergence.
4. Small-body virtual work invariance across generalized coordinate permutations.
5. NamedStateManifest serialization and round-trip identity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.named_state import (
    NAMED_STATE_SCHEMA_VERSION,
    CaptureAttachmentDeclaration,
    NamedStateConformanceAdapter,
    NamedStateManifest,
    QuaternionVelocityMap,
    SmallBodyVirtualWorkOracle,
    validate_named_state_conformance,
)
from src.shared.python.motion_matching.pinocchio_g2_g3 import (
    ClubKind,
    IntegratorConfig,
)

pytestmark = pytest.mark.unit


def test_schema_version_is_stable() -> None:
    assert NAMED_STATE_SCHEMA_VERSION == "named-state-conformance/1.0.0"


def test_permute_named_state_storage_preserves_invariant_fk() -> None:
    manifest = NamedStateManifest(
        schema_version=NAMED_STATE_SCHEMA_VERSION,
        coordinate_names=(
            "pelvis_tx",
            "pelvis_ty",
            "pelvis_tz",
            "hip_flexion",
            "knee_angle",
        ),
        velocity_names=(
            "pelvis_vx",
            "pelvis_vy",
            "pelvis_vz",
            "hip_flexion_rate",
            "knee_rate",
        ),
        control_names=("hip_motor", "knee_motor"),
        armature={"hip_motor": 0.005, "knee_motor": 0.005},
        interpolation={"pelvis_tx": "linear", "knee_angle": "cubic"},
    )
    adapter = NamedStateConformanceAdapter(manifest)

    # Order A
    state_a = {
        "pelvis_tx": 0.1,
        "pelvis_ty": -0.2,
        "pelvis_tz": 0.85,
        "hip_flexion": 0.35,
        "knee_angle": -0.45,
    }
    # Order B: permuted keys
    state_b = {
        "knee_angle": -0.45,
        "pelvis_tz": 0.85,
        "pelvis_tx": 0.1,
        "hip_flexion": 0.35,
        "pelvis_ty": -0.2,
    }

    vec_a = manifest.pack_q(state_a)
    vec_b = manifest.pack_q(state_b)
    np.testing.assert_array_equal(vec_a, vec_b)

    fk_a = adapter.compute_kinematics(state_a)
    fk_b = adapter.compute_kinematics(state_b)
    np.testing.assert_allclose(fk_a["end_effector"], fk_b["end_effector"], atol=1e-12)


def test_named_state_rejects_missing_or_unknown_names() -> None:
    manifest = NamedStateManifest(
        schema_version=NAMED_STATE_SCHEMA_VERSION,
        coordinate_names=("q0", "q1", "q2"),
        velocity_names=("v0", "v1", "v2"),
        control_names=("u0",),
        armature={"u0": 0.001},
        interpolation={"q0": "linear", "q1": "linear", "q2": "linear"},
    )

    # Missing name
    with pytest.raises(ValueError, match="missing required coordinate"):
        manifest.pack_q({"q0": 1.0, "q1": 2.0})

    # Unknown name
    with pytest.raises(ValueError, match="unknown coordinate"):
        manifest.pack_q({"q0": 1.0, "q1": 2.0, "q2": 3.0, "q_extra": 4.0})

    # Non-finite values
    with pytest.raises(ValueError, match="finite"):
        manifest.pack_q({"q0": 1.0, "q1": np.nan, "q2": 3.0})


def test_quaternion_analytical_vs_finite_difference_velocity() -> None:
    q_map = QuaternionVelocityMap()
    # Initial quaternion: arbitrary unit orientation
    q0 = np.array([0.70710678, 0.0, 0.70710678, 0.0])
    q0 = q0 / np.linalg.norm(q0)

    # Known angular velocity in rad/s
    omega_true = np.array([1.5, -2.0, 0.8])

    dt = 1e-5
    # Forward integrated quaternion under constant omega over dt
    angle = np.linalg.norm(omega_true) * dt
    axis = omega_true / np.linalg.norm(omega_true)
    delta_q = np.array([np.cos(angle / 2.0), *(axis * np.sin(angle / 2.0))])

    # Quaternion multiplication delta_q (x) q0 for world frame rotation
    w1, x1, y1, z1 = delta_q
    w2, x2, y2, z2 = q0
    q1 = np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )

    # Analytical velocity
    omega_analytical = q_map.quaternion_to_angular_velocity(q0, q1, dt)
    np.testing.assert_allclose(omega_analytical, omega_true, rtol=1e-4, atol=1e-4)

    # Double-cover sign invariance: -q1 represents the identical 3D orientation
    omega_neg = q_map.quaternion_to_angular_velocity(q0, -q1, dt)
    np.testing.assert_allclose(omega_neg, omega_true, rtol=1e-4, atol=1e-4)


def test_capture_attachment_conformance_rejects_driver_attachments_for_iron() -> None:
    driver_decl = CaptureAttachmentDeclaration(
        club=ClubKind.DRIVER,
        document_id="anthro_driver",
        document_sha256="driver_doc_sha_123",
        attachment_calibration_hash="driver_calib_hash_456",
        grip_frame_id="driver_grip",
    )
    iron_decl = CaptureAttachmentDeclaration(
        club=ClubKind.IRON,
        document_id="anthro_iron",
        document_sha256="iron_doc_sha_789",
        attachment_calibration_hash="iron_calib_hash_999",
        grip_frame_id="iron_grip",
    )

    manifest = NamedStateManifest(
        schema_version=NAMED_STATE_SCHEMA_VERSION,
        coordinate_names=("q0", "q1"),
        velocity_names=("v0", "v1"),
        control_names=("u0",),
        armature={"u0": 0.005},
        interpolation={"q0": "linear", "q1": "linear"},
        capture_declaration=iron_decl,
    )
    adapter = NamedStateConformanceAdapter(manifest)

    # Conforming iron declaration passes
    assert adapter.verify_capture_conformance(iron_decl) is True

    # Reusing driver declaration for iron fails closed
    with pytest.raises(ValueError, match="cross-club attachment contamination"):
        adapter.verify_capture_conformance(driver_decl)


def test_solve_and_replay_tolerance_divergence_fails_closed() -> None:
    manifest = NamedStateManifest(
        schema_version=NAMED_STATE_SCHEMA_VERSION,
        coordinate_names=("q0",),
        velocity_names=("v0",),
        control_names=("u0",),
        armature={"u0": 0.005},
        interpolation={"q0": "linear"},
    )
    adapter = NamedStateConformanceAdapter(manifest)

    solve_cfg = IntegratorConfig(name="rk45", rtol=1e-4, fixed_step=False)
    replay_cfg = IntegratorConfig(name="rk45", rtol=1e-6, fixed_step=False)

    with pytest.raises(ValueError, match="integrator tolerance mismatch"):
        adapter.verify_integrator_conformance(solve_cfg, replay_cfg)


def test_small_body_virtual_work_conformance() -> None:
    oracle = SmallBodyVirtualWorkOracle(link_lengths=(0.5, 0.4), link_masses=(1.5, 1.0))

    q_named = {"shoulder_flex": 0.4, "elbow_flex": -0.6}
    dq_named = {"shoulder_flex": 0.01, "elbow_flex": -0.015}
    tau_named = {"shoulder_flex": 12.5, "elbow_flex": -8.0}

    # Evaluate virtual work in nominal coordinate order
    w_nominal = oracle.compute_virtual_work(q_named, dq_named, tau_named)

    # Permuted coordinate order in input maps
    q_permuted = {"elbow_flex": -0.6, "shoulder_flex": 0.4}
    dq_permuted = {"elbow_flex": -0.015, "shoulder_flex": 0.01}
    tau_permuted = {"elbow_flex": -8.0, "shoulder_flex": 12.5}

    w_permuted = oracle.compute_virtual_work(q_permuted, dq_permuted, tau_permuted)
    assert np.isclose(w_nominal, w_permuted, atol=1e-12)

    # Tip force mapping: tau = J(q)^T F_tip. Virtual work in joint space must equal Cartesian work F^T dx
    f_tip = np.array([10.0, -15.0])
    w_joint, w_cart = oracle.verify_virtual_work_duality(q_named, dq_named, f_tip)
    np.testing.assert_allclose(w_joint, w_cart, atol=1e-12)


def test_named_state_manifest_save_and_reopen_identity(tmp_path: Path) -> None:
    iron_decl = CaptureAttachmentDeclaration(
        club=ClubKind.IRON,
        document_id="anthro_iron",
        document_sha256="doc_sha_iron_abc",
        attachment_calibration_hash="calib_sha_iron_def",
        grip_frame_id="iron_grip_frame",
    )
    manifest = NamedStateManifest(
        schema_version=NAMED_STATE_SCHEMA_VERSION,
        coordinate_names=("pelvis_tx", "hip_flexion", "knee_angle"),
        velocity_names=("pelvis_vx", "hip_rate", "knee_rate"),
        control_names=("hip_motor", "knee_motor"),
        armature={"hip_motor": 0.005, "knee_motor": 0.005},
        interpolation={"pelvis_tx": "linear", "knee_angle": "cubic"},
        capture_declaration=iron_decl,
    )

    save_path = tmp_path / "manifest.json"
    manifest.save_json(save_path)

    reloaded = NamedStateManifest.load_json(save_path)
    assert reloaded.schema_version == manifest.schema_version
    assert reloaded.coordinate_names == manifest.coordinate_names
    assert reloaded.velocity_names == manifest.velocity_names
    assert reloaded.control_names == manifest.control_names
    assert reloaded.armature == manifest.armature
    assert reloaded.interpolation == manifest.interpolation
    assert reloaded.capture_declaration == manifest.capture_declaration
    assert reloaded.manifest_sha256 == manifest.manifest_sha256


# ---------------------------------------------------------------------------
# Review-Gate Regressions (#11122): attachment identity, immutable manifest, exports
# ---------------------------------------------------------------------------


def test_capture_attachment_conformance_requires_full_attachment_identity() -> None:
    """Same club/document/calibration but different document_sha256 or grip_frame_id must fail closed."""
    declared = CaptureAttachmentDeclaration(
        club=ClubKind.IRON,
        document_id="anthro_iron",
        document_sha256="doc_sha_original",
        attachment_calibration_hash="calib_sha_original",
        grip_frame_id="grip_A",
    )
    manifest = NamedStateManifest(
        schema_version=NAMED_STATE_SCHEMA_VERSION,
        coordinate_names=("q0", "q1"),
        velocity_names=("v0", "v1"),
        control_names=("u0",),
        armature={"u0": 0.005},
        interpolation={"q0": "linear", "q1": "linear"},
        capture_declaration=declared,
    )
    adapter = NamedStateConformanceAdapter(manifest)

    # Different document hash (different model contents) must be rejected
    tampered_document = CaptureAttachmentDeclaration.from_dict(
        {**declared.as_dict(), "document_sha256": "doc_sha_other_model"}
    )
    with pytest.raises(ValueError, match="document_sha256"):
        adapter.verify_capture_conformance(tampered_document)

    # A different grip frame must also be rejected
    tampered_grip = CaptureAttachmentDeclaration.from_dict(
        {**declared.as_dict(), "grip_frame_id": "grip_B"}
    )
    with pytest.raises(ValueError, match="grip_frame_id"):
        adapter.verify_capture_conformance(tampered_grip)


def test_manifest_armature_and_interpolation_mappings_are_frozen() -> None:
    """Mutating the caller's mappings must not alter the manifest or its recorded SHA."""
    armature = {"hip_motor": 0.005, "knee_motor": 0.005}
    interpolation = {"pelvis_tx": "linear", "knee_angle": "cubic"}
    manifest = NamedStateManifest(
        schema_version=NAMED_STATE_SCHEMA_VERSION,
        coordinate_names=("pelvis_tx", "knee_angle"),
        velocity_names=("pelvis_vx", "knee_rate"),
        control_names=("hip_motor",),
        armature=armature,
        interpolation=interpolation,
    )
    sha_before = manifest.manifest_sha256

    # Mutating the caller's original dictionaries must not touch the manifest
    armature["hip_motor"] = 99.0
    interpolation["knee_angle"] = "nearest"
    assert manifest.armature["hip_motor"] == 0.005
    assert manifest.interpolation["knee_angle"] == "cubic"
    assert manifest.manifest_sha256 == sha_before

    # The manifest's own stored mappings must reject item assignment
    with pytest.raises(TypeError):
        manifest.armature["knee_motor"] = 1.0
    with pytest.raises(TypeError):
        manifest.interpolation["pelvis_tx"] = "nearest"


def test_motion_matching_package_exports_named_state_symbols() -> None:
    """The curated `__all__` must expose the named-state facade symbols."""
    import src.shared.python.motion_matching as mm

    expected = (
        "NAMED_STATE_SCHEMA_VERSION",
        "NamedStateManifest",
        "NamedStateConformanceAdapter",
        "CaptureAttachmentDeclaration",
        "QuaternionVelocityMap",
        "SmallBodyVirtualWorkOracle",
        "validate_named_state_conformance",
    )
    for symbol in expected:
        assert symbol in mm.__all__, f"{symbol} missing from motion_matching.__all__"
