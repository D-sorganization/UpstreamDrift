"""Tests for the independently executed spatial forward-contact study."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.spatial_forward_contract import (
    CanonicalSpatialState,
    SpatialContactParameters,
    canonical_spatial_state_digest,
    contact_pair,
    transport_wrench,
)


DATA_DIR = (
    Path(__file__).resolve().parents[2]
    / "docs/research/proximal_distal_energy_transfer/data"
)
pytestmark = pytest.mark.scientific


def test_spatial_contact_parameters_fail_closed() -> None:
    with pytest.raises(ValueError, match="contact_stiffness"):
        SpatialContactParameters(contact_stiffness=-1.0)
    with pytest.raises(ValueError, match="time_step"):
        SpatialContactParameters(time_step=0.0)
    with pytest.raises(ValueError, match="grip offsets"):
        SpatialContactParameters(
            lead_grip_offset=(0.0, 0.0, 0.0),
            trail_grip_offset=(0.0, 0.0, 0.0),
        )


def test_canonical_initial_state_requires_a_unit_quaternion() -> None:
    with pytest.raises(ValueError, match="unit length"):
        CanonicalSpatialState(
            hand_positions=np.zeros((2, 3)),
            hand_velocities=np.zeros((2, 3)),
            club_position=np.zeros(3),
            club_quaternion_wxyz=np.array([2.0, 0.0, 0.0, 0.0]),
            club_linear_velocity=np.zeros(3),
            club_angular_velocity=np.zeros(3),
        )


def test_contact_pair_closes_force_and_power() -> None:
    force_on_club, force_on_hand, stored_power, dissipated_power = contact_pair(
        hand_position=np.array([0.10, -0.02, 0.03]),
        hand_velocity=np.array([0.4, -0.1, 0.2]),
        club_point_position=np.array([0.09, -0.01, 0.02]),
        club_point_velocity=np.array([0.1, 0.0, -0.1]),
        stiffness=900.0,
        damping=12.0,
    )
    np.testing.assert_allclose(force_on_club + force_on_hand, 0.0, atol=1e-14)
    relative_velocity = np.array([0.3, -0.1, 0.3])
    interface_power = float(force_on_hand @ relative_velocity)
    assert interface_power == pytest.approx(stored_power + dissipated_power)
    assert dissipated_power <= 0.0


def test_wrench_transport_and_coincident_grip_control() -> None:
    force = np.array([2.0, -5.0, 3.0])
    opposite = -force
    separated = transport_wrench(
        reference=np.zeros(3),
        points=np.array([[-0.08, 0.015, 0.0], [0.08, -0.015, 0.0]]),
        forces=np.array([force, opposite]),
    )
    coincident = transport_wrench(
        reference=np.zeros(3),
        points=np.zeros((2, 3)),
        forces=np.array([force, opposite]),
    )
    np.testing.assert_allclose(separated[:3], 0.0, atol=1e-14)
    assert np.linalg.norm(separated[3:]) > 0.0
    np.testing.assert_allclose(coincident, 0.0, atol=1e-14)


@pytest.mark.requires_pinocchio
def test_real_engine_adapters_execute_same_initial_contract() -> None:
    pytest.importorskip("mujoco")
    pin = pytest.importorskip("pinocchio")
    version = getattr(pin, "__version__", None)
    if (
        not hasattr(pin, "Model")
        or not isinstance(version, str)
        or int(version.split(".")[0]) < 2
    ):
        pytest.skip("Pinocchio import is not the robotics engine")

    from scripts.research.proximal_distal_energy.spatial_forward_engines import (
        make_spatial_forward_adapter,
    )

    params = SpatialContactParameters(
        duration=0.002, time_step=0.00025, killswitch_time=0.001
    )
    mujoco_adapter = make_spatial_forward_adapter("mujoco", params)
    pinocchio_adapter = make_spatial_forward_adapter("pinocchio", params)
    assert mujoco_adapter.engine_identity.library == "mujoco"
    assert pinocchio_adapter.engine_identity.library == "pinocchio"
    assert mujoco_adapter.engine_identity.native_forward_dynamics
    assert pinocchio_adapter.engine_identity.native_forward_dynamics
    assert mujoco_adapter.model_digest == pinocchio_adapter.model_digest
    np.testing.assert_allclose(
        mujoco_adapter.canonical_state().hand_positions,
        pinocchio_adapter.canonical_state().hand_positions,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        mujoco_adapter.canonical_state().club_position,
        pinocchio_adapter.canonical_state().club_position,
        atol=1e-12,
    )
    initial_state = CanonicalSpatialState(
        hand_positions=np.array([[0.2, 0.11, 1.0], [0.2, -0.11, 1.0]]),
        hand_velocities=np.array([[0.3, 0.1, -0.2], [0.2, -0.1, -0.2]]),
        club_position=np.array([0.2, 0.0, 1.03]),
        club_quaternion_wxyz=np.array([1.0, 0.0, 0.0, 0.0]),
        club_linear_velocity=np.array([0.25, 0.0, -0.2]),
        club_angular_velocity=np.array([0.1, -0.2, 0.4]),
    )
    digest = canonical_spatial_state_digest(initial_state)
    mapped_adapters = (
        make_spatial_forward_adapter("mujoco", params, initial_state),
        make_spatial_forward_adapter("pinocchio", params, initial_state),
    )
    for adapter in mapped_adapters:
        assert adapter.initial_state_digest == digest
        achieved = adapter.canonical_state()
        np.testing.assert_allclose(
            achieved.hand_positions, initial_state.hand_positions
        )
        np.testing.assert_allclose(
            achieved.hand_velocities, initial_state.hand_velocities
        )
        np.testing.assert_allclose(achieved.club_position, initial_state.club_position)
        np.testing.assert_allclose(
            achieved.club_linear_velocity, initial_state.club_linear_velocity
        )
        np.testing.assert_allclose(
            achieved.club_angular_velocity, initial_state.club_angular_velocity
        )
    from scripts.research.proximal_distal_energy.spatial_forward_study import (
        compare_engine_traces,
        run_engine_trace,
    )

    publication_params = SpatialContactParameters()
    mujoco_trace = run_engine_trace(
        "mujoco", publication_params, disable_driver_after_killswitch=True
    )
    pinocchio_trace = run_engine_trace(
        "pinocchio", publication_params, disable_driver_after_killswitch=True
    )
    gates = compare_engine_traces(mujoco_trace, pinocchio_trace, publication_params)
    assert gates["trajectory_gate_passed"] is True
    assert gates["wrench_gate_passed"] is True
    assert gates["energy_gate_passed"] is True


def test_committed_spatial_forward_evidence_is_falsifiable() -> None:
    record = json.loads((DATA_DIR / "spatial_forward_contact_study.json").read_text())
    assert record["schema_version"] == "spatial-forward-contact-evidence-v1"
    assert set(record["engine_identities"]) == {"mujoco", "pinocchio"}
    assert all(
        item["native_forward_dynamics"] for item in record["engine_identities"].values()
    )
    assert record["model_contract"]["digest_match"] is True
    assert record["numerical_gates"]["trajectory_gate_passed"] is True
    assert record["numerical_gates"]["wrench_gate_passed"] is True
    assert record["mechanism_tests"]["coincident_grip_couple_max_nm"] < 1e-10
    assert (
        record["mechanism_tests"]["same_state_killswitch_negative_duration_s"] >= 0.03
    )
    assert record["timestep_refinement"]["monotone_residual_reduction"] is True
    for summary in record["mechanism_tests"][
        "baseline_and_killswitch_trace_summaries"
    ].values():
        assert summary["interface_power_residual_max_w"] < 1e-10
        assert summary["wrench_power_residual_max_w"] < 1e-10
    assert record["claim_boundary"]["human_strategy"] == "untested"
    assert record["claim_boundary"]["muscle_mechanism"] == "untested"
    root = Path(__file__).resolve().parents[2]
    for relative_path, expected in record["source_sha256"].items():
        observed = hashlib.sha256((root / relative_path).read_bytes()).hexdigest()
        assert observed == expected


def test_committed_spatial_forward_arrays_are_finite_and_aligned() -> None:
    with np.load(DATA_DIR / "spatial_forward_contact_study.npz") as archive:
        assert archive["mujoco_baseline_time"].shape == (961,)
        assert archive["pinocchio_baseline_time"].shape == (961,)
        np.testing.assert_array_equal(
            archive["mujoco_baseline_time"], archive["pinocchio_baseline_time"]
        )
        for name in archive.files:
            assert np.all(np.isfinite(archive[name])), name
        for engine in ("mujoco", "pinocchio"):
            for branch in ("baseline", "killswitch"):
                assert (
                    np.max(
                        np.abs(archive[f"{engine}_{branch}_interface_power_residual"])
                    )
                    < 1e-10
                )
                assert (
                    np.max(np.abs(archive[f"{engine}_{branch}_wrench_power_residual"]))
                    < 1e-10
                )
