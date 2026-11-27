# fmt: off
"""Tests for immutable serializable SimulationOutput and DynamicsQuery interface (#15794 B2 / #15731)."""
from __future__ import annotations

from dataclasses import FrozenInstanceError
import json
import pytest
import numpy as np

from src.shared.python.hand_load_nullspace.simulation_output import (
    Channel,
    SimulationOutput,
    DynamicsQuery,
    DefaultDynamicsQuery,
    simulation_output,
    serialize_simulation_output,
    deserialize_simulation_output,
    SCHEMA_VERSION,
)

pytestmark = pytest.mark.unit


def test_simulation_output_immutability():
    ch = Channel("available", "N", (1.0, 2.0, 3.0))
    out = SimulationOutput(
        schema_version=SCHEMA_VERSION,
        frame="frame_001",
        time_s=0.01,
        load_kind="total",
        replay_badge="allocation-only",
        net_force_n=ch,
        midhands_moment_nm=ch,
        f_l_n=ch,
        f_r_n=ch,
        m_l_nm=ch,
        m_r_nm=ch,
        ground_force_n=ch,
        ground_moment_nm=ch,
        cop_m=Channel("UNAVAILABLE", "m", None, "F_z < 5 N"),
    )
    with pytest.raises(Exception) as excinfo:
        setattr(out, "time_s", 0.02)  # noqa: B010
    assert "FrozenInstance" in type(excinfo.value).__name__ or isinstance(excinfo.value, (AttributeError, TypeError))

    with pytest.raises(Exception) as excinfo_ch:
        setattr(ch, "values", (4.0, 5.0, 6.0))  # noqa: B010
    assert "FrozenInstance" in type(excinfo_ch.value).__name__ or isinstance(excinfo_ch.value, (AttributeError, TypeError))


def test_simulation_output_serialization_round_trip():
    ch_avail = Channel("available", "N", (10.0, 20.0, 30.0))
    ch_unavail = Channel("UNAVAILABLE", "m", None, "CoP UNAVAILABLE: F_z < 5 N")
    out = SimulationOutput(
        schema_version=SCHEMA_VERSION,
        frame="frame_0042",
        time_s=0.15,
        load_kind="total",
        replay_badge="forward-replayed",
        net_force_n=ch_avail,
        midhands_moment_nm=ch_avail,
        f_l_n=ch_avail,
        f_r_n=ch_avail,
        m_l_nm=ch_avail,
        m_r_nm=ch_avail,
        ground_force_n=ch_avail,
        ground_moment_nm=ch_avail,
        cop_m=ch_unavail,
        match_id="match_20261125_001",
        model_definition_hash="sha256_model_abc",
        subject_profile_hash="sha256_subj_def",
        match_artifact_hash="sha256_art_123",
        cop_wrench_m=ch_avail,
        cop_pressure_centroid_m=ch_unavail,
        bos_valid=False,
        bos_margin_m=-0.015,
        contact_state="left_only",
        cop_segment_id=2,
        gate_alpha=0.85,
        filter_provenance="raw_fallback",
        com_body_m=ch_avail,
        com_body_club_m=ch_avail,
        mass_total_kg=Channel("available", "kg", (78.5,)),
        momentum_residual_n=ch_avail,
    )

    data = out.to_dict()
    assert isinstance(data, dict)
    assert data["schema_version"] == SCHEMA_VERSION
    assert data["bos_valid"] is False
    assert data["bos_margin_m"] == pytest.approx(-0.015)
    assert data["net_force_n"]["availability"] == "available"
    assert data["cop_m"]["availability"] == "UNAVAILABLE"

    json_str = out.to_json()
    assert isinstance(json_str, str)
    parsed = json.loads(json_str)
    assert parsed["frame"] == "frame_0042"

    rebuilt = SimulationOutput.from_dict(data)
    assert rebuilt == out
    assert rebuilt.time_s == out.time_s
    assert rebuilt.match_id == "match_20261125_001"
    assert rebuilt.bos_valid is False
    assert rebuilt.cop_m.availability == "UNAVAILABLE"
    assert rebuilt.cop_m.reason == "CoP UNAVAILABLE: F_z < 5 N"

    rebuilt_json = SimulationOutput.from_json(json_str)
    assert rebuilt_json == out

    # Top-level helper functions test
    serialized = serialize_simulation_output(out)
    assert deserialize_simulation_output(serialized) == out


def test_serialization_fails_on_unsupported_version():
    data = {
        "schema_version": "simulation-output/999",
        "frame": "f1",
        "time_s": 0.0,
        "load_kind": "total",
        "replay_badge": "allocation-only",
        "net_force_n": {"availability": "UNAVAILABLE", "units": "N", "reason": "none"},
        "midhands_moment_nm": {"availability": "UNAVAILABLE", "units": "N*m", "reason": "none"},
        "f_l_n": {"availability": "UNAVAILABLE", "units": "N", "reason": "none"},
        "f_r_n": {"availability": "UNAVAILABLE", "units": "N", "reason": "none"},
        "m_l_nm": {"availability": "UNAVAILABLE", "units": "N*m", "reason": "none"},
        "m_r_nm": {"availability": "UNAVAILABLE", "units": "N*m", "reason": "none"},
        "ground_force_n": {"availability": "UNAVAILABLE", "units": "N", "reason": "none"},
        "ground_moment_nm": {"availability": "UNAVAILABLE", "units": "N*m", "reason": "none"},
        "cop_m": {"availability": "UNAVAILABLE", "units": "m", "reason": "none"},
    }
    with pytest.raises(ValueError, match="schema_version is not supported"):
        SimulationOutput.from_dict(data)


def test_dynamics_query_protocol_and_default_implementation():
    q = np.array([0.1, 0.2, 0.3])
    qdot = np.array([0.0, 0.5, -0.2])
    M_val = np.eye(3) * 2.5
    cg_val = np.array([1.2, 3.4, -0.5])
    J_val = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    dq = DefaultDynamicsQuery(
        mass_matrix_fn=lambda q_in: M_val * (q_in[0] / 0.1),
        bias_forces_fn=lambda q_in, qdot_in: cg_val + q_in[1],
        jacobian_midhands_fn=lambda q_in: J_val,
    )

    assert isinstance(dq, DynamicsQuery)
    M = dq.mass_matrix(q)
    assert np.allclose(M, M_val)
    cg = dq.bias_forces(q, qdot)
    assert np.allclose(cg, cg_val + 0.2)
    J = dq.jacobian_midhands(q)
    assert np.allclose(J, J_val)


def test_dynamics_query_preconditions():
    dq = DefaultDynamicsQuery(
        mass_matrix_fn=lambda q: np.eye(2),
        bias_forces_fn=lambda q, qdot: np.zeros(2),
        jacobian_midhands_fn=lambda q: np.zeros((6, 2)),
    )
    with pytest.raises(ValueError, match="q must contain finite values"):
        dq.mass_matrix(np.array([np.nan, 0.0]))
    with pytest.raises(ValueError, match="qdot must contain finite values"):
        dq.bias_forces(np.array([0.0, 0.0]), np.array([np.inf, 1.0]))
# fmt: on
