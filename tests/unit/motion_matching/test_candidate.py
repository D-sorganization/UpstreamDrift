"""Unit tests for versioned MatchedSwingCandidate (MS-15, #10334).

Verifies:
- Round-trip serialization/deserialization equality.
- Distinct kinematic and dynamic profiles.
- Support for nq != nv (e.g. quaternion floating base).
- Virtual-work consistency across coordinate/effort transforms.
- Strict timestamp monotonicity, length matching, and finite bounds.
- Fail-closed rejection of missing dynamic states in dynamic profiles.
- Fail-closed tamper detection via immutable array SHA-256 checksums.
- Unsupported schema version rejection.
- Lossless conversion of historical returned81 NPZ and OpenSim MOT artifacts.
- Tour matching viewer seamless loading of candidates.
- Schema documentation freshness.
"""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.candidate import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateAuxiliary,
    CandidateMarkers,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
    check_virtual_work_consistency,
)
from src.shared.python.motion_matching.candidate_convert import (
    convert_opensim_mot,
    convert_returned81_replay,
)
from src.shared.python.motion_matching.candidate_io import (
    load_candidate,
    save_candidate,
)
from src.tools.tour_matching_viewer.core import load_replay

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
REPLAYS_DIR = ROOT / "docs/development/full_body_models/evidence/replays"
VIEWER_DIR = ROOT / "docs/development/full_body_models/evidence/viewer"
MUJOCO_REPLAY_NPZ = REPLAYS_DIR / "mujoco_returned81_replay.npz"
OPENSIM_MOT = VIEWER_DIR / "opensim_os3b_ik.mot"
CANDIDATES_DOC = ROOT / "docs/development/full_body_models/CANDIDATES.md"


def _make_dummy_dynamic_candidate(
    n_frames: int = 10, nq: int = 5, nv: int = 5, nu: int = 4, nm: int = 3
) -> MatchedSwingCandidate:
    time_s = np.linspace(0.0, 0.5, n_frames)
    q = np.sin(np.outer(time_s, np.arange(1, nq + 1) * 0.5))
    v = np.cos(np.outer(time_s, np.arange(1, nv + 1) * 0.5))
    tau = np.ones((n_frames, nu), dtype=np.float64) * 2.5
    model_markers = np.zeros((n_frames, nm, 3), dtype=np.float64)
    target_markers = np.ones((n_frames, nm, 3), dtype=np.float64) * 0.1
    validity = np.ones((n_frames, nm), dtype=bool)

    coord_names = tuple(f"joint_{i}" for i in range(nq))
    vel_names = tuple(f"vel_{i}" for i in range(nv))
    act_names = tuple(f"act_{i}" for i in range(nu))
    marker_names = tuple(f"marker_{i}" for i in range(nm))

    meta = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=CandidateProfile.DYNAMIC,
        engine="mujoco",
        model_name="test_model",
        model_sha256="a" * 64,
        coordinate_names=coord_names,
        velocity_names=vel_names,
        actuator_names=act_names,
        marker_names=marker_names,
        event_indices={"address": 0, "impact": n_frames - 1},
        solver_settings={"dt": 0.001, "integrator": "RK4"},
    )
    return MatchedSwingCandidate(
        metadata=meta,
        time_s=time_s,
        q=q,
        v=v,
        tau=tau,
        markers=CandidateMarkers(
            model_markers_m=model_markers,
            target_markers_m=target_markers,
            marker_validity=validity,
        ),
    )


def test_candidate_roundtrip(tmp_path: Path) -> None:
    """Test full serialization and deserialization round-trip equality."""
    candidate = _make_dummy_dynamic_candidate()
    target_path = tmp_path / "candidate.npz"

    save_candidate(candidate, target_path)
    loaded = load_candidate(target_path, validate_checksums=True)

    assert loaded.metadata.schema_version == candidate.metadata.schema_version
    assert loaded.metadata.profile == CandidateProfile.DYNAMIC
    assert loaded.metadata.engine == candidate.metadata.engine
    assert loaded.metadata.coordinate_names == candidate.metadata.coordinate_names
    assert loaded.metadata.velocity_names == candidate.metadata.velocity_names
    assert loaded.metadata.actuator_names == candidate.metadata.actuator_names
    assert loaded.metadata.marker_names == candidate.metadata.marker_names

    np.testing.assert_allclose(loaded.time_s, candidate.time_s)
    np.testing.assert_allclose(loaded.q, candidate.q)
    assert loaded.v is not None and candidate.v is not None
    np.testing.assert_allclose(loaded.v, candidate.v)
    assert loaded.tau is not None and candidate.tau is not None
    np.testing.assert_allclose(loaded.tau, candidate.tau)
    assert loaded.model_markers_m is not None and candidate.model_markers_m is not None
    np.testing.assert_allclose(loaded.model_markers_m, candidate.model_markers_m)
    assert (
        loaded.target_markers_m is not None and candidate.target_markers_m is not None
    )
    np.testing.assert_allclose(loaded.target_markers_m, candidate.target_markers_m)
    assert loaded.marker_validity is not None and candidate.marker_validity is not None
    np.testing.assert_array_equal(loaded.marker_validity, candidate.marker_validity)

    # Immutability check
    assert not loaded.time_s.flags.writeable
    assert not loaded.q.flags.writeable
    assert not loaded.v.flags.writeable
    assert not loaded.tau.flags.writeable


def test_candidate_nq_not_equal_nv(tmp_path: Path) -> None:
    """Test support for models with quaternion floating base where nq != nv."""
    n_frames = 8
    nq = 7  # 3 pos + 4 quat
    nv = 6  # 3 linear + 3 angular vel
    nu = 6

    time_s = np.linspace(0.0, 0.35, n_frames)
    q = np.zeros((n_frames, nq))
    q[:, 6] = 1.0  # unit quaternion w
    v = np.ones((n_frames, nv)) * 0.1
    tau = np.ones((n_frames, nu)) * 5.0

    meta = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=CandidateProfile.DYNAMIC,
        engine="pinocchio",
        model_name="floating_base_model",
        model_sha256="b" * 64,
        coordinate_names=("px", "py", "pz", "qx", "qy", "qz", "qw"),
        velocity_names=("vx", "vy", "vz", "wx", "wy", "wz"),
        actuator_names=tuple(f"tau_{i}" for i in range(nu)),
    )
    candidate = MatchedSwingCandidate(
        metadata=meta,
        time_s=time_s,
        q=q,
        v=v,
        tau=tau,
    )

    path = tmp_path / "floating.npz"
    save_candidate(candidate, path)
    loaded = load_candidate(path)

    assert loaded.q.shape == (n_frames, 7)
    assert loaded.v is not None and loaded.v.shape == (n_frames, 6)
    assert loaded.metadata.coordinate_names == (
        "px",
        "py",
        "pz",
        "qx",
        "qy",
        "qz",
        "qw",
    )
    assert loaded.metadata.velocity_names == ("vx", "vy", "vz", "wx", "wy", "wz")


def test_virtual_work_consistency() -> None:
    """Test virtual-work and instantaneous power consistency across coordinate transforms."""
    # Virtual work: tau_q^T * v == tau_act^T * qd_act
    # For linear transmission tau_q = B * tau_act:
    # v^T * B * tau_act == (B^T * v)^T * tau_act
    nv = 4
    nu = 3
    B = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [0.5, 0.5, 0.0],
        ]
    )
    tau_act = np.array([10.0, -5.0, 2.0])
    tau_q = B @ tau_act
    v = np.array([0.2, -0.4, 0.8, 0.1])

    is_consistent, diff = check_virtual_work_consistency(
        generalized_velocity=v,
        generalized_effort=tau_q,
        actuator_transmission=B,
        actuator_effort=tau_act,
        tolerance=1e-8,
    )
    assert is_consistent
    assert diff < 1e-8

    # Inconsistent check
    tau_act_bad = np.array([10.0, -5.0, 99.0])
    is_bad, _ = check_virtual_work_consistency(
        generalized_velocity=v,
        generalized_effort=tau_q,
        actuator_transmission=B,
        actuator_effort=tau_act_bad,
        tolerance=1e-8,
    )
    assert not is_bad


def test_candidate_rejects_non_monotone_or_invalid_time() -> None:
    """Verify strict fail-closed rejection of non-monotone, nan, or negative timestamps."""
    meta = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=CandidateProfile.KINEMATIC,
        engine="mujoco",
        model_name="m",
        model_sha256="c" * 64,
        coordinate_names=("q0",),
    )

    # Non-monotone time
    with pytest.raises(ValueError, match="strictly monotonically increasing"):
        MatchedSwingCandidate(
            metadata=meta,
            time_s=np.array([0.0, 0.2, 0.1, 0.3]),
            q=np.zeros((4, 1)),
        )

    # Duplicate time
    with pytest.raises(ValueError, match="strictly monotonically increasing"):
        MatchedSwingCandidate(
            metadata=meta,
            time_s=np.array([0.0, 0.1, 0.1, 0.2]),
            q=np.zeros((4, 1)),
        )

    # Negative start time
    with pytest.raises(ValueError, match="negative"):
        MatchedSwingCandidate(
            metadata=meta,
            time_s=np.array([-0.05, 0.0, 0.05]),
            q=np.zeros((3, 1)),
        )

    # NaN in time
    with pytest.raises(ValueError, match="finite"):
        MatchedSwingCandidate(
            metadata=meta,
            time_s=np.array([0.0, np.nan, 0.2]),
            q=np.zeros((3, 1)),
        )

    # Length mismatch
    with pytest.raises(ValueError, match="length mismatch"):
        MatchedSwingCandidate(
            metadata=meta,
            time_s=np.array([0.0, 0.1, 0.2]),
            q=np.zeros((4, 1)),
        )


def test_candidate_rejects_missing_dynamic_state() -> None:
    """Verify dynamic profile rejects missing velocities or efforts."""
    meta = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=CandidateProfile.DYNAMIC,
        engine="mujoco",
        model_name="m",
        model_sha256="d" * 64,
        coordinate_names=("q0",),
        velocity_names=("v0",),
        actuator_names=("tau0",),
    )

    # Missing velocities v
    with pytest.raises(ValueError, match="requires velocity"):
        MatchedSwingCandidate(
            metadata=meta,
            time_s=np.array([0.0, 0.1]),
            q=np.zeros((2, 1)),
            v=None,
            tau=np.zeros((2, 1)),
        )

    # Missing efforts tau
    with pytest.raises(ValueError, match="requires actuator effort"):
        MatchedSwingCandidate(
            metadata=meta,
            time_s=np.array([0.0, 0.1]),
            q=np.zeros((2, 1)),
            v=np.zeros((2, 1)),
            tau=None,
        )


def test_candidate_tamper_detection(tmp_path: Path) -> None:
    """Verify tampering with array bytes triggers SHA-256 checksum mismatch error."""
    candidate = _make_dummy_dynamic_candidate()
    target_path = tmp_path / "tamper_test.npz"
    save_candidate(candidate, target_path)

    # Unpack, tamper with q.npy, repack
    loaded_dict = dict(np.load(target_path))
    tampered_q = loaded_dict["q"].copy()
    tampered_q[0, 0] += 1.0  # modify byte
    loaded_dict["q"] = tampered_q

    tampered_path = tmp_path / "tampered.npz"
    np.savez(tampered_path, **loaded_dict)

    with pytest.raises(
        ValueError, match="Tampering detected: checksum mismatch for 'q'"
    ):
        load_candidate(tampered_path, validate_checksums=True)


def test_candidate_rejects_unsupported_schema(tmp_path: Path) -> None:
    """Verify loading fails closed when schema_version is unsupported."""
    candidate = _make_dummy_dynamic_candidate()
    target_path = tmp_path / "unsupported_schema.npz"
    save_candidate(candidate, target_path)

    loaded_dict = dict(np.load(target_path))
    manifest_bytes = loaded_dict["manifest_json"]
    manifest_str = str(manifest_bytes)
    manifest_str = manifest_str.replace(
        CANDIDATE_SCHEMA_VERSION, "unsupported-schema-v999"
    )
    loaded_dict["manifest_json"] = np.array(manifest_str)

    bad_schema_path = tmp_path / "bad_schema.npz"
    np.savez(bad_schema_path, **loaded_dict)

    with pytest.raises(ValueError, match="Unsupported schema version"):
        load_candidate(bad_schema_path)


def test_convert_returned81_replay_lossless() -> None:
    """Verify converter faithfully translates mujoco_returned81_replay.npz without loss."""
    assert MUJOCO_REPLAY_NPZ.is_file(), f"Missing fixture: {MUJOCO_REPLAY_NPZ}"

    candidate = convert_returned81_replay(MUJOCO_REPLAY_NPZ, engine="mujoco")

    # Historical frame count is 307
    assert len(candidate.time_s) == 307
    assert candidate.metadata.profile == CandidateProfile.KINEMATIC
    assert "tau" in candidate.metadata.missing_fields
    assert candidate.model_markers_m is not None
    assert candidate.model_markers_m.shape == (307, 25, 3)
    assert candidate.target_markers_m is not None
    assert candidate.target_markers_m.shape == (307, 25, 3)
    assert candidate.marker_validity is not None
    assert candidate.marker_validity.shape == (307, 25)

    # State shape: 54 numbers -> 27 q, 27 v
    assert candidate.q.shape == (307, 27)
    assert candidate.v is not None and candidate.v.shape == (307, 27)
    assert candidate.tau is None


def test_convert_opensim_mot_lossless() -> None:
    """Verify converter faithfully translates OpenSim MOT coordinate file."""
    assert OPENSIM_MOT.is_file(), f"Missing fixture: {OPENSIM_MOT}"

    candidate = convert_opensim_mot(OPENSIM_MOT, engine="opensim")

    # Header states nRows=30, nColumns=42 (1 time + 41 coordinates)
    assert len(candidate.time_s) == 30
    assert candidate.metadata.profile == CandidateProfile.KINEMATIC
    assert candidate.q.shape == (30, 41)
    assert len(candidate.metadata.coordinate_names) == 41
    assert "TranslationInputX" in candidate.metadata.coordinate_names
    assert "hip_flexion_r" in candidate.metadata.coordinate_names
    assert candidate.v is None
    assert candidate.tau is None


def test_viewer_loads_candidate(tmp_path: Path) -> None:
    """Verify tour_matching_viewer.core.load_replay seamlessly loads candidate NPZ packages."""
    candidate = _make_dummy_dynamic_candidate(n_frames=20, nq=41, nv=41, nu=38, nm=25)
    cand_path = tmp_path / "candidate_replay.npz"
    save_candidate(candidate, cand_path)

    replay_data = load_replay(cand_path)
    assert replay_data.frame_count == 20
    np.testing.assert_allclose(replay_data.time_s, candidate.time_s)
    np.testing.assert_allclose(replay_data.coordinates, candidate.q)
    assert replay_data.model_markers_m is not None
    assert replay_data.target_markers_m is not None
    assert replay_data.valid_mask is not None


def test_candidate_schema_freshness() -> None:
    """Verify CANDIDATES.md documentation references the active schema version."""
    assert CANDIDATES_DOC.is_file(), f"Missing documentation file: {CANDIDATES_DOC}"
    content = CANDIDATES_DOC.read_text(encoding="utf-8")
    assert CANDIDATE_SCHEMA_VERSION in content
    assert "CandidateMetadata" in content
    assert "MatchedSwingCandidate" in content
    assert "kinematic" in content
    assert "dynamic" in content


def test_candidate_containers_and_immutability() -> None:
    """Verify CandidateMarkers, CandidateAuxiliary containers, properties, and writeable=False."""
    meta = CandidateMetadata(
        schema_version=CANDIDATE_SCHEMA_VERSION,
        profile=CandidateProfile.KINEMATIC,
        engine="mujoco",
        model_name="test",
        coordinate_names=("q0",),
    )
    markers = CandidateMarkers(
        model_markers_m=np.zeros((2, 1, 3)),
        target_markers_m=np.ones((2, 1, 3)),
        marker_validity=np.ones((2, 1), dtype=bool),
    )
    aux = CandidateAuxiliary(
        actuator_states=np.ones((2, 2)),
        external_forces=np.ones((2, 6)),
    )
    cand = MatchedSwingCandidate(
        metadata=meta,
        time_s=np.array([0.0, 0.1]),
        q=np.zeros((2, 1)),
        markers=markers,
        auxiliary=aux,
    )
    assert cand.markers.model_markers_m is not None
    assert cand.auxiliary.actuator_states is not None
    assert cand.model_markers_m is not None
    assert cand.target_markers_m is not None
    assert cand.marker_validity is not None
    assert cand.actuator_states is not None
    assert cand.external_forces is not None
    # Array immutability
    assert not cand.time_s.flags.writeable
    assert not cand.q.flags.writeable
    assert not cand.markers.model_markers_m.flags.writeable
    assert not cand.auxiliary.actuator_states.flags.writeable
