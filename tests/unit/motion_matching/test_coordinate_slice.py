"""Unit tests for 44-to-27 coordinate slice projection (MS-62, #10349)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.contracts import PreconditionError
from src.shared.python.motion_matching.candidate import (
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.coordinate_slice import (
    SliceMap,
    check_slice_virtual_work,
    derive_boundary_wrenches,
    load_slice_map,
    project_kinematic_trajectory,
    slice_candidate,
    workspace_overrides_from_geometry_document,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FULL_BODY_SPEC = (
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
SLICE_MAP_PATH = ROOT / "evidence/matched/driver_g1_simscape_slice/slice_map.json"


@pytest.fixture(scope="module")
def slice_map() -> SliceMap:
    return load_slice_map(SLICE_MAP_PATH)


def _make_full_body_candidate(
    coordinate_names: tuple[str, ...],
    *,
    n_frames: int = 5,
) -> MatchedSwingCandidate:
    nq = len(coordinate_names)
    time_s = np.linspace(0.0, 0.04, n_frames)
    q = np.stack(
        [np.full(n_frames, float(i) * 0.01) for i in range(nq)],
        axis=1,
    )
    v = np.cos(np.outer(time_s, np.arange(1, nq + 1) * 0.5))
    tau = np.ones((n_frames, nq)) * 0.5
    meta = CandidateMetadata(
        profile=CandidateProfile.DYNAMIC,
        engine="pinocchio",
        model_name="anthro_driver",
        coordinate_names=coordinate_names,
        velocity_names=coordinate_names,
        actuator_names=coordinate_names,
    )
    return MatchedSwingCandidate(metadata=meta, time_s=time_s, q=q, v=v, tau=tau)


def test_committed_slice_map_has_27_retained_and_17_omitted(
    slice_map: SliceMap,
) -> None:
    assert slice_map.source_coordinate_count == 44
    assert slice_map.target_coordinate_count == 27
    assert slice_map.omitted_count == 17
    assert "NeckInputX" in slice_map.omitted
    assert "hip_flexion_r" in slice_map.omitted


def test_project_kinematic_trajectory_preserves_shared_names(
    slice_map: SliceMap,
) -> None:
    candidate = _make_full_body_candidate(slice_map.source_coordinates)
    q27, v27, names27 = project_kinematic_trajectory(
        candidate.q,
        candidate.v,
        slice_map.source_coordinates,
        slice_map,
    )
    assert names27 == list(slice_map.target_coordinates)
    for name in slice_map.target_coordinates:
        src_idx = slice_map.coordinate_indices[name]
        tgt_idx = names27.index(name)
        np.testing.assert_allclose(q27[:, tgt_idx], candidate.q[:, src_idx])
        np.testing.assert_allclose(v27[:, tgt_idx], candidate.v[:, src_idx])


def test_check_slice_virtual_work_decomposes_power(slice_map: SliceMap) -> None:
    candidate = _make_full_body_candidate(slice_map.source_coordinates)
    ok, residual = check_slice_virtual_work(
        candidate.v[0],
        candidate.tau[0],
        slice_map.source_coordinates,
        slice_map,
    )
    assert ok
    assert residual == pytest.approx(0.0, abs=1e-12)


def test_derive_boundary_wrenches_accumulates_omitted_efforts(
    slice_map: SliceMap,
) -> None:
    candidate = _make_full_body_candidate(slice_map.source_coordinates)
    boundary = derive_boundary_wrenches(
        candidate.tau,
        candidate.v,
        slice_map.source_coordinates,
        slice_map,
    )
    assert boundary.shape == (candidate.q.shape[0], 6)
    assert np.any(boundary != 0.0)


def test_slice_candidate_writes_receipt(tmp_path: Path, slice_map: SliceMap) -> None:
    candidate = _make_full_body_candidate(slice_map.source_coordinates)
    out = tmp_path / "candidate27.npz"
    receipt = slice_candidate(candidate, out, slice_map=slice_map)
    assert isinstance(receipt, dict)
    assert receipt["target_coordinate_count"] == 27
    assert receipt["omitted_count"] == 17
    assert out.is_file()


def test_slice_candidate_fails_closed_on_coordinate_mismatch(
    slice_map: SliceMap,
) -> None:
    candidate = _make_full_body_candidate(("q0", "q1", "q2"))
    with pytest.raises(ValueError, match="coordinate"):
        slice_candidate(candidate, slice_map=slice_map)


def test_load_slice_map_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PreconditionError, match="slice map JSON must exist"):
        load_slice_map(tmp_path / "missing.json")


def test_workspace_overrides_from_geometry_document() -> None:
    overrides = workspace_overrides_from_geometry_document(FULL_BODY_SPEC)
    assert overrides["UpperArmLength"] > 0.0
    assert overrides["LowerArmLength"] > 0.0


def test_slice_map_rejects_overlap(tmp_path: Path, slice_map: SliceMap) -> None:
    payload = json.loads(SLICE_MAP_PATH.read_text(encoding="utf-8"))
    payload["omitted"]["HipInputX"] = {
        "reason": "bad",
        "boundary_channel": "x",
        "axis": 0,
    }
    bad_path = tmp_path / "bad_slice_map.json"
    bad_path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="overlap"):
        load_slice_map(bad_path)
