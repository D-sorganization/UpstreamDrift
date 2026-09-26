"""Unit tests for unified cross-engine parity reporting (MS-70, #10350).

Verifies:
1. Two fake plants with a known 1 mm marker offset report exact 1 mm difference.
2. Missing / uninstalled engine records 'unavailable' status with reason without aborting.
3. Negative test: equal aggregate RMSE with divergent pointwise trajectories fails parity.
4. Parity schema serialization, validation, and round-trip.
5. Markdown table rendering with comparison classes.
6. CLI execution with --candidate and --out.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pytest

from src.shared.python.motion_matching.candidate import (
    CANDIDATE_SCHEMA_VERSION,
    CandidateAuxiliary,
    CandidateMarkers,
    CandidateMetadata,
    CandidateProfile,
    MatchedSwingCandidate,
)
from src.shared.python.motion_matching.candidate_io import save_candidate
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.full_body_ik import BaseFullBodyIK
from src.shared.python.motion_matching.parity_report import (
    build_parity_report,
    evaluate_pointwise_trajectory_parity,
    run_parity_report_cli,
)
from src.shared.python.motion_matching.parity_schema import (
    PARITY_REPORT_SCHEMA_VERSION,
    ComparisonClass,
    EngineParityRow,
    PointwiseDifference,
    UnifiedParityReport,
)
from src.shared.python.motion_matching.pipeline.plant import (
    MatchingPlant,
    register_plant,
)

pytestmark = pytest.mark.unit


class FakeMatchingPlant:
    """Fake MatchingPlant with configurable marker offset and torques for testing."""

    def __init__(
        self,
        name: str,
        marker_offset: float = 0.0,
        torque_multiplier: float = 1.0,
        divergent_trajectory: bool = False,
        marker_count: int | None = None,
    ) -> None:
        self._name = name
        self._marker_offset = marker_offset
        self._torque_multiplier = torque_multiplier
        self._divergent_trajectory = divergent_trajectory
        self._marker_count = marker_count
        self._ground_plane = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)

    @property
    def engine_name(self) -> str:
        return self._name

    @property
    def plant_sha(self) -> str:
        return f"fake-sha-{self._name}"

    @property
    def coordinate_order(self) -> tuple[str, ...]:
        return ("q0", "q1", "q2")

    @property
    def ground_plane(self) -> GroundPlane:
        return self._ground_plane

    def create_ik(
        self,
        attachments: Mapping[str, tuple[str, Sequence[float]]],
        *,
        ik_backend: str = "lm",
    ) -> BaseFullBodyIK:
        raise NotImplementedError("Not needed for parity test")  # tracked: #10350

    def frame_poses(
        self, mapping: Mapping[str, tuple[str, Sequence[float]]], q: np.ndarray
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        return {}

    def marker_positions(
        self, q: np.ndarray, attachments: Mapping[str, tuple[str, Sequence[float]]]
    ) -> np.ndarray:
        # Base markers of shape (frames, markers, 3)
        n_frames = len(q)
        if self._marker_count is not None:
            n_markers = self._marker_count
        else:
            n_markers = len(attachments) if attachments else 4
        markers = np.zeros((n_frames, n_markers, 3), dtype=np.float64)
        if self._divergent_trajectory:
            # Alternate sign across frames so mean is 0 but distance is large
            signs = np.where(np.arange(n_frames) % 2 == 0, 1.0, -1.0)[:, None, None]
            markers += signs * 0.05
        else:
            markers[:, :, 0] += self._marker_offset
        return markers

    def contact_forces(
        self, coordinates: Mapping[str, float], rates: Mapping[str, float]
    ) -> Any:
        return {"total_normal_force_n": 800.0, "total_friction_force_n": 20.0}

    def closure_residuals(self, q: np.ndarray) -> np.ndarray:
        return np.zeros(6, dtype=np.float64)

    def step(
        self, q: np.ndarray, v: np.ndarray, tau: np.ndarray, dt: float
    ) -> tuple[np.ndarray, np.ndarray]:
        return q + v * dt, v

    def accelerations(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> Mapping[str, float]:
        return dict.fromkeys(coordinates, 0.1)

    def acceleration_derivatives(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> Any:
        return None

    def contact_effort_derivatives(
        self, coordinates: Mapping[str, float], rates: Mapping[str, float]
    ) -> Any:
        return None


def _make_sample_candidate(tmp_path: Path) -> Path:
    """Create a valid MatchedSwingCandidate package for testing."""
    n_frames = 20
    time_s = np.linspace(0.0, 0.85, n_frames)
    q = np.zeros((n_frames, 3), dtype=np.float64)
    v = np.ones((n_frames, 3), dtype=np.float64) * 0.1
    tau = np.ones((n_frames, 3), dtype=np.float64) * 10.0
    markers = np.zeros((n_frames, 4, 3), dtype=np.float64)

    cand = MatchedSwingCandidate(
        metadata=CandidateMetadata(
            schema_version=CANDIDATE_SCHEMA_VERSION,
            profile=CandidateProfile.DYNAMIC,
            engine="fake_engine_a",
            model_name="test_model",
            model_sha256="test-sha-1234",
            coordinate_names=("q0", "q1", "q2"),
            velocity_names=("v0", "v1", "v2"),
            actuator_names=("tau0", "tau1", "tau2"),
            marker_names=("m0", "m1", "m2", "m3"),
            extra={"total_work_j": 42.5},
        ),
        time_s=time_s,
        q=q,
        v=v,
        tau=tau,
        markers=CandidateMarkers(
            model_markers_m=markers,
            target_markers_m=markers,
            marker_validity=np.ones((n_frames, 4), dtype=bool),
        ),
    )
    p = tmp_path / "candidate.npz"
    save_candidate(cand, p)
    return p


def test_two_plants_known_offset(tmp_path: Path) -> None:
    """MS-70 TDD step 1: Two fake plants with a known 1 mm marker offset report exact 1 mm diff."""
    cand_path = _make_sample_candidate(tmp_path)

    plant_a = FakeMatchingPlant("fake_a", marker_offset=0.000)
    plant_b = FakeMatchingPlant("fake_b", marker_offset=0.001)  # 1 mm offset

    report = build_parity_report(
        candidate=cand_path,
        plants={"fake_a": plant_a, "fake_b": plant_b},
        reference_engine="fake_a",
    )

    assert isinstance(report, UnifiedParityReport)
    assert "fake_a" in report.engine_rows
    assert "fake_b" in report.engine_rows

    row_b = report.engine_rows["fake_b"]
    assert row_b.status == "qualified"
    assert "marker_diff_m" in row_b.pointwise_differences

    marker_diff = row_b.pointwise_differences["marker_diff_m"]
    assert pytest.approx(marker_diff.max_abs_diff, abs=1e-6) == 0.001
    assert pytest.approx(marker_diff.rms_diff, abs=1e-6) == 0.001
    assert marker_diff.pass_gate is True  # 1 mm meets the 1 mm target


def test_missing_engine_records_unavailable(tmp_path: Path) -> None:
    """MS-70 TDD step 2: Missing engine -> row 'unavailable' with reason, report still written."""
    cand_path = _make_sample_candidate(tmp_path)
    plant_a = FakeMatchingPlant("fake_a")

    report = build_parity_report(
        candidate=cand_path,
        plants={"fake_a": plant_a},
        engines=("fake_a", "nonexistent_engine"),
        reference_engine="fake_a",
    )

    assert isinstance(report, UnifiedParityReport)
    assert "nonexistent_engine" in report.engine_rows
    unavail_row = report.engine_rows["nonexistent_engine"]
    assert unavail_row.status == "unavailable"
    assert (
        "not installed" in unavail_row.reason.lower()
        or "not registered" in unavail_row.reason.lower()
    )
    # Report itself completed and was written
    assert len(report.engine_rows) == 2


def test_negative_equal_aggregate_different_trajectories_fails() -> None:
    """MS-70 acceptance criteria: Equal aggregate RMSE with different trajectories fails parity."""
    # Construct two trajectories where aggregate RMSE vs ground truth is identical:
    # Trajectory 1: constant +0.02 m offset
    # Trajectory 2: constant -0.02 m offset
    # Both have aggregate RMSE = 0.02 m vs ground truth (zeros).
    # However, pointwise difference between Trajectory 1 and Trajectory 2 is 0.04 m, which exceeds 1 mm!
    n_frames = 10
    n_markers = 4
    traj_1 = np.zeros((n_frames, n_markers, 3))
    traj_2 = np.zeros((n_frames, n_markers, 3))
    traj_1[..., 0] = 0.02
    traj_2[..., 0] = -0.02

    rmse_1 = float(np.sqrt(np.mean(traj_1**2)))
    rmse_2 = float(np.sqrt(np.mean(traj_2**2)))
    assert pytest.approx(rmse_1) == pytest.approx(rmse_2)  # identical aggregate RMSE!

    diff = evaluate_pointwise_trajectory_parity(traj_1, traj_2, tolerance_m=0.001)
    assert diff.pass_gate is False
    assert pytest.approx(diff.max_abs_diff) == 0.04
    assert pytest.approx(diff.rms_diff) == 0.04


def test_parity_schema_serialization_round_trip() -> None:
    """MS-70 TDD step 3: Schema freshness and JSON round-trip."""
    diff = PointwiseDifference(
        metric_name="marker_pos_m",
        max_abs_diff=0.0005,
        rms_diff=0.0003,
        mean_diff=0.0002,
        unit="m",
        pass_gate=True,
    )
    row = EngineParityRow(
        engine="mujoco",
        status="qualified",
        comparison_class=ComparisonClass.SAME_MODEL_NUMERICAL_PARITY,
        model_name="golf_humanoid",
        model_sha256="abc1234",
        assumptions={"mass_kg": 80.0, "integrator": "rk4"},
        pointwise_differences={"marker_pos_m": diff},
        total_work_J=45.0,
        wall_clock_s=0.12,
        reason="",
    )
    report = UnifiedParityReport(
        schema_version=PARITY_REPORT_SCHEMA_VERSION,
        candidate_id="cand-001",
        candidate_sha256="candsha-001",
        reference_engine="pinocchio",
        reference_model_sha256="pin-sha",
        created_at="2026-09-20T12:00:00Z",
        is_parity_accepted=True,
        status="PASSED",
        engine_rows={"mujoco": row},
        pairwise_comparisons={},
    )

    d = report.to_dict()
    assert d["schema_version"] == PARITY_REPORT_SCHEMA_VERSION
    assert d["engine_rows"]["mujoco"]["engine"] == "mujoco"
    assert d["is_parity_accepted"] is True

    json_str = report.to_json()
    assert "mujoco" in json_str
    assert "is_parity_accepted" in json_str

    reloaded = UnifiedParityReport.from_dict(json.loads(json_str))
    assert reloaded.schema_version == PARITY_REPORT_SCHEMA_VERSION
    assert reloaded.engine_rows["mujoco"].engine == "mujoco"
    assert reloaded.engine_rows["mujoco"].total_work_J == 45.0
    assert reloaded.is_parity_accepted is True
    assert (
        reloaded.engine_rows["mujoco"]
        .pointwise_differences["marker_pos_m"]
        .max_abs_diff
        == 0.0005
    )


def test_markdown_rendering_contains_classes_and_rows() -> None:
    """MS-70: render_markdown() outputs formatted tables with all comparison classes."""
    row_same = EngineParityRow(
        engine="mujoco",
        status="qualified",
        comparison_class=ComparisonClass.SAME_MODEL_NUMERICAL_PARITY,
        model_name="golf_humanoid",
        model_sha256="abc1234",
        pointwise_differences={
            "marker_diff_m": PointwiseDifference(
                metric_name="marker_diff_m",
                max_abs_diff=0.0008,
                rms_diff=0.0005,
                mean_diff=0.0004,
                unit="m",
                pass_gate=True,
            )
        },
        total_work_J=42.0,
        wall_clock_s=0.15,
    )
    row_native = EngineParityRow(
        engine="opensim",
        status="unavailable",
        comparison_class=ComparisonClass.NATIVE_MODEL_OBSERVABLE_AGREEMENT,
        model_name="golf_humanoid.osim",
        model_sha256="osim123",
        reason="OpenSim SDK not installed in active environment",
    )
    report = UnifiedParityReport(
        schema_version=PARITY_REPORT_SCHEMA_VERSION,
        candidate_id="test_candidate",
        candidate_sha256="sha-cand-test",
        reference_engine="pinocchio",
        reference_model_sha256="sha-pin",
        created_at="2026-09-20T12:00:00Z",
        is_parity_accepted=False,
        status="PARTIAL",
        engine_rows={"mujoco": row_same, "opensim": row_native},
    )

    md = report.render_markdown()
    assert "# Unified Cross-Engine Parity Report" in md
    assert "Same-Model Numerical Parity" in md
    assert "Native-Model Observable Agreement" in md
    assert "mujoco" in md
    assert "opensim" in md
    assert "unavailable" in md


def test_cli_execution(tmp_path: Path) -> None:
    """MS-70: CLI --candidate and --out generates parity_report.json and parity_report.md."""
    cand_path = _make_sample_candidate(tmp_path)
    out_dir = tmp_path / "output"

    plant_a = FakeMatchingPlant("mujoco")
    exit_code = run_parity_report_cli(
        [
            "--candidate",
            str(cand_path),
            "--out",
            str(out_dir),
            "--reference-engine",
            "mujoco",
        ],
        plants={"mujoco": plant_a},
    )

    assert exit_code == 0
    json_path = out_dir / "parity_report.json"
    md_path = out_dir / "parity_report.md"
    assert json_path.is_file()
    assert md_path.is_file()

    data = json.loads(json_path.read_text(encoding="utf-8"))
    assert data["schema_version"] == PARITY_REPORT_SCHEMA_VERSION
    assert "mujoco" in data["engine_rows"]


def test_empty_row_set_overall_status_partial() -> None:
    """Empty row set must evaluate to PARTIAL, never PASSED (P1-7, #10960)."""
    from src.shared.python.motion_matching.parity_report import (
        _determine_overall_status,
    )

    status = _determine_overall_status({})
    assert status == "PARTIAL"


def test_missing_reference_sets_unverified(tmp_path: Path) -> None:
    """Missing reference engine markers -> row status 'unverified', never 'qualified' (P1-7, #10960)."""
    cand_path = _make_sample_candidate(tmp_path)
    plant_b = FakeMatchingPlant("fake_b", marker_offset=0.0)

    report = build_parity_report(
        candidate=cand_path,
        plants={"fake_b": plant_b},
        engines=("fake_b",),
        reference_engine="missing_ref",
    )

    row_b = report.engine_rows["fake_b"]
    assert row_b.status == "unverified"
    assert "unavailable" in row_b.reason.lower() or "missing" in row_b.reason.lower()
    assert report.is_parity_accepted is False


def test_shape_mismatch_sets_unverified(tmp_path: Path) -> None:
    """Shape mismatch against reference -> row status 'unverified', never 'qualified' (P1-7, #10960)."""
    cand_path = _make_sample_candidate(tmp_path)
    plant_ref = FakeMatchingPlant("fake_ref", marker_count=4)
    plant_b = FakeMatchingPlant("fake_b", marker_count=6)

    report = build_parity_report(
        candidate=cand_path,
        plants={"fake_ref": plant_ref, "fake_b": plant_b},
        engines=("fake_ref", "fake_b"),
        reference_engine="fake_ref",
    )

    row_b = report.engine_rows["fake_b"]
    assert row_b.status == "unverified"
    assert "shape mismatch" in row_b.reason.lower()
    assert report.is_parity_accepted is False


def test_ungated_engine_class_sets_unverified(tmp_path: Path) -> None:
    """Native-model observable agreement engines are never gated -> 'unverified' (P1-7, #10960)."""
    cand_path = _make_sample_candidate(tmp_path)
    plant_ref = FakeMatchingPlant("fake_ref")
    plant_native = FakeMatchingPlant("opensim")  # In NATIVE_MODEL_ENGINES

    report = build_parity_report(
        candidate=cand_path,
        plants={"fake_ref": plant_ref, "opensim": plant_native},
        engines=("fake_ref", "opensim"),
        reference_engine="fake_ref",
    )

    row_native = report.engine_rows["opensim"]
    assert row_native.status == "unverified"
    assert (
        "not gated" in row_native.reason.lower()
        or "never gated" in row_native.reason.lower()
    )


def test_drop_model_markers_fallback_no_self_comparison(tmp_path: Path) -> None:
    """Drop model_markers_m fallback: candidate markers must NOT serve as reference (P1-7, #10960)."""
    cand_path = _make_sample_candidate(tmp_path)
    plant_b = FakeMatchingPlant("fake_b")

    # Candidate has model_markers_m populated, but reference_engine 'nonexistent' has no plant
    report = build_parity_report(
        candidate=cand_path,
        plants={"fake_b": plant_b},
        engines=("fake_b",),
        reference_engine="nonexistent",
    )

    row_b = report.engine_rows["fake_b"]
    assert row_b.status == "unverified"
    assert "unavailable" in row_b.reason.lower() or "missing" in row_b.reason.lower()
    assert report.is_parity_accepted is False


def test_pass_all_gates_false_initially_and_diverged_rejected(tmp_path: Path) -> None:
    """pass_all_gates starts False and diverged trajectory is rejected (P1-7, #10960)."""
    cand_path = _make_sample_candidate(tmp_path)
    plant_ref = FakeMatchingPlant("fake_ref")
    plant_diverged = FakeMatchingPlant(
        "fake_diverged", marker_offset=0.05
    )  # 50 mm > 1 mm tolerance

    report = build_parity_report(
        candidate=cand_path,
        plants={"fake_ref": plant_ref, "fake_diverged": plant_diverged},
        engines=("fake_ref", "fake_diverged"),
        reference_engine="fake_ref",
    )

    row_div = report.engine_rows["fake_diverged"]
    assert row_div.status == "rejected"
    assert "diverged" in row_div.reason.lower()
    assert report.status == "REJECTED"
    assert report.is_parity_accepted is False
