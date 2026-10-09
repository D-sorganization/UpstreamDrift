"""Original native force/length counterexample and fail-closed audits (#11856)."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.tour_matching import (
    compute_path_length_finite_difference_moment_arm,
    validate_moment_arm_consistency,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_path_length_cannot_pass_gate(bad: float) -> None:
    with pytest.raises((ValueError, AssertionError)):
        validate_moment_arm_consistency(0.1, lambda q: bad, 0.0)


@pytest.mark.parametrize("tol", [math.nan, math.inf, -1.0, 0.0])
def test_invalid_tolerance_cannot_pass_gate(tol: float) -> None:
    with pytest.raises((ValueError, AssertionError)):
        validate_moment_arm_consistency(0.1, lambda q: 1.0 - q * 0.1, 0.0, tol)


@pytest.mark.parametrize(
    "q,dq", [(math.nan, 1e-5), (0.0, math.inf), (1e20, 1e-5), (0.0, 1e308)]
)
def test_invalid_or_unresolved_stencil_rejected(q: float, dq: float) -> None:
    with pytest.raises((ValueError, AssertionError)):
        compute_path_length_finite_difference_moment_arm(lambda value: 1.0, q, dq)


def test_original_native_probe_records_workless_identity_failure(
    tmp_path: Path,
) -> None:
    pytest.importorskip("opensim")
    from scripts.diagnostics.native_path_work import TOLERANCE_M, run_probe

    receipt = run_probe(tmp_path)
    assert receipt["scientific_status"] == "unqualified"
    assert receipt["source_binary_equivalence"] == "unverified"
    assert receipt["runtime"]["version"]
    assert receipt["runtime"]["extension_sha256"]
    fixed, moving = receipt["cases"]
    assert fixed["gate"] == "workless_identity_satisfied_at_tested_samples"
    assert moving["gate"] == "workless_identity_not_satisfied"
    assert (
        receipt["interpretation"] == "workless-identity-audit-not-runtime-defect-proof"
    )
    assert moving["guide_work_model"] == "unmodeled-moving-guide"
    for case in (fixed, moving):
        assert len(case["source_sha256"]) == 64
        assert len(case["loaded_sha256"]) == 64
        assert case["source_unchanged"] is True
        assert case["loaded_unchanged"] is True
        assert (case["native_nq"], case["native_nu"], case["native_nz"]) == (1, 1, 0)
        assert case["native_constraints"] == case["native_controllers"] == 0
        assert len(case["samples"]) == 9
        for row in case["samples"]:
            assert row["native_tension_n"] == pytest.approx(1.0)
            assert row["native_moment_arm_m"] == pytest.approx(
                row["native_acceleration_rad_s2"] * case["inertia_kg_m2"], abs=1e-11
            )
            assert row["coordinate_span_rad"] == pytest.approx(2 * row["step_rad"])
            assert row["full_minus_native_m"] == pytest.approx(
                row["same_body_segment_arm_m"], abs=TOLERANCE_M
            )
            assert row["length_speed_m_s"] == pytest.approx(
                -row["full_length_arm_m"], abs=TOLERANCE_M
            )
            if case["moving"]:
                assert row["full_minus_native_m"] == pytest.approx(
                    0.03, abs=TOLERANCE_M
                )
                assert row["gate_rejected"] is True
            else:
                assert abs(row["full_minus_native_m"]) < TOLERANCE_M
                assert row["gate_rejected"] is False
            if row["step_rad"] == 1e-5:
                assert row["full_minus_native_m"] == pytest.approx(
                    0.03 if case["moving"] else 0.0, abs=1e-10
                )


@pytest.mark.parametrize("level", ["warn", "off"])
def test_evidence_gate_cannot_be_disabled(
    monkeypatch: pytest.MonkeyPatch, level: str
) -> None:
    monkeypatch.setenv("DBC_LEVEL", level)
    with pytest.raises(ValueError, match="finite"):
        validate_moment_arm_consistency(0.1, lambda q: math.nan, 0.0)


def test_finite_difference_overflow_rejected() -> None:
    with pytest.raises(ValueError, match="finite-difference moment arm"):
        compute_path_length_finite_difference_moment_arm(
            lambda q: 1e308 if q > 0 else -1e308, 0.0
        )
