"""Evidence-schema tests for the proximal-acceleration dose study."""

from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.run_proximal_acceleration_transfer_study import (
    run_study,
    write_outputs,
)

pytestmark = pytest.mark.scientific


def test_acceleration_study_covers_registered_phases_and_closes() -> None:
    record, arrays = run_study()

    assert record["schema_version"] == "proximal-acceleration-transfer-evidence-v1"
    assert record["case_count"] == 45
    assert len(record["phase_summaries"]) == 5
    assert arrays["proximal_acceleration_rad_s2"].shape == (45,)
    assert np.max(np.abs(arrays["proximal_acceleration_residual_rad_s2"])) < 1e-10
    assert np.max(np.abs(arrays["acceleration_closure_residual_rad_s2"])) < 1e-10
    assert np.max(np.abs(arrays["force_closure_residual_n"])) < 1e-10


def test_acceleration_study_preserves_claim_boundaries() -> None:
    record, _ = run_study()

    assert record["claim_status"]["pointwise_proximal_acceleration"] == "tested"
    assert record["claim_status"]["forward_acceleration_strategy"] == "untested"
    assert record["claim_status"]["human_proximal_acceleration"] == "untested"
    assert record["intervention_contract"]["state_and_kinetic_energy_matched"] is True
    assert record["intervention_contract"]["input_work_matched"] is False


def test_acceleration_artifacts_are_byte_deterministic(tmp_path) -> None:
    first = write_outputs(tmp_path / "first")
    second = write_outputs(tmp_path / "second")
    for left, right in zip(first, second, strict=True):
        assert left.read_bytes() == right.read_bytes()
