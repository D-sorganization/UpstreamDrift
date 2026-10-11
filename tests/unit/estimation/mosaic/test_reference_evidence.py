"""The committed evidence receipt must match a fresh deterministic run."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.reference_evidence import (
    RECEIPT_PATH,
    run_reference,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
RELATIVE_TOLERANCE = 0.05


def test_committed_receipt_matches_fresh_run() -> None:
    committed = json.loads((ROOT / RECEIPT_PATH).read_text(encoding="utf-8"))
    report, fresh = run_reference()

    assert fresh["fixture"] == committed["fixture"]
    assert fresh["structural_rank"] == committed["structural_rank"] == 9
    assert (
        fresh["unobservable_parameters"]
        == committed["unobservable_parameters"]
        == ["m_0"]
    )
    assert (
        fresh["outer_iterations_per_stage"] == committed["outer_iterations_per_stage"]
    )
    assert fresh["all_stages_converged"] and committed["all_stages_converged"]
    assert report.open_loop_accepted

    scalar_keys = (
        "geometry_error_identifiable_m",
        "torque_rms_error",
        "torque_rms_scale",
        "inertia_max_relative_error_observable",
        "inertia_max_relative_error_prior",
    )
    for key in scalar_keys:
        np.testing.assert_allclose(
            fresh[key], committed[key], rtol=RELATIVE_TOLERANCE, atol=1e-9
        )
    np.testing.assert_allclose(
        fresh["kinematic_rms_rad"],
        committed["kinematic_rms_rad"],
        rtol=RELATIVE_TOLERANCE,
    )
    for fresh_replay, committed_replay in zip(
        fresh["replay"], committed["replay"], strict=True
    ):
        for key in ("open_loop_rms_rad", "closed_loop_rms_rad", "feedback_effort_rms"):
            np.testing.assert_allclose(
                fresh_replay[key], committed_replay[key], rtol=RELATIVE_TOLERANCE
            )
        assert fresh_replay["divergence_time_s"] is None
    # the document's headline claims, checked against the receipt itself
    assert committed["geometry_error_identifiable_m"] < 1e-4
    assert committed["torque_rms_error"] / committed["torque_rms_scale"] < 0.25
    assert (
        committed["inertia_max_relative_error_observable"]
        < committed["inertia_max_relative_error_prior"]
    )
    assert all(r["open_loop_rms_rad"] < 0.05 for r in committed["replay"])
