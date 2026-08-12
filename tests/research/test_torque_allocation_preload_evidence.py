"""Machine-readable evidence contracts for torque allocation and preload."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[2]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
pytestmark = pytest.mark.scientific


def test_summary_closes_matched_task_and_preserves_claim_boundaries() -> None:
    record = json.loads(
        (ARTICLE / "data/torque_allocation_preload_study.json").read_text()
    )
    assert record["schema_version"] == "1.0.0"
    assert record["epic"].endswith("/8497")
    assert "neither identifies muscles" in record["claim_boundary"]
    assert record["matched_task"]["maximum_task_error_nm"] < 1e-10
    assert record["matched_task"]["maximum_moment_closure_error_nm"] < 1e-10
    assert record["registered_hypotheses"]["RA-H5"].startswith("No universal")


def test_trace_archive_reproduces_reported_transmission_metrics() -> None:
    record = json.loads(
        (ARTICLE / "data/torque_allocation_preload_study.json").read_text()
    )
    with np.load(
        ARTICLE / "data/torque_allocation_preload_study.npz", allow_pickle=False
    ) as arrays:
        np.testing.assert_allclose(arrays["net_control_moment_nm"], 8.0, atol=1e-10)
        np.testing.assert_allclose(
            arrays["direct_wrist_moment_nm"] + arrays["grip_force_couple_nm"],
            arrays["net_control_moment_nm"],
            atol=1e-10,
        )
        proposed = record["transmission_results"]["persistent_arm_drive_preloaded"]
        opposite = record["transmission_results"][
            "wrist_to_arm_role_reversal_preloaded"
        ]
        assert proposed["arm_zero_transmission_duration_s"] == pytest.approx(0.0)
        assert proposed["wrist_zero_transmission_duration_s"] == pytest.approx(0.0)
        assert opposite["arm_zero_transmission_duration_s"] > 0.0
        assert opposite["wrist_zero_transmission_duration_s"] > 0.0
        assert (
            proposed["net_torque_error_impulse_nms"]
            < opposite["net_torque_error_impulse_nms"]
        )


def test_publication_figures_exist_in_pdf_and_svg() -> None:
    for stem in (
        "fig_torque_allocation_geometry_surface",
        "fig_torque_allocation_moment_closure",
        "fig_torque_role_reversal_transmission",
    ):
        for suffix in (".pdf", ".svg"):
            path = ARTICLE / "figures" / f"{stem}{suffix}"
            assert path.is_file()
            assert path.stat().st_size > 1000
