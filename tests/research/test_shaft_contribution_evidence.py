"""Regression tests for the recorded shaft-contribution evidence package."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

RECORD = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "research"
    / "proximal_distal_energy_transfer"
    / "data"
    / "shaft_contribution_study.json"
)


@pytest.fixture(scope="module")
def evidence() -> dict:
    with RECORD.open(encoding="utf-8") as stream:
        return json.load(stream)


def _variant(evidence: dict, name: str) -> dict:
    return next(row for row in evidence["variant_summaries"] if row["name"] == name)


def test_matched_rigid_and_flexible_reference_cases_both_deliver(
    evidence: dict,
) -> None:
    flexible = _variant(evidence, "flexible_reference")
    rigid = _variant(evidence, "rigid_matched")

    assert flexible["impact"]["impact_found"]
    assert rigid["impact"]["impact_found"]
    assert flexible["impact"]["impact_speed_m_s"] - rigid["impact"][
        "impact_speed_m_s"
    ] == pytest.approx(0.1083367610145265)
    assert flexible["impact"]["shaft_flex_at_impact_deg"] == pytest.approx(
        -3.581146177759325
    )


def test_energy_and_dissipation_contracts_close(evidence: dict) -> None:
    for name in (
        "flexible_reference",
        "rigid_matched",
        "gravity_disabled",
        "joint_damping_disabled",
        "shaft_damping_disabled",
    ):
        row = _variant(evidence, name)
        assert row["energy"]["maximum_abs_energy_closure_error_j"] < 0.07
        assert row["energy"]["shaft_damping_dissipation_j"] <= 1e-12
        assert row["energy"]["joint_damping_dissipation_j"] <= 1e-12


def test_ablations_remove_only_their_declared_terms(evidence: dict) -> None:
    gravity_off = _variant(evidence, "gravity_disabled")
    joint_damping_off = _variant(evidence, "joint_damping_disabled")
    shaft_damping_off = _variant(evidence, "shaft_damping_disabled")

    assert gravity_off["terms"]["gravity"]["peak_abs_generalized_torque_nm"] == 0.0
    assert (
        joint_damping_off["terms"]["joint_damping"]["peak_abs_generalized_torque_nm"]
        == 0.0
    )
    assert (
        shaft_damping_off["terms"]["shaft_damping"]["peak_abs_generalized_torque_nm"]
        == 0.0
    )


def test_timestep_refinement_converges_impact_metrics(evidence: dict) -> None:
    rows = {row["dt_s"]: row for row in evidence["timestep_rows"]}
    fine = rows[0.00025]["impact"]

    assert abs(rows[0.0005]["impact"]["impact_time_s"] - fine["impact_time_s"]) < 5e-5
    assert (
        abs(rows[0.0005]["impact"]["impact_speed_m_s"] - fine["impact_speed_m_s"])
        < 0.006
    )
    assert (
        abs(rows[0.001]["impact"]["impact_speed_m_s"] - fine["impact_speed_m_s"]) < 0.02
    )


def test_robustness_grid_covers_every_declared_combination(evidence: dict) -> None:
    grid = evidence["robustness_grid"]
    expected = (
        len(grid["stiffness_values_nm_rad"])
        * len(grid["damping_values_nms_rad"])
        * len(grid["cut_times_s"])
    )

    assert expected == 120
    assert len(grid["rows"]) == expected
    assert all(row["impact_found"] for row in grid["rows"])
