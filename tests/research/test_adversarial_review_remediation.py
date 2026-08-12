"""Evidence-contract tests for adversarial-review remediation epic #8499."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytestmark = pytest.mark.scientific


DATA = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "research"
    / "proximal_distal_energy_transfer"
    / "data"
)


def _read_json(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def test_primary_sweep_retains_every_candidate_and_status() -> None:
    rows = _read_json("e1_sweep.json")["rows"]
    assert len(rows) == 92
    assert all("impact_status" in row for row in rows)
    assert all("candidate_t_crossing_s" in row for row in rows)

    counts = {
        status: sum(row["impact_status"] == status for row in rows)
        for status in {
            "accepted_registered_delivery_zone",
            "crossing_outside_registered_delivery_zone",
            "no_club_vertical_crossing",
        }
    }
    assert counts == {
        "accepted_registered_delivery_zone": 63,
        "crossing_outside_registered_delivery_zone": 29,
        "no_club_vertical_crossing": 0,
    }
    assert all(row["candidate_t_crossing_s"] is not None for row in rows)


def test_impact_bound_counts_close_and_winners_are_stable() -> None:
    evidence = _read_json("e1c_sensitivity.json")
    winners: dict[str, set[str]] = {"shoulder_60": set(), "shoulder_100": set()}
    accepted_totals: list[int] = []

    for shoulder in winners:
        sensitivity = evidence["results"][shoulder]["c5_validity_rule_sensitivity"]
        for result in sensitivity.values():
            counts = result["status_counts"]
            assert sum(counts.values()) == result["attempted_programs"] == 46
            accepted_totals.append(counts["accepted"])
            winners[shoulder].add(result["winner"])

    assert min(accepted_totals) == 28
    assert max(accepted_totals) == 37
    assert all(len(values) == 1 for values in winners.values())


def test_command_rise_study_is_complete_and_reports_every_ordering() -> None:
    evidence = _read_json("e1e_smooth_command_sensitivity.json")
    assert evidence["time_constants_s"] == [0.0, 0.02, 0.035, 0.05]
    assert len(evidence["rows"]) == 4 * 92

    summaries = evidence["summaries"]
    assert set(summaries) == {"0_ms", "20_ms", "35_ms", "50_ms"}
    for summary in summaries.values():
        for shoulder in ("shoulder_60", "shoulder_100"):
            assert summary[shoulder]["attempted_programs"] == 46
            assert summary[shoulder]["registered_ordering_preserved"] is True
