"""Registered evidence-runner contracts for issue #9125."""

from __future__ import annotations

from copy import deepcopy

import pytest

from scripts.research.proximal_distal_energy.run_event_topology_robustness import (
    ADEQUACY,
    BASE_DT_S,
    COMMON_HORIZON_S,
    REGISTERED_DELAYS_S,
    ScenarioCase,
    registered_scenarios,
    validate_report,
)

pytestmark = pytest.mark.unit


def test_registered_design_is_dimensionless_and_adequacy_gated() -> None:
    scenarios = registered_scenarios()

    assert tuple(item.name for item in scenarios) == (
        "zero",
        "fraction_0p001",
        "fraction_0p005",
        "fraction_0p01",
    )
    assert tuple(item.scale_fraction for item in scenarios) == (0.0, 0.001, 0.005, 0.01)
    assert scenarios[0].replicate_count == 4
    assert all(item.replicate_count == 192 for item in scenarios[1:])
    expected_delays = tuple(round(index * 0.02, 2) for index in range(11))
    assert expected_delays == REGISTERED_DELAYS_S
    assert 0.40 + REGISTERED_DELAYS_S[-1] == pytest.approx(COMMON_HORIZON_S)
    assert pytest.approx(300.0) == COMMON_HORIZON_S / BASE_DT_S
    assert ADEQUACY.required_independent_pairs == 96


@pytest.mark.parametrize(
    "kwargs",
    [
        {"name": "", "scale_fraction": 0.1, "replicate_count": 4},
        {"name": "bad", "scale_fraction": -0.1, "replicate_count": 4},
        {"name": "bad", "scale_fraction": 0.1, "replicate_count": 3},
    ],
)
def test_invalid_registered_scenario_fails_closed(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        ScenarioCase(**kwargs)


def test_validation_rejects_probability_fields_when_adequacy_fails(
    registered_report: dict[str, object],
) -> None:
    corrupted = deepcopy(registered_report)
    scenario = corrupted["scenarios"][0]  # type: ignore[index]
    summary = scenario["delay_summaries"][0]  # type: ignore[index]
    assert summary["adequacy_passed"] is False  # type: ignore[index]
    summary["preservation_fraction"] = 1.0  # type: ignore[index]

    with pytest.raises(ValueError, match="inadequate summary"):
        validate_report(corrupted)


@pytest.fixture(scope="module")
def registered_report() -> dict[str, object]:
    """Minimal structurally valid report without running the full campaign."""

    summaries = [
        {
            "delay_s": delay,
            "topology_counts": {"unique_transverse": 4},
            "independent_pair_count": 2,
            "preserved_pair_count": 2,
            "adequacy_passed": False,
            "preservation_fraction": None,
            "preservation_interval": None,
        }
        for delay in REGISTERED_DELAYS_S
    ]
    return {
        "schema_version": "proximal-distal-event-topology-robustness/v1",
        "source_identity": {"test_fixture": True},
        "registration": {
            "delays_s": list(REGISTERED_DELAYS_S),
            "base_dt_s": BASE_DT_S,
            "common_horizon_s": COMMON_HORIZON_S,
        },
        "scenarios": [
            {
                "name": "zero",
                "scale_fraction": 0.0,
                "replicate_count": 4,
                "delay_summaries": summaries,
            }
        ],
        "availability": {
            "human_motor_noise": "unavailable",
            "fatigue_interpretation": "unavailable",
            "controller_ranking": "suppressed",
            "coaching_recommendation": "unavailable",
        },
        "inference_boundary": (
            "Synthetic model-scenario perturbations do not establish human motor "
            "noise, fatigue, controller superiority, or coaching guidance."
        ),
    }
