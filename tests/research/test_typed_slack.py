from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.typed_slack import (
    SlackParameters,
    energy_residual,
    evaluate_slack,
)

pytestmark = pytest.mark.scientific


def test_each_slack_class_has_distinct_engagement_and_energy_state() -> None:
    x = np.array([-0.02, 0.0, 0.02, 0.04])
    xd = np.gradient(x, 0.01)
    contact = evaluate_slack(
        x, xd, SlackParameters("contact_disengagement", 0.01, 100.0)
    )
    backlash = evaluate_slack(
        x, xd, SlackParameters("transmission_backlash", 0.01, 100.0)
    )
    preload = evaluate_slack(
        x, xd, SlackParameters("structural_preload", 0.0, 100.0, preload=0.01)
    )
    biological = evaluate_slack(
        x, xd, SlackParameters("biological_series_compliance", 0.01, 100.0)
    )
    control = evaluate_slack(x, xd, SlackParameters("control_deadband", 0.01, 100.0))

    assert not contact.engaged[0] and backlash.engaged[0]
    assert np.all(preload.engaged)
    assert np.array_equal(contact.engaged, biological.engaged)
    assert np.all(control.stored_energy == 0.0)
    assert np.any(backlash.stored_energy > 0.0)
    assert np.all(control.stored_energy == 0.0)
    assert np.any(backlash.elastic != 0.0) and np.all(control.elastic == 0.0)


@pytest.mark.parametrize(
    "kind",
    [
        "contact_disengagement",
        "transmission_backlash",
        "structural_preload",
        "biological_series_compliance",
        "control_deadband",
    ],
)
def test_energy_ledger_closes_under_refinement(kind: str) -> None:
    residuals = []
    for count in (1001, 4001):
        time = np.linspace(0.0, 1.0, count)
        x = 0.03 * np.sin(0.5 * np.pi * time)
        xd = 0.03 * 0.5 * np.pi * np.cos(0.5 * np.pi * time)
        params = SlackParameters(
            kind,
            0.005,
            120.0,
            damping=0.2,
            preload=0.004 if kind == "structural_preload" else 0.0,
        )
        trace = evaluate_slack(x, xd, params)
        residuals.append(abs(energy_residual(time, xd, trace)))
    assert residuals[1] <= residuals[0]
    assert residuals[1] < 2e-6


def test_preload_cannot_leak_into_another_slack_class() -> None:
    with pytest.raises(ValueError, match="only for structural_preload"):
        SlackParameters("transmission_backlash", 0.01, 100.0, preload=0.01)
