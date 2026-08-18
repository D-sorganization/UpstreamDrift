from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.typed_slack import SlackParameters
from scripts.research.proximal_distal_energy.typed_slack_dynamic import (
    DynamicSlackParameters,
    excitation,
    scaled_sensitivity_audit,
    simulate_dynamic_slack,
)

pytestmark = pytest.mark.scientific


def _parameters(kind: str) -> DynamicSlackParameters:
    return DynamicSlackParameters(
        constitutive=SlackParameters(
            kind,
            0.004,
            120.0,
            damping=0.25,
            preload=0.003 if kind == "structural_preload" else 0.0,
        ),
        time_constant_s=0.018,
    )


def test_mechanical_classes_close_energy_and_remain_passive() -> None:
    time = np.linspace(0.0, 1.0, 4001)
    signal = excitation(time, "multisine_reversal")
    for kind in (
        "contact_disengagement",
        "transmission_backlash",
        "structural_preload",
        "biological_series_compliance",
    ):
        result = simulate_dynamic_slack(time, signal, _parameters(kind))
        assert result.passivity_applicable
        assert result.input_work_j >= -2e-5
        assert abs(result.energy_residual_j) < 2e-5
        assert result.loop_area_j >= -2e-5


def test_control_deadband_is_not_relabelled_as_passive_mechanics() -> None:
    time = np.linspace(0.0, 1.0, 4001)
    signal = excitation(time, "multisine_reversal")
    result = simulate_dynamic_slack(time, signal, _parameters("control_deadband"))
    assert not result.passivity_applicable
    assert np.all(result.stored_energy_j == 0.0)
    assert result.activation_delay_s > 0.0


def test_identifiability_audit_is_scaled_and_excitation_specific() -> None:
    time = np.linspace(0.0, 1.0, 2001)
    parameters = _parameters("transmission_backlash")
    slow = scaled_sensitivity_audit(time, excitation(time, "slow_sine"), parameters)
    rich = scaled_sensitivity_audit(
        time,
        excitation(time, "multisine_reversal"),
        parameters,
    )
    assert slow.parameter_names == ("threshold", "stiffness", "damping")
    assert 0 <= slow.rank <= len(slow.parameter_names)
    assert rich.rank >= slow.rank
    assert rich.minimum_scaled_singular_value >= 0.0
