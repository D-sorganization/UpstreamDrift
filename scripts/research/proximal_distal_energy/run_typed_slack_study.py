"""Generate the registered one-class-at-a-time typed-slack evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .typed_slack import SlackParameters, energy_residual, evaluate_slack

ROOT = Path(__file__).resolve().parents[3]
OUTPUT = (
    ROOT / "docs/research/proximal_distal_energy_transfer/data/typed_slack_study.json"
)


def main() -> None:
    time = np.linspace(0.0, 0.5, 4001)
    displacement = 0.03 * np.sin(4.0 * np.pi * time)
    rate = 0.03 * 4.0 * np.pi * np.cos(4.0 * np.pi * time)
    definitions = {
        "contact_disengagement": SlackParameters(
            "contact_disengagement", 0.005, 120.0, 0.2
        ),
        "transmission_backlash": SlackParameters(
            "transmission_backlash", 0.005, 120.0, 0.2
        ),
        "structural_preload": SlackParameters(
            "structural_preload", 0.0, 120.0, 0.2, 0.004
        ),
        "biological_series_compliance": SlackParameters(
            "biological_series_compliance", 0.005, 120.0, 0.2
        ),
        "control_deadband": SlackParameters("control_deadband", 0.005, 120.0),
    }
    cases = {}
    for name, parameters in definitions.items():
        trace = evaluate_slack(displacement, rate, parameters)
        cases[name] = {
            "engaged_fraction": float(np.mean(trace.engaged)),
            "peak_abs_transmitted": float(np.max(np.abs(trace.transmitted))),
            "peak_stored_energy": float(np.max(trace.stored_energy)),
            "signed_input_work": float(np.trapezoid(trace.transmitted * rate, x=time)),
            "energy_residual": float(energy_residual(time, rate, trace)),
        }
    record = {
        "schema_version": "typed-slack-study/v1",
        "study_id": "one-class-at-a-time-scalar-slack-contract",
        "model_tier": "synthetic_scalar_constitutive_screen",
        "classes": list(definitions),
        "cases": cases,
        "claims": {
            "global_slack_benefit": "unsupported",
            "class_interchangeability": "rejected_by_state_and_energy_definition",
            "human_strategy": "untested",
        },
        "limitations": [
            "The common displacement is synthetic and not a fitted swing or tissue trajectory.",
            "Contact disengagement and biological series compliance share a unilateral scalar form here but retain different measurement and interpretation requirements.",
            "The control deadband has no elastic storage; its coefficient is an input transmission gain, not physical stiffness.",
            "No delivery-speed benefit, tissue effect, or human strategy is evaluated in this constitutive screen.",
        ],
    }
    OUTPUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(OUTPUT)


if __name__ == "__main__":
    main()
