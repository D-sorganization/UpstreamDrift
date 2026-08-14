"""Generate dynamic, passivity, and identifiability evidence for typed slack."""

from __future__ import annotations

import json
from dataclasses import asdict
from itertools import combinations
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .typed_slack import SlackParameters
from .typed_slack_dynamic import (
    DynamicSlackParameters,
    excitation,
    scaled_sensitivity_audit,
    simulate_dynamic_slack,
)

matplotlib.use("Agg")
plt.rcParams.update(
    {
        "pdf.use14corefonts": True,
        "font.family": "sans-serif",
        "axes.unicode_minus": False,
    }
)

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
JSON_OUTPUT = ARTICLE / "data/typed_slack_dynamic_study.json"
NPZ_OUTPUT = ARTICLE / "data/typed_slack_dynamic_study.npz"
FIGURE_OUTPUT = ARTICLE / "figures/fig_typed_slack_dynamic_audit.pdf"

KINDS = (
    "contact_disengagement",
    "transmission_backlash",
    "structural_preload",
    "biological_series_compliance",
    "control_deadband",
)
EXCITATIONS = ("slow_sine", "multisine_reversal")
MECHANICAL_KINDS = KINDS[:-1]


def _parameters(kind: str) -> DynamicSlackParameters:
    return DynamicSlackParameters(
        constitutive=SlackParameters(
            kind,
            threshold=0.004,
            stiffness=120.0,
            damping=0.25,
            preload=0.003 if kind == "structural_preload" else 0.0,
        ),
        time_constant_s=0.018,
    )


def _distance(left: np.ndarray, right: np.ndarray) -> float:
    scale = max(
        float(np.sqrt(np.mean(left**2))), float(np.sqrt(np.mean(right**2))), 1e-12
    )
    return float(np.sqrt(np.mean((left - right) ** 2)) / scale)


def _render_figure(
    time: np.ndarray,
    signals: dict[str, np.ndarray],
    traces: dict[str, dict[str, np.ndarray]],
    audits: dict[str, dict[str, dict[str, object]]],
) -> None:
    rich = "multisine_reversal"
    figure, axes = plt.subplots(2, 2, figsize=(10.5, 7.4), constrained_layout=True)
    for kind in MECHANICAL_KINDS:
        axes[0, 0].plot(signals[rich], traces[rich][kind], label=kind.replace("_", " "))
    axes[0, 0].set_title("Mechanical Force-Displacement Loops")
    axes[0, 0].set_xlabel("Synthetic Displacement (m)")
    axes[0, 0].set_ylabel("Transmitted Force (N)")
    axes[0, 0].legend(fontsize=7)

    axes[0, 1].plot(time, signals[rich], label="command")
    axes[0, 1].plot(
        time, traces[rich]["control_deadband"] / 120.0, label="delayed output / gain"
    )
    axes[0, 1].set_title("Control Deadband Is a Signal Map")
    axes[0, 1].set_xlabel("Time (s)")
    axes[0, 1].set_ylabel("Command-Equivalent Amplitude")
    axes[0, 1].legend(fontsize=8)

    positions = np.arange(len(KINDS))
    width = 0.36
    for offset, excitation_name in enumerate(EXCITATIONS):
        ranks = [audits[excitation_name][kind]["rank"] for kind in KINDS]
        axes[1, 0].bar(
            positions + (offset - 0.5) * width,
            ranks,
            width,
            label=excitation_name.replace("_", " "),
        )
    axes[1, 0].set_title("Local Scaled Sensitivity Rank")
    axes[1, 0].set_ylabel("Rank of Three Parameters")
    axes[1, 0].set_xticks(
        positions, [kind.replace("_", "\n") for kind in KINDS], fontsize=7
    )
    axes[1, 0].set_ylim(0.0, 3.3)
    axes[1, 0].legend(fontsize=8)

    for kind in KINDS:
        values = np.asarray(traces[rich][kind])
        axes[1, 1].plot(
            time,
            values / max(float(np.max(np.abs(values))), 1e-12),
            label=kind.replace("_", " "),
        )
    axes[1, 1].set_title("Normalized Outputs Are Not Class Labels")
    axes[1, 1].set_xlabel("Time (s)")
    axes[1, 1].set_ylabel("Normalized Transmitted Output")
    axes[1, 1].legend(fontsize=7)
    figure.savefig(FIGURE_OUTPUT)
    plt.close(figure)


def main() -> None:
    time = np.linspace(0.0, 1.0, 4001)
    signals = {name: excitation(time, name) for name in EXCITATIONS}
    traces: dict[str, dict[str, np.ndarray]] = {}
    cases: dict[str, dict[str, dict[str, object]]] = {}
    audits: dict[str, dict[str, dict[str, object]]] = {}
    arrays: dict[str, np.ndarray] = {"time_s": time}
    for excitation_name, signal in signals.items():
        traces[excitation_name] = {}
        cases[excitation_name] = {}
        audits[excitation_name] = {}
        arrays[f"signal_{excitation_name}"] = signal
        for kind in KINDS:
            parameters = _parameters(kind)
            result = simulate_dynamic_slack(time, signal, parameters)
            audit = scaled_sensitivity_audit(time, signal, parameters)
            traces[excitation_name][kind] = result.transmitted
            arrays[f"transmitted_{excitation_name}_{kind}"] = result.transmitted
            arrays[f"engaged_{excitation_name}_{kind}"] = result.engaged.astype(
                np.uint8
            )
            cases[excitation_name][kind] = {
                "engaged_fraction": float(np.mean(result.engaged)),
                "peak_abs_transmitted": float(np.max(np.abs(result.transmitted))),
                "input_work_j": result.input_work_j,
                "dissipative_work_j": result.dissipative_work_j,
                "loop_area_j": result.loop_area_j,
                "energy_residual_j": result.energy_residual_j,
                "passivity_applicable": result.passivity_applicable,
                "activation_delay_s": result.activation_delay_s,
            }
            audits[excitation_name][kind] = asdict(audit)

    rich_traces = traces["multisine_reversal"]
    distances = {
        f"{left}__{right}": _distance(rich_traces[left], rich_traces[right])
        for left, right in combinations(KINDS, 2)
    }
    residuals = [
        abs(float(cases[name][kind]["energy_residual_j"]))
        for name in EXCITATIONS
        for kind in MECHANICAL_KINDS
    ]
    record = {
        "schema_version": "typed-slack-dynamic-study/v1",
        "study_id": "typed-slack-dynamic-passivity-identifiability-audit",
        "model_tier": "synthetic_scalar_dynamic_constitutive_screen",
        "classes": list(KINDS),
        "excitations": list(EXCITATIONS),
        "cases": cases,
        "identifiability": audits,
        "pairwise_normalized_output_rmse_multisine": distances,
        "passivity_summary": {
            "mechanical_classes": {
                "all_pass": all(value < 2e-5 for value in residuals),
                "maximum_abs_energy_residual_j": max(residuals),
            }
        },
        "control_boundary": {
            "passivity_applicable": False,
            "reason": "The control deadband is a delayed command-transmission map, not a mechanical energy-storage element.",
        },
        "claims": {
            "global_slack_benefit": "unsupported",
            "class_identification_from_one_channel": "not_established",
            "excitation_independent_identifiability": "not_supported",
            "human_intentionality": "untested",
        },
        "limitations": [
            "All excitations and parameters are synthetic and are not fitted to a golfer, tissue, grip, shaft, or controller.",
            "Transmission backlash is represented by the existing memoryless dead-zone-plus-damping surrogate; a stateful rate-independent play operator is not represented.",
            "Biological series compliance is a unilateral Kelvin-Voigt surrogate, not a Hill-type tendon or subject-specific anatomical model.",
            "Local sensitivity rank is not global or practical identifiability and does not prove that a class generated measured data.",
            "Output separation under the registered multisine does not identify anatomical intent or a beneficial amount of slack.",
            "The suite tests constitutive channels, not club delivery, injury risk, or performance.",
        ],
        "array_artifact": NPZ_OUTPUT.name,
        "figure_artifact": FIGURE_OUTPUT.name,
    }
    JSON_OUTPUT.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(NPZ_OUTPUT, **arrays)
    _render_figure(time, signals, traces, audits)
    print(JSON_OUTPUT)


if __name__ == "__main__":
    main()
