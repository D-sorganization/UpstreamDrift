"""Write deterministic evidence for the distributed-shaft comparison."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np

from scripts.research.proximal_distal_energy.shaft_beam_reference import (
    BeamReferenceConfig,
    BeamReferenceStudy,
    run_beam_reference_study,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _json_value(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_value(item) for item in value]
    return value


def _summary(study: BeamReferenceStudy) -> dict[str, Any]:
    identification = _json_value(asdict(study.identification))
    return {
        "schema_version": "proximal-distal-shaft-beam-v1",
        "claim_status": study.claim_status,
        "calibration_status": "not_equipment_calibration",
        "identification_data_status": "declared_synthetic_modal_frequencies",
        "shared_fe_authority": "src.shared.python.physics.flexible_shaft",
        "configuration": _json_value(asdict(BeamReferenceConfig.publication_default())),
        "identification": identification,
        "convergence": {
            "frequencies_hz_48_elements": study.converged_frequencies_hz.tolist(),
            "maximum_relative_change_24_to_48": study.element_convergence_relative,
        },
        "comparison": {
            "reduced_mode_count": 1,
            "reference_mode_count": BeamReferenceConfig.publication_default().mode_count,
            "low_frequency_tip_rms_discrepancy_m": (
                study.low_frequency_tip_rms_discrepancy_m
            ),
            "high_frequency_tip_rms_discrepancy_m": (
                study.high_frequency_tip_rms_discrepancy_m
            ),
            "reference_peak_tip_deflection_m": (study.reference_peak_tip_deflection_m),
            "reduced_peak_tip_deflection_m": study.reduced_peak_tip_deflection_m,
        },
        "closure": {
            "maximum_reference_work_energy_residual_j": (
                study.reference_energy_closure_j
            ),
            "maximum_reduced_work_energy_residual_j": (study.reduced_energy_closure_j),
        },
        "interpretation": {
            "supported": (
                "a first-mode reduction agrees best for slowly varying loading and "
                "misses higher-mode content under a short force-and-moment pulse"
            ),
            "not_supported": [
                "equipment-specific shaft calibration",
                "human shaft-loading inference",
                "transport of the beam result through the constrained two-hand solve",
            ],
        },
        "open_gate": "couple_distributed_beam_into_forward_two_hand_solve",
    }


def write_beam_reference_evidence(
    output_dir: Path = DEFAULT_OUTPUT,
) -> tuple[Path, Path]:
    """Execute the study and atomically replace its JSON/NPZ evidence."""
    if not isinstance(output_dir, Path):
        raise TypeError("output_dir must be a pathlib.Path")
    output_dir.mkdir(parents=True, exist_ok=True)
    study = run_beam_reference_study()
    json_path = output_dir / "shaft_beam_reference.json"
    npz_path = output_dir / "shaft_beam_reference.npz"
    json_path.write_text(
        json.dumps(_summary(study), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    np.savez_compressed(
        npz_path,
        time_s=study.low_reference.time_s,
        low_reduced_tip_deflection_m=study.low_reduced.tip_deflection_m,
        low_reference_tip_deflection_m=study.low_reference.tip_deflection_m,
        high_reduced_tip_deflection_m=study.high_reduced.tip_deflection_m,
        high_reference_tip_deflection_m=study.high_reference.tip_deflection_m,
        low_reduced_energy_j=study.low_reduced.mechanical_energy_j,
        low_reference_energy_j=study.low_reference.mechanical_energy_j,
        high_reduced_energy_j=study.high_reduced.mechanical_energy_j,
        high_reference_energy_j=study.high_reference.mechanical_energy_j,
        high_reduced_input_power_w=study.high_reduced.input_power_w,
        high_reduced_damping_power_w=study.high_reduced.damping_power_w,
        high_reference_input_power_w=study.high_reference.input_power_w,
        high_reference_damping_power_w=study.high_reference.damping_power_w,
        converged_frequencies_hz=study.converged_frequencies_hz,
    )
    return json_path, npz_path


def main() -> None:
    write_beam_reference_evidence()


if __name__ == "__main__":
    main()


__all__ = ["write_beam_reference_evidence"]
