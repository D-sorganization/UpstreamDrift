"""Independently reconcile the original results chapter with committed evidence."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
OUTPUT = DATA / "results_chapter_audit.json"


def _json(name: str) -> dict:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def build_audit() -> dict[str, object]:
    """Recompute the chapter's main selection, ledger, and sensitivity claims."""
    sweep = _json("e1_sweep.json")["rows"]
    summary = _json("results_summary.json")
    parameter = _json("e1d_parameter_sensitivity.json")
    smooth = _json("e1e_smooth_command_sensitivity.json")
    impact = _json("e1c_sensitivity.json")
    bounded = _json("e1b_bounded_sweep.json")["rows"]
    traces = np.load(DATA / "representative_traces.npz")

    selected: dict[str, dict[str, object]] = {}
    for shoulder in (60.0, 100.0):
        accepted = [
            row
            for row in sweep
            if row["shoulder_torque_nm"] == shoulder
            and row["impact_status"] == "accepted_registered_delivery_zone"
        ]
        passive = next(row for row in accepted if row["profile"] == "passive")
        early = next(
            row
            for row in accepted
            if row["profile"] == "drive_only" and row["onset_s"] == 0.0
        )
        drive = max(
            (row for row in accepted if row["profile"] == "drive_only"),
            key=lambda row: row["clubhead_speed_mps"],
        )
        restrain = max(
            (row for row in accepted if row["profile"] == "restrain_then_drive"),
            key=lambda row: row["clubhead_speed_mps"],
        )
        selected[str(int(shoulder))] = {
            "passive_speed_m_s": passive["clubhead_speed_mps"],
            "early_speed_m_s": early["clubhead_speed_mps"],
            "best_drive_speed_m_s": drive["clubhead_speed_mps"],
            "best_drive_onset_s": drive["onset_s"],
            "best_restrain_speed_m_s": restrain["clubhead_speed_mps"],
            "best_restrain_onset_s": restrain["onset_s"],
            "best_restrain_nm": restrain["wrist_restrain_nm"],
            "early_vs_passive_percent": 100
            * (early["clubhead_speed_mps"] / passive["clubhead_speed_mps"] - 1),
            "drive_vs_passive_percent": 100
            * (drive["clubhead_speed_mps"] / passive["clubhead_speed_mps"] - 1),
        }

    prefix = "best_restrain__power__"
    residual = traces[prefix + "club_energy_rate"] - (
        traces[prefix + "joint_force_power"] + traces[prefix + "moment_power_on_club"]
    )
    status_counts = {
        status: sum(row["impact_status"] == status for row in sweep)
        for status in sorted({row["impact_status"] for row in sweep})
    }
    bounded_best = {
        profile: max(
            (
                row
                for row in bounded
                if row["profile"] == profile and row["clubhead_speed_mps"] is not None
            ),
            key=lambda row: row["clubhead_speed_mps"],
        )
        for profile in ("passive", "drive_only", "restrain_5", "restrain_10")
    }
    return {
        "schema_version": "proximal-distal-results-chapter-audit-v1",
        "selected_programs": selected,
        "attempted_programs": len(sweep),
        "impact_status_counts": status_counts,
        "summary_representatives_match": all(
            abs(summary["representatives"][name]["impact"][1] - selected["60"][key])
            < 1e-12
            for name, key in (
                ("passive", "passive_speed_m_s"),
                ("early_drive", "early_speed_m_s"),
                ("best_drive", "best_drive_speed_m_s"),
                ("best_restrain", "best_restrain_speed_m_s"),
            )
        ),
        "energy_balance_residual_max_w": float(np.max(np.abs(residual))),
        "energy_balance_residual_rms_w": float(np.sqrt(np.mean(residual**2))),
        "bounded_selected_speed_m_s": {
            name: row["clubhead_speed_mps"] for name, row in bounded_best.items()
        },
        "all_parameter_cases_preserve_ordering": parameter[
            "all_cases_confirm_ordering"
        ],
        "parameter_case_count": len(parameter["cases"]),
        "impact_family_ordering_preserved": impact["ordering_robustness_confirmed"],
        "smooth_time_constants_s": smooth["time_constants_s"],
        "smooth_all_orderings_preserved": all(
            record[shoulder]["registered_ordering_preserved"]
            for record in smooth["summaries"].values()
            for shoulder in ("shoulder_60", "shoulder_100")
        ),
        "interpretation": (
            "deterministic internal consistency of a finite planar command family; "
            "not biological validation, a global optimum, or coaching advice"
        ),
    }


def main() -> None:
    """Write the deterministic audit report."""
    OUTPUT.write_text(json.dumps(build_audit(), indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
