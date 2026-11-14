#!/usr/bin/env python3
"""Regenerate docs/validation/drive_mode_sensitivity_receipt.json (Refs #13886, #13888)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src/engines/pendulum_models/python"))

import numpy as np

from src.shared.python.biomechanics.mtu_joint_actuator import TendonMode
from src.shared.python.loop_nullspace import impedance_spec as isp
from src.shared.python.pendulum_simulator import sensitivity as sens
from src.shared.python.pendulum_simulator.drive_modes import DriveMode
from src.shared.python.pendulum_simulator.hill_mtu_drive import (
    DEFAULT_COCONTRACTION_LEVELS,
    build_matched_excitation,
    default_triple_mtu_actuators,
)

OUT = ROOT / "docs/validation/drive_mode_sensitivity_receipt.json"
WINDOW_S = 0.15
T_END = 0.2
DT = 0.002


def _impedance() -> isp.JointImpedance:
    return isp.JointImpedance(
        (isp.ImpedanceWindow(0.0, T_END, np.full(3, 50.0), np.full(3, 1.5)),)
    )


def main() -> None:
    nom = sens.rollout_nominal_trajectory(t_end=T_END, dt=DT)
    imp = _impedance()
    modes = []
    for mode in (
        DriveMode.TORQUE_UNDAMPED,
        DriveMode.TORQUE_CANONICAL,
        DriveMode.JOINT_IMPEDANCE,
    ):
        extra = imp if mode is DriveMode.JOINT_IMPEDANCE else None
        _, _, _, rec = sens.integrate_drive_mode_stm(
            mode, nom, extra, window_s=WINDOW_S
        )
        modes.append(rec.to_dict())
    sweep: list[dict] = []
    for c in DEFAULT_COCONTRACTION_LEVELS:
        for tendon in (TendonMode.RIGID, TendonMode.COMPLIANT):
            try:
                hill = build_matched_excitation(
                    nom, c, default_triple_mtu_actuators(tendon), tendon_mode=tendon
                )
                _, _, _, rec = sens.integrate_drive_mode_stm(
                    DriveMode.HILL_MTU, nom, window_s=WINDOW_S, hill=hill
                )
                row = rec.to_dict()
            except (RuntimeError, ValueError):
                row = {
                    "drive_mode": DriveMode.HILL_MTU.value,
                    "validation_status": "NOT_QUALIFIED",
                    "cocontraction": c,
                    "tendon_mode": tendon.value,
                    "sigma_max_window": None,
                    "ftle_window": None,
                }
            else:
                row["cocontraction"] = c
                row["tendon_mode"] = tendon.value
            sweep.append(row)
    try:
        hill_ref = build_matched_excitation(
            nom,
            0.2,
            default_triple_mtu_actuators(TendonMode.RIGID),
            tendon_mode=TendonMode.RIGID,
        )
        _, _, _, hill_rec = sens.integrate_drive_mode_stm(
            DriveMode.HILL_MTU, nom, window_s=WINDOW_S, hill=hill_ref
        )
        modes.append(hill_rec.to_dict())
    except (RuntimeError, ValueError):
        modes.append(
            {
                "drive_mode": DriveMode.HILL_MTU.value,
                "validation_status": "NOT_QUALIFIED",
                "impedance_hypotheses": [isp.HYPOTHESIS_ID],
            }
        )
    k_sweep = []
    for k_val in (0.0, 10.0, 25.0, 50.0):
        imp_k = isp.JointImpedance(
            (isp.ImpedanceWindow(0.0, T_END, np.full(3, k_val), np.full(3, 1.5)),)
        )
        _, _, _, rec = sens.integrate_drive_mode_stm(
            DriveMode.JOINT_IMPEDANCE, nom, imp_k, window_s=WINDOW_S
        )
        k_sweep.append(
            {
                "k_nm_per_rad": k_val,
                "d_nm_s_per_rad": 1.5,
                "sigma_max_window": rec.sigma_max_window,
                "ftle_window": rec.ftle_window,
            }
        )
    doc = {
        "schema": "upstreamdrift.drive_mode_sensitivity_receipt.v1",
        "issue": 13888,
        "parent_epic": 13789,
        "validation_status": "UNVALIDATED",
        "impedance_hypotheses": [isp.HYPOTHESIS_ID],
        "window_s": WINDOW_S,
        "modes": modes,
        "impedance_k_sweep": k_sweep,
        "hill_cocontraction_sweep": sweep,
        "smoke_note": "Full sweep uses t_end=0.2 s; CI smoke uses t_end=0.08 s in unit tests.",
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
