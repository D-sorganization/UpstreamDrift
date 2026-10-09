"""Compare contact-grip diagnostic runs against the bushing (issue #11986).

Reproduce after the diagnostic runs of ``run_grip_contact.py`` (tags
``_ft1e-3``, ``_ft5e-3``, ``_trail``, ``_lead``) exist in ``contact/``::

    PYTHONPATH=.:src python3 \\
        docs/development/full_body_models/evidence/grip_kinetics/run_grip_internal_force_compare.py

Writes ``contact/internal_force_<club>.json``: per run, the peak per-hand
force, net force, internal force, axial squeeze, couple, pad normal-force sum
and slip, next to the OpenSim bushing of the same club.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(HERE))

from run_grip_validation import _split, deflection  # noqa: E402

from src.shared.python.grip_contact.contact_run import ContactRun  # noqa: E402
from src.shared.python.grip_contact.parity import GripKineticsSeries  # noqa: E402

OUT = HERE / "contact"
PARITY = HERE / "parity"
RUNS = {
    "baseline_solreffriction_2dt": "",
    "solreffriction_1e-3_s": "_ft1e-3",
    "solreffriction_5e-3_s": "_ft5e-3",
    "trail_hand_follows_club": "_trail",
    "lead_hand_only": "_lead",
}


def contact_metrics(run: ContactRun) -> dict[str, Any]:
    out = _split(run.series)
    out["peak_pad_normal_sum_per_hand_n"] = float(
        max(run.squeeze_n[s].max() for s in "LR")
    )
    out["peak_axial_slip_mm"] = float(
        max(np.abs(run.axial_slip_m[s]).max() for s in "LR") * 1e3
    )
    out["peak_roll_slip_mrad"] = float(
        max(np.abs(run.roll_slip_rad[s]).max() for s in "LR") * 1e3
    )
    out["hand_to_club_deflection"] = deflection(run.series)
    return out


def main() -> int:
    for club in ("driver", "iron7"):
        report: dict[str, Any] = {
            "club": club,
            "opensim_bushing": _split(
                GripKineticsSeries.load_npz(PARITY / f"opensim_{club}_series.npz")
            ),
        }
        for label, tag in RUNS.items():
            path = OUT / f"mujoco_{club}{tag}_series.npz"
            if path.with_suffix(".contact.npz").exists():
                report[label] = contact_metrics(ContactRun.load_npz(path))
        (OUT / f"internal_force_{club}.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8"
        )
        sys.stdout.write(json.dumps(report, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
