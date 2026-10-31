"""Grip-size sweep CLI and receipt (#13892): ``python3 -m scripts.grip.grip_size_sweep``.

Demand: per-hand peak of the club's OpenSim bushing parity series, friction
and grip axis from its full-body spec (as ``run_grip_contact.squeeze_for``).
Default hand lengths sit 0.03 in either side of every Golf Pride bin edge
(crease measure), plus the ANSUR-mean male of 1.75 m stature.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import asdict
from pathlib import Path

import numpy as np

from src.shared.python.core.handedness import subject_handedness
from src.shared.python.grip_contact import GripInterface
from src.shared.python.grip_contact.grip_size_sweep import (
    BIN_EDGE_BAND_IN,
    GOLF_PRIDE_BIN_UPPER_EDGES_IN,
    INCH_M,
    STYLION_TO_CREASE_M,
    grip_size_sweep,
    per_hand_bushing_demand,
)
from src.shared.python.grip_contact.parity import GripKineticsSeries
from src.shared.python.hand_size import HAND_LENGTH_STATURE_FRACTION_MALE as MALE

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "docs/development/full_body_models"
EVIDENCE = MODELS / "evidence/grip_kinetics"
EDGE_CASES_IN = [
    e + d for _, e in GOLF_PRIDE_BIN_UPPER_EDGES_IN[:-1] for d in (-0.03, 0.03)
]
DEFAULT_HAND_LENGTHS_M = sorted(
    [round(x * INCH_M + STYLION_TO_CREASE_M, 5) for x in EDGE_CASES_IN]
    + [round(1.75 * MALE, 5)]
)
DEFAULT_PRESETS = ("mens_standard",)  # midsize is exactly +4 layers


def _receipt_text(header: dict, rows: list[dict]) -> str:
    """One header key and one sweep row per line (diff-friendly JSON)."""
    lines = [f"  {json.dumps(k)}: {json.dumps(v)}" for k, v in header.items()]
    body = ",\n".join(f"    {json.dumps(row)}" for row in rows)
    return "{\n" + ",\n".join([*lines, f'  "rows": [\n{body}\n  ]']) + "\n}\n"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--club", choices=("driver", "iron7"), default="driver")
    ap.add_argument("--hand-lengths", type=float, nargs="+")
    ap.add_argument("--presets", nargs="+", default=list(DEFAULT_PRESETS))
    ap.add_argument("--layers", type=int, nargs="+", default=[0, 2, 4])
    ap.add_argument("--out", type=Path)
    args = ap.parse_args(argv)

    spec_path = MODELS / f"full_body_spec_anthro_{args.club}.json"
    spec = json.loads(spec_path.read_text("utf-8"))
    interface = GripInterface.from_spec(spec)
    friction = interface.contact_material.static_friction
    axis = np.asarray(interface.right.rotation, float)[:, 0]
    ref = GripKineticsSeries.load_npz(
        EVIDENCE / f"parity/opensim_{args.club}_series.npz"
    )
    series = (ref.force_on_club_n, ref.torque_on_club_nm, ref.club_rotation)
    demand = per_hand_bushing_demand(*series, axis, subject_handedness(spec))
    hand_lengths = args.hand_lengths or DEFAULT_HAND_LENGTHS_M
    rows = grip_size_sweep(hand_lengths, args.presets, args.layers, demand, friction)
    header = {
        "issue": 13892,
        "club": args.club,
        "demand_source": f"parity/opensim_{args.club}_series.npz (per-hand peak)",
        "friction": friction,
        "peak_demand_axial_n_torque_nm_transverse_n": demand,
        "stylion_to_crease_m": STYLION_TO_CREASE_M,
        "golf_pride_bin_upper_edges_in": GOLF_PRIDE_BIN_UPPER_EDGES_IN[:-1],
        "bin_edge_band_in": BIN_EDGE_BAND_IN,
        "reproduce": f"python3 -m scripts.grip.grip_size_sweep --club {args.club}",
    }
    out = args.out or EVIDENCE / f"grip_size_sweep_{args.club}.json"
    out.write_text(_receipt_text(header, [asdict(r) for r in rows]), "utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
