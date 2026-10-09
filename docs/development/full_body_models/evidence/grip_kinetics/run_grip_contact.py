"""Contact-grip full-swing runs (issue #11739, OSV-7 phase 3, epic #11726).

Reproduce from the repository root, one heavy simulation at a time::

    MPLBACKEND=Agg PYTHONPATH=.:src nice -n 10 python3 \\
        docs/development/full_body_models/evidence/grip_kinetics/run_grip_contact.py \\
        --club driver --engine mujoco

The squeeze of each hand is derived, not tuned: it is the smallest total pad
normal force that carries the peak per-hand demand of the OpenSim bushing
reference of the same club (``grip_contact.pad_layout.required_squeeze_n``).
Outputs are overwritten by name; nothing is deleted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))

from src.shared.python.grip_contact import (  # noqa: E402
    GripInterface,
    load_coordinate_swing,
)
from src.shared.python.grip_contact.pad_contact import build_pad_model  # noqa: E402
from src.shared.python.grip_contact.pad_layout import (  # noqa: E402
    squeeze_from_bushing_series,
)
from src.shared.python.grip_contact.parity import GripKineticsSeries  # noqa: E402

MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
OUT = HERE / "contact"
REFERENCE = HERE / "parity"


def squeeze_for(club: str, interface: GripInterface) -> float:
    """Squeeze per hand derived from the OpenSim bushing reference of ``club``."""
    ref = GripKineticsSeries.load_npz(REFERENCE / f"opensim_{club}_series.npz")
    return squeeze_from_bushing_series(
        ref.force_on_club_n,
        ref.torque_on_club_nm,
        ref.club_rotation,
        np.asarray(interface.right.rotation, float)[:, 0],
        interface.contact_material.static_friction,
        0.0127,
    )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--club", choices=("driver", "iron7"), required=True)
    ap.add_argument("--engine", choices=("mujoco",), default="mujoco")
    ap.add_argument("--squeeze-scale", type=float, default=1.0)
    ap.add_argument("--friction", type=float, default=None)
    ap.add_argument("--t-end", type=float, default=None)
    ap.add_argument("--tag", default="")
    ap.add_argument(
        "--friction-time",
        type=float,
        default=None,
        help="solreffriction time constant [s] (default 2 timesteps; issue #11986)",
    )
    ap.add_argument(
        "--hand-mode",
        choices=("prescribed", "trail_follows_club", "lead_only"),
        default="prescribed",
        help="diagnostic hand drive (issue #11986)",
    )
    args = ap.parse_args()

    spec_bytes = (MODELS / f"full_body_spec_anthro_{args.club}.json").read_bytes()
    spec = json.loads(spec_bytes)
    names = spec["coordinate_order"]
    swing = load_coordinate_swing(
        FIXTURES / f"swing_q_{args.club}.npz",
        FIXTURES / "address_poses.json",
        args.club,
        names,
    )
    interface = GripInterface.from_spec(spec)
    squeeze = args.squeeze_scale * squeeze_for(args.club, interface)
    pads = build_pad_model(interface, squeeze)
    if args.friction is not None:
        from dataclasses import replace

        pads = replace(
            pads,
            law=replace(
                pads.law,
                static_friction=args.friction,
                dynamic_friction=min(pads.law.dynamic_friction, args.friction),
            ),
        )
    from src.engines.physics_engines.mujoco.python.grip_contact_sim import (
        simulate_grip_contact,
    )

    start = time.perf_counter()
    run = simulate_grip_contact(
        spec_bytes,
        swing,
        pads,
        interface,
        t_end_s=args.t_end,
        friction_time_s=args.friction_time,
        hand_mode=args.hand_mode,
    )
    wall = time.perf_counter() - start
    OUT.mkdir(exist_ok=True)
    stem = f"{args.engine}_{args.club}{args.tag}"
    run.save_npz(OUT / f"{stem}_series.npz")
    info: dict[str, Any] = {
        "club": args.club,
        "engine": args.engine,
        "wall_time_s": wall,
        "squeeze_per_hand_n": squeeze,
        "pad_stiffness_n_m": pads.law.stiffness_n_m,
        "pad_dissipation_s_m": pads.law.dissipation_s_m,
        "preload_penetration_m": pads.layout.preload_penetration_m,
        "friction_static": pads.law.static_friction,
        "friction_dynamic": pads.law.dynamic_friction,
        **dict(run.series.metadata),
    }
    (OUT / f"{stem}_run.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    sys.stdout.write(json.dumps(info, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
