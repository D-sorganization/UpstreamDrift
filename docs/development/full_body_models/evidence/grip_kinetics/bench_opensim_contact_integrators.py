"""Integrator benchmark of the OpenSim contact grip (issue #11739, OSV-7 phase 4).

The 24 elastic-foundation pads carry a stiff regularised-Coulomb friction (the
shared law's transition speed of 1 mm/s makes each sliding pad a damper of
``mu N / v_t`` ~ 8e4 N s/m), so an explicit error-controlled integrator needs
steps of the order of 1e-7 s.  This script times short windows of the address
hold with several integrators and writes ``contact/integrator_bench.json``.
One variant per invocation (run each under ``timeout``)::

    PYTHONPATH=.:src python3 .../bench_opensim_contact_integrators.py \\
        --method CPodes --accuracy 1e-4 --window 0.02 --budget 600
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
sys.path.insert(0, str(ROOT))

from src.shared.python.grip_contact import (  # noqa: E402
    ClubDynamics,
    GripInterface,
    load_coordinate_swing,
)
from src.shared.python.grip_contact.pad_contact import build_pad_model  # noqa: E402

MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
OUT = HERE / "contact"
SQUEEZE_N = 1104.0
SAMPLE_DT_S = 0.002


class BudgetExceededError(RuntimeError):
    """Raised from the progress callback when the wall budget is used up."""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--method", default="RungeKuttaMerson")
    ap.add_argument("--accuracy", type=float, default=1e-4)
    ap.add_argument("--window", type=float, default=0.02)
    ap.add_argument("--budget", type=float, default=600.0)
    ap.add_argument("--transition", type=float, default=None)
    ap.add_argument("--mesh-segments", type=int, default=None)
    args = ap.parse_args()

    from src.engines.physics_engines.opensim.python.grip_contact_osim_sim import (
        ContactGripSimulator,
    )

    spec_bytes = (MODELS / "full_body_spec_anthro_driver.json").read_bytes()
    spec = json.loads(spec_bytes)
    names = spec["coordinate_order"]
    swing = load_coordinate_swing(
        FIXTURES / "swing_q_driver.npz",
        FIXTURES / "address_poses.json",
        "driver",
        names,
    )
    interface = GripInterface.from_spec(spec)
    pads = build_pad_model(interface, SQUEEZE_N)
    if args.transition is not None:
        pads = replace(
            pads, law=replace(pads.law, transition_velocity_m_s=args.transition)
        )
    n = int(round(args.window / SAMPLE_DT_S)) + 1
    q = np.tile(np.asarray(swing.q[0], float), (n, 1))
    start = time.perf_counter()
    sim = ContactGripSimulator(
        spec_bytes, names, np.arange(n) * SAMPLE_DT_S, q, pads, interface
    )
    build_s = time.perf_counter() - start
    t0 = time.perf_counter()

    def progress(t: float) -> None:
        sys.stderr.write(f"t={t:.3f} wall={time.perf_counter() - t0:.1f}\n")
        sys.stderr.flush()
        if time.perf_counter() - t0 > args.budget:
            raise BudgetExceededError(f"budget {args.budget} s exceeded at t={t}")

    result: dict[str, object] = {
        "method": args.method,
        "accuracy": args.accuracy,
        "transition_velocity_m_s": pads.law.transition_velocity_m_s,
        "window_s": args.window,
        "build_s": build_s,
    }
    try:
        run = sim.run(
            accuracy=args.accuracy, on_sample=progress, method=args.method
        ).to_contact_run()
        weight = ClubDynamics.from_spec(spec).mass_kg * 9.80665
        total = run.series.force_on_club_n["L"] + run.series.force_on_club_n["R"]
        result.update(
            completed=True,
            wall_s=time.perf_counter() - t0,
            net_force_over_weight_last=float(np.linalg.norm(total[-1]) / weight),
            axial_slip_mm=float(
                max(np.abs(run.axial_slip_m[s]).max() for s in "LR") * 1e3
            ),
            squeeze_last_n=[float(run.squeeze_n[s][-1]) for s in "LR"],
        )
    except BudgetExceededError as exc:
        result.update(completed=False, wall_s=time.perf_counter() - t0, note=str(exc))
    OUT.mkdir(exist_ok=True)
    tag = f"{args.method}_{args.accuracy:g}_vt{pads.law.transition_velocity_m_s:g}"
    (OUT / f"integrator_bench_{tag}.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    sys.stdout.write(json.dumps(result, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
