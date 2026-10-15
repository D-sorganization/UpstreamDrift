"""Write the NSX-72 spatial closure-alternative record (#12590).

Not run in CI.  Usage::

    python3 scripts/benchmarks/loop_nullspace_closure_alternatives.py \\
        docs/research/loop_nullspace/records/closure_alternatives_spatial.json

For each grip model of ``closure_alternatives`` on the fixed-base spatial
fixture: the admission outcome with its own position, velocity and
acceleration residuals for the same 20-frame trajectories (WELDED, TWIST,
TILT), the position-level probe, the model's own motion, and the dynamics under
three declared torque boxes.  Synthetic fixture, declared priors; nothing here
is evidence about a golfer or a real grip.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from src.shared.python.loop_nullspace.admission import AdmissionGates  # noqa: E402
from src.shared.python.loop_nullspace.closure_alternatives import (  # noqa: E402
    GRIP_MODEL_SPECS,
    GripModel,
    Motion,
    SpatialGripPlant,
    admission_outcome,
    motion_frames,
    position_probe,
)
from src.shared.python.loop_nullspace.closure_alternatives_report import (  # noqa: E402
    GRIP_MODEL_LABEL,
    GRIP_MODEL_NON_CLAIMS,
    box_feasible_set_of,
    closure_component_quantities,
    evaluate_grip_model,
    median_width_ratio,
)
from src.shared.python.loop_nullspace.envelopes import (  # noqa: E402
    joint_torque_quantities,
)
from src.shared.python.loop_nullspace.restriction import (  # noqa: E402
    snapshot_quantities,
)

CAPS_NM = (6.0, 8.0, 12.0)
N_FRAMES, SEED = 20, 7
PROBE_STEPS = (1e-4, 1.5e-4, 2e-4)
GATES = AdmissionGates()  # NSX-00 gate 1e-6 at every level, never loosened


def _admission_row(plant: SpatialGripPlant, motion: Motion) -> dict:
    snaps, times = motion_frames(plant, motion, n_frames=N_FRAMES, seed=SEED)
    adm = admission_outcome(plant, snaps, times, GATES)
    return {
        "admitted": adm.admitted,
        "max_phi": adm.max_phi,
        "max_phi_dot": adm.max_phi_dot,
        "max_phi_ddot": adm.max_phi_ddot,
        "first_reason": adm.reasons[0] if adm.reasons else None,
    }


def _probe_row(plant: SpatialGripPlant, motion: Motion) -> dict:
    snaps, times = position_probe(plant, motion, steps=PROBE_STEPS)
    adm = admission_outcome(plant, snaps, times, GATES)
    return {"admitted": adm.admitted, "max_phi": adm.max_phi}


def _dynamics(model: GripModel, motion: Motion, cap: float):
    plant = SpatialGripPlant(model)
    snaps, times = motion_frames(plant, motion, n_frames=N_FRAMES, seed=SEED)
    adm = admission_outcome(plant, snaps, times, GATES)
    builders = [
        snapshot_quantities(joint_torque_quantities(None)),
        closure_component_quantities(GRIP_MODEL_SPECS[model].components),
    ]
    return evaluate_grip_model(
        adm, quantities=builders, feasible_set_of=box_feasible_set_of(cap)
    )


def build_record() -> dict:
    """The full record (see module docstring)."""
    record: dict = {
        "label": GRIP_MODEL_LABEL,
        "non_claims": list(GRIP_MODEL_NON_CLAIMS),
        "fixture": "spatial_two_arm_club fixed base, seed 7, 20 frames",
        "gates": "1e-6 at position, velocity and acceleration",
        "caps_nm": list(CAPS_NM),
        "models": {},
    }
    for model, spec in GRIP_MODEL_SPECS.items():
        plant = SpatialGripPlant(model)
        record["models"][model.value] = {
            "closure_kind": spec.closure_kind.value,
            "rows": list(spec.components),
            "n_coordinates": plant.n,
            "kinematic_nullity": plant.kinematic_nullity(),
            "summary": spec.summary,
            "limitation": spec.limitation,
            "admission": {
                m.value: _admission_row(plant, m)
                for m in (Motion.WELDED, Motion.TWIST, Motion.TILT, Motion.OWN)
            },
            "position_probe": {
                m.value: _probe_row(plant, m) for m in (Motion.TWIST, Motion.TILT)
            },
            "dynamics": {},
        }
    for cap in CAPS_NM:
        for motion in (Motion.WELDED, Motion.OWN):
            results = {m: _dynamics(m, motion, cap) for m in GripModel}
            for model, res in results.items():
                entry = res.to_dict()
                entry["torque_width_ratio_vs_weld"] = (
                    median_width_ratio(results[GripModel.WELD], res, "tau_")
                    if motion is Motion.WELDED
                    else None
                )
                key = f"{motion.value}@{cap:g}Nm"
                record["models"][model.value]["dynamics"][key] = entry
    return record


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.write_text(json.dumps(build_record(), indent=2, default=str))


if __name__ == "__main__":
    main()
