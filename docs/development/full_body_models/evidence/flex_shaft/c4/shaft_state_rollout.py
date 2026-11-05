"""C4 (#14560) evidence: MuJoCo chain-shaft state through impact on the best-fit driver.

Same replay as R11 (``flex_shaft/r11/chain_deflection.py``): the chain-shaft
bundle of the best-fit driver (``same_input_bundle export --shaft-model chain``)
is replayed open loop over the impact window with the bundle's efforts and ball
force.  ``shaft_state_series`` then reports tip deflection, lead/droop and strain
energy from the MuJoCo model of the same spec.

    python3 shaft_state_rollout.py --out-dir <dir>   # needs the repo import paths
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import matplotlib
import mujoco
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from scripts.same_input_bundle import export  # noqa: E402
from src.engines.physics_engines.mujoco.python.full_body_mjcf import (  # noqa: E402
    export_full_body_mjcf,
)
from src.shared.python.impact_parameters import TargetFrame  # noqa: E402
from src.shared.python.impact_parameters.shaft_state import (  # noqa: E402
    shaft_state_series,
)
from src.shared.python.motion_matching.club_face_target import (  # noqa: E402
    face_axis_in_frame,
)
from src.shared.python.motion_matching.club_shaft_chain import (  # noqa: E402
    passive_coordinates,
)
from src.shared.python.motion_matching.same_input import (  # noqa: E402
    InputBundle,
    VectorPlant,
    open_loop,
)

SEEDS = Path(__file__).resolve().parents[2] / "ground_support/anthro_driver_seeds"
WINDOW_S = (-0.08, 0.0)  # R11 window, ended at contact (the chain run diverges after)


def replay_window(bundle: InputBundle) -> tuple[np.ndarray, np.ndarray, bytes]:
    """Spec-order ``(time_s relative to impact, q)`` of the open-loop window."""
    dt = float(bundle.dt_s)
    impact = float(bundle.provenance["impact_split_time_s"])
    first = max(0, int(round((impact + WINDOW_S[0]) / dt)))
    stop = min(bundle.steps, int(round((impact + WINDOW_S[1]) / dt)))
    ball = bundle.ball_impact()
    rollout = open_loop(
        VectorPlant("mujoco", bundle.spec_bytes),
        bundle.reference_q[first],
        bundle.reference_v[first],
        bundle.replay_efforts(stop)[first * bundle.effort_holds :],
        dt_s=dt,
        policy=bundle.step_policy(),
        impact=None if ball is None else ball.shifted(-first * dt),
    )
    q = np.asarray(rollout.q)
    return (first + np.arange(q.shape[0])) * dt - impact, q, bundle.spec_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    out_dir = parser.parse_args().out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "chain_bundle.npz"
        export(SEEDS, path, 1.33, shaft_model="chain")
        bundle = InputBundle.load(path)
    rel_s, q_spec, spec_bytes = replay_window(bundle)
    spec = json.loads(spec_bytes)
    xml, meta = export_full_body_mjcf(spec_bytes)
    model = mujoco.MjModel.from_xml_string(xml)
    order = list(spec["coordinate_order"])

    def adr(name: str, table: np.ndarray) -> int:
        return int(table[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)])

    cols = [adr(n, model.jnt_qposadr) for n in order]
    q = np.zeros((q_spec.shape[0], model.nq))
    q[:, cols] = q_spec
    v = np.gradient(q, rel_s, axis=0)  # all joints are 1-DOF: qvel = dq/dt
    shaft = [adr(n, model.jnt_dofadr) for n in passive_coordinates(spec)]
    state = shaft_state_series(
        model,
        rel_s,
        q,
        v,
        shaft,
        meta["frame_sites"]["Clubhead"],
        TargetFrame(),
        face_normal_site=face_axis_in_frame(spec),
    )
    idx = int(np.argmin(np.abs(rel_s)))
    fig, ax = plt.subplots(2, 1, sharex=True, figsize=(7, 5))
    ax[0].plot(rel_s * 1e3, state.lead_m * 1e3, label="lead (+ forward)")
    ax[0].plot(rel_s * 1e3, state.droop_m * 1e3, label="droop (+ down)")
    ax[0].set_ylabel("tip deflection (mm)")
    ax[0].legend()
    ax[1].plot(rel_s * 1e3, state.face_delta_deg, label="face change (+ open)")
    ax[1].plot(rel_s * 1e3, state.loft_delta_deg, label="loft change (+ more loft)")
    ax[1].set_ylabel("shaft-induced angle (deg)")
    ax[1].legend()
    ax[1].set_xlabel("time relative to impact (ms)")
    ax[0].set_title("Chain-shaft state through the impact window (MuJoCo)")
    fig.savefig(out_dir / "c4_shaft_state_impact_window.png", dpi=130)
    receipt = {**state.at_impact(idx), "peak_deflection_m": state.peak_deflection_m}
    (out_dir / "c4_shaft_state_result.json").write_text(json.dumps(receipt, indent=2))
    print(json.dumps(receipt, indent=2))  # noqa: T201


if __name__ == "__main__":
    main()
