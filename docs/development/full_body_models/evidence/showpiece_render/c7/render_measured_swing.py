"""SHOW-C7 (#14563): flight card of the MEASURED best-fit driver swing.

This is the measured ~110 mph swing at the detector's impact, not the
optimised 125 mph showpiece (that render waits for the full-body adapter C3c).
Writes ``c7_measured_swing_flight.png``: side and top view of the ball flight
with the impact conditions and carry/total overlaid.  The driver bundle and any
video stay off-repo.  Run under the repository's pytest import context.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from src.engines.physics_engines.mujoco.python.full_body_mjcf import (  # noqa: E402
    export_full_body_mjcf,
)
from src.shared.python.impact_parameters import TargetFrame  # noqa: E402
from src.shared.python.impact_parameters.adapters import (  # noqa: E402
    club_face_spec_from_spec,
    clubhead_series_from_mujoco,
)
from src.shared.python.impact_parameters.conditions import (  # noqa: E402
    Rollout,
    evaluate_rollout,
)
from src.shared.python.impact_parameters.flight_report import (  # noqa: E402
    flight_for_impact,
    overlay_lines,
)

TITLE = "MEASURED best-fit driver swing (~110 mph), not the optimised showpiece"
SEEDS = Path(__file__).resolve().parents[2] / "ground_support" / "anthro_driver_seeds"


def render(out_dir: Path, seeds: Path = SEEDS) -> dict[str, float]:
    """Render the card to ``out_dir``; returns the numbers drawn on it."""
    raw = (seeds / "full_body_spec_hipcal_scaled.json").read_bytes()
    spec = json.loads(raw)
    xml, meta = export_full_body_mjcf(raw)
    model = mujoco.MjModel.from_xml_string(xml)
    rec = np.load(seeds / "dynamics_record.npz")
    names = meta["coordinate_order"]
    joints = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n) for n in names]
    t = rec["time_s"]
    q = np.zeros((len(t), model.nq))
    v = np.zeros((len(t), model.nv))
    q[:, model.jnt_qposadr[joints]] = rec["q"]
    v[:, model.jnt_dofadr[joints]] = rec["v"]
    site = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_SITE, meta["frame_sites"]["Clubhead"]
    )
    body = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_BODY, int(model.site_bodyid[site])
    )
    impact = json.loads((seeds / "replay_post_impact_capture_a.json").read_text())
    idx = int(np.searchsorted(t, impact["impact_time_s"]))
    frame, face = TargetFrame(), club_face_spec_from_spec(spec)
    cond = evaluate_rollout(
        model, Rollout(t, q, v), body, frame, spec=face, impact_index=idx
    )
    series = clubhead_series_from_mujoco(model, t, q, v, body, face)
    flight = flight_for_impact(series, idx, frame)

    path = flight.trajectory_m  # down-range, left, up
    fig, (side, top) = plt.subplots(2, 1, figsize=(10, 7.5), height_ratios=[2, 1])
    if flight.valid:
        side.plot(path[:, 0], path[:, 2], color="tab:blue")
        top.plot(path[:, 0], -path[:, 1], color="tab:red")
    else:  # outside the flight model's validity: say so, draw no flight
        for ax in (side, top):
            ax.text(
                0.5,
                0.5,
                "FLIGHT NOT SHOWN\n" + flight.invalid_reason,
                ha="center",
                va="center",
                color="firebrick",
                transform=ax.transAxes,
            )
    side.set(xlabel="down-range (m)", ylabel="height (m)", title=TITLE)
    top.axhline(0.0, color="grey", lw=0.8, ls="--")
    top.set(xlabel="down-range (m)", ylabel="right of target (m)")
    for ax in (side, top):
        ax.grid(alpha=0.3)
    fig.text(0.01, 0.01, "\n".join(overlay_lines(TITLE, cond, flight)[1:]),
             family="monospace", fontsize=8, va="bottom")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    out_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_dir / "c7_measured_swing_flight.png", dpi=110)
    plt.close(fig)
    return {
        "impact_index": idx,
        "speed_mph": cond.speed_mph,
        "face_deg": cond.face_deg,
        "path_deg": cond.path_deg,
        "aoa_deg": cond.aoa_deg,
        "flight_valid": flight.valid,
        "carry_m": flight.carry_m,
        "lateral_m": flight.lateral_m,
        "ball_speed_mph": flight.ball_speed_mph,
        "launch_deg": flight.launch_deg,
    }


if __name__ == "__main__":
    print(json.dumps(render(Path(__file__).parent), indent=2))  # noqa: T201
