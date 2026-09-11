"""Plot measured error and applied endpoint efforts for a completed native prefix."""

import json
import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("run_dir", type=Path)
parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[5])
args = parser.parse_args()
root = args.run_dir
sys.path.insert(0, str(args.repo))
from src.shared.python.motion_matching.prefix_fit import bernstein_effort_range

report = json.loads((root / "first_prefix_fit.json").read_text())
capture = json.loads((root / "driver_marker_payload.json").read_text())
time = np.asarray(capture["time_s"])
mask = time <= report["duration_s"] + 1e-12
time = time[mask]
indices = [capture["labels"].index(s) for s in report["labels"]]
observed = np.array(capture["points_world_m"])[mask][:, indices]
valid = np.array(capture["valid"])[mask][:, indices]
observed[~valid] = np.nan
pred = np.array(report["final_prediction_m"])
baseline = np.array(report["baseline_prediction_m"])
err = np.linalg.norm(pred - observed, axis=2) * 1000
initial = np.linalg.norm(baseline - observed, axis=2) * 1000
fig, axes = plt.subplots(2, 2, figsize=(13, 8), layout="constrained")
ax = axes[0, 0]
ax.plot(time, np.sqrt(np.nanmean(initial**2, axis=1)), label="Zero Effort")
ax.plot(time, np.sqrt(np.nanmean(err**2, axis=1)), label="Fitted Effort")
ax.set(xlabel="Time (s)", ylabel="Marker RMS (mm)", title="Forward Marker Error")
ax.legend()
ax.grid(alpha=0.2)
ax = axes[0, 1]
rows = report["evaluations"]
ax.semilogy([r["number"] for r in rows], [r["rmse_m"] * 1000 for r in rows], alpha=0.65)
ax.set(
    xlabel="Native Evaluation", ylabel="Marker RMS (mm)", title="Optimizer Evaluations"
)
ax.grid(alpha=0.2)
ax = axes[1, 0]
per = np.sqrt(np.nanmean(err**2, axis=0))
order = np.argsort(per)[-10:]
ax.barh(np.array(report["labels"])[order], per[order])
ax.set(xlabel="Marker RMS (mm)", title="Ten Largest Marker Residuals")
ax = axes[1, 1]
names = np.array(report["fit_identity"]["coordinate_names"])
p = np.array(report["stage"]["parameters"])
effort = (np.array(report["effort_scales"]) * (p - 1)).reshape(len(names), -1)
ranges = bernstein_effort_range(effort)
torques = np.array([not n.startswith("Translation") for n in names])
ix = np.argsort(np.max(abs(ranges[torques]), axis=1))[-10:]
ax.barh(
    names[torques][ix],
    ranges[torques][ix, 1] - ranges[torques][ix, 0],
    left=ranges[torques][ix, 0],
    alpha=0.4,
    label="Full Profile Range",
)
ax.scatter(effort[torques][ix, 0], np.arange(len(ix)), color="steelblue", label="Start")
ax.scatter(effort[torques][ix, -1], np.arange(len(ix)), color="darkorange", label="End")
span = float(np.ptp(ranges[torques][ix]))
padding = max(1.0, span * 0.05)
ax.set_xlim(
    float(ranges[torques][ix].min()) - padding,
    float(ranges[torques][ix].max()) + padding,
)
ax.legend()
ax.set(xlabel="Torque (Nm)", title="Ten Largest Angular Efforts")
fig.suptitle(
    f"Native R2025b Driver Fit: 0–{report['duration_s']:g} Seconds\nQualified Initial State; Fixed Provisional Geometry and Single-Frame Attachments",
    fontsize=13,
)
fig.savefig(root / f"prefix_{round(report['duration_s'] * 1000)}ms_fit.png", dpi=160)
summary = {
    "duration_s": report["duration_s"],
    "samples": len(time),
    "markers": len(indices),
    "rms_mm": float(np.sqrt(np.nanmean(err**2))),
    "p95_mm": float(np.nanpercentile(err, 95)),
    "max_mm": float(np.nanmax(err)),
    "baseline_rms_mm": float(np.sqrt(np.nanmean(initial**2))),
    "optimizer_converged": report["stage"]["optimizer_converged"],
    "accepted_numerically": report["accepted_numerically"],
    "evaluations": len(rows),
    "max_per_marker_rms_mm": float(np.max(per)),
    "force_controls_N": dict(
        zip(names[~torques], effort[~torques].tolist(), strict=True)
    ),
    "force_ranges_N": dict(
        zip(names[~torques], ranges[~torques].tolist(), strict=True)
    ),
    "max_abs_torque_control_bound_Nm": float(np.max(abs(effort[torques]))),
    "max_abs_torque_Nm": float(np.max(abs(ranges[torques]))),
    "max_torque_change_Nm": float(
        np.max(abs(effort[torques, -1] - effort[torques, 0]))
    ),
}
(root / "fit_summary.json").write_text(json.dumps(summary, indent=2))
