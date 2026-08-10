"""E1b — bounded-actuator sensitivity check for the timing sweep.

E1 (:mod:`run_experiments`) uses constant, unbounded torques. Real
actuators obey torque-velocity limits that shrink the late-drive window
[Sprigings & Neal 2000], so this module re-runs the wrist-onset sweep
with a linear torque-velocity bound on every *concentric* (drive)
channel:

    tau(omega) = tau_iso * clip(1 - omega / omega_max, 0, 1)

applied to the shoulder drive (against the arm's angular velocity) and
to the wrist drive (against the wrist's relative opening velocity).
Restraining wrist torque is left constant: eccentric muscle torque does
not fall with shortening velocity the way concentric torque does, and
the restraint channel opposes the motion.

Because the torque now depends on the state, rollouts integrate the
dynamics in closed loop using the backend's own primitives
(``mass_matrix`` / ``bias_forces``) with the same fixed-step RK4
structure and step size as E1.

Outputs: ``data/e1b_bounded_sweep.json`` and
``figures/fig_e1b_bounded_sweep.pdf``.

Usage::

    python3 -m scripts.research.proximal_distal_energy.e1b_bounded_torque
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from scripts.research.proximal_distal_energy.run_experiments import (
    DT,
    HORIZON,
    INITIAL_Q,
    ONSET_GRID,
    _git_sha,
)
from scripts.research.proximal_distal_energy.swing_model import (
    PlanarInertials,
    find_impact,
)
from src.shared.python.simulation_backends import GolfModelParams, make_backend

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[3]
OUTPUT_ROOT = REPO_ROOT / "docs" / "research" / "proximal_distal_energy_transfer"
DATA_DIR = OUTPUT_ROOT / "data"
FIG_DIR = OUTPUT_ROOT / "figures"

#: Isometric (zero-velocity) torque levels and torque-velocity intercepts.
#: Model-scale choices, documented in the report Methods; the shoulder
#: bound is chosen so peak arm speeds in E1 (~15-18 rad/s) sit deep in
#: the tapering region, making the bound genuinely restrictive.
SHOULDER_TAU_ISO = 100.0
SHOULDER_OMEGA_MAX = 20.0
WRIST_TAU_ISO = 20.0
WRIST_OMEGA_MAX = 30.0
RESTRAIN_LEVELS = (5.0, 10.0)


def _bounded(tau_iso: float, omega: float, omega_max: float) -> float:
    """Linear concentric torque-velocity bound, floored at zero."""
    scale = 1.0 - omega / omega_max
    return tau_iso * float(np.clip(scale, 0.0, 1.0))


def _torque(
    t: float, q: np.ndarray, v: np.ndarray, onset_s: float, restrain_nm: float
) -> np.ndarray:
    """State-dependent torque program (bounded drives, constant restraint)."""
    tau_s = _bounded(SHOULDER_TAU_ISO, v[0], SHOULDER_OMEGA_MAX)
    if t < onset_s:
        tau_w = -restrain_nm
    else:
        tau_w = _bounded(WRIST_TAU_ISO, v[1], WRIST_OMEGA_MAX)
    return np.array([tau_s, tau_w])


def rollout_bounded(
    provider, onset_s: float, restrain_nm: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Closed-loop RK4 rollout using the backend's dynamics primitives."""
    n = HORIZON + 1
    t = np.arange(n) * DT
    q = np.empty((n, 2))
    v = np.empty((n, 2))
    q[0] = INITIAL_Q
    v[0] = 0.0

    def f(tk: float, qk: np.ndarray, vk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        tau = _torque(tk, qk, vk, onset_s, restrain_nm)
        m = np.asarray(provider.mass_matrix(qk))
        bias = np.asarray(provider.bias_forces(qk, vk))
        return vk, np.linalg.solve(m, tau - bias)

    for k in range(HORIZON):
        tk, qk, vk = t[k], q[k], v[k]
        k1q, k1v = f(tk, qk, vk)
        k2q, k2v = f(tk + DT / 2, qk + DT / 2 * k1q, vk + DT / 2 * k1v)
        k3q, k3v = f(tk + DT / 2, qk + DT / 2 * k2q, vk + DT / 2 * k2v)
        k4q, k4v = f(tk + DT, qk + DT * k3q, vk + DT * k3v)
        q[k + 1] = qk + DT / 6 * (k1q + 2 * k2q + 2 * k3q + k4q)
        v[k + 1] = vk + DT / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
    return t, q, v


def main() -> None:
    """Run the bounded-actuator sweep and render its figure."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    params = GolfModelParams.default()
    inertials = PlanarInertials.from_params(params)
    provider = make_backend("ode", params)

    rows: list[dict] = []
    variants: list[tuple[str, float, float | None]] = [("passive", 0.0, None)]
    for onset in ONSET_GRID:
        variants.append(("drive_only", 0.0, float(onset)))
        for restrain in RESTRAIN_LEVELS:
            variants.append((f"restrain_{restrain:.0f}", restrain, float(onset)))
    for profile, restrain, onset in variants:
        if onset is None:
            t, q, v = rollout_bounded(provider, float("inf"), 0.0)
        else:
            t, q, v = rollout_bounded(provider, onset, restrain)
        impact = find_impact(t, q, v, inertials)
        rows.append(
            {
                "profile": profile,
                "onset_s": onset,
                "restrain_nm": restrain,
                "t_impact_s": None if impact is None else impact[0],
                "clubhead_speed_mps": None if impact is None else impact[1],
                "theta1_at_impact_rad": None if impact is None else impact[2],
            }
        )
        logger.info("e1b row: %s", rows[-1])

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    provenance = {
        "git_sha": _git_sha(),
        "dt_s": DT,
        "horizon": HORIZON,
        "initial_q_rad": list(INITIAL_Q),
        "shoulder_tau_iso_nm": SHOULDER_TAU_ISO,
        "shoulder_omega_max_rad_s": SHOULDER_OMEGA_MAX,
        "wrist_tau_iso_nm": WRIST_TAU_ISO,
        "wrist_omega_max_rad_s": WRIST_OMEGA_MAX,
        "restrain_levels_nm": list(RESTRAIN_LEVELS),
        "onset_grid_s": [float(x) for x in ONSET_GRID],
        "backend": "ode (primitives, closed-loop RK4)",
    }
    with (DATA_DIR / "e1b_bounded_sweep.json").open("w", encoding="utf-8") as fh:
        json.dump({"provenance": provenance, "rows": rows}, fh, indent=1)

    fig, ax = plt.subplots(figsize=(6.8, 4.2))
    passive_speed = next(
        r["clubhead_speed_mps"] for r in rows if r["profile"] == "passive"
    )
    if passive_speed is not None:
        ax.axhline(
            passive_speed, color="#666666", ls="--", lw=1.2, label="passive baseline"
        )
    labels = {
        "drive_only": "drive only (bounded)",
        "restrain_5": "restrain 5 N·m, then drive (bounded)",
        "restrain_10": "restrain 10 N·m, then drive (bounded)",
    }
    for profile, label in labels.items():
        pts = sorted(
            (r["onset_s"], r["clubhead_speed_mps"])
            for r in rows
            if r["profile"] == profile and r["clubhead_speed_mps"] is not None
        )
        if pts:
            xs, ys = zip(*pts, strict=False)
            ax.plot(xs, ys, marker="o", ms=3.5, lw=1.4, label=label)
    ax.set_xlabel("Wrist drive onset time [s]")
    ax.set_ylabel("Clubhead speed at impact [m/s]")
    ax.set_title("E1b — Timing sweep with linear torque-velocity bounds on drives")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "fig_e1b_bounded_sweep.pdf")
    fig.savefig(FIG_DIR / "fig_e1b_bounded_sweep.svg")
    plt.close(fig)
    logger.info("wrote e1b outputs")


if __name__ == "__main__":
    main()
