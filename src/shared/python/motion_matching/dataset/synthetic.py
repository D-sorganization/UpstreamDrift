"""Generate a small synthetic sweep dataset that passes loader validation.

The real parquet sweep dataset is not yet in the repo. This helper writes
a structurally-valid ``trials.parquet`` + ``timesteps.parquet`` so unit
and end-to-end tests can run before the real data lands. The numbers are
not physically meaningful — they only satisfy the schema and the
shaft-length sanity check.

Coefficient bounds match
``src/engines/.../dataset_generator/generateRandomCoefficients.m``:

    A, B (t^6, t^5):  -1000 .. +1000
    C, D (t^4, t^3):   -500 .. +500
    E, F (t^2, t^1):   -100 .. +100
    G    (constant):    -25 .. +25
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)

_SHAFT_LENGTH_M = 1.10  # nominal driver shaft length

# Half-widths of the uniform distribution per coefficient slot A..G.
_COEFF_HALF_RANGES: tuple[float, ...] = (
    1000.0,  # A
    1000.0,  # B
    500.0,  # C
    500.0,  # D
    100.0,  # E
    100.0,  # F
    25.0,  # G
)


def make_synthetic_sweep(
    path: str | Path,
    *,
    n_trials: int = 10,
    n_joints: int = 14,
    n_timesteps: int = 300,
    seed: int = 0,
) -> Path:
    """Write a valid synthetic sweep dataset to ``path``.

    Args:
        path: Folder to create. Will hold ``trials.parquet`` and
            ``timesteps.parquet``.
        n_trials: Number of simulated trials.
        n_joints: Number of joints in each trial's coefficient/state vectors.
        n_timesteps: Number of timesteps per trial.
        seed: RNG seed for reproducibility.

    Returns:
        The folder path that was written.
    """
    folder = Path(path)
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)

    duration_s = 1.0
    sample_rate_hz = (n_timesteps - 1) / duration_s if n_timesteps > 1 else 1.0
    joint_names = [f"joint_{i:02d}" for i in range(n_joints)]

    trials_df = _build_trials_df(rng, n_trials, joint_names, duration_s, sample_rate_hz)
    timesteps_df = _build_timesteps_df(rng, n_trials, n_joints, n_timesteps, duration_s)

    trials_path = folder / "trials.parquet"
    timesteps_path = folder / "timesteps.parquet"
    trials_df.to_parquet(trials_path, index=False)
    timesteps_df.to_parquet(timesteps_path, index=False)
    logger.info(
        "wrote synthetic sweep: %d trials × %d timesteps to %s",
        n_trials,
        n_timesteps,
        folder,
    )
    return folder


def _build_trials_df(
    rng: np.random.Generator,
    n_trials: int,
    joint_names: list[str],
    duration_s: float,
    sample_rate_hz: float,
) -> pd.DataFrame:
    n_joints = len(joint_names)
    rows = []
    for trial_id in range(n_trials):
        coeffs = _random_coefficients(rng, n_joints).tolist()
        rows.append(
            {
                "trial_id": np.uint32(trial_id),
                "coefficients": coeffs,
                "joint_names": list(joint_names),
                "simulation_time_s": float(duration_s),
                "sample_rate_hz": float(sample_rate_hz),
                "solver_status": "success",
                "clubhead_speed_max_mph": float(rng.uniform(80.0, 120.0)),
                "total_work_J": float(rng.uniform(100.0, 500.0)),
                "dataset_run_id": "synthetic",
                "seed": np.int64(seed_for(trial_id)),
            }
        )
    return pd.DataFrame(rows)


def seed_for(trial_id: int) -> int:
    """Deterministic per-trial seed (kept small to fit int64)."""
    return 1000 + trial_id


def _random_coefficients(rng: np.random.Generator, n_joints: int) -> np.ndarray:
    """Sample a flat ``n_joints * 7`` coefficient vector in documented bounds."""
    out = np.empty(n_joints * 7, dtype=np.float64)
    for joint_idx in range(n_joints):
        for slot, half in enumerate(_COEFF_HALF_RANGES):
            out[joint_idx * 7 + slot] = rng.uniform(-half, half)
    return out


def _build_timesteps_df(
    rng: np.random.Generator,
    n_trials: int,
    n_joints: int,
    n_timesteps: int,
    duration_s: float,
) -> pd.DataFrame:
    t_axis = np.linspace(0.0, duration_s, n_timesteps)
    rows = []
    for trial_id in range(n_trials):
        rows.extend(_one_trial_rows(rng, trial_id, n_joints, t_axis))
    return pd.DataFrame(rows)


def _one_trial_rows(
    rng: np.random.Generator,
    trial_id: int,
    n_joints: int,
    t_axis: np.ndarray,
) -> list[dict]:
    """Build per-timestep rows for a single trial."""
    out = []
    for t in t_axis:
        q = rng.normal(0.0, 0.5, size=n_joints).tolist()
        qd = rng.normal(0.0, 1.0, size=n_joints).tolist()
        qdd = rng.normal(0.0, 5.0, size=n_joints).tolist()
        tau = rng.normal(0.0, 10.0, size=n_joints).tolist()
        butt = np.array([0.0, 0.0, 1.0])
        # Place clubhead at fixed shaft length from butt; rotate slowly in t.
        angle = 2.0 * np.pi * t / max(t_axis[-1], 1e-9)
        head = butt + _SHAFT_LENGTH_M * np.array([np.cos(angle), np.sin(angle), 0.0])
        out.append(
            {
                "trial_id": np.uint32(trial_id),
                "t": float(t),
                "q": q,
                "qd": qd,
                "qdd": qdd,
                "tau": tau,
                "r_butt": butt.tolist(),
                "r_clubhead": head.tolist(),
                "q_club": [1.0, 0.0, 0.0, 0.0],
                "v_clubhead": [0.0, 0.0, 0.0],
                "omega_club": [0.0, 0.0, 0.0],
            }
        )
    return out
