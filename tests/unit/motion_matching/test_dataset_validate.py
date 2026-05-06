"""Direct tests for individual schema validators."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.shared.python.motion_matching.dataset import _validate


def _good_trials() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "trial_id": np.uint32(0),
                "coefficients": [0.0] * 14,
                "joint_names": ["a", "b"],
                "simulation_time_s": 1.0,
                "sample_rate_hz": 100.0,
                "solver_status": "success",
            },
            {
                "trial_id": np.uint32(1),
                "coefficients": [0.0] * 14,
                "joint_names": ["a", "b"],
                "simulation_time_s": 1.0,
                "sample_rate_hz": 100.0,
                "solver_status": "success",
            },
        ]
    )


def _good_timesteps() -> pd.DataFrame:
    rows = []
    for trial_id in (0, 1):
        for t in np.linspace(0.0, 1.0, 5):
            rows.append(
                {
                    "trial_id": np.uint32(trial_id),
                    "t": float(t),
                    "q": [0.0, 0.0],
                    "qd": [0.0, 0.0],
                    "qdd": [0.0, 0.0],
                    "tau": [0.0, 0.0],
                    "r_butt": [0.0, 0.0, 1.0],
                    "r_clubhead": [1.1, 0.0, 1.0],
                    "q_club": [1.0, 0.0, 0.0, 0.0],
                    "v_clubhead": [0.0, 0.0, 0.0],
                    "omega_club": [0.0, 0.0, 0.0],
                }
            )
    return pd.DataFrame(rows)


@pytest.mark.unit
def test_validate_trials_rejects_duplicate_trial_id() -> None:
    df = _good_trials()
    df.loc[1, "trial_id"] = np.uint32(0)
    with pytest.raises(ValueError, match="unique"):
        _validate.validate_trials_table(df)


@pytest.mark.unit
def test_validate_timesteps_rejects_unknown_trial_id() -> None:
    ts = _good_timesteps()
    ts.loc[0, "trial_id"] = np.uint32(99)
    with pytest.raises(ValueError, match="not present in trials"):
        _validate.validate_timesteps_table(ts, trial_ids={0, 1}, n_joints=2)


@pytest.mark.unit
def test_validate_timesteps_rejects_non_monotonic_time() -> None:
    ts = _good_timesteps()
    # Reverse order within trial 0.
    mask = ts["trial_id"] == 0
    ts.loc[mask, "t"] = ts.loc[mask, "t"].iloc[::-1].to_numpy()
    with pytest.raises(ValueError, match="monotonic"):
        _validate.validate_timesteps_table(ts, trial_ids={0, 1}, n_joints=2)


@pytest.mark.unit
def test_validate_timesteps_rejects_wrong_q_length() -> None:
    ts = _good_timesteps()
    ts.at[0, "q"] = [0.0]  # length 1, expected 2
    with pytest.raises(ValueError, match="expected length 2"):
        _validate.validate_timesteps_table(ts, trial_ids={0, 1}, n_joints=2)


@pytest.mark.unit
def test_validate_no_nan_in_success_trials_flags_nan() -> None:
    trials = _good_trials()
    ts = _good_timesteps()
    ts.at[0, "q"] = [float("nan"), 0.0]
    with pytest.raises(ValueError, match="NaN/Inf"):
        _validate.validate_no_nan_in_success_trials(trials, ts)


@pytest.mark.unit
def test_validate_shaft_length_flags_units_bug() -> None:
    ts = _good_timesteps()
    # Move clubhead 100m away — clearly a units bug.
    ts.at[0, "r_clubhead"] = [100.0, 0.0, 1.0]
    with pytest.raises(ValueError, match="shaft length"):
        _validate.validate_shaft_length(ts)
