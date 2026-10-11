"""SimscapeOutput.force_columns / to_force_series (#11304, FTO-19).

All data are synthetic fixtures built here; no MATLAB is involved.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.engines.simscape._errors import SimscapeSimulationError
from src.engines.simscape._output import SimscapeOutput
from src.engines.simscape._simscape_io import logsout_to_simscape_output
from src.engines.simscape.force_channels import load_simscape_force_series

pytestmark = pytest.mark.unit

N = 3
RZ90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])


def synthetic_force_columns() -> dict[str, np.ndarray]:
    """Local LS joint-reaction force/torque, position and rotation (synthetic)."""
    cols: dict[str, np.ndarray] = {}
    for i, k in enumerate("123"):
        cols[f"LSLogs_ConstraintForceLocal_{k}"] = np.full(N, [1.0, 2.0, 3.0][i])
        cols[f"LSLogs_ConstraintTorqueLocal_{k}"] = np.full(N, [0.0, 1.0, 0.0][i])
        cols[f"LSLogs_GlobalPosition_{k}"] = np.full(N, [0.1, 0.2, 1.3][i])
    for i in range(3):
        for j in range(3):
            cols[f"LSLogs_Rotation_Transform_I{i + 1}{j + 1}"] = np.full(N, RZ90[i, j])
    return cols


def _base_kwargs() -> dict[str, Any]:
    return {
        "time": np.linspace(0.0, 0.02, N),
        "q": np.zeros((N, 2)),
        "qd": np.zeros((N, 2)),
        "qdd": np.zeros((N, 2)),
        "tau": np.zeros((N, 2)),
        "omega": np.zeros((N, 2)),
        "r_butt": np.zeros((N, 3)),
        "r_clubhead": np.zeros((N, 3)),
        "q_club": np.tile([1.0, 0.0, 0.0, 0.0], (N, 1)),
        "v_clubhead": np.zeros((N, 3)),
    }


def test_default_is_backwards_compatible() -> None:
    out = SimscapeOutput(**_base_kwargs())
    assert out.force_columns is None
    with pytest.raises(ValueError, match="no force_columns"):
        out.to_force_series()


def test_to_force_series_matches_csv_loader(tmp_path: Path) -> None:
    cols = synthetic_force_columns()
    out = SimscapeOutput(**_base_kwargs(), force_columns=cols)
    path = tmp_path / "synthetic_trial.csv"
    names = ["time", *cols]
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(names)
        for r in range(N):
            w.writerow(
                [repr(float(out.time[r]))] + [repr(float(cols[k][r])) for k in cols]
            )
    expected, expected_missing = load_simscape_force_series(path)
    series, missing = out.to_force_series()
    assert missing == expected_missing
    assert len(series) == len(expected)
    for live, loaded in zip(series.frames, expected.frames, strict=True):
        assert [(w.label, w.force_n, w.torque_nm) for w in live.wrenches] == [
            (w.label, w.force_n, w.torque_nm) for w in loaded.wrenches
        ]
    # Local force (1,2,3) rotated by Rz(90) -> (-2, 1, 3): rotation applied.
    wrench = next(
        w for w in series.frames[0].wrenches if w.label == "joint_reaction:LS"
    )
    assert wrench.force_n == pytest.approx((-2.0, 1.0, 3.0))


def test_missing_rotation_is_unavailable_not_zero() -> None:
    cols = {
        k: v
        for k, v in synthetic_force_columns().items()
        if "Rotation_Transform" not in k
    }
    series, missing = SimscapeOutput(
        **_base_kwargs(), force_columns=cols
    ).to_force_series()
    assert "joint_reaction:LS:force" in missing
    assert all(w.label != "joint_reaction:LS" for w in series.frames[0].wrenches)


def test_non_orthonormal_rotation_rejected() -> None:
    cols = synthetic_force_columns()
    cols["LSLogs_Rotation_Transform_I11"] = np.full(N, 2.0)
    out = SimscapeOutput(**_base_kwargs(), force_columns=cols)
    with pytest.raises(ValueError, match="LS.*not orthonormal"):
        out.to_force_series()


@pytest.mark.parametrize(
    ("column", "exc"),
    [
        (np.zeros(N + 1), ValueError),
        (np.array([0.0, np.nan, 0.0]), ValueError),
        ([0.0, 0.0, 0.0], TypeError),
    ],
)
def test_force_columns_validated(column: object, exc: type[Exception]) -> None:
    with pytest.raises(exc):
        SimscapeOutput(**_base_kwargs(), force_columns={"x": column})  # type: ignore[dict-item]


def test_logsout_forces_optional_and_carried() -> None:
    base = _base_kwargs()
    assert logsout_to_simscape_output(dict(base)).force_columns is None
    out = logsout_to_simscape_output({**base, "forces": synthetic_force_columns()})
    assert out.force_columns is not None
    assert set(out.force_columns) == set(synthetic_force_columns())
    series, _ = out.to_force_series()
    assert len(series) == N


def test_logsout_bad_forces_wrapped() -> None:
    with pytest.raises(SimscapeSimulationError, match="force_columns"):
        logsout_to_simscape_output({**_base_kwargs(), "forces": {"x": np.zeros(N + 2)}})


def test_live_output_wrenches_labelled_distinctly_from_csv() -> None:
    out = SimscapeOutput(**_base_kwargs(), force_columns=synthetic_force_columns())
    series, _ = out.to_force_series()
    sources = {w.source for f in series.frames for w in f.wrenches}
    assert sources == {"simscape_output"}


def test_csv_loader_keeps_csv_source(tmp_path: Path) -> None:
    cols = synthetic_force_columns()
    path = tmp_path / "t.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["time", *cols])
        for r in range(N):
            w.writerow([float(r)] + [float(cols[k][r]) for k in cols])
    series, _ = load_simscape_force_series(path)
    assert {w.source for f in series.frames for w in f.wrenches} == {"simscape_csv"}


@pytest.mark.parametrize("bad", [np.zeros((0, 0)), [1.0, 2.0], "abc", 5])
def test_logsout_non_mapping_forces_wrapped(bad: object) -> None:
    with pytest.raises(SimscapeSimulationError, match="forces"):
        logsout_to_simscape_output({**_base_kwargs(), "forces": bad})


def test_trimmed_simscape_force_fixture_carries_channels() -> None:
    fixture_path = (
        Path(__file__).resolve().parents[2]
        / "fixtures"
        / "simscape"
        / "synthetic_simscape_force_output.json"
    )
    assert fixture_path.exists()
    with fixture_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    assert data["release"] == "2025b"
    assert (
        data["model_sha256"]
        == "daca9a90ad0ab819c7d61641ed594b8f230f8658f2a5f4ce34026db44b52ddc9"
    )
    assert data["channel_count"] == 237

    t_rows = len(data["time"])
    assert t_rows == 5

    base = {
        "time": np.array(data["time"], dtype=np.float64),
        "q": np.zeros((t_rows, 2)),
        "qd": np.zeros((t_rows, 2)),
        "qdd": np.zeros((t_rows, 2)),
        "tau": np.zeros((t_rows, 2)),
        "omega": np.zeros((t_rows, 2)),
        "r_butt": np.zeros((t_rows, 3)),
        "r_clubhead": np.zeros((t_rows, 3)),
        "q_club": np.tile([1.0, 0.0, 0.0, 0.0], (t_rows, 1)),
        "v_clubhead": np.zeros((t_rows, 3)),
        "forces": {k: np.array(v, dtype=np.float64) for k, v in data["forces"].items()},
    }

    out = logsout_to_simscape_output(base)
    assert out.force_columns is not None
    assert len(out.force_columns) == 237

    series, missing = out.to_force_series()
    assert len(series.frames) == t_rows
    # 26 wrenches per frame (all joints and external wrenches except the 2 with missing actuators)
    assert len(series.frames[0].wrenches) == 26
    # The trimmed fixture carries no per-hand columns (#11715) and the
    # canonical model has no feet on the ground (#11709): both unavailable.
    assert missing == (
        "joint_actuator:LF:torque",
        "joint_actuator:RF:torque",
        "grip:hand_left:force",
        "grip:hand_left:torque",
        "grip:hand_right:force",
        "grip:hand_right:torque",
        "contact:grf_left:force",
        "contact:grf_right:force",
    )
