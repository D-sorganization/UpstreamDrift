"""Build the Simscape ground-truth fixture for the OpenSim equivalence test.

Reads ``trial_001_*.csv`` from the canonical Simscape dataset directory and
distils it into a small ``.npz`` that ships with the test suite. Issue
#4131 mandates this distillation step so the equivalence test can run on
CI without MATLAB or the multi-megabyte trial CSV.

Usage
-----
    python3 -m tests.fixtures.opensim_simscape_equivalence.build_fixture \\
        --trial src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/Scripts/Dataset\\ Generator/golf_swing_dataset_20251030/trial_001_20251030_174116.csv

The output ``simscape_reference.npz`` is committed to the repo at
``tests/fixtures/opensim_simscape_equivalence/simscape_reference.npz``.
Re-run only when the canonical Simscape model is regenerated. The data is
in the **Simscape Z-up world** convention (per
``shared/models/golf_humanoid_topology.yaml``).

Pose definitions
----------------
- ``address``        — synthetic neutral pose (q = 0). The Simscape trial
                       starts at top-of-backswing, not address; the
                       address pose is therefore the analytic neutral
                       configuration with zero grip displacement from the
                       model origin (the origin convention used by
                       ``compute_skeleton_fk.m``).
- ``top_of_backswing`` — first frame of the trial (lowest CHSpeed).
- ``impact``         — frame at maximum CHSpeed.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

POSE_NAMES = ("address", "top_of_backswing", "impact")


def _column(headers: list[str], name: str) -> int:
    try:
        return headers.index(name)
    except ValueError as exc:
        raise KeyError(
            f"required column {name!r} not present in Simscape CSV; "
            f"trial may be from an older dataset version"
        ) from exc


def _xyz(headers: list[str], row: list[str], stem: str) -> np.ndarray:
    return np.array(
        [float(row[_column(headers, f"{stem}_{i}")]) for i in (1, 2, 3)],
        dtype=np.float64,
    )


def _grip_orientation(
    grip: np.ndarray,
    lh: np.ndarray,
    rh: np.ndarray,
    clubhead: np.ndarray,
) -> np.ndarray:
    """Build a grip-frame quaternion ``[w, x, y, z]`` from hand markers.

    The grip frame is constructed as:
        +x := (clubhead − grip) normalized   (shaft axis, distal)
        +y := (right_hand − left_hand) normalized projected ⟂ x
        +z := x × y                          (right-hand rule)

    Returns a unit quaternion in canonical ``[w, x, y, z]`` ordering.
    """
    x = clubhead - grip
    n = float(np.linalg.norm(x))
    if n < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    x = x / n

    y_raw = rh - lh
    y_raw = y_raw - x * float(np.dot(y_raw, x))
    n = float(np.linalg.norm(y_raw))
    if n < 1e-9:
        # Pick any vector orthogonal to x (degenerate fallback).
        y_raw = np.array([0.0, 0.0, 1.0]) - x * x[2]
        n = float(np.linalg.norm(y_raw))
    y = y_raw / n
    z = np.cross(x, y)
    rot = np.column_stack([x, y, z])

    # Shepperd's method — rotation matrix → quaternion [w, x, y, z].
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        return np.array(
            [
                0.25 / s,
                (rot[2, 1] - rot[1, 2]) * s,
                (rot[0, 2] - rot[2, 0]) * s,
                (rot[1, 0] - rot[0, 1]) * s,
            ],
            dtype=np.float64,
        )
    diag = (rot[0, 0], rot[1, 1], rot[2, 2])
    i = int(np.argmax(diag))
    j = (i + 1) % 3
    k = (j + 1) % 3
    s = 2.0 * np.sqrt(1.0 + rot[i, i] - rot[j, j] - rot[k, k])
    q = np.zeros(4, dtype=np.float64)
    q[0] = (rot[k, j] - rot[j, k]) / s
    q[i + 1] = 0.25 * s
    q[j + 1] = (rot[j, i] + rot[i, j]) / s
    q[k + 1] = (rot[k, i] + rot[i, k]) / s
    return q


def build_fixture_from_trial(trial_csv: Path) -> dict[str, np.ndarray]:
    """Read a Simscape trial CSV and return the equivalence-test fixture dict."""
    if not trial_csv.is_file():
        raise FileNotFoundError(f"Simscape trial CSV not found: {trial_csv}")

    with open(trial_csv, encoding="utf-8") as f:
        reader = csv.reader(f)
        headers = next(reader)
        rows = list(reader)

    n = len(rows)
    if n < 3:
        raise ValueError(f"Simscape trial too short ({n} rows); need >= 3")

    times = np.array(
        [float(r[_column(headers, "time")]) for r in rows], dtype=np.float64
    )
    chs = np.array(
        [float(r[_column(headers, "ClubLogs_CHSpeed")]) for r in rows],
        dtype=np.float64,
    )

    grip_traj = np.array(
        [_xyz(headers, r, "MidpointCalcsLogs_MPGlobalPosition") for r in rows],
        dtype=np.float64,
    )
    clubhead_traj = np.array(
        [_xyz(headers, r, "ClubLogs_CHGlobalPosition") for r in rows],
        dtype=np.float64,
    )
    lh_traj = np.array(
        [_xyz(headers, r, "LWLogs_LHGlobalPosition") for r in rows],
        dtype=np.float64,
    )
    rh_traj = np.array(
        [_xyz(headers, r, "RWLogs_RHGlobalPosition") for r in rows],
        dtype=np.float64,
    )

    grip_quat_traj = np.array(
        [
            _grip_orientation(grip_traj[i], lh_traj[i], rh_traj[i], clubhead_traj[i])
            for i in range(n)
        ],
        dtype=np.float64,
    )

    impact_idx = int(np.argmax(chs))
    if impact_idx == 0:
        impact_idx = n - 1
    top_idx = int(np.argmin(chs[: max(impact_idx, 1)]))

    # Address pose — synthetic neutral, with grip / clubhead at the
    # canonical zero-pose locations. The Simscape compute_skeleton_fk.m
    # at q = 0 places the grip directly above the model origin with the
    # shaft pointing along +y in the address frame. Since the trial does
    # not include a true address sample, we emit zero-grip / zero-quat
    # and let the test downstream compare against the OpenSim FK at
    # ``OPENSIM_NEUTRAL_POSE``. The address-pose tolerance is therefore
    # an internal-consistency check at q = 0.
    address_grip = np.zeros(3, dtype=np.float64)
    address_clubhead = np.array([0.0, 0.0, -1.0], dtype=np.float64)
    address_grip_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    address_clubhead_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    out: dict[str, np.ndarray] = {
        "trial_path": np.array(str(trial_csv), dtype="<U512"),
        "n_samples": np.array(n, dtype=np.int64),
        "time": times,
        "chs": chs,
        "grip_traj": grip_traj,
        "clubhead_traj": clubhead_traj,
        "grip_quat_traj": grip_quat_traj,
        "lh_traj": lh_traj,
        "rh_traj": rh_traj,
        "impact_idx": np.array(impact_idx, dtype=np.int64),
        "top_idx": np.array(top_idx, dtype=np.int64),
        # Per-pose snapshots for fast random access in the test.
        "pose_names": np.array(POSE_NAMES, dtype="<U32"),
        "address_grip": address_grip,
        "address_clubhead": address_clubhead,
        "address_grip_quat": address_grip_quat,
        "address_clubhead_quat": address_clubhead_quat,
        "top_of_backswing_grip": grip_traj[top_idx],
        "top_of_backswing_clubhead": clubhead_traj[top_idx],
        "top_of_backswing_grip_quat": grip_quat_traj[top_idx],
        "impact_grip": grip_traj[impact_idx],
        "impact_clubhead": clubhead_traj[impact_idx],
        "impact_grip_quat": grip_quat_traj[impact_idx],
    }
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trial", required=True, type=Path)
    parser.add_argument(
        "--out",
        default=Path(__file__).with_name("simscape_reference.npz"),
        type=Path,
    )
    args = parser.parse_args()

    fixture = build_fixture_from_trial(args.trial)
    np.savez_compressed(args.out, **fixture)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes, {len(fixture)} keys)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
