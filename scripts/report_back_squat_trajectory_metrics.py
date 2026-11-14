"""Read bar-path, velocity and phase metrics from the committed squat bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

DEFAULT_BUNDLE = (
    Path(__file__).resolve().parents[1]
    / "docs"
    / "development"
    / "lifting"
    / "evidence"
    / "back_squat"
    / "back_squat_bundle.npz"
)


def _path_metrics(
    time_s: np.ndarray, forward_m: np.ndarray, vertical_m: np.ndarray
) -> dict[str, float]:
    """Summarize one sampled sagittal path using central finite differences."""
    if (
        time_s.ndim != 1
        or forward_m.shape != time_s.shape
        or vertical_m.shape != time_s.shape
    ):
        raise ValueError(
            "time and path coordinates must be matching one-dimensional arrays"
        )
    if (
        time_s.size < 3
        or not np.isfinite(time_s).all()
        or not np.isfinite(forward_m).all()
    ):
        raise ValueError("path needs at least three finite samples")
    if not np.isfinite(vertical_m).all() or np.any(np.diff(time_s) <= 0.0):
        raise ValueError(
            "vertical coordinates must be finite and sample times strictly increasing"
        )

    path = np.column_stack((forward_m, vertical_m))
    velocity = np.gradient(path, time_s, axis=0, edge_order=2)
    bottom = int(np.argmin(vertical_m))
    if bottom == 0 or bottom == time_s.size - 1:
        raise ValueError("trajectory bottom must be inside the sampled interval")
    down_index = int(np.argmin(velocity[:, 1]))
    up_index = int(np.argmax(velocity[:, 1]))
    return {
        "start_time_s": float(time_s[0]),
        "bottom_time_s": float(time_s[bottom]),
        "end_time_s": float(time_s[-1]),
        "descent_s": float(time_s[bottom] - time_s[0]),
        "ascent_s": float(time_s[-1] - time_s[bottom]),
        "vertical_excursion_m": float(vertical_m[0] - vertical_m[bottom]),
        "forward_range_m": float(np.ptp(forward_m)),
        "forward_bottom_minus_start_m": float(forward_m[bottom] - forward_m[0]),
        "sagittal_path_length_m": float(
            np.linalg.norm(np.diff(path, axis=0), axis=1).sum()
        ),
        "mean_down_vertical_speed_m_s": float(
            (vertical_m[bottom] - vertical_m[0]) / (time_s[bottom] - time_s[0])
        ),
        "mean_up_vertical_speed_m_s": float(
            (vertical_m[-1] - vertical_m[bottom]) / (time_s[-1] - time_s[bottom])
        ),
        "peak_down_vertical_speed_m_s": float(velocity[down_index, 1]),
        "peak_down_vertical_speed_time_s": float(time_s[down_index]),
        "peak_up_vertical_speed_m_s": float(velocity[up_index, 1]),
        "peak_up_vertical_speed_time_s": float(time_s[up_index]),
    }


def extract_metrics(bundle_path: Path = DEFAULT_BUNDLE) -> dict[str, Any]:
    """Extract documented metrics without rerunning the optimizer or simulation."""
    with np.load(Path(bundle_path), allow_pickle=False) as archive:
        try:
            manifest = json.loads(str(archive["manifest"].item()))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{bundle_path} has no valid JSON manifest") from exc

    provenance = manifest.get("provenance")
    if not isinstance(provenance, dict):
        raise ValueError("bundle provenance is missing")
    sagittal = provenance.get("sagittal_solution")
    if not isinstance(sagittal, dict):
        raise ValueError("bundle sagittal_solution provenance is missing")
    time_s = np.asarray(sagittal.get("time_s"), dtype=float)
    if time_s.ndim != 1 or time_s.size < 3 or not np.isfinite(time_s).all():
        raise ValueError("time_s must contain at least three finite samples")
    sample_intervals = np.diff(time_s)
    if np.any(sample_intervals <= 0.0):
        raise ValueError("time_s samples must be strictly increasing")

    optimizer_bar = np.asarray(sagittal.get("bar_m"), dtype=float)
    if optimizer_bar.shape != (time_s.size, 2):
        raise ValueError(
            "sagittal_solution.bar_m must be (N, 2) forward/up coordinates"
        )

    ik_bar = np.asarray(provenance.get("ik_bar_path_m"), dtype=float)
    forward_axis = np.asarray(
        provenance.get("ik", {}).get("sagittal_forward"), dtype=float
    )
    if ik_bar.shape != (time_s.size, 3) or forward_axis.shape != (3,):
        raise ValueError(
            "ik_bar_path_m and sagittal_forward must match the full solution"
        )
    if not np.isfinite(forward_axis).all() or not np.isclose(
        np.linalg.norm(forward_axis[:2]), 1.0
    ):
        raise ValueError("sagittal_forward must be a finite horizontal unit vector")

    forward_world = ik_bar[:, :2] @ forward_axis[:2]
    optimizer = provenance.get("optimizer")
    if not isinstance(optimizer, dict):
        raise ValueError("optimizer provenance is missing")
    return {
        "bundle": Path(bundle_path).name,
        "source_commit": provenance.get("source_commit"),
        "tools_pin": provenance.get("tools_pin"),
        "optimizer": optimizer,
        "sampling": {
            "samples": int(time_s.size),
            "start_time_s": float(time_s[0]),
            "end_time_s": float(time_s[-1]),
            "dt_s": float(sample_intervals[0]),
            "uniform_sampling": bool(
                np.allclose(sample_intervals, sample_intervals[0])
            ),
            "velocity_method": "numpy.gradient against saved sample times; no added filtering",
        },
        "paths": {
            "optimizer_planar": {
                "coordinate_frame": "optimizer sagittal coordinates: forward/up relative to ankle (m)",
                "metrics": _path_metrics(
                    time_s, optimizer_bar[:, 0], optimizer_bar[:, 1]
                ),
            },
            "retargeted_high_bar_ik": {
                "coordinate_frame": "world bar-centre position projected onto sagittal_forward; vertical is world z (m)",
                "bar_frame": provenance.get("bar_frame"),
                "metrics": _path_metrics(time_s, forward_world, ik_bar[:, 2]),
            },
        },
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=DEFAULT_BUNDLE)
    args = parser.parse_args()
    print(json.dumps(extract_metrics(args.bundle), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
