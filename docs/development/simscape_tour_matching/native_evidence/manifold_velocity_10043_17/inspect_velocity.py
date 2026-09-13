"""Inspect saved replay velocities and native chart conditioning; no integration."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pinocchio as pin

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.engines.physics_engines.pinocchio.python.native_manifold_model import (
    NativeManifoldPinocchioModel,
)
from src.shared.python.pose_interchange.joint_chart import SerialRotationChart
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for argument in ("model", "candidate", "scalar", "manifold", "output"):
        parser.add_argument("--" + argument, type=Path, required=True)
    args = parser.parse_args()
    spec = json.loads(args.model.read_bytes())
    names = spec["coordinate_order"]
    candidate = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_bytes()), names, digest(args.model)
    )
    scalar = NativePinocchioModel(spec)
    manifold = NativeManifoldPinocchioModel(spec)
    frame_names = [frame["name"] for frame in spec["frames"]]
    frame_ids = [scalar.model.getFrameId(name) for name in frame_names]
    velocity_indices = [
        scalar.model.joints[scalar.model.getJointId(name)].idx_v for name in names
    ]
    with np.load(args.scalar, allow_pickle=False) as archive:
        time, scalar_state = archive["time"], archive["state"]
    with np.load(args.manifold, allow_pickle=False) as archive:
        other_time, native_state = archive["time"], archive["native_state"]
        configurations, velocities = archive["configuration"], archive["velocity"]
    if (
        not np.array_equal(time, other_time)
        or scalar_state.shape != native_state.shape
        or scalar_state.shape != (time.size, 2 * len(names))
    ):
        raise ValueError("Saved replay clocks/state dimensions disagree")
    if not all(
        np.isfinite(x).all()
        for x in (time, scalar_state, native_state, configurations, velocities)
    ):
        raise ValueError("Nonfinite saved replay data")
    native_error = native_state[:, len(names) :] - scalar_state[:, len(names) :]
    physical = np.empty((len(time), len(frame_names), 6))
    scalar_physical = np.empty_like(physical)
    group_rows = {group.name: [] for group in manifold.adapter.groups}
    for index, t in enumerate(time):
        sq, sv = scalar_state[index, : len(names)], scalar_state[index, len(names) :]
        q = scalar.configuration(dict(zip(names, sq, strict=True)))
        v = np.zeros(scalar.model.nv)
        v[velocity_indices] = sv
        pin.forwardKinematics(scalar.model, scalar.data, q, v)
        alternate = manifold.frame_velocities(configurations[index], velocities[index])
        for frame_index, (name, frame_id) in enumerate(
            zip(frame_names, frame_ids, strict=True)
        ):
            expected = pin.getFrameVelocity(
                scalar.model,
                scalar.data,
                frame_id,
                pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            ).vector
            scalar_physical[index, frame_index] = expected
            physical[index, frame_index] = alternate[name] - expected
        for group in manifold.adapter.groups:
            indices = [names.index(name) for name in group.coordinates]
            chart = SerialRotationChart(group.axes)
            a, b = sq[indices], native_state[index, indices]
            va, vb = sv[indices], native_state[index, len(names) + np.array(indices)]
            ea, eb = chart.rate_map(a), chart.rate_map(b)
            sigma = np.linalg.svd(ea, compute_uv=False)
            omega_difference = eb @ vb - ea @ va
            geometry_term = (eb - ea) @ vb
            inverse_prediction = np.linalg.solve(ea, omega_difference - geometry_term)
            dv = vb - va
            if not np.allclose(inverse_prediction, dv, rtol=1e-7, atol=1e-10):
                raise AssertionError(
                    "Angular velocity discrepancy decomposition failed"
                )
            group_rows[group.name].append(
                {
                    "time_s": float(t),
                    "middle_angle_scalar_rad": float(a[1]),
                    "middle_angle_manifold_rad": float(b[1]),
                    "scalar_condition_number": float(sigma[0] / sigma[-1]),
                    "manifold_condition_number": chart.condition_number(b),
                    "scalar_distance_to_middle_pole_rad": float(
                        abs((a[1] % np.pi) - np.pi / 2)
                    ),
                    "native_velocity_difference_norm": float(np.linalg.norm(dv)),
                    "relative_joint_omega_difference_norm_rad_s": float(
                        np.linalg.norm(omega_difference)
                    ),
                    "geometry_term_norm_rad_s": float(np.linalg.norm(geometry_term)),
                    "mapped_rate_difference_norm_rad_s": float(np.linalg.norm(ea @ dv)),
                    "actual_rate_amplification": float(
                        np.linalg.norm(dv)
                        / max(np.linalg.norm(ea @ dv), np.finfo(float).tiny)
                    ),
                    "inverse_gain_bound": float(1 / sigma[-1]),
                    "native_rate_difference": dv.tolist(),
                    "native_coordinate_difference_rad": (b - a).tolist(),
                    "scalar_relative_joint_omega_norm_rad_s": float(
                        np.linalg.norm(ea @ va)
                    ),
                    "manifold_relative_joint_omega_norm_rad_s": float(
                        np.linalg.norm(eb @ vb)
                    ),
                }
            )
    native_peak = np.unravel_index(np.argmax(abs(native_error)), native_error.shape)

    def physical_peak(start: int) -> dict:
        norms = np.linalg.norm(physical[:, :, start : start + 3], axis=2)
        row, column = np.unravel_index(np.argmax(norms), norms.shape)
        return {
            "time_s": float(time[row]),
            "frame": frame_names[column],
            "norm": float(norms[row, column]),
            "vector": physical[row, column, start : start + 3].tolist(),
            "scalar_velocity_norm": float(
                np.linalg.norm(scalar_physical[row, column, start : start + 3])
            ),
            "manifold_velocity_norm": float(
                np.linalg.norm(
                    scalar_physical[row, column, start : start + 3]
                    + physical[row, column, start : start + 3]
                )
            ),
        }

    group_summary = {}
    for name, rows in group_rows.items():
        group_summary[name] = {
            "maximum_condition": max(rows, key=lambda r: r["scalar_condition_number"]),
            "maximum_native_rate_error": max(
                rows, key=lambda r: r["native_velocity_difference_norm"]
            ),
            "at_global_native_peak": rows[native_peak[0]],
        }
    coordinate_summary = []
    for col, name in enumerate(names):
        row = int(np.argmax(abs(native_error[:, col])))
        coordinate_summary.append(
            {
                "coordinate": name,
                "time_s": float(time[row]),
                "signed_error": float(native_error[row, col]),
                "scalar_rate": float(scalar_state[row, len(names) + col]),
                "manifold_native_rate": float(native_state[row, len(names) + col]),
            }
        )
    report = {
        "scope": "Saved run52 scalar versus manifold-1 physical velocity diagnostic; no new integration or acceptance claim.",
        "pinocchio_version": pin.__version__,
        "candidate_sha256": candidate.sha256,
        "input_sha256": {
            name: digest(getattr(args, name))
            for name in ("model", "candidate", "scalar", "manifold")
        },
        "script_sha256": digest(Path(__file__)),
        "sample_count": len(time),
        "frame_count": len(frame_names),
        "max_native_velocity_error": {
            "time_s": float(time[native_peak[0]]),
            "coordinate": names[native_peak[1]],
            "absolute_error": float(abs(native_error[native_peak])),
        },
        "max_physical_linear_velocity_error_m_s": physical_peak(0),
        "max_physical_angular_velocity_error_rad_s": physical_peak(3),
        "physical_velocity_convention": "Frame-origin, world-aligned, linear/angular; actual native scalar Pinocchio FK versus manifold frame API.",
        "coordinate_peaks": sorted(
            coordinate_summary, key=lambda r: abs(r["signed_error"]), reverse=True
        ),
        "group_summary": group_summary,
        "group_time_series": group_rows,
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
