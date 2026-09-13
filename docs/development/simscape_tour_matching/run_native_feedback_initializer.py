"""Native forward tracking initializer, recorded-effort replay and sextic probe."""

from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from scipy.interpolate import CubicSpline, CubicHermiteSpline
from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.motion_matching.acceleration_effort import (
    allocate_acceleration_effort,
)
from src.shared.python.motion_matching.continuous_forward import integrate_forward
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "candidate", "path", "payload", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--frequency", type=float, default=20.0)
    parser.add_argument("--effort-samples-per-frame", type=int, default=2)
    parser.add_argument(
        "--replay-input", choices=("sampled", "reference-state"), default="sampled"
    )
    parser.add_argument("--effort-regularization", type=float, default=0.0)
    parser.add_argument("--feedback-only", action="store_true")
    args = parser.parse_args()
    if not np.isfinite(args.effort_regularization) or args.effort_regularization < 0:
        raise ValueError("Effort regularization must be finite and nonnegative")
    if args.effort_samples_per_frame < 1:
        raise ValueError("Effort samples per frame must be positive")
    if args.output.exists() or not np.isfinite(args.frequency) or args.frequency <= 0:
        raise ValueError(
            "New output directory and positive tracking frequency required"
        )
    raw = args.model.read_bytes()
    spec = json.loads(raw)
    names = spec["coordinate_order"]
    n = len(names)
    candidate = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_text()), names, hashlib.sha256(raw).hexdigest()
    )
    path = json.loads(args.path.read_text())
    payload = json.loads(args.payload.read_text())
    if (
        path["model_sha256"] != candidate.document["model_sha256"]
        or payload["source_sha256"] != candidate.document["capture_sha256"]
    ):
        raise ValueError("Model and capture identities must match")
    q0 = np.array(candidate.document["q0"])
    v0 = np.array(candidate.document["qd0"])
    initial = np.r_[q0, v0]
    records = path["records"]
    reference_times = np.array([r["time_s"] for r in records])
    positions = np.array([r["coordinates"] for r in records])
    clock = np.array(payload["time_s"])
    horizon = float(clock[-1])
    if (
        reference_times[0] != 0
        or reference_times[-1] != horizon
        or np.max(abs(positions[0] - q0)) > 1e-7
    ):
        raise ValueError(
            "Reference must span capture and preserve original initial pose"
        )
    reference = CubicSpline(
        reference_times, positions, axis=0, bc_type=((1, v0), (2, np.zeros(n)))
    )
    model = NativePinocchioModel(spec)
    zero = dict.fromkeys(names, 0.0)
    acceleration_scales = np.r_[np.full(3, 0.1), np.ones(n - 3)]
    effort_scales = np.r_[np.full(3, 100.0), np.full(n - 3, 20.0)]

    def mapping(value: np.ndarray) -> dict[str, float]:
        return dict(zip(names, map(float, value), strict=True))

    latest: dict = {}

    def controlled(
        time: float, state: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, float]:
        latest.update(time_s=float(time), state=state.tolist())
        q, v = mapping(state[:n]), mapping(state[n:])
        free_map = model.accelerations(q, v, zero)
        free = np.array([free_map[name] for name in names])
        response = model.acceleration_derivatives(q, v, zero).deffort
        desired = (
            reference(time, 2)
            + 2 * args.frequency * (reference(time, 1) - state[n:])
            + args.frequency**2 * (reference(time) - state[:n])
        )
        allocation = allocate_acceleration_effort(
            response,
            free,
            desired,
            acceleration_scales=acceleration_scales,
            effort_scales=effort_scales,
            effort_regularization=args.effort_regularization,
        )
        latest.update(
            primitive_effort=allocation.effort.tolist(),
            response_rank=allocation.response_rank,
        )
        actual_map = model.accelerations(q, v, mapping(allocation.effort))
        actual = np.array([actual_map[name] for name in names])
        return (
            np.r_[state[n:], actual],
            allocation.effort,
            float(np.max(abs(actual - allocation.achieved_acceleration))),
        )

    args.output.mkdir()
    dense_clock = np.interp(
        np.linspace(
            0, len(clock) - 1, args.effort_samples_per_frame * (len(clock) - 1) + 1
        ),
        np.arange(len(clock)),
        clock,
    )
    try:
        feedback = integrate_forward(
            initial,
            dense_clock,
            lambda t, x: controlled(t, x)[0],
            rtol=1e-9,
            atol=1e-11,
            max_step=0.0005,
        )
    except (
        ValueError,
        RuntimeError,
        FloatingPointError,
        np.linalg.LinAlgError,
    ) as error:
        latest.update(
            failure=str(error),
            scope="Failed feedback initializer; not a trajectory",
            runner_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        )
        (args.output / "failure.json").write_text(json.dumps(latest, indent=2))
        raise
    derivatives = []
    efforts = []
    affine_errors = []
    closure = []
    predicted = []
    offsets = np.array(candidate.document["marker_offsets_m"])
    bodies = candidate.document["marker_bodies"]
    for time, state in zip(dense_clock, feedback.state, strict=True):
        derivative, u, error = controlled(float(time), state)
        derivatives.append(derivative)
        efforts.append(u)
        affine_errors.append(error)
        closure.append(np.concatenate(model.closure_errors()))
        predicted.append(
            model.marker_derivatives(mapping(state[:n]), bodies, offsets).positions_m
        )
    efforts = np.array(efforts)
    predicted = np.array(predicted)
    ids = [
        payload["labels"].index(label) for label in candidate.document["marker_labels"]
    ]
    targets = np.array(payload["points_world_m"])[:, ids]
    valid = np.array(payload["valid"])[:, ids].astype(bool)

    def metrics(prediction: np.ndarray) -> dict[str, float]:
        errors = np.linalg.norm(prediction - targets, axis=2)
        return {
            "rms_m": float(np.sqrt(np.mean(errors[valid] ** 2))),
            "terminal_rms_m": float(np.sqrt(np.mean(errors[-1, valid[-1]] ** 2))),
        }

    report = {
        "scope": "Feedback initializer and open-loop diagnostics, not final match acceptance",
        "frequency_rad_s": args.frequency,
        "effort_regularization": args.effort_regularization,
        "feedback_only": args.feedback_only,
        "model_sha256": hashlib.sha256(raw).hexdigest(),
        "path_sha256": hashlib.sha256(args.path.read_bytes()).hexdigest(),
        "candidate_sha256": candidate.sha256,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "feedback": metrics(predicted[:: args.effort_samples_per_frame]),
        "effort_sample_count": len(dense_clock),
        "replay_input": args.replay_input,
        "feedback_elapsed_s": feedback.elapsed_s,
        "feedback_closure_max_abs": float(np.max(abs(np.array(closure)))),
        "affine_response_max_abs_error": max(affine_errors),
        "primitive_effort_max_abs": np.max(abs(efforts), axis=0).tolist(),
        "effort_bounds_enforced": False,
    }
    np.savez_compressed(
        args.output / "feedback.npz",
        time=dense_clock,
        state=feedback.state,
        primitive_efforts=efforts,
        state_derivatives=np.asarray(derivatives),
        markers=predicted,
    )
    (args.output / "feedback-report.json").write_text(json.dumps(report, indent=2))
    if args.feedback_only:
        (args.output / "report.json").write_text(json.dumps(report, indent=2))
        return
    curve = CubicSpline(dense_clock, efforts, axis=0)
    reference_state = CubicHermiteSpline(
        dense_clock, feedback.state, np.asarray(derivatives), axis=0
    )

    def time_only_effort(time: float) -> np.ndarray:
        if args.replay_input == "reference-state":
            return controlled(time, reference_state(time))[1]
        return curve(time)

    def recorded(time: float, state: np.ndarray) -> np.ndarray:
        a = model.accelerations(
            mapping(state[:n]), mapping(state[n:]), mapping(time_only_effort(time))
        )
        return np.r_[state[n:], [a[name] for name in names]]

    replay = integrate_forward(
        initial, clock, recorded, rtol=1e-10, atol=1e-12, max_step=0.00025
    )
    replay_markers = np.array(
        [
            model.marker_derivatives(mapping(x[:n]), bodies, offsets).positions_m
            for x in replay.state
        ]
    )
    report["recorded_effort_replay"] = metrics(replay_markers)
    report["recorded_closure_max_abs"] = float(
        max(
            np.max(
                abs(
                    np.concatenate(
                        model.closure_residuals(mapping(x[:n]), mapping(x[n:]))
                    )
                )
            )
            for x in replay.state
        )
    )
    np.savez_compressed(
        args.output / "recorded-replay.npz",
        time=clock,
        state=replay.state,
        markers=replay_markers,
    )
    coefficients = (
        np.polynomial.polynomial.polyfit(dense_clock / horizon, efforts, 6)[::-1].T
        / horizon ** np.arange(6, -1, -1)[None, :]
    )
    root = next(j for j in spec["joints"] if j["parent"] == "world")
    rotation = np.array(root["parent_to_base"])[:3, :3].T
    coefficients[:3] = rotation.T @ coefficients[:3]
    document = candidate.document
    document["coefficients"] = coefficients.tolist()
    document["duration_s"] = horizon
    sextic = NativeReplayCandidate.from_document(
        document, names, hashlib.sha256(raw).hexdigest()
    )
    (args.output / "sextic-candidate.json").write_text(
        json.dumps(sextic.document, indent=2)
    )
    try:
        final = replay_candidate(raw, sextic, clock, max_step=0.00025)
        report["sextic_replay"] = metrics(final.markers_m)
        np.savez_compressed(
            args.output / "sextic-replay.npz",
            time=clock,
            state=final.integration.state,
            markers=final.markers_m,
        )
    except (ValueError, RuntimeError, FloatingPointError) as error:
        report["sextic_replay_failure"] = str(error)
    (args.output / "report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
