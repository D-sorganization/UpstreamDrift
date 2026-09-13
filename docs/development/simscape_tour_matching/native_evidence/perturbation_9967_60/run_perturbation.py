"""Bounded full-prefix input perturbation propagation from the original state."""

import argparse
import hashlib
import json
from pathlib import Path
from time import perf_counter

import numpy as np

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.motion_matching.native_candidate import (
    NativeReplayCandidate,
    increment_native_bernstein,
)
from src.shared.python.pose_interchange.native_joint_state import (
    NativeJointStateAdapter,
)


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BudgetModel(NativePinocchioModel):
    """Count every acceleration call, including the replay's sample audits."""

    def __init__(self, specification):
        super().__init__(specification)
        self.calls = 0
        self.last_coordinates = None

    def accelerations(self, coordinates, rates, primitive_efforts):
        if self.calls >= 120000:
            raise RuntimeError(
                "Acceleration budget exhausted:120000 calls including sample audits"
            )
        self.calls += 1
        self.last_coordinates = dict(coordinates)
        return super().accelerations(coordinates, rates, primitive_efforts)


def angular_velocities(adapter, states):
    names = adapter.coordinate_order
    zeros = dict.fromkeys(names, 0.0)
    values = []
    for state in states:
        pose = adapter.export(
            dict(zip(names, state[: len(names)], strict=True)),
            dict(zip(names, state[len(names) :], strict=True)),
            zeros,
            zeros,
        )
        values.append(
            [pose.rotations[g.name].omega_parent_rad_s for g in adapter.groups]
        )
    return np.asarray(values)


def max_by_time(values, vector=False):
    array = np.linalg.norm(values, axis=-1) if vector else abs(values)
    return np.max(array.reshape((len(array), -1)), axis=1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "candidate", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Require a new immutable output directory")
    raw = args.model.read_bytes()
    spec = json.loads(raw)
    names = spec["coordinate_order"]
    base = NativeReplayCandidate.from_document(
        json.loads(args.candidate.read_bytes()), names, digest(args.model)
    )
    if base.document["duration_s"] != 0.85:
        raise ValueError("This diagnostic requires original .85-second candidate")
    adapter = NativeJointStateAdapter(spec)
    clock = np.linspace(0, 0.85, 307)
    args.output.mkdir()
    (args.output / "original-candidate.json").write_text(
        json.dumps(base.document, indent=2) + "\n"
    )
    report = {
        "scope": "Original-state LSInputX degree6 final Bernstein-control perturbation; no state resets or fit acceptance.",
        "model_sha256": digest(args.model),
        "base_candidate_sha256": base.sha256,
        "script_sha256": digest(Path(__file__)),
        "coordinate": "LSInputX",
        "bernstein_index": 6,
        "basis_duration_s": 0.85,
        "rtol": 1e-12,
        "atol": 1e-14,
        "max_step": 0.000125,
        "max_acceleration_calls_including_audits": 120000,
        "physical_angular_convention": "Relative joint angular velocity, expressed in each native joint base, via NativeJointStateAdapter.",
        "angular_group_names": [g.name for g in adapter.groups],
        "runs": [],
        "comparisons": {},
    }
    outcomes = {}

    def save_report():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save_report()
    for label, delta in [
        ("baseline", 0.0),
        ("plus_1e-6", 1e-6),
        ("minus_1e-6", -1e-6),
        ("plus_1e-4", 1e-4),
        ("minus_1e-4", -1e-4),
        ("baseline_repeat", 0.0),
    ]:
        controls = np.zeros((len(names), 7))
        controls[names.index("LSInputX"), 6] = delta
        candidate = (
            base
            if delta == 0
            else increment_native_bernstein(base, controls, basis_duration_s=0.85)
        )
        candidate_document = candidate.document
        for field in (
            "q0",
            "qd0",
            "marker_labels",
            "marker_bodies",
            "marker_offsets_m",
        ):
            if candidate_document[field] != base.document[field]:
                raise AssertionError(
                    "Perturbation altered protected initial state/attachment"
                )
        (args.output / (label + "-candidate.json")).write_text(
            json.dumps(candidate_document, indent=2) + "\n"
        )
        engines = []

        def factory(specification, owned_engines=engines):
            engine = BudgetModel(specification)
            owned_engines.append(engine)
            return engine

        start = perf_counter()
        run = {"label": label, "delta_nm": delta, "candidate_sha256": candidate.sha256}
        try:
            result = replay_candidate(
                raw,
                candidate,
                clock,
                model_factory=factory,
                rtol=1e-12,
                atol=1e-14,
                max_step=0.000125,
            )
            state = result.integration.state
            omega = angular_velocities(adapter, state)
            q, v = state[:, : len(names)], state[:, len(names) :]
            outcomes[label] = {
                "q": q,
                "v": v,
                "markers": result.markers_m,
                "omega": omega,
            }
            np.savez_compressed(
                args.output / (label + ".npz"),
                time=clock,
                state=state,
                markers=result.markers_m,
                joint_omega=omega,
            )
            run.update(
                status="completed",
                integration_evaluations=result.integration.evaluations,
                acceleration_calls=engines[0].calls,
                closure_pose_max_abs=result.closure_pose_max_abs,
                closure_velocity_max_abs=result.closure_velocity_max_abs,
            )
        except (
            RuntimeError,
            ValueError,
            FloatingPointError,
            np.linalg.LinAlgError,
        ) as error:
            run.update(
                status="failed",
                error_type=type(error).__name__,
                error=str(error),
                acceleration_calls=engines[0].calls if engines else 0,
                latest_trial_coordinates=engines[0].last_coordinates
                if engines
                else None,
            )
        run["elapsed_s"] = perf_counter() - start
        report["runs"].append(run)
        save_report()
    if "baseline" in outcomes:
        baseline = outcomes["baseline"]
        for label, values in outcomes.items():
            if label == "baseline":
                continue
            by_time = {
                name: max_by_time(array - baseline[name], name in ("markers", "omega"))
                for name, array in values.items()
            }
            report["comparisons"][label] = {
                "maximum": {
                    name: float(max(series)) for name, series in by_time.items()
                },
                "time_of_maximum": {
                    name: float(clock[int(np.argmax(series))])
                    for name, series in by_time.items()
                },
                "difference_by_time": {
                    name: series.tolist() for name, series in by_time.items()
                },
            }
        central = {}
        for delta, suffix in [(1e-6, "1e-6"), (1e-4, "1e-4")]:
            plus, minus = "plus_" + suffix, "minus_" + suffix
            if plus not in outcomes or minus not in outcomes:
                continue
            response = {
                name: (outcomes[plus][name] - outcomes[minus][name]) / (2 * delta)
                for name in baseline
            }
            even = {
                name: (
                    outcomes[plus][name] + outcomes[minus][name] - 2 * baseline[name]
                )
                / (2 * delta)
                for name in baseline
            }
            central[suffix] = response
            np.savez_compressed(
                args.output / ("central_" + suffix + ".npz"),
                time=clock,
                **response,
                **{"even_" + n: a for n, a in even.items()},
            )
            report["comparisons"]["central_" + suffix] = {
                "delta_nm": delta,
                "maximum_gain_per_nm": {
                    n: float(np.max(max_by_time(a, n in ("markers", "omega"))))
                    for n, a in response.items()
                },
                "maximum_even_response_per_nm": {
                    n: float(np.max(max_by_time(a, n in ("markers", "omega"))))
                    for n, a in even.items()
                },
                "even_to_odd_maxnorm_ratio": {
                    n: float(
                        np.linalg.norm(even[n].reshape(-1), np.inf)
                        / max(
                            np.linalg.norm(response[n].reshape(-1), np.inf),
                            np.finfo(float).tiny,
                        )
                    )
                    for n in response
                },
            }
        if len(central) == 2:
            a, b = central["1e-6"], central["1e-4"]
            report["central_amplitude_comparison"] = {
                n: {
                    "relative_max_component_disagreement": float(
                        np.max(abs(a[n] - b[n]))
                        / max(np.max(abs(b[n])), np.finfo(float).tiny)
                    ),
                    "difference_by_time": max_by_time(
                        a[n] - b[n], n in ("markers", "omega")
                    ).tolist(),
                }
                for n in a
            }
    report["status"] = (
        "completed"
        if all(r["status"] == "completed" for r in report["runs"])
        else "partial_or_failed"
    )
    report["artifacts_sha256"] = {
        f.name: digest(f) for f in args.output.glob("*") if f.name != "report.json"
    }
    save_report()


if __name__ == "__main__":
    main()
