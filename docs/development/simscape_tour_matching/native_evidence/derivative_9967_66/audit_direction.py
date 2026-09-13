"""One-column original-state replay derivative audit; no provider replacement."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from time import perf_counter
import traceback
import zipfile

import numpy as np
from src.engines.physics_engines.pinocchio.python.native_model import (
    NativePinocchioModel,
)
from src.engines.physics_engines.pinocchio.python.native_replay import replay_candidate
from src.shared.python.motion_matching.native_candidate import (
    NativeReplayCandidate,
    increment_native_bernstein,
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def utc():
    return datetime.now(timezone.utc).isoformat()


class BudgetModel(NativePinocchioModel):
    """Thin actual-acceleration counter; underlying numerical provider unchanged."""

    def __init__(self, specification):
        super().__init__(specification)
        self.calls = 0

    def accelerations(self, coordinates, rates, primitive_efforts):
        if self.calls >= 120000:
            raise RuntimeError("Acceleration budget exhausted: 120000 including audits")
        self.calls += 1
        return super().accelerations(coordinates, rates, primitive_efforts)


def metrics(analytic, finite):
    difference = finite - analytic
    return {
        "max_component_abs_error_m_per_nm": float(np.max(abs(difference))),
        "analytic_max_abs_m_per_nm": float(np.max(abs(analytic))),
        "finite_max_abs_m_per_nm": float(np.max(abs(finite))),
        "relative_l2_error": float(
            np.linalg.norm(difference) / max(np.linalg.norm(analytic), 1e-300)
        ),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "checkpoint", "sensitivity", "receipt", "runtime", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(exist_ok=False)
    raw = args.model.read_bytes()
    spec = json.loads(raw)
    names = spec["coordinate_order"]
    base = NativeReplayCandidate.from_document(
        json.loads(args.checkpoint.read_text())["candidate"], names, sha(args.model)
    )
    if (
        base.sha256
        != "e057a7e977cfe2a163fe4c5e7e5767c17d5025b4ef7f33166672afbf658c358d"
    ):
        raise ValueError("Unexpected fixed candidate")
    prior = json.loads(args.receipt.read_text())
    if (
        prior["candidate_sha256"] != base.sha256
        or prior["arguments"]["first_control"] != 4
    ):
        raise ValueError("Stored sensitivity identity/layout mismatch")
    for name, digest in prior["source_hashes"].items():
        if sha(args.runtime / name) != digest:
            raise ValueError("Runtime differs from exact64: " + name)
    data = np.load(args.sensitivity)
    clock = data["time_s"]
    if hashlib.sha256(clock.tobytes()).hexdigest() != prior["clock_sha256"]:
        raise ValueError("Clock identity mismatch")
    column = 3 * names.index("LSInputX") + 2
    if data["marker_jacobian"].shape != (
        len(clock),
        len(base.document["marker_labels"]),
        3,
        len(names) * 3,
    ):
        raise ValueError("Unexpected sensitivity shape")
    analytic = data["marker_jacobian"][..., column]
    np.savez_compressed(
        args.output / "analytic-direction.npz", time_s=clock, direction=analytic
    )
    for label, path in [
        ("model.json", args.model),
        ("checkpoint.json", args.checkpoint),
        ("sensitivity64-receipt.json", args.receipt),
        ("driver.py", Path(__file__)),
    ]:
        (args.output / label).write_bytes(path.read_bytes())
    with zipfile.ZipFile(
        args.output / "runtime-source.zip", "w", zipfile.ZIP_DEFLATED
    ) as archive:
        for path in args.runtime.rglob("*.py"):
            archive.write(path, path.relative_to(args.runtime))
    report = {
        "status": "running",
        "started_utc": utc(),
        "candidate_sha256": base.sha256,
        "first_control": 4,
        "coordinate": "LSInputX",
        "bernstein_index": 6,
        "column_index": column,
        "column_layout": "coordinate-major ascending B4 B5 B6",
        "rtol": 1e-12,
        "atol": 1e-14,
        "max_step": 0.000125,
        "acceleration_budget_per_replay": 120000,
        "clock_sha256": prior["clock_sha256"],
        "scope": "One marker Jacobian direction only; not whole-Jacobian acceptance",
        "audit_threshold_unchanged": 1e-3,
        "inputs": {
            str(p): sha(p)
            for p in (
                args.model,
                args.checkpoint,
                args.sensitivity,
                args.receipt,
                Path(__file__),
            )
        },
        "runtime": str(args.runtime),
        "source_hashes": prior["source_hashes"],
        "runs": [],
        "comparisons": {},
    }

    def save():
        (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")

    save()
    previous = None
    try:
        for step in (1e-5, 1e-4, 1e-3):
            markers = []
            for sign in (1, -1):
                delta = np.zeros((len(names), 7))
                delta[names.index("LSInputX"), 6] = sign * step
                candidate = increment_native_bernstein(
                    base, delta, basis_duration_s=base.document["duration_s"]
                )
                for field in (
                    "q0",
                    "qd0",
                    "marker_labels",
                    "marker_bodies",
                    "marker_offsets_m",
                ):
                    if candidate.document[field] != base.document[field]:
                        raise ValueError("Protected candidate field changed: " + field)
                label = f"{sign:+d}_{step:g}"
                engines = []

                def factory(specification, owned_engines=engines):
                    engine = BudgetModel(specification)
                    owned_engines.append(engine)
                    return engine

                run = {
                    "label": label,
                    "candidate_sha256": candidate.sha256,
                    "started_utc": utc(),
                    "status": "running",
                }
                report["runs"].append(run)
                save()
                start = perf_counter()
                (args.output / (label + "-candidate.json")).write_text(
                    json.dumps(candidate.document, indent=2) + "\n"
                )
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
                    markers.append(result.markers_m)
                    np.savez_compressed(
                        args.output / (label + ".npz"),
                        time_s=clock,
                        state=result.integration.state,
                        markers_m=result.markers_m,
                    )
                    run.update(
                        status="completed",
                        integration_evaluations=result.integration.evaluations,
                    )
                finally:
                    run.update(
                        ended_utc=utc(),
                        elapsed_s=perf_counter() - start,
                        acceleration_calls=sum(e.calls for e in engines),
                    )
                    save()
            finite = (markers[0] - markers[1]) / (2 * step)
            np.savez_compressed(
                args.output / f"direction_{step:g}.npz",
                finite_difference=finite,
                error=finite - analytic,
            )
            regions = {
                "whole": np.ones(len(clock), dtype=bool),
                "early_t_le_0p6": clock <= 0.6,
                "transition_t_gt_0p6": clock > 0.6,
            }
            comparison = {
                name: metrics(analytic[mask], finite[mask])
                for name, mask in regions.items()
            }
            comparison["whole_relative_below_1e-3"] = (
                comparison["whole"]["relative_l2_error"] <= 1e-3
            )
            if previous is not None:
                comparison["h_dependence_vs_previous"] = metrics(previous, finite)
            previous = finite
            report["comparisons"][str(step)] = comparison
            save()
        report["status"] = "completed"
    except (ValueError, RuntimeError, FloatingPointError) as error:
        report.update(
            status="failed", error=str(error), traceback=traceback.format_exc()
        )
        raise
    finally:
        report["ended_utc"] = utc()
        save()


if __name__ == "__main__":
    main()
