"""Run the MJX knot optimiser head-to-head benchmark (#11058).

Run as ``python -m scripts.benchmark_mjx_knot_solvers --work <scratch>`` with
the JAX/MJX interpreter, so every solver runs on the same MuJoCo build. Each
(capture, solver) is one full pipeline run; its ``receipt.json`` (and the MJX
receipt) is copied under ``--evidence`` with a ``run.json`` of the command,
exit code and wall clock. ``--render-only`` rebuilds ``rows.json`` and
``REPORT.md`` from the committed receipts without running anything.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from src.shared.python.motion_matching.solver_benchmark import (
    MJX_SOLVERS,
    promotion_decision,
    render_report,
    row_from_receipt,
    unavailable_row,
)

REPO = Path(__file__).resolve().parents[1]
FULL_BODY = "docs/development/full_body_models"
EVIDENCE = REPO / FULL_BODY / "evidence/mjx_benchmark"
# The canonical calibrated runs (CANONICAL_RUN.md, section 2A).
CAPTURES: dict[str, list[str]] = {
    "driver": [
        "--spec",
        f"{FULL_BODY}/full_body_spec_anthro_driver.json",
        "--capture",
        "driver",
        "--static-seeds",
    ],
    "iron": [
        "--spec",
        f"{FULL_BODY}/full_body_spec_anthro_iron7.json",
        "--capture",
        "iron",
        "--static-seeds",
        "--zmp-filter",
    ],
}
SOLVER_ORDER = ("none", "shooting", "mjx-adam", "mjx-lbfgs", "ipopt")
# Where the benchmark departs from the #11058 solver list, and why.
SOLVER_NOTES = (
    "SciPy arm: `mjx-lbfgs` runs L-BFGS-B over the same `delta_knots` on the "
    "exact JAX gradient instead of `least_squares` (trf) with a finite-difference "
    "Jacobian through the shared simulator; one FD Jacobian needs one full "
    "shared-simulator replay per knot parameter, so that arm was not run.",
    "Every optimised reference is rescored through the shared "
    "`FullBodySimulator` (`shared_simulator_replay`); the MJX plant's own RMS "
    "is not used for any comparison.",
    "`ipopt` has no pipeline stage; it is recorded as unavailable with the "
    "interpreter's binding status.",
    "Promotion needs the candidate's replay RMS <= shooting fit on every "
    "capture, a `converged` stop inside the iteration budget (Adam has no "
    "convergence test, so it can only stop on the budget) and a downswing "
    "weight fraction above zero. Each MJX method is judged separately.",
)


def solver_args(solver: str, *, shooting_passes: int, mjx_iterations: int) -> list[str]:
    """Pipeline flags that select ``solver``; every other flag is shared."""
    if solver == "none":
        return []
    if solver == "shooting":
        return ["--shooting-fit", str(shooting_passes)]
    if solver in MJX_SOLVERS:
        method = solver.removeprefix("mjx-")
        return [
            "--trajectory-optimiser",
            "mjx-knots",
            "--mjx-method",
            method,
            "--mjx-iterations",
            str(mjx_iterations),
        ]
    raise ValueError(f"solver {solver!r} is not a pipeline solver")


def ipopt_unavailable_reason() -> str | None:
    """None when an IPOPT binding is importable in this interpreter."""
    found = [m for m in ("cyipopt", "pydrake") if importlib.util.find_spec(m)]
    if found:
        return None
    return (
        f"no IPOPT binding (cyipopt or pydrake) importable in Python "
        f"{platform.python_version()} on {platform.system()}"
    )


def run_case(capture: str, solver: str, args: argparse.Namespace) -> None:
    out = Path(args.work) / f"{capture}_{solver}"
    command = [
        sys.executable,
        "-m",
        "src.shared.python.motion_matching.pipeline.cli",
        *CAPTURES[capture],
        "--out",
        str(out),
        *solver_args(
            solver,
            shooting_passes=args.shooting_passes,
            mjx_iterations=args.mjx_iterations,
        ),
    ]
    start = time.perf_counter()
    with open(
        Path(args.work) / f"{capture}_{solver}.log", "w", encoding="utf-8"
    ) as log:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            command, cwd=REPO, stdout=log, stderr=subprocess.STDOUT, check=False
        )
    elapsed = time.perf_counter() - start
    dest = Path(args.evidence) / f"{capture}_{solver}"
    dest.mkdir(parents=True, exist_ok=True)
    for name in ("receipt.json", "mjx_optimisation_receipt.json"):
        if (out / name).is_file():
            shutil.copyfile(out / name, dest / name)
    recorded = ["python", *command[1:]]
    recorded[recorded.index("--out") + 1] = "<work>"
    run = {"command": recorded, "returncode": proc.returncode, "wall_clock_s": elapsed}
    (dest / "run.json").write_text(json.dumps(run, indent=2) + "\n", encoding="utf-8")


def load_row(capture: str, solver: str, evidence: Path) -> dict[str, Any]:
    case = evidence / f"{capture}_{solver}"
    if solver == "ipopt":
        note = case / "unavailable.json"
        reason = json.loads(note.read_text(encoding="utf-8"))["reason"]
        return unavailable_row(capture, solver, reason)
    run = json.loads((case / "run.json").read_text(encoding="utf-8"))
    if run["returncode"] != 0 or not (case / "receipt.json").is_file():
        return unavailable_row(
            capture, solver, f"pipeline exited {run['returncode']}; see run log"
        )
    mjx_path = case / "mjx_optimisation_receipt.json"
    return row_from_receipt(
        capture,
        solver,
        json.loads((case / "receipt.json").read_text(encoding="utf-8")),
        wall_clock_s=float(run["wall_clock_s"]),
        mjx_receipt=(
            json.loads(mjx_path.read_text(encoding="utf-8"))
            if mjx_path.is_file()
            else None
        ),
    )


def render(evidence: Path) -> list[dict[str, Any]]:
    run = json.loads((evidence / "provenance.json").read_text(encoding="utf-8"))
    # The notes describe the method, not the run, so they come from this file.
    provenance = {**run, "notes": list(SOLVER_NOTES)}
    rows = [
        load_row(capture, solver, evidence)
        for capture in provenance["captures"]
        for solver in provenance["solvers"]
    ]
    decisions = [
        promotion_decision(rows, candidate=solver)
        for solver in sorted(MJX_SOLVERS)
        if solver in provenance["solvers"]
    ]
    (evidence / "rows.json").write_text(
        json.dumps({"rows": rows, "decisions": decisions}, indent=2) + "\n",
        encoding="utf-8",
    )
    (evidence / "REPORT.md").write_text(
        render_report(rows, decisions, provenance), encoding="utf-8"
    )
    return decisions


def _git_head() -> str:
    return subprocess.run(  # noqa: S603, S607 - fixed argv
        ["git", "rev-parse", "HEAD"],
        cwd=REPO,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", help="scratch directory for full run outputs")
    parser.add_argument("--evidence", default=str(EVIDENCE))
    parser.add_argument(
        "--captures", nargs="+", default=list(CAPTURES), choices=list(CAPTURES)
    )
    parser.add_argument(
        "--solvers", nargs="+", default=list(SOLVER_ORDER), choices=SOLVER_ORDER
    )
    parser.add_argument("--shooting-passes", type=int, default=8)
    parser.add_argument("--mjx-iterations", type=int, default=40)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args(argv)
    evidence = Path(args.evidence)
    if not args.render_only:
        if not args.work:
            parser.error("--work is required unless --render-only")
        import mujoco  # the build every solver ran on

        Path(args.work).mkdir(parents=True, exist_ok=True)
        evidence.mkdir(parents=True, exist_ok=True)
        provenance = {
            "commit": _git_head(),
            "python": platform.python_version(),
            "mujoco": mujoco.__version__,
            "host": platform.node(),
            "captures": args.captures,
            "solvers": args.solvers,
            "shooting_passes": args.shooting_passes,
            "mjx_iterations": args.mjx_iterations,
        }
        (evidence / "provenance.json").write_text(
            json.dumps(provenance, indent=2) + "\n", encoding="utf-8"
        )
        for capture in args.captures:
            for solver in args.solvers:
                if solver == "ipopt":
                    reason = ipopt_unavailable_reason()
                    case = evidence / f"{capture}_ipopt"
                    case.mkdir(parents=True, exist_ok=True)
                    # An importable binding still has no pipeline stage here.
                    reason = reason or "IPOPT importable but has no pipeline stage"
                    (case / "unavailable.json").write_text(
                        json.dumps({"reason": reason}, indent=2) + "\n",
                        encoding="utf-8",
                    )
                    continue
                run_case(capture, solver, args)
    decisions = render(evidence)
    print(json.dumps(decisions, indent=2))  # noqa: T201 - CLI output
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
