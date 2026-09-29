#!/usr/bin/env python3
"""CLI: diagnose the initial-state delta for a Simscape input MAT.

Pipeline:
    1. If ``matlab.engine`` is importable, call
       ``diagnose_initial_state.m`` to produce a fresh MAT report.
       Otherwise fall back to a precomputed report supplied via
       ``--report``.
    2. Load the report into the Python diff model.
    3. Render skeleton overlay, joint-delta bars, Cartesian delta plot.
    4. Write ``report.md`` and ``report.json`` for downstream consumers.

The MATLAB step is optional; the Python visualizer side runs anywhere.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless by default; CLI never opens a window

# Make the in-repo package importable when run from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
_PKG_PARENT = _REPO_ROOT / "src" / "shared" / "python"
if str(_PKG_PARENT) not in sys.path:
    sys.path.insert(0, str(_PKG_PARENT))

from motion_matching.diagnostics.initial_state_diff import (  # noqa: E402
    load_diff_report,
    plot_cartesian_delta_summary,
    plot_per_joint_delta_bars,
    plot_skeleton_overlay,
    report_to_json,
    summarize_for_pr_comment,
)


def _run_matlab_driver(input_file: Path, out_mat: Path) -> bool:
    """Try to invoke the MATLAB diagnostic. Return True on success."""
    try:
        import matlab.engine  # type: ignore[import-not-found]
    except ImportError:
        return False

    eng = matlab.engine.start_matlab()
    try:
        diag_dir = (
            _REPO_ROOT
            / "src"
            / "engines"
            / "Simscape_Multibody_Models"
            / "3D_Golf_Model"
            / "matlab"
            / "diagnostics"
        )
        eng.addpath(str(diag_dir), nargout=0)
        eng.diagnose_initial_state(
            str(input_file),
            "save_to",
            str(out_mat),
            nargout=0,
        )
    finally:
        eng.quit()
    return out_mat.is_file()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        help="Path to the Simscape input MAT (e.g. 3DModelInputs_Impact.mat).",
    )
    parser.add_argument(
        "--report",
        type=Path,
        help="Precomputed MAT report (skips the MATLAB step).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory for skeleton_overlay.png, joint_deltas.png, "
        "cartesian_deltas.png, report.md, report.json.",
    )
    args = parser.parse_args(argv)

    if not args.input and not args.report:
        parser.error("Provide --input (to run MATLAB) or --report (precomputed).")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    report_mat = args.report
    if report_mat is None:
        if args.input is None or not args.input.is_file():
            parser.error(f"--input does not exist: {args.input}")
        report_mat = args.out_dir / "report.mat"
        ok = _run_matlab_driver(args.input, report_mat)
        if not ok:
            print(
                "matlab.engine not available and no --report supplied; "
                "cannot produce a fresh diagnostic.",
                file=sys.stderr,
            )
            return 2

    report = load_diff_report(report_mat)

    fig_overlay = plot_skeleton_overlay(report)
    fig_overlay.savefig(
        args.out_dir / "skeleton_overlay.png", dpi=150, bbox_inches="tight"
    )

    fig_bars = plot_per_joint_delta_bars(report)
    fig_bars.savefig(args.out_dir / "joint_deltas.png", dpi=150, bbox_inches="tight")

    fig_cart = plot_cartesian_delta_summary(report)
    fig_cart.savefig(
        args.out_dir / "cartesian_deltas.png", dpi=150, bbox_inches="tight"
    )

    (args.out_dir / "report.md").write_text(
        summarize_for_pr_comment(report), encoding="utf-8"
    )
    (args.out_dir / "report.json").write_text(report_to_json(report), encoding="utf-8")

    verdict = "SIGNIFICANT" if report.is_significant else "negligible"
    print(f"Initial-state diff: {verdict}. Wrote artifacts to {args.out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
