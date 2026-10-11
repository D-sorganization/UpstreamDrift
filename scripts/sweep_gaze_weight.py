"""Gaze-weight sweep (OSV-3b, #11729): one IK-stage run per (capture, weight).

    MUJOCO_GL=egl PYTHONPATH=.:src python3 -m scripts.sweep_gaze_weight \\
        --capture driver --weight 3 --out RUN

Runs the shared ground-support pipeline up to the trajectory inverse
kinematics (the dynamics stage is replaced by a stub, so a run takes minutes,
not the 40 min of a fixture run) with the flags of the club-face fixture runs
and writes ``sweep_row.json``: marker RMS, closure, the OSV-10 face fit and
the ``head_gaze`` block, all measured on the reference trajectory ``q_ref``.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from scripts.regenerate_club_face_fixtures import pipeline_argv


def row_from(ik_report: dict[str, Any], head_gaze: dict[str, Any]) -> dict[str, Any]:
    """The sweep metrics of one run (measured on the reference trajectory)."""
    ref = ik_report["reference"]
    face = ik_report.get("face_orientation", {}).get("reference", {})
    return {
        "marker_rms_mm": 1000.0 * ref["marker_rms_m"],
        "closure_error_max_mm": 1000.0 * ref["closure_error_max_m"],
        "face_fit_deg": face,
        "head_gaze": head_gaze,
    }


def _fixed_scales(femur: float, tibia: float):  # noqa: ANN202
    """A drop-in for ``search_segment_scales`` that evaluates one scale pair."""
    from src.shared.python.motion_matching.segment_scaling import scale_segments

    if femur <= 0 or tibia <= 0:
        raise ValueError("segment scales must be positive")

    def search(_lane, hip_spec, *_args, **_kwargs):  # noqa: ANN202
        scales = {f"femur_{s}": femur for s in "rl"} | {
            f"tibia_{s}": tibia for s in "rl"
        }
        return scale_segments(hip_spec, scales), femur, tibia, []

    return search


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", choices=["driver", "iron"], required=True)
    parser.add_argument("--weight", type=float, required=True)
    parser.add_argument("--face-weight", type=float, default=3.0)
    parser.add_argument(
        "--scales",
        type=float,
        nargs=2,
        metavar=("FEMUR", "TIBIA"),
        help=(
            "skip the 3x3 segment-scale grid search and use this pair (the "
            "winner of the fixture run; the gaze weight does not enter it)"
        ),
    )
    parser.add_argument("--out", type=Path, required=True)
    ns = parser.parse_args(argv)
    ns.out.mkdir(parents=True, exist_ok=True)

    from src.shared.python.motion_matching.pipeline import cli
    from src.shared.python.motion_matching.pipeline.gaze_residual import (
        head_gaze_receipt,
    )

    captured: dict[str, Any] = {}

    def stub(_ctx, lane, kin, *rest):  # noqa: ANN001, ANN202
        # rest = (sim, adapter, labels, q_ref, calibration, spec, ik_report), the
        # positional tail of cli._simulate_and_receipt.
        q_ref, ik_report = rest[3], rest[-1]
        captured["row"] = row_from(ik_report, head_gaze_receipt(lane, kin, q_ref))
        return {}

    cli._simulate_and_receipt = stub  # type: ignore[assignment]
    if ns.scales is not None:
        cli.search_segment_scales = _fixed_scales(*ns.scales)  # type: ignore[assignment]
    argv_pipe = pipeline_argv(ns.capture, ns.out, ns.face_weight)
    args = cli.build_parser().parse_args([*argv_pipe, "--gaze-weight", repr(ns.weight)])
    cli.run_pipeline(args)
    row = {"capture": ns.capture, "gaze_weight": ns.weight, **captured["row"]}
    (ns.out / "sweep_row.json").write_text(json.dumps(row, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
