"""CLI driver for clubhead trace comparison.

Loads two ``ClubTarget`` artifacts (Excel/C3D for the measured side, CSV or
``.npz`` for the simulated side), runs :func:`compare_clubhead_traces`, and
writes four PNGs plus a JSON report into ``--out-dir``.

Usage:
    python3 scripts/compare_clubhead_traces.py \
        --measured path/to/Wiffle.xlsx --measured-sheet TW_ProV1 \
        --simulated path/to/sim_out.csv \
        --out-dir results/comparisons/swing_001
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
from src.shared.python.motion_matching.club_target import (  # noqa: E402
    ClubTarget,
    SourceProvenance,
)
from src.shared.python.motion_matching.diagnostics.clubhead_trace import (  # noqa: E402
    TraceCompareOptions,
    compare_clubhead_traces,
    plot_3d_overlay,
    plot_per_axis_timeseries,
    plot_setup_pose_skeletons,
    plot_speed_comparison,
)

logger = logging.getLogger("compare_clubhead_traces")


def _load_measured(path: Path, sheet: str | None) -> ClubTarget:
    suf = path.suffix.lower()
    if suf in (".xlsx", ".xlsm", ".xls"):
        from src.shared.python.motion_matching.loaders.excel import (  # noqa: PLC0415
            load_club_target_excel,
        )

        return load_club_target_excel(path, sheet=sheet or "")
    if suf == ".c3d":
        from src.shared.python.motion_matching.loaders.c3d import (  # noqa: PLC0415
            load_club_target_c3d,
        )

        return load_club_target_c3d(path)
    raise ValueError(f"unsupported measured suffix {suf!r} for {path}")


def _load_simulated_csv(path: Path) -> ClubTarget:
    """Minimal CSV loader: columns time,butt_x..z,clubhead_x..z,qw,qx,qy,qz."""
    arr = np.genfromtxt(path, delimiter=",", names=True, dtype=float)
    time = np.asarray(arr["time"], dtype=np.float64)
    butt = np.column_stack([arr["butt_x"], arr["butt_y"], arr["butt_z"]])
    head = np.column_stack([arr["clubhead_x"], arr["clubhead_y"], arr["clubhead_z"]])
    quat = np.column_stack([arr["qw"], arr["qx"], arr["qy"], arr["qz"]])

    # ⚡ Bolt: np.sqrt(np.einsum(...)) avoids intermediate temporary array allocation
    # and is ~2x faster than np.linalg.norm(..., axis=1)
    norms = np.sqrt(np.einsum("ij,ij->i", quat, quat))[:, np.newaxis]
    norms[norms == 0.0] = 1.0
    quat = quat / norms
    sha = hashlib.sha256(path.read_bytes()).hexdigest()

    # ⚡ Bolt: argmax is invariant to monotonic transformations, so we omit sqrt entirely
    # and use einsum for a ~2x speedup avoiding memory allocations and sqrt overhead
    _head_diff = np.diff(head, axis=0)

    return ClubTarget(
        time=time - time[0],
        butt=butt,
        clubhead=head,
        club_quat=quat,
        impact_idx=int(np.argmax(np.einsum("ij,ij->i", _head_diff, _head_diff))) + 1,
        source=SourceProvenance(
            filename=path.name,
            format="simscape_csv",
            subject_id="SIM",
            trial_id=path.stem,
            sha256=sha,
        ),
    )


def _load_simulated(path: Path) -> ClubTarget:
    if path.suffix.lower() == ".csv":
        return _load_simulated_csv(path)
    raise ValueError(f"unsupported simulated suffix {path.suffix!r}")


def _parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--measured", required=True, type=Path)
    p.add_argument("--measured-sheet", default=None)
    p.add_argument("--simulated", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument(
        "--time-alignment",
        choices=("impact", "address", "none"),
        default="impact",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args(argv if argv is not None else sys.argv[1:])
    args.out_dir.mkdir(parents=True, exist_ok=True)

    measured = _load_measured(args.measured, args.measured_sheet)
    simulated = _load_simulated(args.simulated)
    rep = compare_clubhead_traces(
        measured,
        simulated,
        TraceCompareOptions(time_alignment=args.time_alignment),
    )

    plot_3d_overlay(rep).savefig(args.out_dir / "3d_overlay.png", dpi=120)
    plot_per_axis_timeseries(rep).savefig(args.out_dir / "per_axis.png", dpi=120)
    plot_speed_comparison(rep).savefig(args.out_dir / "speed.png", dpi=120)
    plot_setup_pose_skeletons(measured, simulated).savefig(
        args.out_dir / "setup_pose.png", dpi=120
    )
    (args.out_dir / "report.json").write_text(
        json.dumps(rep.to_json_dict(), indent=2, sort_keys=True)
    )
    logger.info(
        "wrote 4 PNGs and report.json into %s (RMSE %.2f mm)",
        args.out_dir,
        rep.total_rmse_mm,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
