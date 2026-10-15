"""Clubhead-speed decomposition against the capture (R5, #12682).

    MUJOCO_GL=egl PYTHONPATH=.:src python3 -m scripts.diagnose_clubhead_speed \\
        --fixture driver
    ... --run-dir RUNS/driver
    ... --bundle RUNS/bundles/driver.npz --receipt RUNS/driver/receipt.json

Prints, per series, the impact speed, the percentage against the capture's
face-centre speed, the peak speed, the peak time relative to impact and the
peak lead relative to the capture's own peak:

* capture: face centre (calibrated head triad) and the raw head-cluster
  centroid;
* model: every available trajectory (IK, reference, tracked reference,
  forward-dynamics swing, bundle reference) at the face centre and at the
  ``Clubhead`` frame origin.

Every series is measured with ``fd_phase.speed_series`` on its own clock
relative to its own ball passage (the receipt rule). A series whose point
never passes the ball is reported as unavailable with its closest approach,
never as a zero speed. Captures are named only capture-A/B/O; no capture data
is written anywhere.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "club_face"
SPEC_DIR = ROOT / "docs" / "development" / "full_body_models"
SCALED_SPEC = "full_body_spec_hipcal_scaled.json"
#: The fixture holds every second sample of the 1 kHz reference.
FIXTURE_DT_S = 0.002
#: Fixture club -> capture name of ``pipeline.constants.capture_path``.
FIXTURE_CAPTURES = {"driver": "driver", "iron7": "iron"}
#: Anonymous capture labels (privacy: never the capture's own name).
CAPTURE_LABELS = {"driver": "capture-A", "iron": "capture-B", "owner": "capture-O"}


class MujocoFrameKinematics:
    """``body_poses(q, frames)`` of spec frames from the MuJoCo export.

    The minimal kinematics the face-centre helpers need
    (``club_face_target.model_face_centres``). Raises ``ImportError``
    without MuJoCo.
    """

    def __init__(self, spec_bytes: bytes, coordinate_order: Sequence[str]) -> None:
        from src.engines.physics_engines.mujoco.python.overlay_source import (
            MujocoOverlaySource,
        )

        self._source = MujocoOverlaySource(spec_bytes)
        self._order = list(coordinate_order)

    def body_poses(
        self, q: np.ndarray, frames: Sequence[str]
    ) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        poses = self._source.frame_poses(
            dict(zip(self._order, map(float, q), strict=True))
        )
        return {f: (poses[f][:3, :3], poses[f][:3, 3]) for f in frames}


def frame_paths(
    kin: Any, q: np.ndarray, spec: Mapping[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    """``(face centre, Clubhead origin)`` world paths of ``q``."""
    from src.shared.python.motion_matching import club_face_target as cft

    poses = [kin.body_poses(row, [cft.FACE_FRAME])[cft.FACE_FRAME] for row in q]
    rotations = np.array([r for r, _ in poses], dtype=float).reshape(-1, 3, 3)
    origins = np.array([p for _, p in poses], dtype=float).reshape(-1, 3)
    centre = cft.face_centre_in_frame(spec)
    return cft.face_centre_path(rotations, origins, centre), origins


def capture_paths(
    capture_name: str, attachments: Mapping[str, tuple[str, Sequence[float]]], spec: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(time, face centre, head-cluster centroid)`` of the capture triad."""
    from src.shared.python.motion_matching import club_face_target as cft
    from src.shared.python.motion_matching.ground_support import (
        capture_to_native_world,
    )
    from src.shared.python.motion_matching.pipeline.constants import capture_path
    from src.shared.python.motion_matching.tour_capture_contract import (
        load_tour_capture,
    )

    triad = tuple(cft.triad_offsets(attachments))
    capture = load_tour_capture(capture_path(capture_name)).subset(triad)
    points = capture_to_native_world(capture.points_m)
    t = np.asarray(capture.time_s, dtype=float)
    t = t - t[0]
    _, face = cft.observe_capture_face(
        points, capture.valid, tuple(capture.labels), attachments, spec
    )
    seen = np.asarray(capture.valid, dtype=bool)[..., None]
    centroid = np.where(seen, points, np.nan).mean(axis=1)
    return t, cft.fill_unobserved(t, face), cft.fill_unobserved(t, centroid)


def _model_rows(
    label: str, kin: Any, q: np.ndarray, t: np.ndarray, spec: Mapping[str, Any]
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    face, origin = frame_paths(kin, q, spec)
    yield f"{label}, face centre", t, face
    yield f"{label}, Clubhead origin", t, origin


def fixture_series(club: str) -> tuple[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    """Capture and model series of a committed club-face fixture."""
    spec_bytes = (SPEC_DIR / f"full_body_spec_anthro_{club}.json").read_bytes()
    spec = json.loads(spec_bytes)
    poses = json.loads((FIXTURES / "address_poses.json").read_text("utf-8"))
    kin = MujocoFrameKinematics(spec_bytes, poses["poses"][club]["coordinate_order"])
    record = json.loads((FIXTURES / "provenance.json").read_text("utf-8"))
    offsets = record["clubs"][club]["capture_triad_offsets_m"]
    attachments = {k: ("Clubhead", tuple(v)) for k, v in offsets.items()}
    capture = FIXTURE_CAPTURES[club]
    rows = list(_capture_rows(capture, attachments, spec))
    q = np.load(FIXTURES / f"swing_q_{club}.npz")["q"].astype(float)
    rows += _model_rows("FD fixture", kin, q, np.arange(len(q)) * FIXTURE_DT_S, spec)
    tracked = FIXTURES / f"tracked_q_{club}.npz"
    if tracked.is_file():
        with np.load(tracked) as data:
            q_t, t_t = data["q"].astype(float), data["time_s"].astype(float)
        rows += _model_rows("tracked reference", kin, q_t, t_t, spec)
    return capture, rows


def _capture_rows(
    capture: str, attachments: Mapping[str, tuple[str, Sequence[float]]], spec: Any
) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    t, face, centroid = capture_paths(capture, attachments, spec)
    yield "capture, face centre", t, face
    yield "capture, head-cluster centroid", t, centroid


def _receipt_club(
    receipt_path: Path,
) -> tuple[str, dict[str, tuple[str, tuple[float, ...]]]]:
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    attachments = {
        label: (a["body"], tuple(a["offset_m"]))
        for label, a in receipt["ik"]["attachments_m"].items()
    }
    return str(receipt["capture"]), attachments


def run_dir_series(
    run_dir: Path,
) -> tuple[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    """Capture and model series of a pipeline run directory."""
    spec_bytes = (run_dir / SCALED_SPEC).read_bytes()
    spec = json.loads(spec_bytes)
    capture, attachments = _receipt_club(run_dir / "receipt.json")
    kin = MujocoFrameKinematics(spec_bytes, spec["coordinate_order"])
    rows = list(_capture_rows(capture, attachments, spec))
    with np.load(run_dir / "ik_trajectory.npz") as ik:
        t = ik["time_s"].astype(float) - float(ik["time_s"][0])
        rows += _model_rows("IK", kin, ik["q"].astype(float), t, spec)
        rows += _model_rows("reference", kin, ik["q_ref"].astype(float), t, spec)
    with np.load(run_dir / "dynamics_record.npz") as dyn:
        t_track = dyn["track_time_s"].astype(float) - float(dyn["track_time_s"][0])
        rows += _model_rows("tracked", kin, dyn["q_track"].astype(float), t_track, spec)
        t_fd = dyn["time_s"].astype(float) - float(dyn["time_s"][0])
        rows += _model_rows("FD", kin, dyn["q"].astype(float), t_fd, spec)
    return capture, rows


def bundle_series(
    bundle_path: Path, receipt_path: Path
) -> tuple[str, list[tuple[str, np.ndarray, np.ndarray]]]:
    """Capture and model series of a same-input bundle's reference."""
    from src.shared.python.motion_matching.same_input.bundle import InputBundle

    bundle = InputBundle.load(bundle_path)
    spec = json.loads(bundle.spec_bytes)
    capture, attachments = _receipt_club(receipt_path)
    kin = MujocoFrameKinematics(bundle.spec_bytes, bundle.coordinate_order)
    q = np.asarray(bundle.reference_q, dtype=float)
    rows = list(_capture_rows(capture, attachments, spec))
    rows += _model_rows(
        "bundle reference", kin, q, np.arange(len(q)) * bundle.dt_s, spec
    )
    return capture, rows


def decomposition(
    rows: Sequence[tuple[str, np.ndarray, np.ndarray]],
) -> list[dict[str, Any]]:
    """Speed record per row, against the first row (the capture face centre)."""
    from src.shared.python.motion_matching.pipeline.fd_phase import (
        FacePath,
        speed_series,
    )

    records = [
        {"series": name, **speed_series(FacePath(t, path))} for name, t, path in rows
    ]
    truth = records[0]
    for rec in records:
        both = rec["status"] == truth["status"] == "ok"
        rec["error_pct"] = (
            100.0 * (rec["impact_speed_mps"] / truth["impact_speed_mps"] - 1.0)
            if both
            else None
        )
        rec["peak_lead_s"] = (
            rec["peak_minus_impact_s"] - truth["peak_minus_impact_s"] if both else None
        )
    return records


def _fmt(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def render_table(capture: str, records: Sequence[Mapping[str, Any]]) -> str:
    """Markdown decomposition table; unavailable rows carry their reason."""
    label = CAPTURE_LABELS.get(capture, "capture")
    lines = [
        f"Clubhead speed at impact against {label} (face centre, m/s)",
        "",
        "| Series | Impact | vs capture | Peak | Peak - impact | Peak lead | Note |",
        "|---|---|---|---|---|---|---|",
    ]
    for rec in records:
        ms = (
            None
            if rec["peak_minus_impact_s"] is None
            else 1e3 * rec["peak_minus_impact_s"]
        )
        lead = None if rec["peak_lead_s"] is None else 1e3 * rec["peak_lead_s"]
        lines.append(
            f"| {rec['series']} | {_fmt(rec['impact_speed_mps'], '.2f')} "
            f"| {_fmt(rec['error_pct'], '+.1f')} % | {_fmt(rec['peak_speed_mps'], '.2f')} "
            f"| {_fmt(ms, '+.1f')} ms | {_fmt(lead, '+.1f')} ms "
            f"| {rec['reason'] or ''} |"
        )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--fixture", choices=sorted(FIXTURE_CAPTURES))
    source.add_argument("--run-dir", type=Path)
    source.add_argument("--bundle", type=Path)
    parser.add_argument(
        "--receipt",
        type=Path,
        help="run receipt naming the capture and its triad offsets (with --bundle)",
    )
    parser.add_argument("--json", action="store_true", help="print JSON records")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.fixture:
        capture, rows = fixture_series(args.fixture)
    elif args.run_dir:
        capture, rows = run_dir_series(args.run_dir)
    else:
        if args.receipt is None:
            raise SystemExit("--bundle needs --receipt for the capture triad")
        capture, rows = bundle_series(args.bundle, args.receipt)
    records = decomposition(rows)
    if args.json:
        sys.stdout.write(json.dumps(records, indent=1) + "\n")
    else:
        sys.stdout.write(render_table(capture, records) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
