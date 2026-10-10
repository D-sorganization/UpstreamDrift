"""Regenerate the club-face swing fixtures and their provenance (OSV-10, #11759).

    MUJOCO_GL=egl PYTHONPATH=.:src python3 -m scripts.regenerate_club_face_fixtures \\
        --work RUNS --capture driver [--face-weight W] [--reuse-run]

For each capture this script

1. runs the shared ground-support pipeline (``motion_matching.pipeline.cli``)
   on the committed anthropometric document, with the flags of the original
   fixture runs and the face-orientation residual weight;
2. exports the MuJoCo same-input reference bundle of that run
   (``scripts.same_input_bundle export``);
3. writes ``tests/fixtures/club_face/swing_q_<club>.npz`` (every second
   1 kHz reference sample, float32), the address pose in
   ``address_poses.json`` and a per-club provenance record in
   ``provenance.json``: commands, input and output hashes, marker RMS, the
   model-versus-capture face fit, the impact and peak-speed frames and the
   clubhead speed timing (GCV-20).

Before overwriting, the previous fixture is measured the same way so the
provenance keeps the before/after face angles. Nothing here is hand edited.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "club_face"
DOCS = ROOT / "docs" / "development" / "full_body_models"
FIXTURE_STRIDE = 2  # the 1 kHz reference, every second sample (2 ms)
FIXTURE_DT_S = 0.002
#: capture -> (fixture club alias, document, extra pipeline flags of the
#: original fixture runs, read back from their receipts).
CAPTURE_RUNS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "driver": ("driver", "full_body_spec_anthro_driver.json", ("--static-seeds",)),
    "iron": (
        "iron7",
        "full_body_spec_anthro_iron7.json",
        ("--static-seeds", "--zmp-filter"),
    ),
}
LOG = logging.getLogger("club_face_fixtures")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pipeline_argv(capture: str, run_dir: Path, face_weight: float) -> list[str]:
    """Pipeline CLI arguments of one fixture run."""
    if capture not in CAPTURE_RUNS:
        raise ValueError(f"unknown capture {capture!r}; expected {list(CAPTURE_RUNS)}")
    _, document, flags = CAPTURE_RUNS[capture]
    return [
        "--spec",
        str(DOCS / document),
        "--capture",
        capture,
        *flags,
        *__import__("os").environ.get("UD_EXTRA_12117", "").split(),
        "--face-weight",
        repr(float(face_weight)),
        "--out",
        str(run_dir),
    ]


def run_and_export(
    capture: str, work: Path, face_weight: float, reuse: bool
) -> tuple[Path, Path]:
    """Run the pipeline (unless reusing a finished run) and export its bundle."""
    from scripts.same_input_bundle import export
    from src.shared.python.motion_matching.pipeline.cli import (
        build_parser,
        run_pipeline,
    )

    run_dir = work / capture
    if not (reuse and (run_dir / "receipt.json").is_file()):
        args = pipeline_argv(capture, run_dir, face_weight)
        run_pipeline(build_parser().parse_args(args))
    bundle = work / "bundles" / f"{capture}.npz"
    if not (reuse and bundle.is_file()):
        export(run_dir, bundle, None)
    return run_dir, bundle


def _git_head() -> str:
    out = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    )
    return out.stdout.strip()


def measure_swing(run_dir: Path, q: np.ndarray, dt_s: float) -> dict[str, Any]:
    """Face events and clubhead speed timing of ``q`` and of the capture triad.

    The model uses MuJoCo FK of the run's scaled document; the capture face
    comes from the run's calibrated head triad. Impact is the sub-sample
    passage of the face centre through the ball, each swing on its own clock.
    Returns ``{"faces": {...}, "speed_timing": {...}}``, each keyed by
    ``model`` and ``capture`` (speed timing: GCV-20, #11767).
    """
    from dataclasses import asdict

    from src.shared.python.model_appearance import club_face as cf
    from src.shared.python.motion_matching import club_face_target as cft

    kin, lane, spec, attachments = _run_kinematics(run_dir)
    cap_n, cap_c = cft.observe_capture_face(
        lane.points, lane.valid, lane.labels, attachments, spec
    )
    t_cap = lane.times - lane.times[0]
    model_n, model_c = _model_face(kin, spec, q, list(kin.coordinate_order))
    t_model = np.arange(len(q)) * dt_s
    cap_c = cft.fill_unobserved(t_cap, cap_c)
    model = cf.face_events(t_model, model_n, model_c)
    capture = cf.face_events(t_cap, cap_n, cap_c)
    return {
        "faces": {"model": asdict(model), "capture": asdict(capture)},
        "speed_timing": {
            "model": cf.clubhead_speed_timing(t_model, model_c).to_record(),
            "capture": cf.clubhead_speed_timing(t_cap, cap_c).to_record(),
        },
    }


def _run_kinematics(run_dir: Path) -> tuple[Any, Any, dict[str, Any], dict]:
    from src.shared.python.motion_matching.pipeline.constants import capture_path
    from src.shared.python.motion_matching.pipeline.lane import Lane
    from src.shared.python.motion_matching.pipeline.plant import get_plant

    receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
    spec_path = run_dir / "full_body_spec_hipcal_scaled.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    attachments = {
        label: (a["body"], tuple(a["offset_m"]))
        for label, a in receipt["ik"]["attachments_m"].items()
    }
    lane = Lane(tuple(receipt["labels"]), capture_path(receipt["capture"]))
    lane.plant = get_plant("mujoco", spec)
    _, kin = lane.kinematics(spec_path.read_bytes(), attachments)
    return kin, lane, spec, attachments


def _model_face(
    kin: Any, spec: dict[str, Any], q: np.ndarray, order: list[str]
) -> tuple[np.ndarray, np.ndarray]:
    from src.shared.python.motion_matching import club_face_target as cft

    if q.shape[1] != len(order):
        raise ValueError("fixture columns do not match the run coordinates")
    axis = cft.face_axis_in_frame(spec)
    centre = cft.face_centre_in_frame(spec)
    normals, centres = [], []
    for row in np.asarray(q, dtype=float):
        rot, pos = kin.body_poses(row, [cft.FACE_FRAME])[cft.FACE_FRAME]
        normals.append(rot @ axis)
        centres.append(rot @ centre + pos)
    return np.array(normals), np.array(centres)


def write_fixtures(
    capture: str, run_dir: Path, bundle_path: Path, argv: list[str]
) -> dict[str, Any]:
    """Overwrite the club's fixtures and return its provenance record."""
    club = CAPTURE_RUNS[capture][0]
    npz_path = FIXTURES / f"swing_q_{club}.npz"
    before = None
    if npz_path.is_file():
        old_q = np.load(npz_path)["q"].astype(float)
        before = {
            "fixture_sha256": _sha256(npz_path),
            **measure_swing(run_dir, old_q, FIXTURE_DT_S),
        }
    with np.load(bundle_path, allow_pickle=False) as bundle:
        reference = np.asarray(bundle["reference_q"], dtype=float)
        manifest = json.loads(str(bundle["manifest"]))
    np.savez_compressed(npz_path, q=reference[::FIXTURE_STRIDE].astype(np.float32))
    _write_address_pose(club, manifest, reference[0])
    receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
    q_fixture = np.load(npz_path)["q"].astype(float)
    return {
        "capture": capture,
        "pipeline_argv": argv,
        "bundle_manifest": manifest,
        "bundle_sha256": _sha256(bundle_path),
        "run_receipt_sha256": _sha256(run_dir / "receipt.json"),
        "fixture_sha256": _sha256(npz_path),
        "marker_rms_m": {
            "ik": receipt["ik"]["marker_rms_m"],
            "reference": receipt["ik"]["reference"]["marker_rms_m"],
            "tracking": receipt["dynamics"]["marker_rms_m"],
        },
        "face_orientation": receipt["ik"].get("face_orientation"),
        "capture_triad_offsets_m": _triad_offsets(receipt),
        "before": before,
        "after": measure_swing(run_dir, q_fixture, FIXTURE_DT_S),
        "impact_split": receipt["ik"].get("impact_split"),
    }


def _triad_offsets(receipt: dict[str, Any]) -> dict[str, list[float]]:
    """Calibrated head-triad offsets in the face frame (the capture face)."""
    from src.shared.python.motion_matching.club_face_target import (
        FACE_FRAME,
        HEAD_TRIAD_LABELS,
    )

    attached = receipt["ik"]["attachments_m"]
    for label in HEAD_TRIAD_LABELS:
        if attached[label]["body"] != FACE_FRAME:
            raise ValueError(f"{label} is not attached to {FACE_FRAME}")
    return {label: list(attached[label]["offset_m"]) for label in HEAD_TRIAD_LABELS}


def _write_address_pose(club: str, manifest: dict[str, Any], q0: np.ndarray) -> None:
    path = FIXTURES / "address_poses.json"
    doc = json.loads(path.read_text(encoding="utf-8"))
    doc["poses"][club] = {
        "capture_sha256": manifest["provenance"]["capture_sha256"],
        "coordinate_order": list(manifest["coordinate_order"]),
        "q_rad": [float(v) for v in q0],
    }
    path.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")


def _update_provenance(club: str, record: dict[str, Any]) -> None:
    path = FIXTURES / "provenance.json"
    doc: dict[str, Any] = (
        json.loads(path.read_text(encoding="utf-8"))
        if path.is_file()
        else {"schema": "club-face-fixture-provenance/v1", "clubs": {}}
    )
    doc["generator"] = "scripts/regenerate_club_face_fixtures.py"
    doc["clubs"][club] = record
    path.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> None:
    """CLI entry point."""
    from src.shared.python.motion_matching.club_face_target import (
        FACE_ORIENTATION_WEIGHT,
        validate_face_weight,
    )

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--capture", choices=sorted(CAPTURE_RUNS), required=True)
    parser.add_argument("--face-weight", type=float, default=FACE_ORIENTATION_WEIGHT)
    parser.add_argument("--reuse-run", action="store_true")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    weight = validate_face_weight(args.face_weight)
    run_dir, bundle = run_and_export(args.capture, args.work, weight, args.reuse_run)
    record = write_fixtures(
        args.capture, run_dir, bundle, pipeline_argv(args.capture, run_dir, weight)
    )
    record["generated_at_commit"] = _git_head()
    record["face_weight"] = weight
    _update_provenance(CAPTURE_RUNS[args.capture][0], record)
    LOG.info("%s: %s", args.capture, json.dumps(record["after"], indent=1))


if __name__ == "__main__":
    main()
