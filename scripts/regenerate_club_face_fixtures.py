"""Regenerate the club-face swing fixtures and their provenance (OSV-10, #11759).

    MUJOCO_GL=egl PYTHONPATH=.:src python3 -m scripts.regenerate_club_face_fixtures \\
        --work RUNS --capture driver [--face-weight W] [--reuse-run]

For each capture this script

1. runs the shared ground-support pipeline (``motion_matching.pipeline.cli``)
   on the committed anthropometric document, with the flags of the original
   fixture runs and the face-orientation residual weight;
2. exports the MuJoCo same-input reference bundle of that run
   (``scripts.same_input_bundle export``) with the 0.125 ms effort hold
   (``CLUB_FACE_EFFORT_HOLD_MS``, #12279); a reused bundle recorded with
   another hold is re-exported;
3. writes ``tests/fixtures/club_face/swing_q_<club>.npz`` (every second
   1 kHz reference sample, float32), the run's tracked swing in
   ``tracked_q_<club>.npz`` (#12279), the address pose in
   ``address_poses.json``, the run's scaled document in ``spec_<club>.json``
   and a per-club provenance record in ``provenance.json``: commands, input
   and output hashes, marker RMS, the model-versus-capture face fit, the
   impact and peak-speed frames, the clubhead speed timing (GCV-20) and the
   address toe-out against the capture (#12222).

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

from src.shared.python.motion_matching.same_input.bundle import (
    FINE_EFFORT_HOLD_DT_S,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "tests" / "fixtures" / "club_face"
DOCS = ROOT / "docs" / "development" / "full_body_models"
FIXTURE_STRIDE = 2  # the 1 kHz reference, every second sample (2 ms)
FIXTURE_DT_S = 0.002
#: Foot torsional-friction patch radius (m) of the canonical runs: the value
#: #11671 fixed on the driver and validated on the 7-iron
#: (FOOT_CONTACT_TORSION.md), adopted against the ground yaw slip (#11961).
TORSION_PATCH_M = "0.05"
#: The run's hip-calibrated, segment-scaled document: the fixture's model. It
#: is committed beside the fixture as ``spec_<club>.json`` because the hip
#: frames set the foot yaw: the raw document turns the lead foot 29-30 deg (#12222).
SCALED_SPEC = "full_body_spec_hipcal_scaled.json"
#: capture -> (fixture club alias, document, extra pipeline flags of the
#: original fixture runs, read back from their receipts, plus the torsion patch). The driver's static
#: trial holds the measured club cluster fixed: its per-capture grip rotation
#: is fitted on it, and a free cluster moves the capture's own face (#12221).
#: The 7-iron refit is adopted the same way now that the 0.125 ms export hold
#: keeps its tracked swing (impact face 8.7 -> 5.6 deg against 3.2, #12279).
CAPTURE_RUNS: dict[str, tuple[str, str, tuple[str, ...]]] = {
    "driver": (
        "driver",
        "full_body_spec_anthro_driver.json",
        (
            "--static-seeds",
            "--fix-club-markers",
            "--torsional-patch-m",
            TORSION_PATCH_M,
        ),
    ),
    "iron": (
        "iron7",
        "full_body_spec_anthro_iron7.json",
        (
            "--static-seeds",
            "--zmp-filter",
            "--fix-club-markers",
            "--torsional-patch-m",
            TORSION_PATCH_M,
        ),
    ),
}
#: Controller effort hold of the club-face same-input export (ms). The 1 ms
#: zero-order hold slides the 7-iron feet from about 0.67 s; 0.125 ms keeps
#: them planted. Owner ruling on #12279 (2026-10-11): opt-in for these
#: fixtures only; the global ``same_input_bundle`` default stays 1 ms. It is
#: the shared fine hold of the same-input policy (#12502), in ms.
CLUB_FACE_EFFORT_HOLD_MS = round(FINE_EFFORT_HOLD_DT_S * 1e3, 9)
#: Address foot progression of every fixture run: the OSV-4 capture toe-out
#: calibration (#11730), held through the swing by the planted stance (#12222).
FOOT_PROGRESSION = "capture"
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
        "--foot-progression",
        FOOT_PROGRESSION,
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
    if not (reuse and _bundle_hold_ms(bundle) == CLUB_FACE_EFFORT_HOLD_MS):
        export(run_dir, bundle, None, effort_hold_ms=CLUB_FACE_EFFORT_HOLD_MS)
    return run_dir, bundle


def _bundle_hold_ms(bundle: Path) -> float | None:
    """Effort hold (ms) recorded in a bundle's manifest; ``None`` if absent.

    A manifest without the hold key is a legacy 1 ms bundle (#12279).
    """
    if not bundle.is_file():
        return None
    with np.load(bundle, allow_pickle=False) as data:
        policy = json.loads(str(data["manifest"])).get("policy", {})
    return round(float(policy.get("effort_hold_s", 1e-3)) * 1e3, 9)


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


def measure_swing(
    run_dir: Path, q: np.ndarray, dt_s: float, model_spec: Path | None = None
) -> dict[str, Any]:
    """Face events and clubhead speed timing of ``q`` and of the capture triad.

    The model uses MuJoCo FK of the run's scaled document, or of
    ``model_spec`` when given (a previous fixture's own document); the
    capture face comes from the run's calibrated head triad. Impact is the
    sub-sample passage of the face centre through the ball, each swing on its
    own clock. Returns ``{"faces": {...}, "speed_timing": {...}}``, each keyed
    by ``model`` and ``capture`` (speed timing: GCV-20, #11767).
    """
    from dataclasses import asdict

    from src.shared.python.model_appearance import club_face as cf
    from src.shared.python.motion_matching import club_face_target as cft

    kin, lane, spec, attachments = _run_kinematics(run_dir, model_spec)
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


def _run_kinematics(
    run_dir: Path, spec_path: Path | None = None
) -> tuple[Any, Any, dict[str, Any], dict]:
    from src.shared.python.motion_matching.pipeline.constants import capture_path
    from src.shared.python.motion_matching.pipeline.lane import Lane
    from src.shared.python.motion_matching.pipeline.plant import get_plant

    receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
    spec_path = spec_path or run_dir / SCALED_SPEC
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


def _previous_fixture_record(
    npz_path: Path, spec_path: Path, run_dir: Path
) -> dict[str, Any] | None:
    """Measure the fixture being replaced on its own committed document.

    Its joint angles describe the old swing only on the old scaled document
    (``spec_<club>.json``); on a changed document they can miss the ball.
    Returns ``None`` when there is no previous fixture.
    """
    if not npz_path.is_file():
        return None
    old_q = np.load(npz_path)["q"].astype(float)
    return {
        "fixture_sha256": _sha256(npz_path),
        "model_spec_sha256": _sha256(spec_path),
        **measure_swing(run_dir, old_q, FIXTURE_DT_S, model_spec=spec_path),
    }


def write_fixtures(
    capture: str, run_dir: Path, bundle_path: Path, argv: list[str]
) -> dict[str, Any]:
    """Overwrite the club's fixtures and return its provenance record."""
    club = CAPTURE_RUNS[capture][0]
    npz_path = FIXTURES / f"swing_q_{club}.npz"
    before = _previous_fixture_record(npz_path, FIXTURES / f"spec_{club}.json", run_dir)
    with np.load(bundle_path, allow_pickle=False) as bundle:
        reference = np.asarray(bundle["reference_q"], dtype=float)
        manifest = json.loads(str(bundle["manifest"]))
    np.savez_compressed(npz_path, q=reference[::FIXTURE_STRIDE].astype(np.float32))
    _write_address_pose(club, manifest, reference[0])
    spec_path = FIXTURES / f"spec_{club}.json"
    spec_path.write_bytes((run_dir / SCALED_SPEC).read_bytes())
    receipt = json.loads((run_dir / "receipt.json").read_text(encoding="utf-8"))
    q_fixture = np.load(npz_path)["q"].astype(float)
    return {
        "capture": capture,
        "pipeline_argv": argv,
        "bundle_manifest": manifest,
        "bundle_sha256": _sha256(bundle_path),
        "run_receipt_sha256": _sha256(run_dir / "receipt.json"),
        "fixture_sha256": _sha256(npz_path),
        "tracked_sha256": _sha256(_write_tracked_swing(club, run_dir)),
        "spec_sha256": _sha256(spec_path),
        "foot_progression": _foot_record(run_dir, receipt, q_fixture[0]),
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


def _foot_record(
    run_dir: Path, receipt: dict[str, Any], q0: np.ndarray
) -> dict[str, Any]:
    """Capture toe-out targets, the calibrated address and fixture frame 0.

    The targets are the capture's address measurement from the run receipt
    (OSV-4, #11730); the model toe-out is MuJoCo FK of the run's scaled
    document. Raises ``ValueError`` when the run had foot progression off.
    """
    from src.shared.python.motion_matching.pipeline.address_feet import (
        FootTargets,
        model_feet_deg,
    )

    block = receipt["address"].get("foot_progression") or {}
    if not block.get("enabled"):
        raise ValueError("the fixture run has no foot-progression targets")
    feet = block["feet"]
    targets = FootTargets(
        {s: float(e["target_deg"]) for s, e in feet.items()},
        {s: bool(e["target_is_default"]) for s, e in feet.items()},
        block["source"],
    )
    kin, _, _, _ = _run_kinematics(run_dir)
    return {
        "source": block["source"],
        "target_deg": dict(targets.target_deg),
        "address_model_deg": {s: float(e["model_deg"]) for s, e in feet.items()},
        "fixture_frame0_model_deg": model_feet_deg(kin, q0, targets),
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


def _write_tracked_swing(club: str, run_dir: Path) -> Path:
    """Commit the run's tracked swing beside the fixture (#12279).

    ``tracked_q_<club>.npz`` holds ``q_track`` (float32) and ``time_s`` on
    the bundle clock (zero at the first tracked frame), so the fixture's feet
    can be checked against the swing it was exported from.
    """
    with np.load(run_dir / "dynamics_record.npz") as record:
        time_s = np.asarray(record["track_time_s"], dtype=float)
        q_track = np.asarray(record["q_track"], dtype=np.float32)
    path = FIXTURES / f"tracked_q_{club}.npz"
    np.savez_compressed(path, q=q_track, time_s=time_s - time_s[0])
    return path


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
