"""OS-9: anatomical-prior marker registration on the held address (#13434).

Replaces the OS-3/OS-3b free calibration, which seeded the model facing +x
(the golfer faces -x) and allowed 0.5 m marker offsets, so it converged on a
mirrored, unregistered marker set. Here:

1. The model is scaled exactly as OS-3b scales it.
2. The calibration frames are the quiet address hold at the start of the
   capture (owner decision on #13434: option (a), a static segment).
3. Every body marker starts at an anatomical prior anchored to the scaled
   model's joints (``anatomical_marker_priors``) and may move at most its
   prior radius; club markers are placed freely.
4. The pelvis heading and tilt are seeded from the waist cluster and written
   into the model defaults, so OpenSim IK starts facing the right way.
5. The registered model is re-solved over the swing with the OS-8 continuous
   IK, and the torso-in-pelvis rotation is reported at address, top and impact.

Kinematics only. The anatomical priors are declared estimates, not
measurements; the receipt records every marker's deviation from its prior.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from docs.development.opensim_tour_matching.os3b_scale_and_full_ik import (  # noqa: E402
    _build_pose_fn,
    _run_ik_motion,
    _scale_and_setup_model,
    tracking_variant,
)
from docs.development.opensim_tour_matching.os8_continuous_ik import (  # noqa: E402
    OsimKinematics,
)
from src.engines.physics_engines.opensim.python.tour_matching import (  # noqa: E402
    GOLF_HUMANOID_MARKER_BODIES,
    read_trc,
    write_model,
)
from src.engines.physics_engines.opensim.python.tour_matching.anatomical_marker_priors import (  # noqa: E402
    AnatomicalPrior,
    anatomical_marker_priors,
    clamp_to_priors,
    lateral_side_ok,
)
from src.engines.physics_engines.opensim.python.tour_matching.marker_plausibility import (  # noqa: E402
    TRUNK_ROTATION_LIMITS_DEG,
    PlausibilityReport,
    assess_marker_placements,
    assess_trunk_rotation,
)
from src.engines.physics_engines.opensim.python.tour_matching.marker_set import (  # noqa: E402
    parse_model,
)
from src.engines.physics_engines.opensim.python.tour_matching.static_address_registration import (  # noqa: E402
    quiet_segment_frames,
    solve_pelvis_rotations,
    waist_cluster_frame,
)
from src.shared.python.motion_matching.marker_calibration import (  # noqa: E402
    PlacementPrior,
    calibrate_constrained_marker_offsets,
)
from src.shared.python.motion_matching.tour_capture_contract import (  # noqa: E402
    TourCapture,
)
from src.shared.python.tour_baselines.events import SWING_EVENTS_DRIVER  # noqa: E402

EVIDENCE = ROOT / "docs/development/opensim_tour_matching/evidence"
DEFAULT_TRC = EVIDENCE / "tour_average_tracked.trc"
OS8_DIR = EVIDENCE / "os8_continuous_ik"
OS8_MODEL = EVIDENCE / "os7_moco_g1/inputs/golf_humanoid_scaled_tour_markers.osim"
LOGGER = logging.getLogger("os9_address_registration")
WAIST = ("WaistLeft", "WaistRight", "WaistLBack", "WaistRBack")
# The #13434 diagnosis sampled address at 0.211 s (frame 76), after the hold.
ISSUE_ADDRESS_FRAME = 76
LUMBAR = ("lumbar_extension", "lumbar_bending", "lumbar_rotation")
#: Plausibility-gate event -> receipt event.
GATE_EVENTS = {"address": "address_hold", "top": "top", "impact": "impact"}


def sha256_file(path: Path) -> str:
    """Hex sha256 of a file."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frames_subset(capture: TourCapture, frames: np.ndarray) -> TourCapture:
    """The capture restricted to ``frames`` (all labels kept)."""
    return TourCapture(
        time_s=capture.time_s[frames],
        labels=capture.labels,
        points_m=capture.points_m[frames],
        valid=capture.valid[frames],
    )


def labels_subset(capture: TourCapture, labels: list[str]) -> TourCapture:
    """The capture restricted to ``labels``."""
    idx = [capture.index(k) for k in labels]
    return TourCapture(
        time_s=capture.time_s,
        labels=tuple(labels),
        points_m=capture.points_m[:, idx],
        valid=capture.valid[:, idx],
    )


def set_default_values(osim: Path, values: dict[str, float], out: Path) -> Path:
    """Write ``osim`` with coordinate default values replaced by ``values``."""
    tree = parse_model(osim)
    found = set()
    for coord in tree.getroot().iter("Coordinate"):
        name = coord.get("name")
        if name in values:
            node = coord.find("default_value")
            if node is None:
                raise ValueError(f"coordinate {name} has no default_value")
            node.text = repr(float(values[name]))
            found.add(name)
    missing = set(values) - found
    if missing:
        raise ValueError(f"coordinates not in model: {sorted(missing)}")
    return write_model(tree, out)


def pelvis_seed(scaled_osim: Path, cal: TourCapture, priors: dict) -> dict:
    """Pelvis rotations from the waist axes, and the translation that puts the
    prior belt centroid on the observed one (means over the hold)."""
    import opensim  # type: ignore[import-not-found]

    mean = {
        k: np.mean(cal.points_m[cal.valid[:, cal.index(k)], cal.index(k)], axis=0)
        for k in WAIST
    }
    rot = waist_cluster_frame(mean)
    coords, residual = solve_pelvis_rotations(opensim.Model(str(scaled_osim)), rot)
    belt_local = np.mean([priors[k].offset_m for k in WAIST], axis=0)
    origin = np.mean(list(mean.values()), axis=0) - rot @ belt_local
    coords.update(pelvis_tx=origin[0], pelvis_ty=origin[1], pelvis_tz=origin[2])
    return {"coords": {k: float(v) for k, v in coords.items()}, "residual": residual}


def prior_deviation_report(
    offsets: dict, priors: dict[str, AnatomicalPrior]
) -> dict[str, Any]:
    """Per-marker deviation from its prior, bound and side checks."""
    rows: dict[str, Any] = {}
    for name, prior in priors.items():
        if name not in offsets:
            rows[name] = {"status": "not in capture"}
            continue
        off = np.asarray(offsets[name][1], float)
        dev = float(np.linalg.norm(off - np.asarray(prior.offset_m)))
        rows[name] = {
            "body": prior.body,
            "landmark": prior.landmark,
            "prior_m": list(prior.offset_m),
            "offset_m": off.tolist(),
            "deviation_mm": dev * 1e3,
            "radius_mm": prior.radius_m * 1e3,
            "at_bound": dev >= prior.radius_m - 1e-6,
            "side_ok": lateral_side_ok(name, (off[0], off[1], off[2])),
        }
    return rows


def torso_in_pelvis_deg(sto: Path, osim: Path, frames: dict[str, int]) -> dict:
    """Torso-in-pelvis rotation (deg) at the given frames.

    ``magnitude`` is the total rotation angle; the lumbar coordinates give its
    flexion/extension, lateral bending and axial rotation components.
    """
    kin = OsimKinematics(osim)
    lines = sto.read_text().splitlines()
    head = next(i for i, ln in enumerate(lines) if ln.startswith("time"))
    cols = [h.split("/")[-2] for h in lines[head].split("\t")[1:]]
    out = {}
    for event, f in frames.items():
        row = [float(x) for x in lines[head + 1 + f].split("\t")[1:]]
        q = np.array([row[cols.index(n)] for n in kin.names])
        poses = kin.poses(q)
        rel = poses["pelvis"][0].T @ poses["torso"][0]
        out[event] = {
            "magnitude": float(np.degrees(Rotation.from_matrix(rel).magnitude())),
            **{
                n: float(np.degrees(q[kin.names.index(n)]))
                for n in LUMBAR
                if n in kin.names
            },
        }
    return out


def event_frames() -> dict[str, int]:
    """Address (hold), address at 0.211 s, top and impact frame indices."""
    return {
        "address_hold": SWING_EVENTS_DRIVER.address.frame_index,
        "address_0p211s": ISSUE_ADDRESS_FRAME,
        "top": SWING_EVENTS_DRIVER.top_of_backswing.frame_index,
        "impact": SWING_EVENTS_DRIVER.impact.frame_index,
    }


def run_continuous_ik(
    osim: Path, trc: Path, out_dir: Path, reg_weight: float
) -> dict[str, Any]:
    """OS-8 continuous IK on ``osim``; its receipt is kept as ik_receipt.json.

    OS-8 exits 1 when the judged window is discontinuous; that is recorded in
    its receipt, not raised, so a failed registration is still reported.
    """
    cmd = [
        sys.executable,
        str(ROOT / "docs/development/opensim_tour_matching/os8_continuous_ik.py"),
        "--osim",
        str(osim),
        "--trc",
        str(trc),
        "--output",
        str(out_dir),
        "--last",
        "529",
        "--judge-last",
        "475",
        "--reg-weight",
        repr(reg_weight),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode not in (0, 1):
        raise RuntimeError(f"OS-8 IK failed ({proc.returncode}): {proc.stderr[-2000:]}")
    raw = out_dir / "receipt.json"
    data = json.loads(raw.read_text())
    (out_dir / "ik_receipt.json").write_text(json.dumps(data, indent=2) + "\n")
    raw.unlink()
    return data


def calibrate(
    osim: Path, trc: Path, out_dir: Path, iterations: int, threshold: float
) -> dict[str, Any]:
    """Scale, seed, calibrate on the hold and write the registered model."""
    full = read_trc(trc)
    scaled, _scale, base_model, base_state, names, rotational, bodies = (
        _scale_and_setup_model(osim, full, out_dir)
    )
    quiet = quiet_segment_frames(full, threshold_mps=threshold)
    held = frames_subset(full, quiet)
    seen = [k for k in held.labels if held.valid[:, held.index(k)].any()]
    cal = labels_subset(held, seen)
    priors = anatomical_marker_priors(scaled)
    unseen = sorted(k for k in priors if k not in seen)
    seed = pelvis_seed(scaled, cal, priors)
    seeded = set_default_values(scaled, seed["coords"], out_dir / "scaled_seeded.osim")
    pose_fn = _build_pose_fn(base_model, base_state, names, bodies)

    def ik_fn(offsets: dict, cap: TourCapture) -> np.ndarray:
        return _run_ik_motion(seeded, offsets, cap, names, rotational)

    body_labels = [k for k in cal.labels if k in priors]
    prior_offsets = {k: (priors[k].body, priors[k].offset_m) for k in body_labels}
    q_prior = ik_fn(prior_offsets, labels_subset(cal, body_labels))
    result = calibrate_constrained_marker_offsets(
        cal,
        GOLF_HUMANOID_MARKER_BODIES,
        pose_fn,
        ik_fn,
        initial_q=np.mean(q_prior, axis=0),
        iterations=iterations,
        prior=PlacementPrior(
            offsets={k: priors[k].offset_m for k in body_labels},
            weight=float(cal.frames),
            constrain=lambda o: clamp_to_priors(o, priors),
        ),
    )
    # A marker never seen in the hold keeps its prior, flagged in the receipt.
    offsets = {**{k: (priors[k].body, priors[k].offset_m) for k in unseen}}
    offsets.update(result.offsets)
    registered = write_model(
        tracking_variant(seeded, offsets, []),
        out_dir / "golf_humanoid_scaled_tour_markers_address_registered.osim",
    )
    return {
        "full": full,
        "quiet": quiet,
        "priors": priors,
        "seed": seed,
        "result": result,
        "offsets": offsets,
        "unseen": unseen,
        "registered": registered,
    }


def plausibility(cal: dict, rot: dict[str, dict]) -> dict[str, Any]:
    """Screening gate verdict (#13434): marker placements and trunk rotation."""
    seen = {k: v for k, v in cal["offsets"].items() if k not in cal["unseen"]}
    priors = {k: p for k, p in cal["priors"].items() if k in seen}
    trunk = {e: rot[GATE_EVENTS[e]]["magnitude"] for e in TRUNK_ROTATION_LIMITS_DEG}
    report = PlausibilityReport(
        checks=assess_marker_placements(seen, priors),
        trunk=assess_trunk_rotation(trunk),
    )
    return {
        "passed": report.passed,
        "failures": list(report.failures()),
        "trunk": [vars(t) for t in report.trunk],
        "limits": "declared screening bounds, citation pending (#13434)",
    }


def write_receipt(
    osim: Path, trc: Path, out_dir: Path, cal: dict, reg_weight: float
) -> Path:
    """Run the swing IK on the registered model and write receipt.json."""
    registered: Path = cal["registered"]
    ik = run_continuous_ik(registered, trc, out_dir, reg_weight)
    frames = event_frames()
    sto = out_dir / "ik_states_continuous.sto"
    rot_new = torso_in_pelvis_deg(sto, registered, frames)
    rot_old = torso_in_pelvis_deg(
        OS8_DIR / "ik_states_continuous.sto", OS8_MODEL, frames
    )
    result, quiet, full = cal["result"], cal["quiet"], cal["full"]
    report = prior_deviation_report(cal["offsets"], cal["priors"])
    os8 = json.loads((OS8_DIR / "receipt.json").read_text())
    receipt = {
        "schema": "os9-address-registration/v1",
        "issue": "#13434",
        "qualification": "kinematics only; anatomical priors declared, not validated",
        "inputs": {
            "model": osim.name,
            "model_sha256": sha256_file(osim),
            "trc": trc.name,
            "trc_sha256": sha256_file(trc),
        },
        "method": {
            "calibration_frames": "quiet address hold (owner decision, #13434 a)",
            "quiet_first_frame": int(quiet[0]),
            "quiet_last_frame": int(quiet[-1]),
            "quiet_duration_s": float(full.time_s[quiet[-1]] - full.time_s[quiet[0]]),
            "prior_weight_frame_equivalents": len(quiet),
            "iterations": result.iterations,
            "pelvis_seed": cal["seed"]["coords"],
            "pelvis_seed_residual_rad": cal["seed"]["residual"],
            "assumptions": [
                "Body markers lie within their declared radius of an anatomical "
                "landmark anchored to the scaled model's joints.",
                "Waist-belt axes coincide with the pelvis body axes (heading seed).",
                "The subject holds one posture during the quiet address segment.",
                "Club markers have no anatomical prior and are placed freely.",
            ],
        },
        "calibration": {
            "rms_per_iteration_m": list(result.rms_per_iteration_m),
            "best_iteration": result.best_iteration,
            "per_marker_rms_m": result.per_marker_rms_m,
        },
        "prior_deviation": report,
        "placed_at_prior_unseen_in_hold": cal["unseen"],
        "markers_at_bound": sorted(k for k, r in report.items() if r.get("at_bound")),
        "side_failures": sorted(
            k for k, r in report.items() if r.get("side_ok") is False
        ),
        "swing_ik": {
            "mean_marker_rms_m": ik["marker_rms_m"]["mean"],
            "max_marker_rms_m": ik["marker_rms_m"]["max"],
            "os8_mean_marker_rms_m": os8["marker_rms_m"]["mean"],
            "reg_weight": reg_weight,
            "continuous": ik["continuity"]["continuous"],
            "judged_largest_step": ik["continuity"]["judged_largest_steps"][0],
        },
        "torso_in_pelvis_rotation_deg": {
            "frames": frames,
            "registered": rot_new,
            "os8_baseline": rot_old,
        },
        "plausibility": plausibility(cal, rot_new),
        "outputs": {
            "model": registered.name,
            "model_sha256": sha256_file(registered),
            "states_sha256": sha256_file(sto),
        },
    }
    path = out_dir / "receipt.json"
    path.write_text(json.dumps(receipt, indent=2, default=float) + "\n")
    return path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--osim", type=Path, required=True, help="unscaled model")
    parser.add_argument("--trc", type=Path, default=DEFAULT_TRC)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=6)
    parser.add_argument("--threshold", type=float, default=0.05)
    # The left hand carries no marker and is not attached to the club, so
    # pro_sup_l is barely observable; at OS-8's 1e-3 tie-break it hopped
    # between its range limits (frame 406). 1e-2 holds the branch with the
    # same marker RMS and trunk angles (#13434).
    parser.add_argument("--reg-weight", type=float, default=1e-2)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO)
    args.output.mkdir(parents=True, exist_ok=True)
    cal = calibrate(args.osim, args.trc, args.output, args.iterations, args.threshold)
    LOGGER.info(
        "wrote %s",
        write_receipt(args.osim, args.trc, args.output, cal, args.reg_weight),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
