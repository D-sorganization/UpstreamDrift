"""Conditional fixed-attachment rigidity floors for the actual full capture."""

import hashlib
import json
from pathlib import Path
import zipfile

import numpy as np
from src.shared.python.motion_matching.rigidity import rigid_attachment_residuals


def rms_mm(values: np.ndarray) -> float:
    return float(np.sqrt(np.nanmean(values**2)) * 1000)


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    raw = (root / "regularized_fit_9967_62/returned-candidate.json").read_bytes()
    candidate = json.loads(raw)
    with zipfile.ZipFile(root / "regularized_fit_9967_63/raw-run.zip") as archive:
        capture_raw = archive.read("inputs/2-driver_marker_payload_9967.json")
    capture = json.loads(capture_raw)
    if candidate["capture_sha256"] != capture["source_sha256"]:
        raise ValueError("Capture identity mismatch")
    indices = [capture["labels"].index(label) for label in candidate["marker_labels"]]
    target = np.asarray(capture["points_world_m"])[:, indices]
    target[~np.asarray(capture["valid"], dtype=bool)[:, indices]] = np.nan
    clock = np.asarray(capture["time_s"])
    bodies = np.asarray(candidate["marker_bodies"])
    errors = rigid_attachment_residuals(
        np.asarray(candidate["marker_offsets_m"]), target, bodies
    )
    split = [
        "hypothetical_independent_head"
        if name in ("HeadTop", "HeadFront", "HeadSide")
        else body
        for name, body in zip(candidate["marker_labels"], bodies, strict=True)
    ]
    relaxed = rigid_attachment_residuals(
        np.asarray(candidate["marker_offsets_m"]), target, split
    )
    report = {
        "qualification": "independent-body rigid relaxation conditional on fixed attachments; not forward dynamics",
        "candidate_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "capture_payload_sha256": hashlib.sha256(capture_raw).hexdigest(),
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "provider_sha256": hashlib.sha256(
            Path(rigid_attachment_residuals.__code__.co_filename).read_bytes()
        ).hexdigest(),
        "frames": len(clock),
        "duration_s": float(clock[-1]),
        "markers": len(indices),
        "full_rms_mm": rms_mm(errors),
        "terminal_rms_mm": rms_mm(errors[-1]),
        "prefixes": {
            str(end): {
                "rms_mm": rms_mm(errors[clock <= end]),
                "terminal_mm": rms_mm(errors[clock <= end][-1]),
            }
            for end in (0.6, 0.85, 1.0, 1.2, 1.4, 1.6, float(clock[-1]))
        },
        "body_rms_mm": {
            str(body): rms_mm(errors[:, bodies == body]) for body in np.unique(bodies)
        },
        "hypothetical_head_relaxation": {
            "full_rms_mm": rms_mm(relaxed),
            "terminal_rms_mm": rms_mm(relaxed[-1]),
            "status": "diagnostic only; permits independent six-DOF head pose and changes original model assumptions",
        },
        "limitations": [
            "No joint connectivity, dynamics, temporal continuity or effort constraints.",
            "Changing offsets/body assignments changes the bound; no original model changes were made.",
        ],
    }
    Path(__file__).with_name("report.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
