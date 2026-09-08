"""Evidence for #9802: reconstruction and model fit from every k-th frame.

Manual annotation is sparse by nature. This builds the synthetic lab
session, turns its (noisy) detections into manual sets that keep only every
k-th frame, reconstructs and fits the golfer from each, and reports the
accuracy against the truth and against the dense (k = 1) fit. Writes
``docs/motion_capture/evidence/sparse_annotations.{md,json}``.

    python3 -m scripts.motion_capture.sparse_fit_evidence [--out DIR]
"""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

from src.motion_capture.annotate import AnnotationSet
from src.motion_capture.annotate.to_observations import (
    to_view_observations,
    write_observation_set,
)
from src.motion_capture.reconstruct.model import FitOptions
from src.motion_capture.reconstruct.model.golfer import GOLFER_LANDMARK_MAP, GOLFER_SPEC
from src.motion_capture.reconstruct.model.session import fit_session_model
from src.motion_capture.reconstruct.pipeline import (
    MatchSpec,
    reconstruct_session,
    start_cameras_from,
)
from src.motion_capture.reconstruct.skeleton import JOINT_NAMES
from src.motion_capture.variants import variant_dir

from .camera_subsets_evidence import build_session

STRIDES = (1, 3, 5, 10)
VIEWS = ("face_on", "down_line", "overhead")
ITERATIONS = 40


def manual_set_from_detections(
    session: Path, views: tuple[str, ...], stride: int, out_set: str
) -> Path:
    """Every ``stride``-th detected frame of each view as clicks (all joints)."""
    payloads = {}
    inputs = []
    for view in views:
        path = session / "observations" / f"{view}.json"
        inputs.append(path)
        payload = json.loads(path.read_text(encoding="utf-8"))
        fps = float(payload["fps"])
        store = AnnotationSet(
            view,
            int(payload["width"] or 1920),
            int(payload["height"] or 1200),
            fps,
            annotator="synthetic",
        )
        for row in payload["frames"]:
            frame = int(round(float(row["time_s"]) * fps))
            if frame % stride:
                continue
            for k, joint in enumerate(JOINT_NAMES):
                x, y = row["keypoints_px"][k]
                if (
                    row["confidence"][k] > 0
                    and 0 <= x < store.width
                    and 0 <= y < store.height
                ):
                    store.set_point(frame, joint, x, y)
        payloads[view] = to_view_observations(
            store, frames_total=payload["frames_total"]
        )
    return write_observation_set(
        session,
        out_set,
        payloads,
        plan_name="synthetic",
        inputs=inputs,
        parameters={"estimator": "manual", "stride": stride},
    )


def landmark_rms_mm(
    fit_landmarks: np.ndarray, truth: np.ndarray, names: list[str]
) -> float:
    """Model landmarks that map 1:1 onto truth joints, RMS in mm."""
    from src.motion_capture.reconstruct.model.kinematics import ArticulatedModel

    model = ArticulatedModel(GOLFER_SPEC)
    rows, cols = [], []
    for i, landmark in enumerate(model.landmark_names):
        source = GOLFER_LANDMARK_MAP.to_reconstruct.get(landmark)
        if isinstance(source, str) and source in names:
            rows.append(i)
            cols.append(names.index(source))
    t = min(fit_landmarks.shape[0], truth.shape[0])
    d = np.linalg.norm(fit_landmarks[:t][:, rows] - truth[:t][:, cols], axis=2)
    return float(1000 * np.sqrt(np.mean(d**2)))


def run(session: Path, cameras: Path) -> dict[str, Any]:
    truth = json.loads((session / "truth.json").read_text(encoding="utf-8"))
    truth_joints = np.asarray(truth["joints_3d_m"], dtype=float)
    names = list(truth["joint_names"])
    rows = []
    dense: np.ndarray | None = None
    for stride in STRIDES:
        out_set = f"observations_manual_k{stride}"
        manual_set_from_detections(session, VIEWS, stride, out_set)
        variant = f"sparse_k{stride}"
        summary = reconstruct_session(
            session,
            start_cameras=start_cameras_from(cameras),
            scale_anchor=("neck", 0.5),
            match=MatchSpec(observation_set=out_set, variant=variant),
        )
        root = variant_dir(session, variant)
        fit, _ = fit_session_model(
            root,
            GOLFER_SPEC,
            GOLFER_LANDMARK_MAP,
            options=FitOptions(max_iterations=ITERATIONS),
            session_root=session,
        )
        joints = np.load(root / "reconstruct" / "joints_3d_m.npy")
        observed = (np.abs(joints).sum(axis=2) > 0).any(axis=1)
        if dense is None:
            dense = fit.landmarks_m
        t = min(dense.shape[0], fit.landmarks_m.shape[0])
        vs_dense = float(
            1000
            * np.sqrt(
                np.mean(np.linalg.norm(fit.landmarks_m[:t] - dense[:t], axis=2) ** 2)
            )
        )
        rows.append(
            {
                "stride": stride,
                "frames_observed": int(observed.sum()),
                "frames_total": int(joints.shape[0]),
                "reconstruction_rms_px": summary.rms_px,
                "model_landmark_rms_vs_truth_mm": landmark_rms_mm(
                    fit.landmarks_m, truth_joints, names
                ),
                "model_landmark_rms_vs_dense_mm": vs_dense,
                "velocity_violations": fit.velocity_violations,
                "rejected": len(fit.rejected),
            }
        )
    return {"strides": rows, "views": list(VIEWS), "iterations": ITERATIONS}


def markdown(payload: dict[str, Any]) -> str:
    head = (
        "| stride k | frames with clicks | reconstruction RMS px | model vs truth mm | "
        "model vs dense fit mm | velocity violations | rejected |\n"
        "| --- | --- | --- | --- | --- | --- | --- |\n"
    )
    lines = [
        f"| {r['stride']} | {r['frames_observed']}/{r['frames_total']} | "
        f"{r['reconstruction_rms_px']:.2f} | {r['model_landmark_rms_vs_truth_mm']:.1f} | "
        f"{r['model_landmark_rms_vs_dense_mm']:.1f} | {r['velocity_violations']} | "
        f"{r['rejected']} |"
        for r in payload["strides"]
    ]
    return head + "\n".join(lines) + "\n"


def write(payload: dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sparse_annotations.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    text = (
        "# Sparse Annotations: Fitting From Sparse Frames (Synthetic Lab Rig)\n\n"
        "Epic #9791, child #9802. The synthetic three-view detections were turned into "
        "manual sets that keep every k-th frame (all joints clicked, 1 px noise), then "
        "reconstructed and fitted with the golfer through the normal path "
        "(`rig reconstruct --observations observations_manual_kK`, `rig fit-model`). "
        "Model landmark RMS is against the truth and against the dense (k = 1) fit. "
        "Regenerate with `python3 -m scripts.motion_capture.sparse_fit_evidence`.\n\n"
        + markdown(payload)
        + "\nFrames without clicks are still produced by the continuity prior; the "
        "reconstruction places them by the segment priors alone, so their accuracy is the "
        "model fit's, not the triangulation's.\n"
    )
    (out_dir / "sparse_annotations.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out", type=Path, default=Path("docs/motion_capture/evidence")
    )
    args = parser.parse_args(argv)
    work = Path(tempfile.mkdtemp(prefix="sparse_fit_"))
    try:
        session, cameras = build_session(work)
        write(run(session, cameras), args.out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
