"""Evidence for #9796: what limiting the cameras costs, on the synthetic lab rig.

Builds the three-view synthetic session, matches it with every subset
(three cameras, the three pairs, the three singles in image space with the
full match's cameras), runs ``compare-variants`` and writes
``docs/motion_capture/evidence/camera_subsets.{md,json}``.

    python3 -m scripts.motion_capture.camera_subsets_evidence [--out DIR]
"""

from __future__ import annotations

import argparse
import itertools
import json
import shutil
import tempfile
from pathlib import Path

from src.motion_capture.compare_variants import compare_variants, markdown
from src.motion_capture.reconstruct import __main__ as recon_cli
from src.motion_capture.reconstruct.model import FitOptions
from src.motion_capture.reconstruct.model.fit2d import (
    ImageSpaceSource,
    fit_session_model_2d,
)
from src.motion_capture.reconstruct.model.golfer import GOLFER_LANDMARK_MAP, GOLFER_SPEC
from src.motion_capture.reconstruct.model.session import fit_session_model
from src.motion_capture.reconstruct.pipeline import (
    MatchSpec,
    reconstruct_session,
    start_cameras_from,
)
from src.motion_capture.variants import variant_dir

VIEWS = ("face_on", "down_line", "overhead")
FRAMES = 48
SEED = 0
NOISE_PX = 1.0
OUTLIERS = 0.02
ITERATIONS = 40


def build_session(root: Path) -> tuple[Path, Path]:
    session = root / "take"
    assert (
        recon_cli.main(
            [
                "synth",
                "--out",
                str(session),
                "--frames",
                str(FRAMES),
                "--noise-px",
                str(NOISE_PX),
                "--outliers",
                str(OUTLIERS),
                "--seed",
                str(SEED),
            ]
        )
        == 0
    )
    truth = json.loads((session / "truth.json").read_text(encoding="utf-8"))
    cameras = session / "cameras.json"
    cameras.write_text(json.dumps(truth["cameras"]), encoding="utf-8")
    return session, cameras


def run(session: Path, cameras: Path) -> dict:
    options = FitOptions(max_iterations=ITERATIONS)
    reconstruct_session(
        session, start_cameras=start_cameras_from(cameras), scale_anchor=("neck", 0.5)
    )
    fit_session_model(session, GOLFER_SPEC, GOLFER_LANDMARK_MAP, options=options)
    for a, b in itertools.combinations(VIEWS, 2):
        name = f"pair_{a[0]}{b[0]}"
        reconstruct_session(
            session,
            start_cameras=start_cameras_from(cameras),
            scale_anchor=("neck", 0.5),
            match=MatchSpec(views=(a, b), variant=name),
        )
        fit_session_model(
            variant_dir(session, name),
            GOLFER_SPEC,
            GOLFER_LANDMARK_MAP,
            options=options,
            session_root=session,
        )
    for view in VIEWS:
        fit_session_model_2d(
            session,
            GOLFER_SPEC,
            GOLFER_LANDMARK_MAP,
            ImageSpaceSource((view,), "", "observations", f"single_{view}"),
            options=options,
        )
    return compare_variants(session)


def write(payload: dict, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "camera_subsets.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    text = (
        "# Camera Subsets: What Fewer Cameras Cost (Synthetic Lab Rig)\n\n"
        f"Epic #9790, child #9796. Synthetic three-view session (`reconstruct synth`, "
        f"{FRAMES} frames, {NOISE_PX} px noise, {OUTLIERS:.0%} outliers, seed {SEED}), "
        "matched with every camera subset: the full match (reference), the three pairs "
        "(triangulated) and the three singles (image-space fit with the full match's "
        "cameras). Reprojection RMS is per view, held-out views marked; 3-D RMS and "
        "joint-angle RMS are against the reference over the frames both have. Regenerate "
        "with `python3 -m scripts.motion_capture.camera_subsets_evidence`.\n\n"
        + markdown(payload)
        + "\nReading it: a pair's held-out view is the honest test of that pair; a single "
        "view's reprojection is small by construction and its angle error against the "
        "reference shows what one camera cannot see (depth-direction motion).\n"
    )
    (out_dir / "camera_subsets.md").write_text(text, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("docs/motion_capture/evidence"),
        help="directory for camera_subsets.{md,json}",
    )
    args = parser.parse_args(argv)
    work = Path(tempfile.mkdtemp(prefix="camera_subsets_"))
    try:
        session, cameras = build_session(work)
        payload = run(session, cameras)
        payload.pop("provenance", None)
        write(payload, args.out)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
