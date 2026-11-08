"""Reproduce SHOW-A10 evidence and summary deliverable (#14735).

Generates:
1. Key-frame still PNGs with owner's A7 grip clock inset.
2. Side-by-side comparison images with 11-04 baseline (fa7a9b7075).
3. SUMMARY.md documenting inputs, outputs, hashes, and acceptance comparison.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from src.shared.python.motion_matching.show_a9_refit import (
    BASELINE_METRICS,
    CANDIDATE_METRICS,
    compare_against_baseline,
    format_comparison_table,
)
from src.tools.native_viewer_export.key_frames import (
    compose_side_by_side,
    draw_grip_clock,
    render_a10_summary,
)

ROOT = Path(__file__).resolve().parents[5]
OUT_DIR = ROOT / "docs" / "development" / "full_body_models" / "evidence" / "show_a10"
SHOW_A8_DIR = (
    ROOT / "docs" / "development" / "full_body_models" / "evidence" / "show_a8"
)
TRUNK_DIR = (
    ROOT / "docs" / "development" / "full_body_models" / "evidence" / "trunk_segments"
)
SHOW_A9_DIR = (
    ROOT / "docs" / "development" / "full_body_models" / "evidence" / "show_a9"
)
ROLLS = {"lead": 15.0, "trail": -100.73}


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def render_key_frames() -> None:
    """Render face-on frames with grip clock inset and down-the-line frames."""
    face_on_frames = {
        "face_on_address_1280.png": SHOW_A8_DIR / "face_on_address_1280.png",
        "face_on_top_1280.png": SHOW_A8_DIR / "face_on_top_1280.png",
        "face_on_impact_1280.png": SHOW_A8_DIR / "face_on_impact_1280.png",
    }
    for out_name, src in face_on_frames.items():
        img = Image.open(src)
        draw_grip_clock(img, ROLLS, r=70).save(
            OUT_DIR / out_name, format="PNG", optimize=True
        )

    dtl_frames = {
        "down_the_line_address_1280.png": SHOW_A8_DIR
        / "down_the_line_address_1280.png",
        "down_the_line_top_1280.png": SHOW_A8_DIR / "down_the_line_top_1280.png",
        "down_the_line_impact_1280.png": SHOW_A8_DIR / "down_the_line_impact_1280.png",
    }
    for out_name, src in dtl_frames.items():
        Image.open(src).save(OUT_DIR / out_name, format="PNG", optimize=True)


def render_side_by_side() -> None:
    """Render side-by-side comparison images against 11-04 baseline."""
    sbs_pairs = [
        (
            "side_by_side_address_face_on.png",
            TRUNK_DIR / "before_v1_rigid_address_face_on.png",
            TRUNK_DIR / "after_v2_segmented_address_face_on.png",
            "11-04 Baseline (fa7a9b7075, Rigid v1)",
            "A10 Improved Candidate (A7 Grip + A8 Trunk)",
        ),
        (
            "side_by_side_top_face_on.png",
            TRUNK_DIR / "before_v1_rigid_top_face_on.png",
            TRUNK_DIR / "after_v2_segmented_top_face_on.png",
            "11-04 Baseline (fa7a9b7075, Rigid v1)",
            "A10 Improved Candidate (A7 Grip + A8 Trunk)",
        ),
        (
            "side_by_side_address_dtl.png",
            TRUNK_DIR / "before_v1_rigid_address_dtl.png",
            TRUNK_DIR / "after_v2_segmented_address_dtl.png",
            "11-04 Baseline (fa7a9b7075, Rigid v1)",
            "A10 Improved Candidate (A7 Grip + A8 Trunk)",
        ),
        (
            "side_by_side_top_dtl.png",
            TRUNK_DIR / "before_v1_rigid_top_dtl.png",
            TRUNK_DIR / "after_v2_segmented_top_dtl.png",
            "11-04 Baseline (fa7a9b7075, Rigid v1)",
            "A10 Improved Candidate (A7 Grip + A8 Trunk)",
        ),
    ]
    for out_name, base_p, cand_p, lbl_l, lbl_r in sbs_pairs:
        sbs = compose_side_by_side(Image.open(base_p), Image.open(cand_p), lbl_l, lbl_r)
        sbs.save(OUT_DIR / out_name, format="PNG", optimize=True)


def write_summary() -> None:
    """Write SUMMARY.md documenting inputs, outputs, and metrics."""
    outputs = {f.name: sha256_file(f) for f in sorted(OUT_DIR.glob("*.png"))}
    a9_receipt = json.loads((SHOW_A9_DIR / "receipt.json").read_text(encoding="utf-8"))
    inputs = {
        "capture_sha256": a9_receipt["input_hashes"]["capture_sha256"],
        "v1_spec_sha256": a9_receipt["input_hashes"]["v1_spec_sha256"],
        "v2_spec_sha256": a9_receipt["input_hashes"]["v2_spec_sha256"],
        "show_a9_receipt_sha256": sha256_file(SHOW_A9_DIR / "receipt.json"),
        "baseline_commit": a9_receipt["input_hashes"]["baseline_commit"],
    }

    summary = render_a10_summary(
        model_id="mujoco_driver",
        source_commit="4bd0ae5099bca6f51c26e7bd0b356d2a6176b0c7",
        receipt_verdict="REJECTED (G3) - recorded as candidate per AC 3",
        input_hashes=inputs,
        output_hashes=outputs,
        lead_roll_deg=15.0,
        trail_roll_deg=-100.73,
    )

    comp = compare_against_baseline(CANDIDATE_METRICS, BASELINE_METRICS)
    extra = [
        "## Acceptance Comparison vs 11-04 Baseline (`fa7a9b7075`)",
        "",
        format_comparison_table(comp),
        "",
        "## Visual Artifacts",
        "",
        "- `face_on_*_1280.png`: Face-on key frames (address, top, impact) with A7 grip clock inset (12 o'clock = 0°, green tick at +15°, red needle lead, blue needle trail).",
        "- `down_the_line_*_1280.png`: Down-the-line key frames (address, top, impact).",
        "- `side_by_side_*`: Side-by-side comparison with 11-04 baseline render showing grip and trunk changes.",
        "",
        "## Reproduction Command",
        "",
        "```bash",
        "python -m src.tools.native_viewer_export --bundle <A9 bundle> --swing driver --club Driver --engines mujoco --speeds 1,0.5 --segment-shading",
        "```",
        "",
    ]
    full = summary.rstrip() + "\n\n" + "\n".join(extra)
    (OUT_DIR / "SUMMARY.md").write_text(full, encoding="utf-8")


def render_all() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    render_key_frames()
    render_side_by_side()
    write_summary()


if __name__ == "__main__":
    render_all()
