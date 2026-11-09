"""Reproduce SHOW-A10 evidence, provenance manifest, and summary deliverable (#14735 / #14906).

Generates:
1. Key-frame still PNGs with owner's A7 grip clock inset.
2. Side-by-side comparison images with 11-04 baseline (fa7a9b7075).
3. SUMMARY.md and provenance_manifest.json documenting inputs, outputs, hashes, and attribution.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
from typing import Any
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
FACE_ON_NAMES = (
    "face_on_address_1280.png",
    "face_on_top_1280.png",
    "face_on_impact_1280.png",
)
DTL_NAMES = (
    "down_the_line_address_1280.png",
    "down_the_line_top_1280.png",
    "down_the_line_impact_1280.png",
)
SBS_SPECS = (
    ("side_by_side_address_face_on.png", "address_face_on"),
    ("side_by_side_top_face_on.png", "top_face_on"),
    ("side_by_side_address_dtl.png", "address_dtl"),
    ("side_by_side_top_dtl.png", "top_dtl"),
)


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _rel_posix(path: Path, repo_root: Path) -> str:
    try:
        return path.resolve().relative_to(repo_root.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


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


def build_provenance_manifest(
    out_dir: Path = OUT_DIR,
    show_a8_dir: Path = SHOW_A8_DIR,
    trunk_dir: Path = TRUNK_DIR,
    a9_receipt_path: Path | None = None,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Derive truthful provenance manifest from supplied inputs and artifacts."""
    r_file = a9_receipt_path or (SHOW_A9_DIR / "receipt.json")
    a9 = json.loads(r_file.read_text(encoding="utf-8"))
    r_in = a9["input_hashes"]
    grip = a9.get("address_grip", {})
    gates = a9.get("acceptance_gates", {})
    comp = a9.get("comparison_with_baseline", {})
    lead_deg = float(grip.get("lead_roll_deg", 15.0))
    trail_deg = float(grip.get("trail_roll_deg", -100.73))
    target_deg = float(grip.get("target_deg", 15.0))

    src_stills: dict[str, str] = {}
    out_arts: dict[str, dict[str, Any]] = {}

    for name in sorted(FACE_ON_NAMES + DTL_NAMES):
        s_path = show_a8_dir / name
        s_rel = _rel_posix(s_path, repo_root)
        src_stills[s_rel] = sha256_file(s_path)
        is_fo = name in FACE_ON_NAMES
        overlay = (
            {
                "type": "grip_clock_inset",
                "lead_roll_deg": lead_deg,
                "trail_roll_deg": trail_deg,
                "target_tick_deg": target_deg,
                "is_measured_from_frame": False,
            }
            if is_fo
            else None
        )
        out_arts[name] = {
            "sha256": sha256_file(out_dir / name),
            "derivation": (
                "copied_from_show_a8_with_illustrative_grip_clock_overlay"
                if is_fo
                else "copied_from_show_a8_reencoded_png"
            ),
            "source_stills": [s_rel],
            "is_fresh_a9_render": False,
            "overlay_applied": overlay,
        }

    for out_name, suffix in SBS_SPECS:
        b_path = trunk_dir / f"before_v1_rigid_{suffix}.png"
        c_path = trunk_dir / f"after_v2_segmented_{suffix}.png"
        b_rel, c_rel = _rel_posix(b_path, repo_root), _rel_posix(c_path, repo_root)
        src_stills[b_rel] = sha256_file(b_path)
        src_stills[c_rel] = sha256_file(c_path)
        out_arts[out_name] = {
            "sha256": sha256_file(out_dir / out_name),
            "derivation": "composed_side_by_side_from_trunk_segments_synthetic_stills",
            "source_stills": [b_rel, c_rel],
            "is_fresh_a9_render": False,
            "overlay_applied": None,
        }

    cand_id = a9.get("model_id", "mujoco_driver_candidate_a9")
    verdict = f"{gates.get('status', 'REJECTED')} ({gates.get('horizon', 'G3')})"
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "deliverable": "SHOW-A10",
        "tracking_issues": {
            "delivery_issue": 14735,
            "epic_14906": 14906,
            "epic_status": "OPEN",
        },
        "best_match_registry_model_id": "mujoco_driver",
        "candidate_refit_model_id": cand_id,
        "visual_frame_attribution": {
            "is_verified_a9_bundle_render": False,
            "key_frame_source_dir": _rel_posix(show_a8_dir, repo_root),
            "side_by_side_source_dir": _rel_posix(trunk_dir, repo_root),
        },
        "angle_attribution": {
            "clock_overlay_values_deg": {
                "lead": lead_deg,
                "trail": trail_deg,
                "target_tick": target_deg,
                "classification": "illustrative_target_and_modelling_choice",
                "is_measured_from_rendered_frames": False,
            },
            "rendered_frame_hand_measurements": {
                "status": "unavailable",
                "lead_roll_deg": None,
                "trail_roll_deg": None,
            },
        },
        "quantitative_refit_metrics": {
            "source_receipt": _rel_posix(r_file, repo_root),
            "receipt_model_id": cand_id,
            "verdict": verdict,
            "is_physically_accepted": bool(gates.get("is_physically_accepted", False)),
            "candidate_metrics": {
                "marker_rms_mm": float(comp["marker_rms_mm"]["candidate"]),
                "closure_residual_mm": float(comp["closure_residual_mm"]["candidate"]),
            },
        },
        "receipt_inputs": {**r_in, "show_a9_receipt_sha256": sha256_file(r_file)},
        "source_stills": dict(sorted(src_stills.items())),
        "output_artifacts": dict(sorted(out_arts.items())),
        "video_delivery_manifest": {
            "in_repo_video_artifacts": [],
            "a9_bundle_video_render_status": "unavailable_in_repo",
        },
    }
    validate_provenance_manifest(manifest)
    return manifest


def validate_provenance_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject false A9 render claims, fake frame measurements, or promoted REJECTED verdicts."""
    frame_attr = manifest.get("visual_frame_attribution", {})
    if frame_attr.get("is_verified_a9_bundle_render") is True:
        raise ValueError("Invalid: verified_a9_bundle_render cannot be True.")
    for fname, meta in manifest.get("output_artifacts", {}).items():
        if meta.get("is_fresh_a9_render") is True or not meta.get("source_stills"):
            raise ValueError(f"Invalid artifact attribution for {fname!r}.")
    overlay = manifest.get("angle_attribution", {}).get("clock_overlay_values_deg", {})
    if overlay.get("is_measured_from_rendered_frames") is True:
        raise ValueError("Invalid: overlay not measured_from_rendered_frames.")
    if overlay.get("classification") != "illustrative_target_and_modelling_choice":
        raise ValueError("Invalid: overlay classification must be illustrative.")
    quant = manifest.get("quantitative_refit_metrics", {})
    if (
        "REJECTED" in str(quant.get("verdict", ""))
        and quant.get("is_physically_accepted") is True
    ):
        raise ValueError("Invalid: REJECTED candidate cannot be physically_accepted.")


def write_summary(
    out_dir: Path = OUT_DIR,
    show_a8_dir: Path = SHOW_A8_DIR,
    trunk_dir: Path = TRUNK_DIR,
    a9_receipt_path: Path | None = None,
    repo_root: Path = ROOT,
) -> dict[str, Any]:
    """Write provenance_manifest.json and SUMMARY.md documenting inputs, outputs, and metrics."""
    manifest = build_provenance_manifest(
        out_dir, show_a8_dir, trunk_dir, a9_receipt_path, repo_root
    )
    (out_dir / "provenance_manifest.json").write_text(
        json.dumps(manifest, separators=(",", ": "), sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    outputs = {k: v["sha256"] for k, v in sorted(manifest["output_artifacts"].items())}
    overlay = manifest["angle_attribution"]["clock_overlay_values_deg"]
    quant = manifest["quantitative_refit_metrics"]

    summary = render_a10_summary(
        model_id="mujoco_driver",
        receipt_model_id=quant["receipt_model_id"],
        source_commit="6392e76112 (SHOW-A8 stills) / f304efdcd4 (SHOW-A9 receipt)",
        receipt_verdict=f"{quant['verdict']} - recorded as candidate per AC 3 (not physically accepted)",
        input_hashes=manifest["receipt_inputs"],
        output_hashes=outputs,
        lead_roll_deg=float(overlay["lead"]),
        trail_roll_deg=float(overlay["trail"]),
        source_still_hashes=manifest["source_stills"],
    )

    comp = compare_against_baseline(CANDIDATE_METRICS, BASELINE_METRICS)
    extra = [
        "## Angle & Measurement Attribution",
        "",
        "- **Clock overlay (`+15.0°` lead / `-100.73°` trail):** Illustrative target / modelling choice from SHOW-A7 / SHOW-A9 receipt drawn over SHOW-A8 stills; **not measured from rendered frames**.",
        "- **Rendered-frame hand roll & wrist angles:** `unavailable` (SHOW-A10 reuses pre-rendered SHOW-A8 PNGs).",
        "- **Quantitative refit metrics:** Imported from `show_a9/receipt.json` (`mujoco_driver_candidate_a9`, `REJECTED (G3)`).",
        "- **Video delivery:** `unavailable_in_repo` (no A9 `.npz` bundle or video set generated here; epic `#14906` remains open).",
        "",
        "## Acceptance Comparison vs 11-04 Baseline (`fa7a9b7075`)",
        "",
        format_comparison_table(comp),
        "",
        "## Visual Artifacts",
        "",
        "- `face_on_*_1280.png`: Face-on key frames (address, top, impact) with A7 grip clock inset (12 o'clock = 0°, green tick at +15°, red needle lead, blue needle trail; copied from `show_a8/`).",
        "- `down_the_line_*_1280.png`: Down-the-line key frames (address, top, impact) copied from `show_a8/`.",
        "- `side_by_side_*`: Side-by-side comparison with 11-04 baseline render showing grip and trunk changes (from `trunk_segments/`).",
        "- `provenance_manifest.json`: Per-file provenance and angle attribution manifest.",
        "",
        "## Reproduction Command",
        "",
        "```bash",
        "python docs/development/full_body_models/evidence/show_a10/render_show_a10_evidence.py",
        "```",
        "",
    ]
    full = summary.rstrip() + "\n\n" + "\n".join(extra)
    (out_dir / "SUMMARY.md").write_text(full, encoding="utf-8", newline="\n")
    return manifest


def render_all(*, regenerate_images: bool = False) -> dict[str, Any]:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    if regenerate_images:
        render_key_frames()
        render_side_by_side()
    return write_summary()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regenerate-images", action="store_true")
    render_all(regenerate_images=parser.parse_args(argv).regenerate_images)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
