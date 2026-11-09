"""Reproduce SHOW-A10 evidence and summary deliverable (#14735).

Generates:
1. Key-frame still PNGs with owner's A7 grip clock inset.
2. Side-by-side comparison images with 11-04 baseline (fa7a9b7075).
3. SUMMARY.md documenting inputs, outputs, hashes, and acceptance comparison.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, NamedTuple
from PIL import Image

ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(ROOT))

from src.shared.python.motion_matching.show_a9_refit import (
    BASELINE_METRICS,
    CANDIDATE_METRICS,
    RefitMetrics,
    compare_against_baseline,
    format_comparison_table,
)
from src.tools.native_viewer_export.key_frames import (
    compose_side_by_side,
    draw_grip_clock,
    render_a10_summary,
)

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
PHASES = ("address", "top", "impact")
FACE_ON_NAMES = tuple(f"face_on_{p}_1280.png" for p in PHASES)
DTL_NAMES = tuple(f"down_the_line_{p}_1280.png" for p in PHASES)
SBS_SPECS = tuple(
    (f"side_by_side_{s}.png", s)
    for s in ("address_face_on", "top_face_on", "address_dtl", "top_dtl")
)
HEX_CHARS = frozenset("0123456789abcdef")


class EvidencePaths(NamedTuple):
    """Filesystem locations consumed and written by SHOW-A10 evidence generation."""

    out_dir: Path = OUT_DIR
    show_a8_dir: Path = SHOW_A8_DIR
    trunk_dir: Path = TRUNK_DIR
    a9_receipt_path: Path = SHOW_A9_DIR / "receipt.json"
    repo_root: Path = ROOT


def sha256_file(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def _rel_posix(path: Path, repo_root: Path) -> str:
    p_res, r_res = path.resolve(), repo_root.resolve()
    return (
        p_res.relative_to(r_res).as_posix()
        if p_res.is_relative_to(r_res)
        else p_res.as_posix()
    )


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


def _collect_still_artifacts(
    paths: EvidencePaths, grip: Mapping[str, Any]
) -> tuple[dict[str, str], dict[str, dict[str, Any]]]:
    """Collect source-still hashes and per-output artifact provenance records."""
    ov_meta = {
        "type": "grip_clock_inset",
        "lead_roll_deg": float(grip.get("lead_roll_deg", 15.0)),
        "trail_roll_deg": float(grip.get("trail_roll_deg", -100.73)),
        "target_tick_deg": float(grip.get("target_deg", 15.0)),
        "is_measured_from_frame": False,
    }
    src_stills: dict[str, str] = {}
    out_arts: dict[str, dict[str, Any]] = {}
    for name in sorted(FACE_ON_NAMES + DTL_NAMES):
        s_path = paths.show_a8_dir / name
        s_rel = _rel_posix(s_path, paths.repo_root)
        src_stills[s_rel] = sha256_file(s_path)
        is_fo = name in FACE_ON_NAMES
        out_arts[name] = {
            "sha256": sha256_file(paths.out_dir / name),
            "derivation": (
                "copied_from_show_a8_with_illustrative_grip_clock_overlay"
                if is_fo
                else "copied_from_show_a8_reencoded_png"
            ),
            "source_stills": [s_rel],
            "is_fresh_a9_render": False,
            "overlay_applied": dict(ov_meta) if is_fo else None,
        }
    for out_name, suffix in SBS_SPECS:
        rels = []
        for prefix in ("before_v1_rigid_", "after_v2_segmented_"):
            p = paths.trunk_dir / f"{prefix}{suffix}.png"
            rel = _rel_posix(p, paths.repo_root)
            src_stills[rel] = sha256_file(p)
            rels.append(rel)
        out_arts[out_name] = {
            "sha256": sha256_file(paths.out_dir / out_name),
            "derivation": "composed_side_by_side_from_trunk_segments_synthetic_stills",
            "source_stills": rels,
            "is_fresh_a9_render": False,
            "overlay_applied": None,
        }
    return dict(sorted(src_stills.items())), dict(sorted(out_arts.items()))


def build_provenance_manifest(
    paths: EvidencePaths = EvidencePaths(),
) -> dict[str, Any]:
    """Derive truthful provenance manifest from supplied inputs and artifacts."""
    a9 = json.loads(paths.a9_receipt_path.read_text(encoding="utf-8"))
    grip, gates, comp = (
        a9.get("address_grip", {}),
        a9.get("acceptance_gates", {}),
        a9.get("comparison_with_baseline", {}),
    )
    src_stills, out_arts = _collect_still_artifacts(paths, grip)
    cand_id = str(a9.get("model_id", "mujoco_driver_candidate_a9"))
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
            "key_frame_source_dir": _rel_posix(paths.show_a8_dir, paths.repo_root),
            "side_by_side_source_dir": _rel_posix(paths.trunk_dir, paths.repo_root),
        },
        "angle_attribution": {
            "clock_overlay_values_deg": {
                "lead": float(grip.get("lead_roll_deg", 15.0)),
                "trail": float(grip.get("trail_roll_deg", -100.73)),
                "target_tick": float(grip.get("target_deg", 15.0)),
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
            "source_receipt": _rel_posix(paths.a9_receipt_path, paths.repo_root),
            "receipt_model_id": cand_id,
            "verdict": verdict,
            "is_physically_accepted": bool(gates.get("is_physically_accepted", False)),
            "candidate_metrics": {
                "marker_rms_mm": float(comp["marker_rms_mm"]["candidate"]),
                "closure_residual_mm": float(comp["closure_residual_mm"]["candidate"]),
            },
            "comparison_with_baseline": comp,
        },
        "receipt_inputs": {
            **a9["input_hashes"],
            "show_a9_receipt_sha256": sha256_file(paths.a9_receipt_path),
        },
        "source_stills": src_stills,
        "output_artifacts": out_arts,
        "video_delivery_manifest": {
            "in_repo_video_artifacts": [],
            "a9_bundle_video_render_status": "unavailable_in_repo",
        },
    }
    validate_provenance_manifest(manifest)
    return manifest


def _is_sha256_hex(val: Any) -> bool:
    return isinstance(val, str) and len(val) == 64 and set(val) <= HEX_CHARS


def _validate_stills_and_outputs(manifest: Mapping[str, Any]) -> None:
    """Verify source_stills hashes and output_artifacts references."""
    stills = manifest.get("source_stills", {})
    if not stills or any(not _is_sha256_hex(d) for d in stills.values()):
        raise ValueError("Invalid sha256 digest in source_stills.")
    for fname, meta in manifest.get("output_artifacts", {}).items():
        refs = meta.get("source_stills")
        if (
            not _is_sha256_hex(meta.get("sha256"))
            or meta.get("is_fresh_a9_render") is True
            or not refs
        ):
            raise ValueError(f"Invalid artifact attribution or sha256 for {fname!r}.")
        if any(ref not in stills for ref in refs):
            raise ValueError(f"Unindexed source_stills reference in {fname!r}.")


def validate_provenance_manifest(manifest: Mapping[str, Any]) -> None:
    """Reject false A9 render claims, fake frame measurements, or promoted REJECTED verdicts."""
    if manifest.get("tracking_issues", {}).get("epic_status") != "OPEN":
        raise ValueError("Invalid epic_status: epic #14906 must remain OPEN.")
    frame_attr = manifest.get("visual_frame_attribution", {})
    if frame_attr.get("is_verified_a9_bundle_render") is True:
        raise ValueError("Invalid: verified_a9_bundle_render cannot be True.")
    _validate_stills_and_outputs(manifest)
    angles = manifest.get("angle_attribution", {})
    overlay = angles.get("clock_overlay_values_deg", {})
    if overlay.get("is_measured_from_rendered_frames") is True or (
        overlay.get("classification") != "illustrative_target_and_modelling_choice"
    ):
        raise ValueError("Invalid: overlay not measured_from_rendered_frames.")
    measured = angles.get("rendered_frame_hand_measurements", {})
    if not frame_attr.get("is_verified_a9_bundle_render", False) and (
        measured.get("status") != "unavailable"
        or measured.get("lead_roll_deg") is not None
        or measured.get("trail_roll_deg") is not None
    ):
        raise ValueError("Invalid rendered_frame_hand_measurements for unverified A9.")
    quant = manifest.get("quantitative_refit_metrics", {})
    if (
        "REJECTED" in str(quant.get("verdict", ""))
        and quant.get("is_physically_accepted") is True
    ):
        raise ValueError("Invalid: REJECTED candidate cannot be physically_accepted.")


def _metrics_from_comp(
    comp: Mapping[str, Any], role: str, fb: RefitMetrics, verdict: str
) -> RefitMetrics:
    """Extract RefitMetrics for baseline or candidate from receipt comparison block."""
    wf = comp.get("weight_fraction", {}).get(role, list(fb.weight_fraction_range))
    return RefitMetrics(
        float(comp.get("marker_rms_mm", {}).get(role, fb.marker_rms_mm)),
        float(comp.get("joint_rms_deg", {}).get(role, fb.joint_rms_deg)),
        float(comp.get("closure_residual_mm", {}).get(role, fb.closure_residual_mm)),
        (float(wf[0]), float(wf[1])),
        float(comp.get("peak_joint_torque_nm", {}).get(role, fb.peak_joint_torque_nm)),
        float(comp.get("in_support_fraction", {}).get(role, fb.in_support_fraction)),
        verdict,
    )


def write_summary(paths: EvidencePaths = EvidencePaths()) -> dict[str, Any]:
    """Write provenance_manifest.json and SUMMARY.md documenting inputs, outputs, and metrics."""
    manifest = build_provenance_manifest(paths)
    (paths.out_dir / "provenance_manifest.json").write_text(
        json.dumps(manifest, separators=(",", ": "), sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    quant, r_in = manifest["quantitative_refit_metrics"], manifest["receipt_inputs"]
    base_commit = str(r_in.get("baseline_commit", "unknown"))
    r_sha12 = str(r_in["show_a9_receipt_sha256"])[:12]
    fa = manifest["visual_frame_attribution"]
    ov = manifest["angle_attribution"]["clock_overlay_values_deg"]
    lead_deg, trail_deg = float(ov["lead"]), float(ov["trail"])
    outputs = {k: v["sha256"] for k, v in sorted(manifest["output_artifacts"].items())}
    summary = render_a10_summary(
        model_id="mujoco_driver",
        source_commit=f"{fa['key_frame_source_dir']} + {quant['source_receipt']} ({r_sha12}), baseline {base_commit}",
        receipt_verdict=f"{quant['verdict']} - recorded as candidate `{quant['receipt_model_id']}` per AC 3 (not physically accepted)",
        input_hashes=r_in,
        output_hashes=outputs,
        lead_roll_deg=lead_deg,
        trail_roll_deg=trail_deg,
    )
    summary = summary.replace(
        f"- Address grip roll: lead {lead_deg:+.1f}°, trail {trail_deg:+.1f}°",
        f"- Clock overlay angles (illustrative target, not measured from rendered frames): lead {lead_deg:+.1f}°, trail {trail_deg:+.2f}°",
    ).replace(
        "- Shading: axial tension/compression segment shading enabled",
        "- Shading & overlays: SHOW-A8 styled-mesh stills (`show_a8/`); segment shading and force/torque arcs are target SHOW-A10 styling (not a verified A9 bundle render)",
    )
    c_raw = quant.get("comparison_with_baseline", {})
    comp = compare_against_baseline(
        _metrics_from_comp(c_raw, "candidate", CANDIDATE_METRICS, quant["verdict"]),
        _metrics_from_comp(c_raw, "baseline", BASELINE_METRICS, quant["verdict"]),
    )
    still_rows = [
        f"| `{k}` | `{v}` |" for k, v in sorted(manifest["source_stills"].items())
    ]
    extra = [
        "## Source Stills Consumed (SHA-256)",
        "",
        "| Source Still | SHA-256 |",
        "|---|---|",
        *still_rows,
        "",
        "## Angle & Measurement Attribution",
        "",
        f"- **Clock overlay (`{lead_deg:+.1f}°` / `{trail_deg:+.2f}°`):** Illustrative target from `{quant['source_receipt']}`; **not measured from rendered frames**.",
        "- **Rendered-frame hand roll & video delivery:** `unavailable` / `unavailable_in_repo` (epic `#14906` remains open).",
        "",
        f"## Acceptance Comparison vs 11-04 Baseline (`{base_commit}`)",
        "",
        format_comparison_table(comp).replace("fa7a9b7075", base_commit),
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
        "python docs/development/full_body_models/evidence/show_a10/render_show_a10_evidence.py # --regenerate-images opt-in",
        "```",
        "",
    ]
    full = summary.rstrip() + "\n\n" + "\n".join(extra)
    (paths.out_dir / "SUMMARY.md").write_text(full, encoding="utf-8", newline="\n")
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
