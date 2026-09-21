"""Candidate animation video and fit-quality report export (MS-86, #10359).

Provides:
- export_video: Renders 3D marker overlay trajectories into GIF or MP4 animations
  with strict fail-closed validation on candidate and marker trajectory integrity.
- export_report: Generates comprehensive Markdown and PDF fit-quality reports
  embedding standardized metrics, gate evaluations, physical constraints,
  and cryptographic provenance conforming to #8820 / Industrial Readiness U3.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import logging
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.candidate import MatchedSwingCandidate
from src.shared.python.motion_matching.candidate_io import load_candidate
from src.shared.python.motion_matching.provenance import (
    engine_package_version,
    git_commit_short,
)

logger = get_logger(__name__)

__all__ = [
    "export_report",
    "export_video",
]

_SUPPORTED_VIDEO_EXTS = {".gif", ".mp4"}
_SUPPORTED_REPORT_EXTS = {".md", ".pdf"}


def _validate_candidate_for_video(
    candidate: MatchedSwingCandidate | Path | str,
) -> MatchedSwingCandidate:
    """Load and validate candidate has required marker trajectories."""
    if isinstance(candidate, (str, Path)):
        cand_path = Path(candidate)
        if not cand_path.is_file():
            raise FileNotFoundError(f"Candidate file not found: {cand_path}")
        cand = load_candidate(cand_path)
    elif isinstance(candidate, MatchedSwingCandidate):
        cand = candidate
    else:
        raise TypeError(
            f"Expected MatchedSwingCandidate or path, got {type(candidate).__name__}"
        )

    if cand.model_markers_m is None or cand.target_markers_m is None:
        raise ValueError(
            "Candidate lacks marker trajectory arrays required for video export"
        )
    if len(cand.time_s) < 2:
        raise ValueError(
            f"Candidate time_s has insufficient frames: {len(cand.time_s)}"
        )
    return cand


def _render_video_frames(
    candidate: MatchedSwingCandidate,
    engine: str,
    stride: int,
) -> list[np.ndarray]:
    """Render 3D marker overlay frames comparing target vs model markers."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    target_markers_m = candidate.target_markers_m
    model_markers_m = candidate.model_markers_m
    valid_mask = candidate.marker_validity
    time_s = candidate.time_s

    assert target_markers_m is not None
    assert model_markers_m is not None

    n_frames = len(time_s)
    frames: list[np.ndarray] = []
    color_map = {
        "mujoco": "#d62728",
        "pinocchio": "#1f77b4",
        "drake": "#2ca02c",
        "opensim": "#9467bd",
        "myosuite": "#8c564b",
        "matlab": "#17becf",
    }
    m_color = color_map.get(engine.lower(), "#ff7f0e")

    fig = plt.figure(figsize=(6, 6), dpi=80)
    ax: Any = fig.add_subplot(111, projection="3d")
    t0 = target_markers_m[0]
    center = np.nanmean(t0, axis=0) if np.isnan(t0).any() else np.mean(t0, axis=0)
    box_half = 1.0

    for k in range(0, n_frames, stride):
        ax.clear()
        t_k = target_markers_m[k]
        m_k = model_markers_m[k]
        mask_k = (
            valid_mask[k] if valid_mask is not None else np.ones(len(t_k), dtype=bool)
        )

        t_valid = t_k[mask_k]
        ax.scatter(
            t_valid[:, 0],
            t_valid[:, 2],
            zs=t_valid[:, 1],
            c="black",
            s=20,
            alpha=0.7,
            label="Capture (C3D)",
        )
        m_valid = m_k[mask_k]
        ax.scatter(
            m_valid[:, 0],
            m_valid[:, 2],
            zs=m_valid[:, 1],
            c=m_color,
            s=25,
            alpha=0.9,
            label=f"Model ({engine})",
        )

        diff = m_valid - t_valid
        err = (
            float(np.sqrt(np.mean(np.einsum("...i,...i->...", diff, diff))))
            if len(t_valid) > 0
            else 0.0
        )

        ax.set_xlim(center[0] - box_half, center[0] + box_half)
        ax.set_ylim(center[2] - box_half, center[2] + box_half)
        ax.set_zlim(center[1] - box_half, center[1] + box_half)
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Z (m)")
        ax.set_zlabel("Y (m)")
        ax.set_title(
            f"{engine.upper()} | t={time_s[k]:.3f}s | RMSE={err * 1000.0:.1f} mm"
        )
        ax.legend(loc="upper right", fontsize=8)

        canvas: Any = fig.canvas
        canvas.draw()
        rgba = np.asarray(canvas.buffer_rgba())
        frames.append(rgba[:, :, :3].copy())

    plt.close(fig)
    return frames


def _write_gif(
    frames: list[np.ndarray],
    path: Path,
    fps: int,
    duration_ms: float | None = None,
) -> Path:
    """Save rendered RGB frames as GIF animation."""
    import imageio.v2 as imageio

    path.parent.mkdir(parents=True, exist_ok=True)
    d_ms = duration_ms if duration_ms is not None else (1000.0 / max(1, fps))
    imageio.mimsave(str(path), frames, duration=d_ms, loop=0)
    logger.info("Saved GIF animation to %s (%d frames)", path, len(frames))
    return path


def _write_mp4(frames: list[np.ndarray], path: Path, fps: int) -> Path:
    """Save rendered RGB frames as MP4 video."""
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        raise ValueError("Cannot write empty frames list to MP4")

    h, w = frames[0].shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(path), fourcc, float(fps), (w, h))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open OpenCV VideoWriter for {path}")

    for frame in frames:
        bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        writer.write(bgr)

    writer.release()
    logger.info("Saved MP4 video to %s (%d frames)", path, len(frames))
    return path


@precondition(
    lambda candidate, engine, path, **kwargs: (
        isinstance(engine, str)
        and len(engine.strip()) > 0
        and Path(path).suffix.lower() in _SUPPORTED_VIDEO_EXTS
    ),
    "engine must be non-empty and path extension must be .gif or .mp4",
)
@postcondition(
    lambda r: Path(r).is_file() and Path(r).stat().st_size > 0,
    "exported video file must exist and be non-empty",
)
def export_video(
    candidate: MatchedSwingCandidate | Path | str,
    engine: str,
    path: Path | str,
    *,
    fps: int = 30,
    stride: int = 5,
    view: str = "marker_overlay",
) -> Path:
    """Export 3D marker overlay trajectory animation as GIF or MP4.

    Args:
        candidate: MatchedSwingCandidate instance or path to .npz package.
        engine: Physics engine name (e.g., "mujoco", "drake", "pinocchio").
        path: Destination file path (.gif or .mp4).
        fps: Frames per second for output video.
        stride: Frame subsampling stride for rendering.
        view: Visual rendering preset ("marker_overlay").

    Returns:
        Resolved output Path.
    """
    out_p = Path(path).resolve()
    cand = _validate_candidate_for_video(candidate)
    stride_safe = max(1, stride)
    frames = _render_video_frames(cand, engine, stride_safe)

    ext = out_p.suffix.lower()
    if ext == ".gif":
        dt_s = cand.time_s[1] - cand.time_s[0] if len(cand.time_s) > 1 else 0.05
        duration_ms = float(1000.0 * stride_safe * dt_s)
        return _write_gif(frames, out_p, fps, duration_ms=duration_ms)
    if ext == ".mp4":
        return _write_mp4(frames, out_p, fps)
    raise ValueError(f"Unsupported video format '{ext}', expected .gif or .mp4")


def _load_receipt_data(
    receipt: dict[str, Any] | Path | str,
) -> tuple[dict[str, Any], str, str]:
    """Parse receipt dictionary and compute deterministic receipt SHA-256."""
    if isinstance(receipt, (str, Path)):
        rec_path = Path(receipt).resolve()
        if not rec_path.is_file():
            raise FileNotFoundError(f"Receipt file not found: {rec_path}")
        raw_bytes = rec_path.read_bytes()
        sha = hashlib.sha256(raw_bytes).hexdigest()
        data = json.loads(raw_bytes.decode("utf-8"))
        return data, sha, str(rec_path)
    if isinstance(receipt, dict):
        canonical_bytes = json.dumps(receipt, sort_keys=True).encode("utf-8")
        sha = hashlib.sha256(canonical_bytes).hexdigest()
        return receipt, sha, ""
    raise TypeError(f"Expected receipt dict or path, got {type(receipt).__name__}")


def _extract_report_provenance(
    receipt_data: dict[str, Any],
    receipt_sha: str,
    receipt_rel_path: str = "",
) -> dict[str, Any]:
    """Assemble complete cryptographic and runtime provenance dictionary."""
    from src.shared.python.motion_matching.ledger import classify_receipt

    cls_engine, cls_lane, cls_capture, cls_reason = (
        classify_receipt(receipt_rel_path, receipt_data)
        if receipt_rel_path
        else ("unknown", "unclassified", None, None)
    )

    engine = (
        receipt_data.get("backend")
        or receipt_data.get("engine")
        or (cls_engine if cls_engine != "unknown" else None)
    )
    if not engine or engine == "unknown":
        qual = str(receipt_data.get("qualification", "")).lower()
        for eng in ("mujoco", "drake", "pinocchio", "opensim", "myosuite", "simscape"):
            if eng in qual or eng in receipt_rel_path.lower():
                engine = eng
                break
    engine = engine or "unknown"

    lane = (
        receipt_data.get("lane")
        or (cls_lane if cls_lane != "unclassified" else None)
        or "unknown"
    )
    capture = (
        receipt_data.get("capture")
        or receipt_data.get("capture_name")
        or cls_capture
        or "unknown"
    )

    eng_ver = receipt_data.get("engine_version") or engine_package_version(None, engine)
    cand_sha = (
        receipt_data.get("candidate_sha256")
        or receipt_data.get("candidate_sha")
        or "unknown"
    )

    acceptance = receipt_data.get("acceptance", {})
    verdict = (
        acceptance.get("verdict")
        or receipt_data.get("verdict")
        or (
            "REJECTED"
            if "qualification" in receipt_data
            and "not a fit" in receipt_data["qualification"]
            else "UNVERIFIED"
        )
    ).upper()

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    return {
        "engine": engine,
        "engine_version": eng_ver,
        "git_commit": git_commit_short(),
        "candidate_sha256": cand_sha,
        "receipt_sha256": receipt_sha,
        "capture": capture,
        "lane": lane,
        "verdict": verdict,
        "timestamp_utc": now_utc,
        "qualification": receipt_data.get("qualification", ""),
        "reason": acceptance.get("reason")
        or receipt_data.get("rejection_reason")
        or cls_reason
        or "",
    }


def _build_metrics_block(receipt_data: dict[str, Any]) -> str:
    """Format standardized metrics markdown table."""
    acceptance = receipt_data.get("acceptance", {})
    metrics = acceptance.get("metrics") or receipt_data.get("metrics", {})
    ik = receipt_data.get("ik", {})

    whole_rmse = metrics.get("whole_marker_rmse_m", ik.get("marker_rms_m"))
    early_rmse = metrics.get("early_marker_rmse_m")
    term_rmse = metrics.get("terminal_marker_rmse_m")
    club_rmse = metrics.get(
        "club_marker_rmse_m", ik.get("segment_rms_m", {}).get("club")
    )
    yaw_rmse = metrics.get("pelvis_yaw_rmse_rad")

    def _fmt_m(val: float | None) -> str:
        return f"{val * 1000.0:.2f} mm ({val:.4f} m)" if val is not None else "—"

    def _fmt_rad(val: float | None) -> str:
        return f"{math.degrees(val):.2f}° ({val:.4f} rad)" if val is not None else "—"

    return (
        "| Standardized Metric | Measured Value |\n"
        "|---|---|\n"
        f"| Whole Marker RMSE | {_fmt_m(whole_rmse)} |\n"
        f"| Early Marker RMSE | {_fmt_m(early_rmse)} |\n"
        f"| Terminal Marker RMSE | {_fmt_m(term_rmse)} |\n"
        f"| Club Marker RMSE | {_fmt_m(club_rmse)} |\n"
        f"| Pelvis Yaw RMSE | {_fmt_rad(yaw_rmse)} |\n"
    )


def _build_gates_block(receipt_data: dict[str, Any]) -> str:
    """Format acceptance gates markdown table."""
    acceptance = receipt_data.get("acceptance", {})
    gates = acceptance.get("gates", [])
    if not gates:
        return "*No formal quantitative acceptance gates evaluated for this receipt.*\n"

    lines = [
        "| Gate Name | Status | Measured | Threshold | Unit |",
        "|---|---|---|---|---|",
    ]
    for g in gates:
        name = g.get("name", "gate")
        status = g.get("status", "UNKNOWN").upper()
        ms = g.get("measured")
        ms_str = f"{ms:.4f}" if ms is not None else "—"
        th = g.get("threshold", 0.0)
        unit = g.get("unit", "")
        lines.append(f"| `{name}` | **{status}** | {ms_str} | {th:.4f} | {unit} |")
    return "\n".join(lines) + "\n"


def _build_provenance_block(prov: dict[str, Any]) -> str:
    """Format cryptographic provenance markdown block (#8820)."""
    return (
        "| Provenance Field | Value |\n"
        "|---|---|\n"
        f"| Physics Engine | `{prov['engine']}` (version: `{prov['engine_version']}`) |\n"
        f"| Git Commit | `{prov['git_commit']}` |\n"
        f"| Candidate SHA-256 | `{prov['candidate_sha256']}` |\n"
        f"| Receipt SHA-256 | `{prov['receipt_sha256']}` |\n"
        f"| Capture / Lane | `{prov['capture']}` / `{prov['lane']}` |\n"
        f"| Overall Verdict | **{prov['verdict']}** |\n"
        f"| Generated (UTC) | `{prov['timestamp_utc']}` |\n"
    )


def _format_markdown_report(
    receipt_data: dict[str, Any],
    prov: dict[str, Any],
) -> str:
    """Generate complete Markdown fit-quality report document."""
    title = f"# Fit-Quality & Acceptance Report: {prov['engine'].upper()} ({prov['capture'].title()})"
    verdict_badge = f"> **Overall Run Verdict: {prov['verdict']}**\n"

    qual_section = ""
    if prov["qualification"] or prov["reason"]:
        qual_section = (
            "## Qualification & Audit Notes\n\n"
            f"> [!IMPORTANT]\n"
            f"> **Qualification:** {prov['qualification'] or 'None'}\n"
            f"> \n"
            f"> **Rejection / Gate Reason:** {prov['reason'] or 'None'}\n\n"
        )

    metrics_section = (
        f"## Shared Standardized Metrics\n\n{_build_metrics_block(receipt_data)}\n"
    )
    gates_section = (
        f"## Quantitative Acceptance Gates\n\n{_build_gates_block(receipt_data)}\n"
    )
    prov_section = (
        f"## Cryptographic Provenance (#8820 / U3)\n\n{_build_provenance_block(prov)}\n"
    )

    return "\n".join(
        [
            title,
            "",
            verdict_badge,
            qual_section,
            metrics_section,
            gates_section,
            prov_section,
        ]
    )


def _render_pdf_report(md_content: str, out_path: Path, title: str) -> Path:
    """Render Markdown report into a clean, paginated PDF via matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.backends.backend_pdf import PdfPages

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with PdfPages(str(out_path)) as pdf:
        fig = plt.figure(figsize=(8.5, 11), dpi=100)
        plt.axis("off")

        # Strip markdown syntax for clean text rendering on canvas
        plain_lines: list[str] = []
        for line in md_content.splitlines():
            clean = line.replace("#", "").replace("**", "").replace("`", "")
            clean = clean.removeprefix("> ")
            plain_lines.append(clean)

        y_pos = 0.95
        line_height = 0.022
        for line in plain_lines:
            if not line.strip():
                y_pos -= line_height * 0.5
                continue
            if y_pos < 0.08:
                pdf.savefig(fig)
                plt.close(fig)
                fig = plt.figure(figsize=(8.5, 11), dpi=100)
                plt.axis("off")
                y_pos = 0.95

            is_header = any(
                line.strip().startswith(h)
                for h in (
                    "Fit-Quality",
                    "Shared Standardized",
                    "Quantitative",
                    "Cryptographic",
                    "Qualification",
                )
            )
            weight = "bold" if is_header else "normal"
            size = 11 if is_header else 8.5
            color = "#111111" if not is_header else "#0d47a1"

            fig.text(
                0.08,
                y_pos,
                line[:105],
                fontsize=size,
                fontweight=weight,
                color=color,
                family="monospace",
            )
            y_pos -= line_height

        pdf.savefig(fig)
        plt.close(fig)

    logger.info("Saved PDF report to %s", out_path)
    return out_path


@precondition(
    lambda receipt, path, **kwargs: Path(path).suffix.lower() in _SUPPORTED_REPORT_EXTS,
    "output path extension must be .md or .pdf",
)
@postcondition(
    lambda r: Path(r).is_file() and Path(r).stat().st_size > 0,
    "exported report file must exist and be non-empty",
)
def export_report(
    receipt: dict[str, Any] | Path | str,
    path: Path | str,
    *,
    candidate: MatchedSwingCandidate | Path | str | None = None,
    include_provenance: bool = True,
) -> Path:
    """Generate comprehensive fit-quality and acceptance report in Markdown or PDF.

    Args:
        receipt: Execution receipt dictionary or path to receipt JSON.
        path: Destination file path (.md or .pdf).
        candidate: Optional MatchedSwingCandidate instance or package path.
        include_provenance: Whether to stamp full cryptographic provenance block.

    Returns:
        Resolved output Path.
    """
    out_p = Path(path).resolve()
    data, sha, rec_path_str = _load_receipt_data(receipt)
    prov = _extract_report_provenance(data, sha, receipt_rel_path=rec_path_str)

    # Allow candidate argument to override candidate_sha if provided
    if candidate is not None:
        if isinstance(candidate, (str, Path)):
            c_p = Path(candidate)
            if c_p.is_file():
                prov["candidate_sha256"] = hashlib.sha256(c_p.read_bytes()).hexdigest()
        elif isinstance(candidate, MatchedSwingCandidate):
            cand_meta = candidate.metadata
            if cand_meta.checksums and "manifest" in cand_meta.checksums:
                prov["candidate_sha256"] = cand_meta.checksums["manifest"]

    md_content = _format_markdown_report(data, prov)
    ext = out_p.suffix.lower()
    if ext == ".md":
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(md_content, encoding="utf-8")
        logger.info("Saved Markdown report to %s", out_p)
        return out_p
    if ext == ".pdf":
        return _render_pdf_report(
            md_content, out_p, title=f"{prov['engine']} Fit Report"
        )
    raise ValueError(f"Unsupported report format '{ext}', expected .md or .pdf")
