"""Validate and govern companion screenshot assets and registries (#9191).

This module is the single authority for screenshot verification, metadata
governance, and representative asset generation for the UpstreamDrift
companion catalog. Every captured visual artifact must carry exact SHA-256
digests, pixel dimensions parsed directly from the PNG IHDR header, viewport,
theme, capture provenance, alt text, caption, and visible limitations.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import struct
import sys
import zlib
from collections.abc import Mapping, Sequence, Set
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

REGISTRY_PATH = Path("scripts/config/companion_screenshots.v1.json")
REGISTRY_ID = "upstreamdrift-companion-screenshots"
REGISTRY_VERSION = "1.0.0"
SCREENSHOTS_DIR = Path("docs/screenshots")
SCHEMA_PATH = Path(
    "docs/api/contracts/upstreamdrift-companion-screenshots-v1.schema.json"
)
SCREENSHOT_PENDING_REASON = "No governed capture exists at this commit (#9191)."

_PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
_IDENTIFIER = re.compile(r"^[a-z0-9][a-z0-9._-]*$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_PENDING_NULL_FIELDS = (
    "path",
    "sha256",
    "width",
    "height",
    "viewport",
    "theme",
    "capture_workflow_id",
    "capture_step_id",
    "capture_environment",
    "alt_text",
    "caption",
)
_ARTIFACT_CLASSES = frozenset({"illustrative", "raw_plot", "qualification_evidence"})
_THEMES = frozenset({"light", "dark"})


class ScreenshotContractError(ValueError):
    """Raised when screenshot metadata or image assets violate contract rules."""


def _non_empty_str(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def png_size(data: bytes) -> tuple[int, int] | None:
    """Extract width and height from PNG IHDR chunk using standard library."""
    if not data.startswith(_PNG_SIGNATURE) or len(data) < 24 or data[12:16] != b"IHDR":
        return None
    width, height = struct.unpack(">II", data[16:24])
    return width, height


def sha256_bytes(data: bytes) -> str:
    """Compute 64-character lowercase hexadecimal SHA-256 digest."""
    return hashlib.sha256(data).hexdigest()


def _resolve_asset(path_val: Any, repo_root: Path) -> tuple[Path | None, str]:
    if not _non_empty_str(path_val):
        return None, "captured screenshot path must be a non-empty string"
    if (
        str(path_val).startswith(("/", "\\"))
        or re.match(r"^[A-Za-z]:", str(path_val))
        or Path(path_val).is_absolute()
    ):
        return None, f"captured screenshot path must not be absolute: {path_val}"
    resolved_repo = repo_root.resolve()
    target = (repo_root / str(path_val)).resolve()
    if not target.is_relative_to(resolved_repo) or target == resolved_repo:
        return None, f"captured screenshot path resolves outside repo_root: {path_val}"
    if not target.is_file():
        return None, f"screenshot file does not exist: {path_val}"
    return target, ""


def verify_screenshot_record(
    record: Mapping[str, Any],
    *,
    repo_root: Path,
    workflow_ids: Set[str] | None = None,
) -> list[str]:
    """Verify a single screenshot record against fail-closed contract rules."""
    violations: list[str] = []
    record_id = str(record.get("id", "<unknown>"))
    status = record.get("status")

    if status == "pending":
        for field in _PENDING_NULL_FIELDS:
            if record.get(field) is not None:
                violations.append(
                    f"record {record_id!r}: pending screenshot {field} must be null, "
                    f"got {record.get(field)!r}"
                )
        if not _non_empty_str(record.get("reason")):
            violations.append(
                f"record {record_id!r}: pending screenshot reason must be a non-empty string"
            )
        return violations

    if status != "captured":
        violations.append(
            f"record {record_id!r}: unknown status {status!r} (must be 'pending' or 'captured')"
        )
        return violations

    # Captured status checks
    if record.get("reason") is not None:
        violations.append(
            f"record {record_id!r}: captured screenshot reason must be null, got {record.get('reason')!r}"
        )
    if not _non_empty_str(record.get("alt_text")):
        violations.append(
            f"record {record_id!r}: captured screenshot alt_text must be a non-empty string"
        )
    if not _non_empty_str(record.get("caption")):
        violations.append(
            f"record {record_id!r}: captured screenshot caption must be a non-empty string"
        )
    if record.get("theme") not in _THEMES:
        violations.append(
            f"record {record_id!r}: captured screenshot theme must be 'light' or 'dark', got {record.get('theme')!r}"
        )
    if record.get("artifact_class") not in _ARTIFACT_CLASSES:
        violations.append(
            f"record {record_id!r}: captured screenshot artifact_class must be one of {sorted(_ARTIFACT_CLASSES)}, "
            f"got {record.get('artifact_class')!r}"
        )
    if not _non_empty_str(record.get("capture_environment")):
        violations.append(
            f"record {record_id!r}: captured screenshot capture_environment must be a non-empty string"
        )
    if not _non_empty_str(record.get("capture_workflow_id")):
        violations.append(
            f"record {record_id!r}: captured screenshot capture_workflow_id must be a non-empty string"
        )
    elif workflow_ids is not None and record["capture_workflow_id"] not in workflow_ids:
        violations.append(
            f"record {record_id!r}: capture_workflow_id {record['capture_workflow_id']!r} does not resolve to an existing workflow"
        )
    if not _non_empty_str(record.get("capture_step_id")):
        violations.append(
            f"record {record_id!r}: captured screenshot capture_step_id must be a non-empty string"
        )

    viewport = record.get("viewport")
    if not isinstance(viewport, Mapping):
        violations.append(
            f"record {record_id!r}: captured screenshot viewport must be a mapping"
        )
    else:
        vw = viewport.get("width")
        vh = viewport.get("height")
        if not isinstance(vw, int) or vw <= 0 or not isinstance(vh, int) or vh <= 0:
            violations.append(
                f"record {record_id!r}: viewport width and height must be positive integers"
            )

    target, reason = _resolve_asset(record.get("path"), repo_root)
    if target is None:
        violations.append(f"record {record_id!r}: {reason}")
        return violations

    data = target.read_bytes()
    computed_sha = sha256_bytes(data)
    if computed_sha != record.get("sha256"):
        violations.append(
            f"record {record_id!r}: sha256 mismatch (declared {record.get('sha256')!r}, computed {computed_sha!r})"
        )

    dims = png_size(data)
    if dims is None:
        violations.append(
            f"record {record_id!r}: unsupported image format: expected a PNG with IHDR header"
        )
    else:
        w, h = dims
        if record.get("width") != w:
            violations.append(
                f"record {record_id!r}: width mismatch (declared {record.get('width')!r}, image is {w})"
            )
        if record.get("height") != h:
            violations.append(
                f"record {record_id!r}: height mismatch (declared {record.get('height')!r}, image is {h})"
            )

    return violations


def verify_screenshot_records(
    payload: Mapping[str, Any],
    repo_root: Path,
    workflow_ids: Set[str] | None = None,
) -> list[str]:
    """Verify all screenshot records in a manifest or payload."""
    if not isinstance(payload, Mapping):
        raise TypeError(f"payload must be a Mapping, got {type(payload).__name__}")
    records = payload.get("records")
    if not isinstance(records, list):
        raise TypeError(f"records must be a list, got {type(records).__name__}")

    violations: list[str] = []
    for idx, record in enumerate(records):
        if not isinstance(record, Mapping):
            violations.append(
                f"record [{idx}]: screenshot record must be a mapping, got {type(record).__name__}"
            )
            continue
        violations.extend(
            verify_screenshot_record(
                record, repo_root=repo_root, workflow_ids=workflow_ids
            )
        )
    return violations


def parse_registry(
    payload: bytes | str,
    *,
    repo_root: Path,
    source_commit: str,
    program_ids: Set[str],
    workflow_ids: Set[str] | None = None,
) -> dict[str, Any]:
    """Parse, validate, and index the governed screenshot registry."""
    raw_text = payload.decode("utf-8") if isinstance(payload, bytes) else payload

    try:
        data = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ScreenshotContractError(
            f"corrupt screenshot registry JSON: {exc}"
        ) from exc

    if not isinstance(data, Mapping):
        raise ScreenshotContractError("screenshot registry must be a JSON object")

    if data.get("registry_id") != REGISTRY_ID:
        raise ScreenshotContractError(
            f"registry_id must be {REGISTRY_ID!r}, got {data.get('registry_id')!r}"
        )
    if data.get("version") != REGISTRY_VERSION:
        raise ScreenshotContractError(
            f"registry version must be {REGISTRY_VERSION!r}, got {data.get('version')!r}"
        )

    raw_records = data.get("records")
    if not isinstance(raw_records, list):
        raise ScreenshotContractError("screenshot registry 'records' must be a list")

    seen_ids: set[str] = set()
    records: list[dict[str, Any]] = []
    catalog_records: list[dict[str, Any]] = []

    for idx, raw in enumerate(raw_records):
        if not isinstance(raw, Mapping):
            raise ScreenshotContractError(f"record [{idx}] must be an object")

        rec_id = str(raw.get("id", "")).strip()
        if not rec_id or not _IDENTIFIER.fullmatch(rec_id):
            raise ScreenshotContractError(
                f"record [{idx}] id must be a valid identifier, got {rec_id!r}"
            )
        if rec_id in seen_ids:
            raise ScreenshotContractError(f"duplicate screenshot id: {rec_id!r}")
        seen_ids.add(rec_id)

        prog_id = str(raw.get("program_id", "")).strip()
        if prog_id not in program_ids:
            raise ScreenshotContractError(
                f"record {rec_id!r} references unknown or hidden program_id: {prog_id!r}"
            )

        status = raw.get("status")
        if status not in ("pending", "captured"):
            raise ScreenshotContractError(
                f"record {rec_id!r} status must be 'pending' or 'captured', got {status!r}"
            )

        record_dict: dict[str, Any] = {
            "id": rec_id,
            "program_id": prog_id,
            "status": status,
            "path": raw.get("path"),
            "sha256": raw.get("sha256"),
            "width": raw.get("width"),
            "height": raw.get("height"),
            "viewport": dict(raw["viewport"])
            if isinstance(raw.get("viewport"), Mapping)
            else None,
            "theme": raw.get("theme"),
            "capture_workflow_id": raw.get("capture_workflow_id"),
            "capture_step_id": raw.get("capture_step_id"),
            "capture_environment": raw.get("capture_environment"),
            "alt_text": raw.get("alt_text"),
            "caption": raw.get("caption"),
            "visible_limitations": list(raw.get("visible_limitations", [])),
            "artifact_class": raw.get("artifact_class", "illustrative"),
            "reason": raw.get("reason"),
        }

        # Check violations
        violations = verify_screenshot_record(
            record_dict, repo_root=repo_root, workflow_ids=workflow_ids
        )
        if violations:
            raise ScreenshotContractError(
                f"screenshot record {rec_id!r} failed validation: {'; '.join(violations)}"
            )

        records.append(record_dict)

        # Catalog record includes source_commit
        catalog_rec = dict(record_dict)
        catalog_rec["source_commit"] = source_commit
        catalog_records.append(catalog_rec)

    # Sort deterministically by id
    records.sort(key=lambda r: r["id"])
    catalog_records.sort(key=lambda r: r["id"])

    summary = {
        "screenshot_records": len(records),
        "captured_screenshot_records": sum(r["status"] == "captured" for r in records),
        "pending_screenshot_records": sum(r["status"] == "pending" for r in records),
    }

    return {
        "records": records,
        "catalog_records": catalog_records,
        "summary": summary,
    }


def load_and_parse_registry(
    repo_root: Path,
    source_commit: str,
    program_ids: Set[str],
    workflow_ids: Set[str] | None = None,
) -> dict[str, Any]:
    """Load registry file from repo_root and parse it."""
    registry_file = repo_root / REGISTRY_PATH
    if not registry_file.is_file():
        raise ScreenshotContractError(f"screenshot registry missing: {REGISTRY_PATH}")
    payload = registry_file.read_bytes()
    return parse_registry(
        payload,
        repo_root=repo_root,
        source_commit=source_commit,
        program_ids=program_ids,
        workflow_ids=workflow_ids,
    )


def build_screenshots_payload(
    records: Sequence[Mapping[str, Any]],
    *,
    source: Mapping[str, Any],
) -> dict[str, Any]:
    """Build standalone screenshots.json payload."""
    return {
        "$schema": "https://upstreamdrift.dev/schemas/upstreamdrift-companion-screenshots-v1.schema.json",
        "schema_version": "1.0.0",
        "manifest_id": "upstreamdrift-companion-screenshots",
        "source": {
            "repository": source["repository"],
            "commit": source["commit"],
        },
        "records": [dict(r) for r in sorted(records, key=lambda r: r["id"])],
    }


def generate_representative_assets(target_dir: Path) -> dict[str, Path]:
    """Generate representative deterministic visual artifacts for governed screenshots.

    Uses Matplotlib with headless Agg backend (zero X11/display dependencies).
    Generates exact-dimension, accessible PNG assets with clear visual indicators.
    """
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    target_dir.mkdir(parents=True, exist_ok=True)
    generated: dict[str, Path] = {}

    # 1. Pendulum Simulator - Light Desktop (1280x720)
    p_light_path = target_dir / "pendulum_simulator_light_1280x720.png"
    fig = Figure(
        figsize=(12.8, 7.2), dpi=100, layout="constrained", facecolor="#f8f9fa"
    )
    axs = fig.subplots(1, 2)
    t = [i * 0.01 for i in range(200)]
    theta1 = [math.sin(ti * 4.0) * math.exp(-0.2 * ti) for ti in t]
    phi = [math.cos(ti * 6.0) * math.exp(-0.15 * ti) for ti in t]

    axs[0].plot(t, theta1, label="theta1 (rad)", color="#0d6efd", linewidth=2.0)
    axs[0].plot(
        t, phi, label="phi (rad)", color="#dc3545", linewidth=2.0, linestyle="--"
    )
    axs[0].set_title(
        "Double Pendulum Joint Angles (Light Theme)", color="#212529", fontsize=14
    )
    axs[0].set_xlabel("Time (s)", color="#212529")
    axs[0].set_ylabel("Angle (rad)", color="#212529")
    axs[0].grid(True, alpha=0.3, color="#adb5bd")
    axs[0].legend(loc="upper right")

    axs[1].plot(theta1, phi, color="#198754", linewidth=2.0)
    axs[1].set_title("State-Space Phase Trajectory", color="#212529", fontsize=14)
    axs[1].set_xlabel("theta1 (rad)", color="#212529")
    axs[1].set_ylabel("phi (rad)", color="#212529")
    axs[1].grid(True, alpha=0.3, color="#adb5bd")

    fig.savefig(p_light_path, format="png", facecolor=fig.get_facecolor())
    generated["pendulum_simulator_light_1280x720"] = p_light_path

    # 2. Pendulum Simulator - Dark Desktop (1280x720)
    p_dark_path = target_dir / "pendulum_simulator_dark_1280x720.png"
    fig_d = Figure(
        figsize=(12.8, 7.2), dpi=100, layout="constrained", facecolor="#1a1a2e"
    )
    axs_d = fig_d.subplots(1, 2)
    for ax in axs_d:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#e0e0e0")
        for spine in ax.spines.values():
            spine.set_color("#4ecca3")

    axs_d[0].plot(t, theta1, label="theta1 (rad)", color="#4ecca3", linewidth=2.0)
    axs_d[0].plot(
        t, phi, label="phi (rad)", color="#e94560", linewidth=2.0, linestyle="--"
    )
    axs_d[0].set_title(
        "Double Pendulum Joint Angles (Dark Theme)", color="#e0e0e0", fontsize=14
    )
    axs_d[0].set_xlabel("Time (s)", color="#e0e0e0")
    axs_d[0].set_ylabel("Angle (rad)", color="#e0e0e0")
    axs_d[0].grid(True, alpha=0.25, color="#2a2a40")
    axs_d[0].legend(
        loc="upper right",
        facecolor="#16213e",
        edgecolor="#4ecca3",
        labelcolor="#e0e0e0",
    )

    axs_d[1].plot(theta1, phi, color="#00fff5", linewidth=2.0)
    axs_d[1].set_title("State-Space Phase Trajectory", color="#e0e0e0", fontsize=14)
    axs_d[1].set_xlabel("theta1 (rad)", color="#e0e0e0")
    axs_d[1].set_ylabel("phi (rad)", color="#e0e0e0")
    axs_d[1].grid(True, alpha=0.25, color="#2a2a40")

    fig_d.savefig(p_dark_path, format="png", facecolor=fig_d.get_facecolor())
    generated["pendulum_simulator_dark_1280x720"] = p_dark_path

    # 3. Pendulum Simulator - Dark Mobile Responsive (375x667)
    p_mob_path = target_dir / "pendulum_simulator_dark_375x667.png"
    fig_m = Figure(
        figsize=(3.75, 6.67), dpi=100, layout="constrained", facecolor="#1a1a2e"
    )
    axs_m = fig_m.subplots(2, 1)
    for ax in axs_m:
        ax.set_facecolor("#16213e")
        ax.tick_params(colors="#e0e0e0", labelsize=8)
        for spine in ax.spines.values():
            spine.set_color("#4ecca3")

    axs_m[0].plot(t[:100], theta1[:100], color="#4ecca3", linewidth=1.5)
    axs_m[0].set_title("Pendulum Angle (Mobile)", color="#e0e0e0", fontsize=10)
    axs_m[0].set_xlabel("Time (s)", color="#e0e0e0", fontsize=8)
    axs_m[0].set_ylabel("Angle (rad)", color="#e0e0e0", fontsize=8)
    axs_m[0].grid(True, alpha=0.25, color="#2a2a40")

    axs_m[1].plot(theta1[:100], phi[:100], color="#00fff5", linewidth=1.5)
    axs_m[1].set_title("Phase Space Portrait", color="#e0e0e0", fontsize=10)
    axs_m[1].set_xlabel("theta1", color="#e0e0e0", fontsize=8)
    axs_m[1].set_ylabel("phi", color="#e0e0e0", fontsize=8)
    axs_m[1].grid(True, alpha=0.25, color="#2a2a40")

    fig_m.savefig(p_mob_path, format="png", facecolor=fig_m.get_facecolor())
    generated["pendulum_simulator_dark_375x667"] = p_mob_path

    # 4. Tour Matching Viewer - Dark Desktop (1280x720)
    tm_path = target_dir / "tour_matching_viewer_dark_1280x720.png"
    fig_tm = Figure(
        figsize=(12.8, 7.2), dpi=100, layout="constrained", facecolor="#121212"
    )
    axs_tm = fig_tm.subplots(1, 2)
    for ax in axs_tm:
        ax.set_facecolor("#1e1e1e")
        ax.tick_params(colors="#e0e0e0")
        for spine in ax.spines.values():
            spine.set_color("#bb86fc")

    ranks = [1, 2, 3, 4, 5]
    scores = [12.4, 18.2, 22.5, 29.1, 34.0]
    axs_tm[0].bar(ranks, scores, color="#bb86fc", width=0.6)
    axs_tm[0].set_title(
        "Candidate Match Residuals (RMS mm)", color="#e0e0e0", fontsize=14
    )
    axs_tm[0].set_xlabel("Candidate Rank", color="#e0e0e0")
    axs_tm[0].set_ylabel("Position RMSE (mm)", color="#e0e0e0")
    axs_tm[0].grid(True, alpha=0.2, color="#444444")

    phase_x = [0.1 * i for i in range(11)]
    vel_y = [5.0 + 35.0 * (px**2) for px in phase_x]
    axs_tm[1].plot(phase_x, vel_y, color="#03dac6", linewidth=2.5, marker="o")
    axs_tm[1].set_title(
        "Clubhead Velocity vs Swing Phase", color="#e0e0e0", fontsize=14
    )
    axs_tm[1].set_xlabel("Normalized Swing Phase", color="#e0e0e0")
    axs_tm[1].set_ylabel("Clubhead Speed (m/s)", color="#e0e0e0")
    axs_tm[1].grid(True, alpha=0.2, color="#444444")

    fig_tm.savefig(tm_path, format="png", facecolor=fig_tm.get_facecolor())
    generated["tour_matching_viewer_dark_1280x720"] = tm_path

    # 5. Project Map - Dark Desktop (1280x720)
    pm_path = target_dir / "project_map_dark_1280x720.png"
    fig_pm = Figure(
        figsize=(12.8, 7.2), dpi=100, layout="constrained", facecolor="#0f172a"
    )
    ax_pm = fig_pm.subplots()
    ax_pm.set_facecolor("#1e293b")
    ax_pm.tick_params(colors="#94a3b8")
    for spine in ax_pm.spines.values():
        spine.set_color("#38bdf8")

    categories = [
        "Physics Engines",
        "Motion Matching",
        "GUI Tools",
        "Robotics",
        "CI & Automation",
    ]
    counts = [6, 12, 18, 5, 14]
    ax_pm.barh(categories, counts, color="#38bdf8", height=0.55)
    ax_pm.set_title(
        "UpstreamDrift Architecture & Subsystem Map", color="#f1f5f9", fontsize=14
    )
    ax_pm.set_xlabel("Active Module Count", color="#f1f5f9")
    ax_pm.grid(True, alpha=0.2, color="#334155")

    fig_pm.savefig(pm_path, format="png", facecolor=fig_pm.get_facecolor())
    generated["project_map_dark_1280x720"] = pm_path

    # 6. Rate of Closure - Dark Desktop (1280x720)
    roc_path = target_dir / "rate_of_closure_dark_1280x720.png"
    fig_roc = Figure(
        figsize=(12.8, 7.2), dpi=100, layout="constrained", facecolor="#18181b"
    )
    ax_roc = fig_roc.subplots()
    ax_roc.set_facecolor("#27272a")
    ax_roc.tick_params(colors="#d4d4d8")
    for spine in ax_roc.spines.values():
        spine.set_color("#f59e0b")

    dt_ms = [i - 50 for i in range(71)]
    roc_deg_s = [1200.0 * math.exp(-((ms) ** 2) / 250.0) for ms in dt_ms]
    ax_roc.plot(dt_ms, roc_deg_s, color="#f59e0b", linewidth=2.5)
    ax_roc.axvline(0, color="#ef4444", linestyle="--", label="Impact")
    ax_roc.set_title(
        "Rate of Closure (RoC) Around Impact Window", color="#fafafa", fontsize=14
    )
    ax_roc.set_xlabel("Time from Impact (ms)", color="#fafafa")
    ax_roc.set_ylabel("Face Closure Rate (deg/s)", color="#fafafa")
    ax_roc.grid(True, alpha=0.25, color="#3f3f46")
    ax_roc.legend(facecolor="#27272a", edgecolor="#f59e0b", labelcolor="#fafafa")

    fig_roc.savefig(roc_path, format="png", facecolor=fig_roc.get_facecolor())
    generated["rate_of_closure_dark_1280x720"] = roc_path

    return generated
