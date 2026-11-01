"""HRG-D3 (#13174): labelled address, top and impact stills of the v2 trunk visuals.

Builds a full-body-v2 same-input bundle by migrating the MuJoCo ``q_track`` from
``ground_support/anthro_driver_drake`` (via ``scripts.same_input_bundle export``),
renders through ``scripts.render_golfer_realism_showcase --v2-spec``, and writes
``receipt.json`` beside the PNGs. Event frames come from the v1 bundle's shared
``club_face`` rules; trunk migration can move the face centre enough that impact
detection on the v2 trajectory fails, so the v1 indices are reused at equal times.

Usage (Linux, MuJoCo; ``MUJOCO_GL=osmesa`` when EGL is unavailable)::

    MUJOCO_GL=osmesa python3 \\
        docs/development/full_body_models/evidence/trunk_visuals/render_trunk_visuals_evidence.py

Stills are visual evidence only, not physics validation.
"""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
EVIDENCE = Path(__file__).resolve().parent
V1_SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
DRAKE_RUN = ROOT / (
    "docs/development/full_body_models/evidence/ground_support/anthro_driver_drake"
)
V2_SPEC = EVIDENCE / "full_body_v2_anthro_driver.json"
V2_BUNDLE = EVIDENCE / "driver_v2_bundle.npz"
OUT = EVIDENCE / "render_out"
RECEIPT = EVIDENCE / "receipt.json"
EXPORT_DURATION_S = 1.65


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _git_head() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True, cwd=ROOT)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"
    return out.strip()


def _export_v1_bundle(tmp: Path) -> Path:
    out = tmp / "anthro_driver_drake_v1.npz"
    cmd = [
        sys.executable,
        "-m",
        "scripts.same_input_bundle",
        "export",
        "--run-dir",
        str(DRAKE_RUN),
        "--out",
        str(out),
        "--duration",
        str(EXPORT_DURATION_S),
    ]
    subprocess.run(cmd, check=True, cwd=ROOT)
    return out


def _build_v2_bundle(v1_bundle: Path) -> tuple[bytes, Path]:
    from src.shared.python.motion_matching.full_body_migration import TrunkMigrator
    from src.shared.python.motion_matching.full_body_v2_geometry import (
        derive_landmark_v2_spec,
    )
    from src.shared.python.motion_matching.same_input import InputBundle
    from src.shared.python.motion_matching.same_input.reference import (
        generate_reference_bundle,
    )

    v1_doc = json.loads(V1_SPEC.read_text(encoding="utf-8"))
    v2_doc = derive_landmark_v2_spec(
        v1_doc, provenance="HRG-D3 trunk stills evidence (#13174)"
    )
    v2_bytes = json.dumps(v2_doc, sort_keys=True).encode("utf-8")
    V2_SPEC.write_bytes(v2_bytes)
    loaded = InputBundle.load(v1_bundle)
    q_v2 = TrunkMigrator(v1_doc, v2_doc).migrate_coordinates(
        loaded.reference_q, loaded.coordinate_order
    )
    times = loaded.dt_s * np.arange(len(q_v2))
    bundle = generate_reference_bundle(
        v2_bytes,
        times,
        q_v2,
        provenance={
            "issue": "13174",
            "source_run": "ground_support/anthro_driver_drake",
            "source_v1_bundle_sha256": _sha256(v1_bundle),
            "event_frames_source": (
                "v1 MuJoCo face-centre rules on the same-input export"
            ),
        },
    )
    bundle.save(V2_BUNDLE)
    return v2_bytes, V2_BUNDLE


def _render_stills(v1_bundle: Path, argv_tail: list[str]) -> Path:
    from scripts import render_golfer_realism_showcase as showcase
    from scripts import render_v2_trunk_stills as stills
    from src.shared.python.motion_matching.same_input import InputBundle
    from src.tools.native_viewer_export.core import SwingInput

    v1_loaded = InputBundle.load(v1_bundle)
    v1_swing = SwingInput(
        v1_loaded, v1_loaded.reference_q, "capture-A_driver", "Driver", "mujoco"
    )
    frames = stills.detect_event_frames(v1_swing)

    argv = [
        "--v2-spec",
        str(V2_SPEC),
        "--swing",
        f"driver={V2_BUNDLE}",
        "--engines",
        "mujoco",
        "--views",
        "face_on,dtl",
        "--out",
        str(OUT),
        "--capture",
        "capture-A",
    ]
    plan = showcase.plan_from_args(showcase.build_parser().parse_args(argv))
    argv_join = argv_tail or argv
    command = shlex.join(
        ["python3", "-m", "scripts.render_golfer_realism_showcase", *argv_join]
    )
    plan = replace(plan, command=command)
    settings = replace(showcase.settings_for(plan), overlays=False, hud=False)
    manifest = stills.render_v2_stills(
        plan, settings, event_finder=lambda _swing: dict(frames)
    )
    return manifest


def _write_receipt(
    manifest: Path, spec_sha256: str, event_frames: dict[str, int]
) -> None:
    doc = json.loads(manifest.read_text(encoding="utf-8"))
    stills = doc.get("stills", [])
    by_event: dict[str, dict[str, str]] = {}
    for row in stills:
        by_event.setdefault(row["event"], {})[row["view"]] = _sha256(
            OUT / Path(row["file"])
        )
    body: dict[str, Any] = {
        "issue": 13174,
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "verified_main_sha": _git_head(),
        "command": doc.get("command"),
        "spec_sha256": spec_sha256,
        "event_frames": event_frames,
        "manifest": manifest.relative_to(EVIDENCE).as_posix(),
        "stills_sha256": by_event,
        "note": (
            "Event frames from v1 same-input export face-centre rules; "
            "v2 trunk migration preserves sample times."
        ),
    }
    RECEIPT.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        v1_bundle = _export_v1_bundle(Path(tmpdir))
        spec_bytes, _ = _build_v2_bundle(v1_bundle)
        from src.shared.python.motion_matching.same_input import InputBundle
        from src.tools.native_viewer_export.core import SwingInput
        from scripts.render_v2_trunk_stills import detect_event_frames

        v1_loaded = InputBundle.load(v1_bundle)
        frames = detect_event_frames(
            SwingInput(
                v1_loaded, v1_loaded.reference_q, "capture-A_driver", "Driver", "mujoco"
            )
        )
        manifest = _render_stills(v1_bundle, sys.argv[1:])
        _write_receipt(manifest, hashlib.sha256(spec_bytes).hexdigest(), frames)
    print(f"wrote receipt {RECEIPT}")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
