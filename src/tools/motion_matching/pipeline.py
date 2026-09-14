"""Command construction and receipt summaries for the Motion Matching tool.

The matching pipeline lives in the evidence driver
``docs/development/full_body_models/evidence/ground_support/run_ground_support.py``
and the document builder ``docs/development/full_body_models/build_anthropometric_spec.py``;
this module knows how to call them for a capture and a club and how to read
back the receipt, so the GUI (``gui.py``) and any script share one path
through the pipeline. Pure functions, no Qt, tested without running the
pipeline. Epic #10113, child #10106.
"""

from __future__ import annotations

import json
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[3]
FULL_BODY = REPO_ROOT / "docs/development/full_body_models"
BUILDER = FULL_BODY / "build_anthropometric_spec.py"
DRIVER_SCRIPT = FULL_BODY / "evidence/ground_support/run_ground_support.py"
NATIVE = (
    REPO_ROOT
    / "docs/development/simscape_tour_matching/native_evidence/native_geometry_spec_9967.json"
)
OSIM = REPO_ROOT / "src/engines/physics_engines/opensim/models/golf_humanoid.osim"
CANDIDATE = FULL_BODY / "evidence/native_candidates/returned81_candidate.json"
CAPTURES = ("driver", "iron")
CLUBS = ("driver", "iron7")
CLUB_FOR_CAPTURE = {"driver": "driver", "iron": "iron7"}


@dataclass(frozen=True)
class MatchRequest:
    """One matching run: which capture, which club, which subject."""

    capture: str
    club: str
    stature_m: float = 1.71
    mass_kg: float = 78.0
    trunk_scale: float = 1.15
    arm_scale: float = 1.10
    shoulder_scale: float = 1.0
    output_name: str | None = None

    def __post_init__(self) -> None:
        if self.capture not in CAPTURES:
            raise ValueError(f"Capture must be one of {CAPTURES}")
        if self.club not in CLUBS:
            raise ValueError(f"Club must be one of {CLUBS}")
        for value in (
            self.stature_m,
            self.mass_kg,
            self.trunk_scale,
            self.arm_scale,
            self.shoulder_scale,
        ):
            if not value > 0:
                raise ValueError("Subject values and scale factors must be positive")

    @property
    def document_name(self) -> str:
        return f"full_body_spec_anthro_{self.club}"

    @property
    def run_name(self) -> str:
        return self.output_name or f"anthro_{self.capture}"

    @property
    def document_path(self) -> Path:
        return FULL_BODY / f"{self.document_name}.json"

    @property
    def output_dir(self) -> Path:
        return FULL_BODY / "evidence/ground_support" / self.run_name


def build_command(request: MatchRequest) -> list[str]:
    """Command line that writes the anthropometric document for ``request``."""
    return [
        sys.executable,
        str(BUILDER),
        "--native",
        str(NATIVE),
        "--osim",
        str(OSIM),
        "--native-candidate",
        str(CANDIDATE),
        "--stature",
        str(request.stature_m),
        "--mass",
        str(request.mass_kg),
        "--trunk-scale",
        str(request.trunk_scale),
        "--arm-scale",
        str(request.arm_scale),
        "--shoulder-scale",
        str(request.shoulder_scale),
        "--club",
        request.club,
        "--output",
        str(FULL_BODY),
    ]


def match_command(request: MatchRequest) -> list[str]:
    """Command line that matches the capture on the built document."""
    return [
        sys.executable,
        str(DRIVER_SCRIPT),
        "--spec",
        str(request.document_path),
        "--skip-hip-calibration",
        "--static-seeds",
        "--capture",
        request.capture,
        "--out",
        str(request.output_dir),
    ]


def summarise_receipt(receipt: dict[str, Any]) -> dict[str, Any]:
    """The numbers a user reads first, from a ground-support receipt.

    Precondition: the receipt carries ``address``, ``ik`` and ``dynamics``
    blocks as written by the driver. Postcondition: every value is a plain
    float, bool or string; missing optional blocks read as ``None``.
    """
    for key in ("address", "ik", "dynamics"):
        if key not in receipt:
            raise ValueError(f"Receipt lacks the {key} block")
    address = receipt["address"].get("calibrated", {})
    com = address.get("centre_of_mass", {})
    dynamics = receipt["dynamics"]
    backswing = dynamics.get("backswing_to_1s", {})
    return {
        "capture": receipt.get("capture"),
        "club": (receipt.get("club") or {}).get("name"),
        "address_marker_rms_mm": _mm(address.get("marker_rms_m")),
        "full_capture_ik_rms_mm": _mm(receipt["ik"].get("marker_rms_m")),
        "backswing_root_error_max_mm": _mm(backswing.get("root_error_max_m")),
        "whole_run_root_rms_mm": _mm(dynamics.get("root_tracking_rms_m")),
        "inside_support_polygon_fraction": dynamics.get(
            "inside_support_polygon_fraction"
        ),
        "com_inside_polygon_at_address": com.get("inside_support_polygon"),
        "range_of_motion_flags_ik": sorted(
            (receipt["ik"].get("range_of_motion_flags") or {}).keys()
        ),
        "range_of_motion_flags_simulation": sorted(
            (dynamics.get("range_of_motion_flags") or {}).keys()
        ),
    }


def _mm(value: Any) -> float | None:
    return None if value is None else round(float(value) * 1e3, 1)


def read_summary(output_dir: Path) -> dict[str, Any]:
    """Summary of the receipt in ``output_dir`` (ValueError when absent)."""
    receipt = output_dir / "receipt.json"
    if not receipt.exists():
        raise ValueError(f"No receipt in {output_dir}")
    return summarise_receipt(json.loads(receipt.read_text(encoding="utf-8")))


def artefacts(output_dir: Path) -> Sequence[Path]:
    """Playback GIFs of a finished run, in display order."""
    return tuple(
        p
        for p in (output_dir / "ik_playback.gif", output_dir / "tracking_playback.gif")
        if p.exists()
    )
