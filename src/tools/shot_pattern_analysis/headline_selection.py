"""The 24 selected zero-lean headline conditions and their saved source cells."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final

from .matched_report import load_validated_cell

CLUB_ORDER: Final = ("driver", "seven_iron", "pitching_wedge")
"""Report order of the three club presets."""

PROFILE_ORDER: Final = ("shaft_only", "combined2")
"""Report order of the two headline delivery-noise profiles."""

CLUB_LABELS: Final = {
    "driver": "Driver",
    "seven_iron": "7-Iron",
    "pitching_wedge": "Pitching Wedge",
}
PROFILE_LABELS: Final = {"shaft_only": "Shaft Only", "combined2": "Combined"}

PRIMARY_PROTOCOL: Final = "original_primary/1"
PW_FACE_BOUNDS_PROTOCOL: Final = "pw-primary-face-bounds/1"


def headline_cell_names() -> tuple[str, ...]:
    """Return the 24 selected cell names in report order (C01..C24)."""
    return tuple(
        f"{club}_lean0_curve{curve}_rotation{sd}_{profile}"
        for club in CLUB_ORDER
        for curve in (1, 2)
        for sd in (1, 2)
        for profile in PROFILE_ORDER
    )


def headline_source(cell: str) -> str:
    """Return the protocol that supplies a headline cell.

    The four pitching-wedge curve-2 conditions were unavailable under the
    original face bounds and come from the declared supplemental protocol.
    """
    if cell not in headline_cell_names():
        raise ValueError(f"{cell!r} is not a headline condition")
    if cell.startswith("pitching_wedge_lean0_curve2"):
        return PW_FACE_BOUNDS_PROTOCOL
    return PRIMARY_PROTOCOL


PROTOCOL_KEYS: Final = {
    "original_primary/1": "P",
    "original_control/1": "OC",
    "pw-primary-face-bounds/1": "WP",
    "pw-control-face-bounds/1": "WC",
    "driver-control-speed-bounds/1": "DC",
}
"""Short report keys for the five source protocols."""


def condition_parts(cell: str) -> tuple[str, str, str, str]:
    """Split a zero-lean cell name into (club label, curve, twist SD, profile)."""
    club, separator, rest = cell.partition("_lean0_curve")
    curve, _, rest = rest.partition("_rotation")
    sd, _, profile = rest.partition("_")
    if not separator or club not in CLUB_LABELS or profile not in PROFILE_LABELS:
        raise ValueError(f"{cell!r} is not a zero-lean headline cell name")
    return CLUB_LABELS[club], curve, sd, PROFILE_LABELS[profile]


def readable_condition(cell: str) -> str:
    """Return a reader-facing label such as ``Driver, Curve ×2, 1° SD, Combined``."""
    club, curve, sd, profile = condition_parts(cell)
    return f"{club}, Curve ×{curve}, {sd}° Twist SD, {profile}"


def load_headline_cells(
    primary_root: Path, pw_face_bounds_root: Path
) -> dict[str, dict[str, Any]]:
    """Load and integrity-check every headline cell's saved plan config.

    Each cell must be listed in its saved matrix plan, pass manifest checks
    and be completed. Returns ``{cell: {"dir", "protocol", "config"}}``.
    """
    roots = {
        PRIMARY_PROTOCOL: Path(primary_root),
        PW_FACE_BOUNDS_PROTOCOL: Path(pw_face_bounds_root),
    }
    plans = {
        protocol: json.loads((root / "matrix_plan.json").read_text(encoding="utf-8"))
        for protocol, root in roots.items()
    }
    selected = {}
    for cell in headline_cell_names():
        protocol = headline_source(cell)
        plan = plans[protocol]
        if cell not in plan.get("cells", {}):
            raise ValueError(f"{cell} is not listed in the {protocol} matrix plan")
        cell_dir = roots[protocol] / cell
        summary = load_validated_cell(cell_dir, plan["cells"][cell], plan["execution"])
        if summary.get("status") != "completed":
            raise ValueError(f"{cell} is not a completed saved cell")
        selected[cell] = {
            "dir": cell_dir,
            "protocol": protocol,
            "config": plan["cells"][cell],
        }
    return selected


INTERACTION_CONTRASTS: Final = tuple(
    (
        f"E{number:02d}",
        kind,
        f"{club}_lean0_{first}_{profile}",
        f"{club}_lean0_{second}_{profile}",
    )
    for number, (club, kind, first, second, profile) in enumerate(
        (
            (club, kind, first, second, profile)
            for club in CLUB_ORDER
            for profile in PROFILE_ORDER
            for kind, first, second in (
                ("shaft_rotation_sd", "curve1_rotation1", "curve1_rotation2"),
                ("curve_strength", "curve1_rotation2", "curve2_rotation2"),
            )
        ),
        start=1,
    )
)
"""Ordered direct interactions: twist SD 1 to 2, then curve 1 to 2 at SD 2."""

FIXED_TARGET_CONTRASTS: Final = (
    (
        "F1",
        "across_lean",
        "driver_lean0_curve2_rotation2_combined2",
        "driver_lean5_curve2_rotation2_combined2",
    ),
    (
        "F2",
        "across_lean",
        "pitching_wedge_lean0_curve2_rotation2_combined2",
        "pitching_wedge_lean5_curve2_rotation2_combined2",
    ),
)
"""Fixed-target lean 0 to 5 degree controls at curve 2, twist SD 2, combined."""
