"""OSL-14 (#13057): realistic hip, pelvis and upper-body mapping for lift inputs.

This module encodes the *scoping* acceptance for rebuilding LIFT-5 lift bundles
on the AN-1 lift body. It validates JSON fixtures and the committed
``docs/development/lifting/evidence/lift_body/lift_body_spec.json`` document.

**TODO (needs motion assets / OpenSim):** Regenerating each lift bundle,
OpenSim IK/ID receipts, Hicks residuals and cross-engine parity. Those steps
stay on the issue until merged evidence exists.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np
from numpy.typing import NDArray

from src.shared.python.lifting.lift_body import lift_bounds
from src.shared.python.motion_matching.anthropometric_geometry import (
    HIP_HALF_WIDTH_FRACTION_OF_STATURE,
)
from src.shared.python.motion_matching.range_of_motion import HUMAN_RANGES_DEG
from src.shared.python.motion_matching.same_input import InputBundle

from ._json_schema import validate_payload

Array = NDArray[np.float64]

_REPO = Path(__file__).resolve().parents[4]
#: Same path as :data:`~src.shared.python.lifting.lift_body.LIFT_BODY_SPEC`.
LIFT_BODY_SPEC_PATH: Final = (
    _REPO / "docs/development/lifting/evidence/lift_body/lift_body_spec.json"
)


ISSUE_ID: Final = 13057
OSL_SLUG: Final = "OSL-14"

FIXTURE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["schema_version", "issue", "lift_body"],
    "additionalProperties": True,
    "properties": {
        "schema_version": {"const": "ud.osl_scoping.osl14.v1"},
        "issue": {
            "type": "object",
            "required": ["id", "slug"],
            "properties": {
                "id": {"const": ISSUE_ID},
                "slug": {"const": OSL_SLUG},
            },
        },
        "lift_body": {
            "type": "object",
            "required": [
                "stature_m",
                "mass_kg",
                "hip_half_width_m",
                "lumbar_coordinates",
                "has_pelvis_as_observe_landmarks",
            ],
            "properties": {
                "stature_m": {"type": "number", "exclusiveMinimum": 0},
                "mass_kg": {"type": "number", "exclusiveMinimum": 0},
                "hip_half_width_m": {"type": "number", "exclusiveMinimum": 0},
                "lumbar_coordinates": {
                    "type": "array",
                    "minItems": 1,
                    "items": {"type": "string"},
                },
                "has_pelvis_as_observe_landmarks": {"type": "boolean"},
            },
        },
    },
}

ISSUE_TODOS_WITHOUT_ASSETS: Final[tuple[str, ...]] = (
    "Confirm whether legacy 159–172° hip flexion is a real angle or a pelvis-frame "
    "convention offset; record the finding with measurement (issue step 1).",
    "Add ASIS/PSIS or equivalent pelvis skin landmarks to the lift spec and track "
    "them in bundle IK targets so pelvis tilt is observed (issue steps 2–3).",
    "Map shoulder centres and scapulae onto Rajagopal segment frames for upper-body "
    "retargeting; regenerate squat, deadlift and clean-first-pull bundles (issue steps 3–4).",
    "Run OpenSim IK/ID and scripts/measure_opensim_lift_ik_id.py; report marker RMS, "
    "Hicks residuals and cross-engine bar-path parity (issue step 5, Acceptance).",
)


def validate_osl14_fixture(record: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a scoping fixture dict; returns a normalised copy."""
    validate_payload(record, FIXTURE_SCHEMA)
    return dict(record)


def issue_todo_block() -> str:
    """GitHub-issue comment block listing work that needs assets."""
    lines = [f"{OSL_SLUG} (#{ISSUE_ID}) — blocked until motion/OpenSim evidence:", ""]
    lines.extend(f"- TODO: {t}" for t in ISSUE_TODOS_WITHOUT_ASSETS)
    return "\n".join(lines)


def lift_body_matches_osl14_acceptance(
    spec: Mapping[str, Any], *, fixture: Mapping[str, Any] | None = None
) -> None:
    """Assert *spec* satisfies OSL-14 structural acceptance (no OpenSim).

    Raises:
        AssertionError: when a documented invariant fails.
    """
    if fixture is not None:
        validate_payload(fixture, FIXTURE_SCHEMA)
    subj = spec["subject"]
    stature = float(subj["stature_m"])
    mass = float(subj["mass_kg"])
    hip_centre_spacing_m = _hip_centre_spacing_m(spec)
    expected_width = 2.0 * HIP_HALF_WIDTH_FRACTION_OF_STATURE * stature

    assert "SpineInputX" in spec["coordinate_order"] and "SpineInputY" in spec["coordinate_order"]
    for name in ("SpineInputX", "SpineInputY"):
        assert tuple(spec["coordinate_ranges_deg"][name]) == HUMAN_RANGES_DEG[name]

    assert math.isclose(hip_centre_spacing_m, expected_width, rel_tol=0.0, abs_tol=1e-5)

    bounds = lift_bounds(spec)
    for side in ("r", "l"):
        lo, hi = bounds[f"hip_flexion_{side}"]
        assert math.isclose(float(np.degrees(lo)), -30.0, rel_tol=0.0, abs_tol=1e-3)
        assert math.isclose(float(np.degrees(hi)), 120.0, rel_tol=0.0, abs_tol=1e-3)

    if fixture is not None:
        lb = fixture["lift_body"]
        assert float(lb["stature_m"]) == stature
        assert float(lb["mass_kg"]) == mass
        assert math.isclose(
            float(lb["hip_half_width_m"]), hip_centre_spacing_m, rel_tol=0.0, abs_tol=1e-5
        )
        assert lb["has_pelvis_as_observe_landmarks"] is False


def committed_lift_body_spec() -> dict[str, Any]:
    """Parse the committed lift-body JSON at :data:`LIFT_BODY_SPEC_PATH`."""
    return json.loads(LIFT_BODY_SPEC_PATH.read_bytes())


def committed_lift_body_spec_sha256() -> str:
    """SHA-256 of the committed lift-body document bytes."""
    return hashlib.sha256(LIFT_BODY_SPEC_PATH.read_bytes()).hexdigest()


def bundle_spec_matches_committed_lift_body_document(bundle_path: Path) -> None:
    """Assert the bundle's embedded spec bytes hash to the committed lift body file.

    **Red anchor for #13057:** committed bundles may still carry an older
    spec revision; remove :func:`bundle_structurally_matches_lift_body` only
    after regenerating bundles on the current document.
    """
    bundle = InputBundle.load(bundle_path)
    expected = committed_lift_body_spec_sha256()
    assert bundle.spec_sha256 == expected, (bundle.spec_sha256, expected)
    prov = bundle.provenance.get("spec_sha256")
    assert prov == expected, (prov, expected)


def bundle_structurally_matches_lift_body(bundle_path: Path) -> None:
    """Structural OSL-14 checks: parsed bundle spec matches committed invariants."""
    bundle = InputBundle.load(bundle_path)
    spec = json.loads(bundle.spec_bytes)
    lift_body_matches_osl14_acceptance(spec, fixture=None)


def _hip_centre_spacing_m(spec: Mapping[str, Any]) -> float:
    left = _hip_translation(spec, "l")
    right = _hip_translation(spec, "r")
    return float(np.linalg.norm(np.asarray(left, dtype=float) - np.asarray(right, dtype=float)))


def _hip_translation(spec: Mapping[str, Any], side: str) -> Array:
    hip = next(j for j in spec["joints"] if j["name"] == f"hip_{side}")
    return np.asarray(hip["parent_to_base"], dtype=float)[:3, 3]
