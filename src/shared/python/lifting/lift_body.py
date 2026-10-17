"""The lifter body the same-input lift bundles replay (OSL-14, #13057).

LIFT-5 (#11745) first replayed the golf ``full_body_spec_hipcal_scaled.json``
body.  That body has no lumbar joint (its spine universal sits at the
shoulders), so the whole trunk-thigh fold of a squat or pull landed on the
hip: the bundle hip flexion read 159-172 deg against the 120 deg human limit,
and the OpenSim lift IK (OSL-12, #13048) could not track its pelvis.

This module composes the AN-1 anthropometric body
(:func:`~src.shared.python.motion_matching.execution.spec_builder.compose_anthropometric_document`)
at the lift subject instead:

* de Leva (1996) segments scaled to :data:`LIFT_BODY_STATURE_M` and
  :data:`LIFT_BODY_MASS_KG`, a lumbar universal joint at L5/S1
  (``SpineInputX`` lateral bend, ``SpineInputY`` flexion) and hip centres
  ``2 * HIP_HALF_WIDTH_FRACTION_OF_STATURE * stature`` apart;
* cited human ranges in the document (``coordinate_ranges_deg``, from
  :data:`~src.shared.python.motion_matching.range_of_motion.HUMAN_RANGES_DEG`:
  Kapandji, Rajagopal 2016 for the legs), which :func:`lift_bounds` turns into
  IK bounds;
* the golf grip rotations (``GRIP_ROTATION_DEG``) unchanged: with the left
  wrist moved to the barbell grip width (``back_bar.apply_grip_width``) both
  hands reach a pronated barbell grip inside their cited ranges.

The document is committed at :data:`LIFT_BODY_SPEC` so bundles hash a stable
spec; :func:`compose_lift_body_document` regenerates it.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]

__all__ = [
    "HEAD_COORDINATES",
    "LIFT_BODY_MASS_KG",
    "LIFT_BODY_SPEC",
    "LIFT_BODY_STATURE_M",
    "SEED_PELVIS_HEIGHT_M",
    "compose_lift_body_document",
    "lift_body_bytes",
    "lift_bounds",
    "lift_seed_pose",
]

_REPO = Path(__file__).resolve().parents[4]
#: The committed lift-body document (regenerate with ``--write`` below).
LIFT_BODY_SPEC = (
    _REPO / "docs/development/lifting/evidence/lift_body/lift_body_spec.json"
)
#: The planar optimizer's default subject (``scripts/generate_lift_bundle.py``).
LIFT_BODY_STATURE_M = 1.78
LIFT_BODY_MASS_KG = 80.0
#: Root height of the seed pose, metres; only a starting point for the
#: standing IK, which solves the root.
SEED_PELVIS_HEIGHT_M = 0.95
#: Head coordinates: no landmark observes them, so the lift solves hold them.
HEAD_COORDINATES: tuple[str, ...] = ("NeckInputX", "NeckInputY", "NeckInputZ")


def compose_lift_body_document() -> dict[str, Any]:
    """The AN-1 lift-body document (de Leva, lumbar joint, cited ranges).

    Postcondition: the document carries ``coordinate_ranges_deg`` for every
    leg, spine and arm coordinate it bounds.
    """
    from src.shared.python.motion_matching.execution.spec_builder import (
        compose_anthropometric_document,
    )

    built = compose_anthropometric_document(
        LIFT_BODY_STATURE_M,
        LIFT_BODY_MASS_KG,
        "driver",
        match_subject_mass=True,
    )
    return dict(built.document)


def _serialise(document: Mapping[str, Any]) -> bytes:
    return (json.dumps(document, indent=1) + "\n").encode("utf-8")


def lift_body_bytes(path: Path = LIFT_BODY_SPEC) -> bytes:
    """Bytes of the committed lift-body document."""
    return Path(path).read_bytes()


def lift_bounds(spec: Mapping[str, Any]) -> dict[str, tuple[float, float]]:
    """IK bounds (radians) from a document's cited ``coordinate_ranges_deg``.

    A document without ranges (the legacy golf spec) gets no bounds.

    Raises:
        ValueError: for a range whose lower limit is not below its upper one.
    """
    bounds: dict[str, tuple[float, float]] = {}
    for name, (low, high) in dict(spec.get("coordinate_ranges_deg", {})).items():
        if not float(low) < float(high):
            raise ValueError(f"range of {name!r} is empty: {low}..{high} deg")
        bounds[name] = (float(np.radians(low)), float(np.radians(high)))
    return bounds


def lift_seed_pose(spec: Mapping[str, Any]) -> Array:
    """Seed pose in spec order: the document's address seed, root raised.

    Raises:
        ValueError: when the document has no ``TranslationInputZ`` root.
    """
    order = list(spec["coordinate_order"])
    if "TranslationInputZ" not in order:
        raise ValueError("lift-body document has no TranslationInputZ root")
    seed = np.zeros(len(order))
    for name, value in dict(spec.get("address_seed_deg", {})).items():
        seed[order.index(name)] = np.radians(float(value))
    seed[order.index("TranslationInputZ")] = SEED_PELVIS_HEIGHT_M
    return seed


def main(argv: list[str] | None = None) -> int:
    """``--write`` regenerates :data:`LIFT_BODY_SPEC`; ``--check`` compares."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args(argv)
    content = _serialise(compose_lift_body_document())
    if args.write:
        LIFT_BODY_SPEC.parent.mkdir(parents=True, exist_ok=True)
        LIFT_BODY_SPEC.write_bytes(content)
        return 0
    return 0 if LIFT_BODY_SPEC.read_bytes() == content else 1


if __name__ == "__main__":
    raise SystemExit(main())
