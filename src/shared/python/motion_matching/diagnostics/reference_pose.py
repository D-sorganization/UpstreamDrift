"""Reference 'credible golfer' pose for sanity-checking input MAT files.

The numbers below codify a generic right-handed adult golfer at address
with a driver. They are deliberately conservative — they exist to flag
*gross* deviations (spine fully upright, lead wrist hinged 90 deg at
address) rather than to claim a single correct pose. Sources:
biomechanics review literature for amateur golfers (e.g. McTeigue 1994,
Gluck 2008) plus visual reference to common driver-address photos.

All values are in DEGREES.
"""

from __future__ import annotations

from collections.abc import Mapping

# Field names match the Simulink.Parameter names found in
# 3DModelInputs_Impact.mat (verified via SCRIPT_TransferStartPositionVelocityIntoModelFromMATFile.m
# and via the model_* columns in the Dataset Generator CSVs).
REFERENCE_GOLFER_FIELDS: tuple[str, ...] = (
    "HipStartPositionX",
    "HipStartPositionY",
    "HipStartPositionZ",
    "SpineStartPositionX",
    "SpineStartPositionY",
    "TorsoStartPosition",
    "LScapStartPositionX",
    "LScapStartPositionY",
    "RScapStartPositionX",
    "RScapStartPositionY",
    "LSStartPositionX",
    "LSStartPositionY",
    "LSStartPositionZ",
    "RSStartPositionX",
    "RSStartPositionY",
    "RSStartPositionZ",
    "LEStartPosition",
    "REStartPosition",
    "LFStartPosition",
    "RFStartPosition",
    "LWStartPositionX",
    "LWStartPositionY",
    "RWStartPositionX",
    "RWStartPositionY",
)


def reference_golfer_setup() -> dict[str, float]:
    """Return a reference golfer ADDRESS pose (degrees, all-zero velocities).

    Postcondition: every value lies inside an anatomically plausible
    range, knees / spine reflect a real address position (forward tilt
    and slight knee flex), arms hang in front of the body, and the lead
    wrist is essentially flat (small extension).
    """
    return {
        # Pelvis (Hip) — slight pelvic forward tilt, square to target.
        "HipStartPositionX": 5.0,
        "HipStartPositionY": 0.0,
        "HipStartPositionZ": 0.0,
        # Spine — ~30 deg forward tilt (X), small lead-side bend (Y).
        "SpineStartPositionX": 30.0,
        "SpineStartPositionY": -5.0,
        # Torso axial rotation — square to target at address.
        "TorsoStartPosition": 0.0,
        # Scapulae — neutral.
        "LScapStartPositionX": 0.0,
        "LScapStartPositionY": 0.0,
        "RScapStartPositionX": 0.0,
        "RScapStartPositionY": 0.0,
        # Shoulders — arms hanging, slight internal rotation as both
        # hands meet on the grip in front of the body.
        "LSStartPositionX": -10.0,
        "LSStartPositionY": -15.0,
        "LSStartPositionZ": -25.0,
        "RSStartPositionX": -10.0,
        "RSStartPositionY": -15.0,
        "RSStartPositionZ": 25.0,
        # Elbows — both nearly straight at address.
        "LEStartPosition": 5.0,
        "REStartPosition": 10.0,
        # Forearm pronation/supination — neutral.
        "LFStartPosition": 0.0,
        "RFStartPosition": 0.0,
        # Wrists — flat lead wrist, mild radial deviation on trail.
        "LWStartPositionX": 0.0,
        "LWStartPositionY": 5.0,
        "RWStartPositionX": 0.0,
        "RWStartPositionY": -5.0,
    }


# Plausible (deg) ranges for ADDRESS / setup. Used by ``compare_to_reference``
# to flag outliers. Ranges are wide on purpose — we only want to catch
# values that are clearly not an address pose.
ADDRESS_RANGES: dict[str, tuple[float, float]] = {
    "HipStartPositionX": (-5.0, 15.0),
    "HipStartPositionY": (-15.0, 15.0),
    "HipStartPositionZ": (-15.0, 15.0),
    "SpineStartPositionX": (15.0, 45.0),  # forward tilt is non-negotiable
    "SpineStartPositionY": (-15.0, 5.0),
    "TorsoStartPosition": (-20.0, 20.0),
    "LSStartPositionZ": (-60.0, 0.0),  # large negative would be top-of-backswing
    "RSStartPositionZ": (0.0, 60.0),
    "LEStartPosition": (-5.0, 30.0),
    "REStartPosition": (-5.0, 30.0),  # at top-of-backswing this jumps to ~100
    "LWStartPositionX": (-30.0, 30.0),  # top-of-backswing has -90 (full hinge)
    "RWStartPositionX": (-30.0, 30.0),
}


def compare_to_reference(
    angles: Mapping[str, float],
    ranges: Mapping[str, tuple[float, float]] | None = None,
) -> list[dict[str, float | str]]:
    """Flag joint-angle fields whose values are outside the address ranges.

    Returns a list of dicts: ``{field, value, low, high, deviation}`` where
    ``deviation`` is the signed distance from the nearest range bound (0 if
    inside the range).

    Raises
    ------
    TypeError
        If ``angles`` is not a Mapping.
    """
    if not isinstance(angles, Mapping):
        raise TypeError(f"angles must be a Mapping, got {type(angles).__name__}")
    if ranges is None:
        ranges = ADDRESS_RANGES

    flags: list[dict[str, float | str]] = []
    for f, (lo, hi) in ranges.items():
        if f not in angles:
            continue
        v = float(angles[f])
        if v < lo:
            flags.append(
                {"field": f, "value": v, "low": lo, "high": hi, "deviation": v - lo}
            )
        elif v > hi:
            flags.append(
                {"field": f, "value": v, "low": lo, "high": hi, "deviation": v - hi}
            )
    return flags
