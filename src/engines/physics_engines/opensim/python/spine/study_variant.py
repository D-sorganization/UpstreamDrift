"""Movement variants in the spine study (SPN-13d, #12767).

Internal to :mod:`spine.study` and :mod:`spine.study_stages`.  A study request
may name one shipped movement variant (``variant_id``) and its parameter.  This
module validates that choice, describes it for the request fingerprint, applies
it to the golf trunk input through :func:`variant_golf_input` (the composition
is not re-implemented here) and checks the driven level coordinates against the
segmental model's own coordinate ranges with :func:`check_model_ranges`.

Violations and an exceeded term bound are reported in the receipt and make the
study ``unqualified``; they are never clipped.  Coupling laws are untouched.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from src.engines.physics_engines.opensim.python.spine.drive import (
    DriveResult,
    PelvisCoordinates,
)
from src.engines.physics_engines.opensim.python.spine.golf_input import (
    GolfTrunkInput,
)
from src.engines.physics_engines.opensim.python.spine.study_receipt import round_sig
from src.engines.physics_engines.opensim.python.spine.variant_inputs import (
    check_model_ranges,
    variant_golf_input,
)
from src.shared.python.contracts import ensure
from src.shared.python.pose_estimation.model_files import sha256_of
from src.shared.python.spine.movement_variants import (
    DEFAULT_VARIANTS_PATH,
    MovementVariant,
    movement_variant,
)

__all__ = [
    "HANDEDNESS",
    "VARIANTS_CONFIG_NAME",
    "apply_request_variant",
    "check_driven_ranges",
    "request_record",
    "validate_request_variant",
]

HANDEDNESS = ("right", "left")
VARIANTS_CONFIG_NAME = "spine_movement_variants.json"
_MAX_REPORTED_VIOLATIONS = 10


def _resolved_parameter(variant: MovementVariant, parameter: object) -> float | int:
    """The parameter value (default when ``None``), checked against its bounds."""
    if parameter is None:
        return variant.parameter_default
    name = variant.parameter_name
    if isinstance(parameter, bool) or not isinstance(parameter, int | float):
        raise TypeError(f"{name} must be a real number, got {type(parameter).__name__}")
    if variant.parameter_integer and parameter != int(parameter):
        raise ValueError(f"{name} must be an integer, got {parameter!r}")
    value = int(parameter) if variant.parameter_integer else float(parameter)
    low, high = variant.parameter_bounds
    if not low <= value <= high:
        raise ValueError(f"{name}={value} outside bounds [{low}, {high}]")
    return value


def validate_request_variant(
    variant_id: object, parameter: object, handedness: object
) -> None:
    """Validate the variant fields of a study request.

    Raises:
        TypeError: ``variant_id`` is not a ``str``/``None`` or ``parameter`` is
            not a real number.
        ValueError: unknown variant id, ``parameter`` or ``handedness`` given
            without (or outside) the variant's declared bounds / choices.
    """
    if variant_id is not None and not isinstance(variant_id, str):
        raise TypeError(f"variant_id must be a str or None, got {variant_id!r}")
    if handedness not in HANDEDNESS:
        raise ValueError(f"variant_handedness must be one of {HANDEDNESS}")
    if variant_id is None:
        if parameter is not None:
            raise ValueError("variant_parameter requires variant_id")
        return
    _resolved_parameter(movement_variant(variant_id), parameter)


def request_record(
    variant_id: str, parameter: float | int | None, handedness: str
) -> dict[str, Any]:
    """Fingerprint/receipt description of the requested variant (path-free).

    Postcondition: ``id`` is ``<variant_id>@<version>``, ``parameter`` the
    resolved value and ``config_sha256`` the hash of the variants file.
    """
    variant = movement_variant(variant_id)
    return {
        "id": f"{variant.variant_id}@{variant.version}",
        "parameter_name": variant.parameter_name,
        "parameter": _resolved_parameter(variant, parameter),
        "handedness": handedness,
        "config_sha256": sha256_of(Path(DEFAULT_VARIANTS_PATH)),
    }


def apply_request_variant(
    request: Any, golf: GolfTrunkInput, out: Any
) -> GolfTrunkInput:
    """Return ``golf`` varied as ``request`` asks and record it in ``out.variant``.

    Events come from the motion file (``event_*`` header lines).  A missing
    event raises ``ValueError`` (the golf-input stage is then recorded as
    failed).  Postcondition: ``out.variant`` holds the request record plus the
    variant ``distribution_source``, target error and term-bound flag; a
    violated term bound adds a status reason.
    """
    out.variant = request_record(
        request.variant_id, request.variant_parameter, request.variant_handedness
    )
    made = variant_golf_input(
        golf,
        movement_variant(request.variant_id),
        handedness=request.variant_handedness,
        parameter=request.variant_parameter,
    )
    result = made.result
    out.variant.update(
        distribution_source=result.distribution_source,
        target_error_deg=round_sig(result.target_error_deg),
        term_bounds_ok=bool(result.term_bounds_ok),
    )
    if not result.term_bounds_ok:
        out.reasons.append("variant term exceeds its declared bounds")
    return made.golf_input


def check_driven_ranges(
    model: Path, drive: DriveResult, pelvis: PelvisCoordinates, out: Any
) -> None:
    """Check the driven level coordinates against the model's own ranges.

    Pelvis coordinates are excluded (the variant changes the trunk only).
    Postcondition: ``out.variant["range_check"]`` reports ``ok``, the count and
    the first violations; any violation adds a status reason and nothing is
    clipped.
    """
    skip = {*pelvis.rotation, *pelvis.translation}
    levels = {k: v for k, v in drive.coordinates.items() if k not in skip}
    check = check_model_ranges(model, levels)
    out.variant["range_check"] = {
        "ok": bool(check.ok),
        "n_violations": len(check.violations),
        "checked": list(check.checked),
        "violations": [
            {
                "coordinate": v.coordinate,
                "frame": v.frame,
                "value": round_sig(v.value),
                "lower": round_sig(v.lower),
                "upper": round_sig(v.upper),
            }
            for v in check.violations[:_MAX_REPORTED_VIOLATIONS]
        ],
    }
    if not check.ok:
        out.reasons.append(
            f"{len(check.violations)} variant samples outside the model coordinate ranges"
        )
    ensure(check.ok == (not check.violations), "range report is consistent")
