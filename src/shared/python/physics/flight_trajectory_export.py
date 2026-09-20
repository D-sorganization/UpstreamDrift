"""Export ``flight_models`` results as ``ball_flight_trajectory/1`` (#9350).

ADR-0047 H1, UpstreamDrift half. The record itself is defined in Tools
(``swing_sim.flight_interchange``); this module builds it **from the
documented contract**, constructing the payload dict directly rather
than importing the vendored package at runtime — the same posture as
Tools' own ``swing_sim.putting.ud_adapter``, which parses this repo's
green topography without ever importing UpstreamDrift. A format
adapter that imports the other side's runtime is not an interchange
seam; it is a dependency.

Why this exists
---------------
Two independent flight-model families are live across the fleet by
design (issue #8978, ADR-0045): this repo's named published models —
Waterloo/Penner, MacDonald-Hanzely, and the constant-coefficient set in
:mod:`.flight_models` — and Tools' ``swing_sim.flight``. Neither is
merged into the other. What ADR-0047 adds is a common export format, so
a Shot Tracer curve and a ``swing_sim`` capability flight can be drawn
on the same axes *because each carries its own label*.

The wire, restated from the Tools contract
------------------------------------------
``swing_sim.ball_flight_trajectory/1``, a JSON object with exactly six
top-level keys:

``format``
    The literal :data:`BALL_FLIGHT_TRAJECTORY_FORMAT`.
``source_id``
    Trimmed nonempty identifier of the producing run.
``frame_id``
    One of the two declared frames. This module always emits
    :data:`FLIGHT_FRAME_ID` — x forward (downrange), y left, z up,
    ground at ``z = 0`` — because that is the frame
    :mod:`.flight_models` integrates in. The record never guesses a
    frame: an undeclared one silently mirrors a shot.
``channels``
    Sorted list of the optional per-sample channels **every** sample
    carries. This module emits ``["velocity_mps"]``: every retained
    :class:`~.flight_models.TrajectoryPoint` holds a velocity, and
    neither family retains a per-sample spin vector (both decay a
    scalar spin analytically inside the derivative function), so
    ``spin_rad_s`` is omitted rather than reconstructed.
``provenance``
    Exactly ``model_family``, ``model_name``, ``parameter_digest`` —
    all mandatory. See below.
``samples``
    At least two objects whose keys are exactly ``time_s``,
    ``position_m``, and the declared channels. ``time_s`` is
    non-negative and strictly increasing; every vector is a finite
    three-element list.

Units are SI throughout: seconds, metres, metres per second.

Provenance
----------
``model_family`` is :data:`UD_FLIGHT_FAMILY`; ``model_name`` is the
model's own display name (``"Waterloo/Penner"``), which is what a
viewer labels a curve with and what the sibling family uses for the
same model; and ``parameter_digest`` is
:func:`trajectory_parameter_digest` over
:attr:`~.flight_models.FlightResult.coefficients`, the aero-coefficient
set the producing model actually integrated with (issue #8978). A
result with no declared coefficients is **refused** rather than
exported with an empty digest — an unattributable trajectory is exactly
the confusion the record exists to prevent.

The digest is comparable only *within* a family: this family names the
Penner lift fit ``lift_scale``/``lift_exponent`` where Tools' names the
same two numbers ``cl1``/``cl2``, so equal digests across families mean
nothing and unequal ones prove nothing.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .flight_models import FlightModelRegistry, FlightModelType, FlightResult

BALL_FLIGHT_TRAJECTORY_FORMAT = "swing_sim.ball_flight_trajectory/1"
"""The wire version this module emits."""

UD_FLIGHT_FAMILY = "ud.flight_models"
"""``model_family`` for every record produced by this repo's models."""

FLIGHT_FRAME_ID = "flight_xfwd_yleft_zup"
"""x forward, y left, z up; ground at z = 0 — what this family integrates in."""

APP_FRAME_ID = "app_xtarget_yup_zright"
"""The other declared frame in the wire; this module never emits it."""

VELOCITY_CHANNEL = "velocity_mps"
"""The single optional channel this family retains per sample."""

TRAJECTORY_RECORD_FIELDS = (
    "channels",
    "format",
    "frame_id",
    "provenance",
    "samples",
    "source_id",
)
"""Exactly the wire's top-level keys, sorted."""

PROVENANCE_FIELDS = ("model_family", "model_name", "parameter_digest")
"""Exactly the wire's provenance keys, sorted; all three are mandatory."""

__all__ = [
    "APP_FRAME_ID",
    "BALL_FLIGHT_TRAJECTORY_FORMAT",
    "FLIGHT_FRAME_ID",
    "PROVENANCE_FIELDS",
    "TRAJECTORY_RECORD_FIELDS",
    "UD_FLIGHT_FAMILY",
    "VELOCITY_CHANNEL",
    "flight_result_to_trajectory_record",
    "trajectory_parameter_digest",
    "trajectory_record_to_json",
]


def _finite_triplet(vector: Any, name: str) -> list[float]:
    """Return a finite three-element SI vector as a plain float list.

    Raises:
        TypeError: If ``vector`` is not a sized sequence of numbers.
        ValueError: If it does not have three finite components.
    """
    try:
        values = [float(component) for component in vector]
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a numeric 3-vector") from error
    if len(values) != 3:
        raise ValueError(f"{name} must have three components; got {len(values)}")
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must be finite; got {values!r}")
    return values


def trajectory_parameter_digest(parameters: Mapping[str, float | int | str]) -> str:
    """Return the wire's SHA-256 provenance digest for a parameter set.

    The algorithm is part of the documented contract, reproduced here
    rather than imported so the export path has no runtime dependency
    on the vendored package: canonical JSON of the flat mapping —
    sorted keys, compact separators, no NaN — hashed as UTF-8.

    Args:
        parameters: A flat, nonempty mapping of parameter name to a
            finite number or a string.

    Returns:
        The 64-character lowercase hex digest.

    Raises:
        TypeError: If ``parameters`` is not a mapping, or carries a
            value that is neither a finite number nor a string.
        ValueError: If it is empty, has a blank key, or carries a
            non-finite number.
    """
    if not isinstance(parameters, Mapping):
        raise TypeError("parameters must be a mapping")
    if not parameters:
        raise ValueError(
            "parameters must be nonempty: a record whose physics cannot be "
            "named cannot carry honest provenance"
        )
    payload: dict[str, float | str] = {}
    for key, value in parameters.items():
        if not isinstance(key, str) or key.strip() != key or not key:
            raise ValueError(f"parameter names must be trimmed strings; got {key!r}")
        if isinstance(value, str):
            payload[key] = value
            continue
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(
                f"parameter {key!r} must be a finite number or a string; "
                f"got {type(value).__name__}"
            )
        number = float(value)
        if not math.isfinite(number):
            raise ValueError(f"parameter {key!r} must be finite; got {value!r}")
        payload[key] = number
    text = json.dumps(payload, allow_nan=False, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _validated_samples(result: FlightResult) -> list[dict[str, Any]]:
    """Return the retained trajectory as wire samples, or refuse.

    The samples are the integrator's own retained points, never
    resampled or re-simulated — the P8 playback transport replays
    exactly these.
    """
    points: Sequence[Any] = result.trajectory
    if len(points) < 2:
        raise ValueError(
            "a flight with fewer than two retained samples is not a trajectory; "
            f"got {len(points)}"
        )
    samples: list[dict[str, Any]] = []
    previous: float | None = None
    for index, point in enumerate(points):
        time_s = float(point.time)
        if not math.isfinite(time_s) or time_s < 0.0:
            raise ValueError(
                f"sample {index} time must be finite and non-negative; got {time_s!r}"
            )
        if previous is not None and time_s <= previous:
            raise ValueError(
                "sample times must be strictly increasing; sample "
                f"{index} at {time_s!r} does not follow {previous!r}"
            )
        previous = time_s
        samples.append(
            {
                "time_s": time_s,
                "position_m": _finite_triplet(point.position, "position_m"),
                VELOCITY_CHANNEL: _finite_triplet(point.velocity, VELOCITY_CHANNEL),
            }
        )
    return samples


def _model_display_name(model_type: FlightModelType) -> str:
    """Return the registry's display name for a declared model type."""
    return FlightModelRegistry.get_model(model_type).name


def flight_result_to_trajectory_record(
    result: FlightResult,
    *,
    source_id: str | None = None,
    model_type: FlightModelType | None = None,
) -> dict[str, Any]:
    """Build the ``ball_flight_trajectory/1`` payload for one flight.

    Args:
        result: A simulated flight from any model in
            :mod:`.flight_models`. Its retained ``trajectory`` becomes
            the record's samples verbatim, and its ``coefficients``
            become the provenance digest.
        source_id: Optional run identifier. Defaults to
            ``"ud.flight_models:<model name>"``.
        model_type: Optional declared model type. When given it is
            cross-checked against ``result.model_name`` and the export
            is refused on disagreement, so a record can never label one
            model's samples with another model's identity.

    Returns:
        A plain ``dict`` matching the documented wire: six top-level
        keys, the flight frame declared, the ``velocity_mps`` channel
        declared, mandatory provenance, and strictly increasing
        non-negative sample times. Serialize it with
        :func:`trajectory_record_to_json` for byte-deterministic bytes.

    Raises:
        TypeError: If ``result`` is not a
            :class:`~.flight_models.FlightResult`, or a sample vector
            is not numeric.
        ValueError: If the result retains fewer than two samples, its
            times are not strictly increasing, any value is
            non-finite, its model name is blank, its coefficients are
            empty, or ``model_type`` disagrees with the result.
    """
    if not isinstance(result, FlightResult):
        raise TypeError("result must be a FlightResult")
    model_name = result.model_name
    if not isinstance(model_name, str) or model_name.strip() != model_name:
        raise TypeError("result.model_name must be a trimmed string")
    if not model_name:
        raise ValueError("result.model_name must name the producing model")
    if model_type is not None:
        declared = _model_display_name(model_type)
        if declared != model_name:
            raise ValueError(
                "model_type disagrees with the result: "
                f"{declared!r} declared, {model_name!r} produced"
            )
    digest = trajectory_parameter_digest(result.coefficients)
    return {
        "channels": [VELOCITY_CHANNEL],
        "format": BALL_FLIGHT_TRAJECTORY_FORMAT,
        "frame_id": FLIGHT_FRAME_ID,
        "provenance": {
            "model_family": UD_FLIGHT_FAMILY,
            "model_name": model_name,
            "parameter_digest": digest,
        },
        "samples": _validated_samples(result),
        "source_id": source_id or f"{UD_FLIGHT_FAMILY}:{model_name}",
    }


def trajectory_record_to_json(record: Mapping[str, Any]) -> str:
    """Serialize a record deterministically, per the wire's posture.

    Sorted keys, compact separators, and ``allow_nan=False``, so two
    equal records serialize to identical bytes within a runtime.

    Raises:
        TypeError: If ``record`` is not a mapping.
        ValueError: If it carries a non-finite number.
    """
    if not isinstance(record, Mapping):
        raise TypeError("record must be a mapping")
    return json.dumps(
        dict(record), allow_nan=False, separators=(",", ":"), sort_keys=True
    )
