"""Immutable authored trace admission; producer assertions are not qualification."""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from src.shared.python.simulation_backends import SCHEMA_VERSION, Trace
from src.shared.python.motion_matching.full_body_forward_dynamics import (
    ROOT_COORDINATES,
)
from .necromatcher_replay import ReplayOptions

if TYPE_CHECKING:
    from .necromatcher import NecromatcherLibrary
    from .necromatcher_efforts import AuthoredEffortProfile

REPLAY_TRACE_SCHEMA = f"simulation_backend.trace/{SCHEMA_VERSION}"


def _json_value(meta: dict[str, Any], key: str) -> Any:
    value = meta.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Replay requires JSON string metadata: {key}")
    return json.loads(value)


def read_authored_replay(
    source: Path, library: NecromatcherLibrary, swing_id: str
) -> Trace:
    """Check canonical trace bytes, exact parent bindings and authored semantics."""
    from src.shared.python.simulation_backends.trace_io import read_trace

    try:
        trace = read_trace(source)
    except (OSError, KeyError) as exc:
        raise ValueError("Replay file is not a valid canonical trace") from exc
    if not isinstance(trace, Trace) or trace.schema_version != SCHEMA_VERSION:
        raise ValueError("Authored replay requires a canonical single trace v2.1.0")
    meta: dict[str, Any] = dict(trace.meta)
    if (
        meta.get("schema") != "necromatcher/authored-replay/1"
        or meta.get("scientific_qualified") is not False
        or meta.get("physical_source_time_qualified") is not False
        or meta.get("independent_replay_executed") is not True
        or meta.get("root_policy") != "unactuated"
        or meta.get("initial_state_policy") != "exact_saved_pose_and_authored_rates"
    ):
        raise ValueError(
            "Replay admission requires explicit unqualified authored provenance"
        )
    for name, kind in (
        ("profile", "torque_profile"),
        ("fit", "kinematic_fit"),
        ("model", "native_model"),
        ("capture", "image_capture"),
    ):
        identity = meta.get(f"{name}_id")
        if not isinstance(identity, str):
            raise ValueError("Replay parent identity must be a string")
        try:
            asset = library.load_asset(identity)
        except KeyError as exc:
            raise ValueError("Replay parent does not exist") from exc
        if (
            asset.kind != kind
            or asset.session_id != swing_id
            or meta.get(f"{name}_hash") != asset.metadata["hash"]
        ):
            raise ValueError("Replay parent kind, session or immutable hash mismatch")
    fit = library.load_fit(meta["fit_id"])
    controls = library.load_effort_profile(meta["profile_id"], meta["model_id"])
    if controls.fit_id != meta["fit_id"] or fit["capture_id"] != meta["capture_id"]:
        raise ValueError("Replay parents do not form one source-bound profile chain")
    if trace.backend != library.load_asset(meta["model_id"]).metadata["engine"]:
        raise ValueError("Replay backend differs from its bound native model")
    for key, expected in (
        ("coordinate_order", controls.dofs),
        ("coordinate_units", controls.coordinate_units),
        ("effort_units", controls.effort_units),
    ):
        if _json_value(meta, key + "_json") != list(expected):
            raise ValueError(
                "Replay ordered coordinates or units differ from its profile"
            )
    rates = _json_value(meta, "initial_rates_json")
    if not isinstance(rates, list):
        raise ValueError("Replay initial rates require a numeric array")
    for key in (
        "source_frame_index",
        "duration_s",
        "record_every",
        "verification_refinement",
        "max_steps",
    ):
        if key not in meta:
            raise ValueError(f"Replay requires explicit option metadata: {key}")
    options = ReplayOptions(
        meta["source_frame_index"],
        tuple(rates),
        meta["duration_s"],
        trace.dt,
        meta["record_every"],
        meta["verification_refinement"],
        meta["max_steps"],
    )
    _check_samples(trace, controls, fit, options)
    return trace


def _check_samples(
    trace: Trace,
    controls: AuthoredEffortProfile,
    fit: dict[str, Any],
    options: ReplayOptions,
) -> None:
    """Validate clock, finite arrays, exact initial sample and time-aligned controls."""
    meta: dict[str, Any] = dict(trace.meta)
    if controls.dofs[: len(ROOT_COORDINATES)] != ROOT_COORDINATES:
        raise ValueError("Replay requires the canonical native root coordinate order")
    try:
        position = fit["frame_indices"].index(options.source_frame_index)
    except ValueError as exc:
        raise ValueError("Replay initial source frame is not in its fit") from exc
    if _json_value(meta, "source_frame_json") != fit["frames"][position]:
        raise ValueError("Replay source frame identity differs from its fit")
    steps = round(options.duration_s / options.dt_s)
    expected_count = (
        steps // options.record_every + 1 + (steps % options.record_every != 0)
    )
    shape = (expected_count, len(controls.dofs))
    if (
        trace.t.shape != (expected_count,)
        or any(
            value is None or value.shape != shape or not np.isfinite(value).all()
            for value in (trace.q, trace.v, trace.u)
        )
        or not np.isfinite(trace.t).all()
    ):
        raise ValueError(
            "Replay requires finite state and effort arrays on its recording grid"
        )
    expected_times = (
        np.minimum(np.arange(expected_count) * options.record_every, steps) * trace.dt
        + controls.start_s
    )
    if (
        not np.allclose(trace.t, expected_times, rtol=0, atol=1e-12)
        or trace.t[-1] > controls.end_s + 1e-12
    ):
        raise ValueError(
            "Replay clock differs from its authored bounded recording grid"
        )
    if not np.array_equal(trace.q[0], fit["q"][position]) or not np.array_equal(
        trace.v[0], options.initial_rates
    ):
        raise ValueError(
            "Replay initial state differs from its saved pose or supplied rates"
        )
    if not controls.channels_are_zero(ROOT_COORDINATES) or trace.torques is not None:
        raise ValueError(
            "Replay requires unactuated roots and unit-preserving generalized efforts"
        )
    expected_efforts = np.asarray(
        [controls.evaluate(float(min(t, controls.end_s))) for t in trace.t]
    )
    if not np.allclose(trace.u, expected_efforts, rtol=0, atol=1e-12):
        raise ValueError("Replay commands differ from their bound authored profile")
    for key in (
        "initial_grip_gap_m",
        "max_grip_gap_m",
        "verification_max_translation_difference_m",
        "verification_max_rotation_difference_rad",
    ):
        value = meta.get(key)
        if (
            value is None
            or type(value) not in (float, int)
            or not np.isfinite(value)
            or value < 0
        ):
            raise ValueError("Replay diagnostics require finite nonnegative values")
    if meta["max_grip_gap_m"] < meta["initial_grip_gap_m"]:
        raise ValueError("Replay maximum grip gap must include its initial state")
