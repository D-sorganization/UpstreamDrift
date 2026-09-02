"""Import ``ball_flight_trajectory/1`` records for the BallFlight route (ADR-0047 H3, #9352).

Private helper for :mod:`src.api.routes.ball_flight` — the import half of
the H1 wire (:mod:`src.shared.python.physics.flight_trajectory_export` is
the UD export half; Tools' ``flight_interchange.adapters`` is the other).
Lives under ``src/api/routes/`` rather than ``src/shared/python/`` because
it resolves the vendored Tools checkout through the canonical
application-layer facade
(:func:`src.launchers.tools_repo_path.resolve_tools_repo`), and
``src/shared/python/`` may never import upward from ``src.launchers`` or
``src.api`` (``tests/unit/repo_hygiene/test_import_boundaries.py`` enforces
this). This module never reimplements the wire's own validation: it
resolves the checkout via that facade — the same posture every other
production consumer of vendored Tools code uses, e.g.
:mod:`src.shared.python.biomechanics.force_source_attribution` — and calls
the vendored reader,
``shared.python.swing_sim.flight_interchange.ball_flight_trajectory_from_json``,
directly. A record produced by either flight-model family — UD's
``physics/flight_models.py`` or Tools' ``swing_sim.flight`` — parses
identically, because neither producer's runtime is imported here.

Every refusal below is either the vendored reader's own
``ContractViolationError`` message, surfaced verbatim (it is a
``ValueError`` subclass, so :class:`TrajectoryImportError` is too, and
the API's ``handle_api_errors`` decorator maps it to a 400 with that
exact text as the response ``detail``), or this module's own explicit
frame-support refusal. Nothing is ever dropped silently.

Frame handling
--------------
The BallFlight page plots directly in the frame
:mod:`.flight_trajectory_export` declares — :data:`FLIGHT_FRAME_ID`, x
forward (downrange), y left, z up — the same frame Shot Tracer's 3D
view uses (ADR-0047 H2). :data:`_PLOT_FRAME_CONVERTERS` is the single
closed dispatch table from a record's declared ``frame_id`` to a
converter into that plot frame. A ``frame_id`` valid on the wire but
absent from this table (today, the wire's other declared frame,
``app_xtarget_yup_zright``) is refused explicitly with
:class:`TrajectoryImportError` rather than silently mis-plotted.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.launchers.tools_repo_path import resolve_tools_repo
from src.shared.python.physics.flight_trajectory_export import FLIGHT_FRAME_ID

_REPO_ROOT = Path(__file__).resolve().parents[3]

__all__ = [
    "ImportedBallFlightSummary",
    "ImportedBallFlightTrajectory",
    "ImportedTrajectorySample",
    "TrajectoryImportError",
    "import_trajectory_record",
    "summarize_imported_trajectory",
]


class TrajectoryImportError(ValueError):
    """Refused an imported ball-flight trajectory record.

    A :class:`ValueError` subclass so it composes with
    ``src.api.middleware.error_handler.handle_api_errors`` without any
    extra glue: ``str(error)`` is always the specific, user-facing
    reason — the vendored reader's own contract-violation message, a
    JSON/shape failure, an unresolvable Tools checkout, or this
    module's frame-support refusal — never a generic "import failed".
    """


@dataclass(frozen=True)
class ImportedTrajectorySample:
    """One sample of an imported trajectory, already in the plot frame."""

    time_s: float
    position_m: tuple[float, float, float]
    velocity_mps: tuple[float, float, float] | None


@dataclass(frozen=True)
class ImportedBallFlightTrajectory:
    """A validated, plot-frame-converted import with its provenance.

    Attributes:
        source_id: The record's own producing-run identifier.
        model_family: Provenance family, e.g. ``"swing_sim.flight"`` or
            ``"ud.flight_models"``. Never merged across families
            (ADR-0047) — always shown beside ``model_name``.
        model_name: Provenance model name within that family.
        parameter_digest: The record's coefficient digest, comparable
            only within ``model_family``.
        frame_id: The record's original declared wire frame (kept for
            transparency even after conversion into the plot frame).
        samples: Samples converted into the BallFlight plot frame.
    """

    source_id: str
    model_family: str
    model_name: str
    parameter_digest: str
    frame_id: str
    samples: tuple[ImportedTrajectorySample, ...]


@dataclass(frozen=True)
class ImportedBallFlightSummary:
    """Best-effort scalar metrics derived from an imported trajectory.

    Computed purely from sample positions and times — never from the
    optional ``velocity_mps`` channel, which the wire allows a record
    to omit entirely — so every accepted import can report one,
    regardless of which optional channels it declares.
    """

    carry_m: float
    apex_m: float
    flight_time_s: float
    landing_angle_deg: float
    lateral_deviation_m: float


def _flight_frame_samples(
    samples: Sequence[Any],
) -> tuple[ImportedTrajectorySample, ...]:
    """Identity conversion: the wire's flight frame is the page's own."""
    return tuple(
        ImportedTrajectorySample(
            time_s=float(sample.time_s),
            position_m=sample.position_m,
            velocity_mps=sample.velocity_mps,
        )
        for sample in samples
    )


_PLOT_FRAME_CONVERTERS: dict[
    str, Callable[[Sequence[Any]], tuple[ImportedTrajectorySample, ...]]
] = {
    FLIGHT_FRAME_ID: _flight_frame_samples,
}
"""Closed dispatch table: wire ``frame_id`` -> converter into the plot frame.

Every entry here is implemented and tested. A ``frame_id`` the wire
allows but this table does not list (currently
``app_xtarget_yup_zright``) is refused by :func:`import_trajectory_record`
rather than guessed at.
"""


def _load_vendored_reader() -> Callable[[str], Any]:
    """Resolve ``vendor/ud-tools`` and import its trajectory reader.

    Uses the same canonical resolution facade as every other production
    consumer of vendored Tools code (env override, then the pinned
    ``vendor/ud-tools`` gitlink, then dev-mode sibling discovery).

    Returns:
        The vendored ``ball_flight_trajectory_from_json`` function.

    Raises:
        TrajectoryImportError: If the Tools checkout cannot be resolved,
            or the vendored ``flight_interchange`` package fails to
            import from it.
    """
    try:
        resolution = resolve_tools_repo(_REPO_ROOT, os.environ.get("TOOLS_REPO_PATH"))
    except RuntimeError as exc:
        raise TrajectoryImportError(f"Tools checkout unavailable: {exc}") from exc
    if resolution is None:
        raise TrajectoryImportError(
            "Tools repository not found: no TOOLS_REPO_PATH, no vendored "
            "vendor/ud-tools, no sibling checkout. Initialize the vendored "
            "submodule with 'git submodule update --init vendor/ud-tools'."
        )

    src_dir = str(resolution.path / "src")
    if src_dir not in sys.path:
        sys.path.insert(0, src_dir)

    try:
        from shared.python.swing_sim.flight_interchange import (  # type: ignore[import-not-found]
            ball_flight_trajectory_from_json,
        )
    except ImportError as exc:
        raise TrajectoryImportError(
            f"vendored flight_interchange reader unavailable at {resolution.path}: "
            f"{exc}"
        ) from exc
    return ball_flight_trajectory_from_json


def import_trajectory_record(record: Mapping[str, Any]) -> ImportedBallFlightTrajectory:
    """Validate one ``ball_flight_trajectory/1`` record and convert its frame.

    Args:
        record: A parsed JSON object — the raw wire record, from either
            flight-model family. Re-serialized and handed to the
            vendored reader unmodified, so every field the reader
            checks (unknown fields, missing fields, malformed
            provenance, non-monotone samples, non-finite values) is
            enforced exactly as the wire defines it.

    Returns:
        The validated trajectory, converted into the BallFlight page's
        plot frame.

    Raises:
        TrajectoryImportError: On any refusal: the vendored reader is
            unavailable, ``record`` is not a JSON object, the record
            violates the wire contract (the reader's own message is
            surfaced verbatim), or the record declares a frame this
            function has not implemented.
    """
    if not isinstance(record, Mapping):
        raise TrajectoryImportError("record must be a JSON object")

    reader = _load_vendored_reader()

    try:
        text = json.dumps(record)
    except (TypeError, ValueError) as exc:
        raise TrajectoryImportError(f"record is not valid JSON: {exc}") from exc

    try:
        parsed = reader(text)
    except (TypeError, ValueError) as exc:
        # ContractViolationError (the vendored reader's own refusal type)
        # is itself a ValueError; this also catches json.JSONDecodeError.
        raise TrajectoryImportError(str(exc)) from exc

    converter = _PLOT_FRAME_CONVERTERS.get(parsed.frame_id)
    if converter is None:
        raise TrajectoryImportError(
            f"unsupported frame {parsed.frame_id!r}: the BallFlight page only "
            f"plots {sorted(_PLOT_FRAME_CONVERTERS)!r}"
        )

    samples = converter(parsed.samples)
    provenance = parsed.provenance
    return ImportedBallFlightTrajectory(
        source_id=parsed.source_id,
        model_family=provenance.model_family,
        model_name=provenance.model_name,
        parameter_digest=provenance.parameter_digest,
        frame_id=parsed.frame_id,
        samples=samples,
    )


def summarize_imported_trajectory(
    trajectory: ImportedBallFlightTrajectory,
) -> ImportedBallFlightSummary:
    """Derive scalar metrics from an imported trajectory's samples alone.

    Mirrors the shape of :class:`~src.api.routes.ball_flight.BallFlightSummary`
    so imported curves can populate the same metrics table as computed
    ones, but the recipe here is purely geometric (positions and
    times) — never the optional ``velocity_mps`` channel — so it works
    identically whether or not a record declares that channel.

    ``landing_angle_deg`` is the descent angle below horizontal between
    the last two retained samples; 0.0 for a two-sample trajectory with
    no vertical change between them.
    """
    samples = trajectory.samples
    first, last = samples[0], samples[-1]
    apex_m = max(sample.position_m[2] for sample in samples)
    previous = samples[-2]
    dx = last.position_m[0] - previous.position_m[0]
    dy = last.position_m[1] - previous.position_m[1]
    dz = last.position_m[2] - previous.position_m[2]
    horizontal = math.hypot(dx, dy)
    landing_angle_deg = (
        math.degrees(math.atan2(-dz, horizontal))
        if horizontal > 0.0 or dz != 0.0
        else 0.0
    )
    return ImportedBallFlightSummary(
        carry_m=last.position_m[0],
        apex_m=apex_m,
        flight_time_s=last.time_s - first.time_s,
        landing_angle_deg=landing_angle_deg,
        lateral_deviation_m=last.position_m[1],
    )
