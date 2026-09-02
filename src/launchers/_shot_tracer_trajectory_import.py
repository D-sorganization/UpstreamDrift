"""Import ``swing_sim.ball_flight_trajectory/1`` records into Shot Tracer.

ADR-0047 H2 (issue #9351): Shot Tracer's multi-model comparison learns to
read trajectory records produced by *either* flight-model family, so a
Tools ``swing_sim`` flight can be plotted on the same axes as a native
Waterloo/Penner curve — cross-family comparison finally spans both
families, each curve labeled with its own provenance. This module is the
import half; the export halves already live in Tools'
``flight_interchange.adapters`` and this repo's
:mod:`src.shared.python.physics.flight_trajectory_export` (H1, #9360).

Where the record is actually read
----------------------------------
This module never reimplements the wire's validation. It resolves the
vendored Tools checkout through the canonical facade
(:func:`src.launchers.tools_repo_path.resolve_tools_repo`) — the same
posture every other production consumer of vendored Tools code uses (see
``src/launchers/external_tools_adapter.py``) — and calls the vendored
reader, ``shared.python.swing_sim.flight_interchange.
ball_flight_trajectory_from_json``, directly. Every refusal below is
either that reader's own ``ContractViolationError`` message, surfaced
verbatim, or this module's own frame-support refusal, equally explicit.
Nothing is ever dropped silently.

Frame handling
---------------
Shot Tracer's 3D view plots directly in the frame
:mod:`~src.shared.python.physics.flight_models` integrates in — x
forward (downrange), y left, z up — which the wire calls
``flight_xfwd_yleft_zup`` (:data:`FLIGHT_FRAME_ID`, reused from the H1
export module rather than redeclared). :data:`_PLOT_FRAME_CONVERTERS` is
the single closed dispatch table from a record's declared ``frame_id`` to
a converter into that plot frame. A ``frame_id`` that is valid on the
wire but absent from this table (today, the wire's other declared frame,
``app_xtarget_yup_zright``) is refused explicitly with
:class:`TrajectoryImportError` rather than silently mis-plotted — the
record declares its frame from a closed enum, and Shot Tracer only
speaks a subset of it so far.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.launchers.tools_repo_path import ensure_tools_importable
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.physics.flight_trajectory_export import FLIGHT_FRAME_ID

logger = get_logger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]

__all__ = [
    "ImportedTrajectoryCurve",
    "TrajectoryImportError",
    "import_trajectory_record",
]


class TrajectoryImportError(Exception):
    """Refused an imported trajectory record.

    ``str(error)`` is always the specific, user-facing reason: the
    vendored reader's own contract-violation message, a JSON/IO failure,
    or this module's frame-support refusal. Never a generic "import
    failed" — callers show ``str(error)`` directly to the user.
    """


@dataclass(frozen=True)
class ImportedTrajectoryCurve:
    """One imported trajectory record, converted and ready to plot.

    Attributes:
        label: ``"<model_family> / <model_name>"``. Always set — an
            imported curve is never plotted unlabeled (ADR-0047).
        positions: Nx3 array of sample positions in Shot Tracer's plot
            frame (:data:`FLIGHT_FRAME_ID`), in metres.
        source_id: The record's own producing-run identifier.
        model_family: Provenance family, e.g. ``"swing_sim.flight"`` or
            ``"ud.flight_models"``.
        model_name: Provenance model name within that family.
        frame_id: The record's original declared wire frame.
    """

    label: str
    positions: np.ndarray
    source_id: str
    model_family: str
    model_name: str
    frame_id: str


def _flight_frame_positions(samples: Sequence[Any]) -> np.ndarray:
    """Identity conversion: the wire's flight frame is Shot Tracer's own."""
    return np.array([list(sample.position_m) for sample in samples], dtype=float)


_PLOT_FRAME_CONVERTERS: dict[str, Callable[[Sequence[Any]], np.ndarray]] = {
    FLIGHT_FRAME_ID: _flight_frame_positions,
}
"""Closed dispatch table: wire ``frame_id`` -> converter into the plot frame.

Every entry here is implemented and tested. A ``frame_id`` the wire
allows but this table does not list (currently
``app_xtarget_yup_zright``) is refused by :func:`import_trajectory_record`
rather than guessed at.
"""


def _load_vendored_reader() -> tuple[Callable[[str], Any], type[Exception]]:
    """Resolve ``vendor/ud-tools`` and import its trajectory reader.

    Uses the same canonical resolution facade as every other production
    consumer of vendored Tools code (env override, then the pinned
    ``vendor/ud-tools`` gitlink, then dev-mode sibling discovery).

    Returns:
        The vendored ``ball_flight_trajectory_from_json`` function and
        the vendored ``ContractViolationError`` class, so callers can
        catch exactly the exceptions that function raises.

    Raises:
        TrajectoryImportError: If the Tools checkout cannot be resolved,
            or the vendored ``flight_interchange`` package fails to
            import from it.
    """
    try:
        resolution = ensure_tools_importable(
            _REPO_ROOT, os.environ.get("TOOLS_REPO_PATH")
        )
    except RuntimeError as exc:
        raise TrajectoryImportError(str(exc)) from exc

    try:
        from shared.python.contracts import (  # type: ignore[import-not-found]
            ContractViolationError,
        )
        from shared.python.swing_sim.flight_interchange import (  # type: ignore[import-not-found]
            ball_flight_trajectory_from_json,
        )
    except ImportError as exc:
        raise TrajectoryImportError(
            f"vendored flight_interchange reader unavailable at {resolution.path}: "
            f"{exc}"
        ) from exc
    return ball_flight_trajectory_from_json, ContractViolationError


def import_trajectory_record(path: Path) -> ImportedTrajectoryCurve:
    """Read, validate, and plot-frame-convert one trajectory record file.

    Args:
        path: Path to a ``swing_sim.ball_flight_trajectory/1`` JSON file,
            produced by either flight-model family.

    Returns:
        The imported curve, labeled with its provenance and converted
        into Shot Tracer's plot frame.

    Raises:
        TrajectoryImportError: On any refusal: the vendored reader is
            unavailable, the file cannot be read, the record violates
            the wire contract (unknown/missing fields, malformed
            provenance, non-monotone samples, an undeclared frame — the
            reader's own message is surfaced verbatim), or the record
            declares a frame this function has not implemented.
    """
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")

    reader, contract_violation_error = _load_vendored_reader()

    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise TrajectoryImportError(f"cannot read {path}: {exc}") from exc

    try:
        record = reader(text)
    except (
        contract_violation_error,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as exc:
        raise TrajectoryImportError(str(exc)) from exc

    converter = _PLOT_FRAME_CONVERTERS.get(record.frame_id)
    if converter is None:
        raise TrajectoryImportError(
            f"unsupported frame {record.frame_id!r}: Shot Tracer's plot-frame "
            f"conversion is only implemented for {sorted(_PLOT_FRAME_CONVERTERS)!r}"
        )

    positions = converter(record.samples)
    provenance = record.provenance
    label = f"{provenance.model_family} / {provenance.model_name}"
    return ImportedTrajectoryCurve(
        label=label,
        positions=positions,
        source_id=record.source_id,
        model_family=provenance.model_family,
        model_name=provenance.model_name,
        frame_id=record.frame_id,
    )
