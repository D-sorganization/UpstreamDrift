"""Simscape per-frame ``linearize`` payload gate (NSX-89, #12718).

The NSX-55 export (:mod:`simscape_export`) is a snapshot grid: it cannot be
re-evaluated at perturbed states, so the closed-loop derivative capability of
Simscape is ``NONE`` (see :mod:`derivative_capability`).  The deferred path,
recorded in ``DV-12718``, is a MATLAB R2025b export of per-frame ``linearize``
``A`` / ``B`` matrices.  This module defines that payload
(``simscape-trajectory-linearization/v1``) and the loader that decides whether
a file may even be read as evidence: it **refuses** anything that is not a
golf-model ``matlab_export`` from exactly R2025b (R2026a, older releases, the
verification rig and synthetic fixtures are all refused), and anything lacking
the solver settings and state mapping that make ``A``/``B`` reproducible.

Accepting a payload here does not flip the registry: the capability stays
``NONE`` until the record's acceptance evidence exists.  No MATLAB is imported.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.evidence import Array, readonly_float
from src.shared.python.loop_nullspace.simscape_export import (
    ACCEPTANCE_RELEASE,
    SCOPE_GOLF,
    SimscapeProvenance,
    _parse_provenance,
    _require_key,
    _require_mapping,
)

__all__ = [
    "LINEARIZATION_SCHEMA_ID",
    "LinearizationFrame",
    "SimscapeLinearization",
    "load_simscape_linearization_doc",
    "parse_simscape_linearization",
]

LINEARIZATION_SCHEMA_ID = "simscape-trajectory-linearization/v1"
_MAPPING_KEYS = ("q_names", "v_names", "input_names")


@dataclass(frozen=True)
class LinearizationFrame:
    """``dx = A dx + B du`` at one state, ``x = (q, v)``."""

    q: Array
    v: Array
    tau: Array
    A: Array
    B: Array


@dataclass(frozen=True)
class SimscapeLinearization:
    """A validated R2025b linearization payload (provenance already gated)."""

    provenance: SimscapeProvenance
    solver_settings: dict[str, Any]
    state_mapping: dict[str, tuple[str, ...]]
    frames: tuple[LinearizationFrame, ...]


def _require_release(prov: SimscapeProvenance) -> None:
    if prov.provenance_kind != "matlab_export":
        raise ValueError(
            "a linearization payload must be a matlab_export, got "
            f"provenance_kind={prov.provenance_kind!r}"
        )
    if prov.model_scope != SCOPE_GOLF:
        raise ValueError(
            f"a linearization payload must have model_scope {SCOPE_GOLF!r}, "
            f"got {prov.model_scope!r}"
        )
    if prov.matlab_release != ACCEPTANCE_RELEASE:
        raise ValueError(
            f"matlab_release={prov.matlab_release!r} is refused; only "
            f"{ACCEPTANCE_RELEASE} is acceptance evidence"
        )


def _parse_mapping(raw: Any) -> dict[str, tuple[str, ...]]:
    doc = _require_mapping(raw, "state_mapping")
    out: dict[str, tuple[str, ...]] = {}
    for key in _MAPPING_KEYS:
        names = _require_key(doc, key, "state_mapping")
        if not isinstance(names, list) or not all(isinstance(n, str) for n in names):
            raise ValueError(f"state_mapping.{key} must be a list of strings")
        out[key] = tuple(names)
    if len(out["q_names"]) != len(out["v_names"]):
        raise ValueError("state_mapping q_names and v_names must have equal length")
    return out


def _parse_frame(raw: Any, index: int, mapping: dict[str, tuple[str, ...]]) -> Any:
    doc = _require_mapping(raw, f"frames[{index}]")
    nq, nu = len(mapping["q_names"]), len(mapping["input_names"])
    shapes = {"q": (nq,), "v": (nq,), "tau": (nu,), "A": (2 * nq, 2 * nq)}
    shapes["B"] = (2 * nq, nu)
    arrays = {}
    for key, shape in shapes.items():
        value = readonly_float(_require_key(doc, key, f"frames[{index}]"), key)
        if value.shape != shape:
            raise ValueError(
                f"frames[{index}].{key} must have shape {shape}, got {value.shape}"
            )
        arrays[key] = value
    return LinearizationFrame(**arrays)


def parse_simscape_linearization(doc: Any) -> SimscapeLinearization:
    """Validate a ``simscape-trajectory-linearization/v1`` document.

    Raises:
        ValueError: Wrong schema, a missing ``solver_settings`` /
            ``state_mapping`` / ``frames``, a matrix whose shape disagrees with
            the state mapping, or provenance that is not a golf-model
            ``matlab_export`` from exactly R2025b (R2026a is refused).
    """
    top = _require_mapping(doc, "linearization")
    if top.get("schema") != LINEARIZATION_SCHEMA_ID:
        raise ValueError(f"schema must be {LINEARIZATION_SCHEMA_ID!r}")
    prov = _parse_provenance(_require_key(top, "provenance", "linearization"))
    _require_release(prov)
    solver = dict(
        _require_mapping(
            _require_key(top, "solver_settings", "linearization"), "solver_settings"
        )
    )
    if not solver:
        raise ValueError("solver_settings must not be empty")
    mapping = _parse_mapping(_require_key(top, "state_mapping", "linearization"))
    raw_frames = _require_key(top, "frames", "linearization")
    if not isinstance(raw_frames, list) or not raw_frames:
        raise ValueError("frames must be a non-empty list")
    frames = tuple(_parse_frame(f, i, mapping) for i, f in enumerate(raw_frames))
    return SimscapeLinearization(prov, solver, mapping, frames)


def load_simscape_linearization_doc(path: str | Path) -> SimscapeLinearization:
    """Read and validate a linearization file.

    Raises:
        ValueError: Not valid JSON, or fails :func:`parse_simscape_linearization`.
    """
    try:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    return parse_simscape_linearization(doc)
