"""Simscape closed-loop snapshot export format and loader (NSX-55, #12583).

A MATLAB R2025b exporter (``simscape_closed_loop_export.m``) writes a JSON file
of schema ``simscape-closed-loop-snapshot/v1``; this module reads it, validates
it with the NSX-10 contracts and decides which capability the file can support.

The decision is deliberately conservative (AGENTS.md "Required MATLAB
Release"): ``CLOSED_LOOP_SNAPSHOT`` is declared ``FULL`` only for a file whose
``provenance_kind`` is ``"matlab_export"`` and whose ``matlab_release`` is
exactly ``"R2025b"``.  A ``"synthetic_test_fixture"`` (serialised from the
analytic planar model, never from MATLAB) and any other MATLAB release,
including R2026a, are capability ``NONE``; their data may only be turned into a
snapshot with ``test_only=True``, which tags the model provenance accordingly.

File layout (all SI, canonical grip-frame closure rows ``[fx fy fz tx ty tz]``
per ``docs/research/loop_nullspace/contracts.md``)::

    schema        "simscape-closed-loop-snapshot/v1"
    provenance    provenance_kind, matlab_release, matlab_version, model_file,
                  model_sha256, exported_at, exporter, note
    topology      closure_kind, closing_side, holding_side, coordinate_order,
                  actuated_indices, unactuated_indices, loop_support_indices
    inertias      bare_club {mass, com, inertia}, welded_hand_inertias [...]
    states[]      q, v, a, M, h, J_c, Jdot_v, grip_frame {rotation, origin}

No MATLAB, Simulink or other physics engine is imported.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.engine_core.capabilities import (
    Capability,
    CapabilityLevel,
    CapabilityUnavailable,
)
from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    ClosureKind,
    GripFrame,
    InputProvenance,
    LoopTopology,
    ModelProvenance,
    SpatialInertia,
)
from src.shared.python.loop_nullspace.evidence import Array

__all__ = [
    "ACCEPTANCE_RELEASE",
    "SCHEMA_ID",
    "SimscapeExport",
    "SimscapeProvenance",
    "SimscapeState",
    "export_from_snapshot",
    "export_to_dict",
    "load_simscape_export",
    "parse_simscape_export",
    "snapshot_from_export",
    "verify_model_file",
]

SCHEMA_ID = "simscape-closed-loop-snapshot/v1"
ACCEPTANCE_RELEASE = "R2025b"
_KIND_MATLAB = "matlab_export"
_KIND_SYNTHETIC = "synthetic_test_fixture"
_KINDS = (_KIND_MATLAB, _KIND_SYNTHETIC)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_MATLAB_REQUIRED = ("matlab_release", "matlab_version", "model_file", "model_sha256")
_SYNTHETIC_FORBIDDEN = ("matlab_release", "matlab_version", "model_sha256")
_STATE_KEYS = ("q", "v", "a", "M", "h", "J_c", "Jdot_v")


@dataclass(frozen=True)
class SimscapeProvenance:
    """Where an export came from; the evidence the capability decision uses."""

    provenance_kind: str
    matlab_release: str | None
    matlab_version: str | None
    model_file: str | None
    model_sha256: str | None
    exported_at: str | None
    exporter: str = ""
    note: str = ""

    def __post_init__(self) -> None:
        if self.provenance_kind not in _KINDS:
            raise ValueError(
                f"provenance_kind must be one of {_KINDS}, got {self.provenance_kind!r}"
            )
        if self.provenance_kind == _KIND_MATLAB:
            self._check_matlab()
        else:
            for key in _SYNTHETIC_FORBIDDEN:
                if getattr(self, key) is not None:
                    raise ValueError(
                        f"a synthetic_test_fixture must not carry {key} "
                        "(it was not produced by MATLAB)"
                    )

    def _check_matlab(self) -> None:
        for key in (*_MATLAB_REQUIRED, "exported_at"):
            value = getattr(self, key)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"matlab_export provenance requires {key}")
        if not _SHA256.fullmatch(str(self.model_sha256)):
            raise ValueError("model_sha256 must be 64 lowercase hex characters")
        if f"({self.matlab_release})" not in str(self.matlab_version):
            raise ValueError(
                f"matlab_version {self.matlab_version!r} does not name the "
                f"claimed matlab_release {self.matlab_release!r}"
            )


@dataclass(frozen=True)
class SimscapeState:
    """One exported dynamic state (arrays already contract-validated)."""

    q: Array
    v: Array
    a: Array
    M: Array
    h: Array
    J_c: Array
    Jdot_v: Array
    grip_rotation: Array
    grip_origin: Array


@dataclass(frozen=True)
class SimscapeExport:
    """A validated Simscape export; see the module docstring for the layout."""

    schema: str
    provenance: SimscapeProvenance
    topology: LoopTopology
    bare_club: SpatialInertia
    welded_hand_inertias: tuple[SpatialInertia, ...]
    states: tuple[SimscapeState, ...]

    def capability_reason(self) -> str:
        """Why :meth:`capability` is what it is (empty-evidence safe)."""
        prov = self.provenance
        if prov.provenance_kind != _KIND_MATLAB:
            return (
                f"provenance_kind={prov.provenance_kind!r} is test-only, not "
                "a MATLAB export"
            )
        if prov.matlab_release != ACCEPTANCE_RELEASE:
            return (
                f"matlab_release={prov.matlab_release!r} is not acceptance "
                f"evidence; only {ACCEPTANCE_RELEASE} counts"
            )
        return f"matlab_export from {ACCEPTANCE_RELEASE}"

    def capability(self) -> CapabilityLevel:
        """``FULL`` only for a ``matlab_export`` from R2025b, else ``NONE``."""
        prov = self.provenance
        ok = (
            prov.provenance_kind == _KIND_MATLAB
            and prov.matlab_release == ACCEPTANCE_RELEASE
        )
        return CapabilityLevel.FULL if ok else CapabilityLevel.NONE

    def declared_capability(self, capability: Capability) -> CapabilityLevel:
        """Capability query: only ``CLOSED_LOOP_SNAPSHOT`` can be non-``NONE``."""
        if capability is Capability.CLOSED_LOOP_SNAPSHOT:
            return self.capability()
        return CapabilityLevel.NONE


# ------------------------------------------------------------------ parsing --


def _require_mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _require_key(doc: dict[str, Any], key: str, where: str) -> Any:
    if key not in doc:
        raise ValueError(f"{where} is missing required field {key!r}")
    return doc[key]


def _parse_provenance(raw: Any) -> SimscapeProvenance:
    doc = _require_mapping(raw, "provenance")
    kind = _require_key(doc, "provenance_kind", "provenance")
    return SimscapeProvenance(
        provenance_kind=kind,
        matlab_release=doc.get("matlab_release"),
        matlab_version=doc.get("matlab_version"),
        model_file=doc.get("model_file"),
        model_sha256=doc.get("model_sha256"),
        exported_at=doc.get("exported_at"),
        exporter=str(doc.get("exporter", "")),
        note=str(doc.get("note", "")),
    )


def _parse_topology(raw: Any) -> LoopTopology:
    doc = _require_mapping(raw, "topology")
    try:
        kind = ClosureKind(_require_key(doc, "closure_kind", "topology"))
    except ValueError as exc:
        raise ValueError(f"topology.closure_kind is not a ClosureKind: {exc}") from exc
    return LoopTopology(
        closure_kind=kind,
        closing_side=str(_require_key(doc, "closing_side", "topology")),
        holding_side=str(_require_key(doc, "holding_side", "topology")),
        actuated_indices=tuple(
            int(i) for i in _require_key(doc, "actuated_indices", "topology")
        ),
        unactuated_indices=tuple(
            int(i) for i in _require_key(doc, "unactuated_indices", "topology")
        ),
        loop_support_indices=tuple(
            int(i) for i in _require_key(doc, "loop_support_indices", "topology")
        ),
        coordinate_order=tuple(
            str(c) for c in _require_key(doc, "coordinate_order", "topology")
        ),
    )


def _parse_inertia(raw: Any, name: str) -> SpatialInertia:
    doc = _require_mapping(raw, name)
    return SpatialInertia(
        mass=float(_require_key(doc, "mass", name)),
        com=np.asarray(_require_key(doc, "com", name), dtype=np.float64),
        inertia=np.asarray(_require_key(doc, "inertia", name), dtype=np.float64),
    )


def _parse_state(raw: Any, index: int) -> SimscapeState:
    where = f"states[{index}]"
    doc = _require_mapping(raw, where)
    arrays = {
        key: np.array(_require_key(doc, key, where), dtype=np.float64)
        for key in _STATE_KEYS
    }
    grip = _require_mapping(_require_key(doc, "grip_frame", where), "grip_frame")
    return SimscapeState(
        grip_rotation=np.array(_require_key(grip, "rotation", where), dtype=np.float64),
        grip_origin=np.array(_require_key(grip, "origin", where), dtype=np.float64),
        **arrays,
    )


def _state_snapshot(
    state: SimscapeState,
    export: SimscapeExport,
    model: ModelProvenance,
    source: InputProvenance,
) -> ClosedLoopSnapshot:
    topo = export.topology
    return ClosedLoopSnapshot(
        q=state.q,
        v=state.v,
        a=state.a,
        M=state.M,
        h=state.h,
        S=np.eye(topo.n)[list(topo.actuated_indices)],
        J_c=state.J_c,
        Jdot_v=state.Jdot_v,
        grip_frame=GripFrame(state.grip_rotation, state.grip_origin),
        bare_club=export.bare_club,
        welded_hand_inertias=export.welded_hand_inertias,
        model_provenance=model,
        input_provenance=source,
        topology=topo,
    )


def parse_simscape_export(doc: Any) -> SimscapeExport:
    """Validate a decoded export document and return a :class:`SimscapeExport`.

    Preconditions are checked, never assumed: the schema id, provenance rules
    (a ``matlab_export`` needs release, version string naming that release,
    model file, lowercase SHA-256 and timestamp; a ``synthetic_test_fixture``
    must carry no MATLAB identity), the topology, and every state through the
    NSX-10 snapshot contracts (finite values, shapes, symmetric positive
    definite ``M``).

    Raises:
        ValueError: On any violation, naming the offending field.
    """
    root = _require_mapping(doc, "export document")
    if root.get("schema") != SCHEMA_ID:
        raise ValueError(f"schema must be {SCHEMA_ID!r}, got {root.get('schema')!r}")
    inertias = _require_mapping(_require_key(root, "inertias", "export"), "inertias")
    raw_states = _require_key(root, "states", "export")
    if not isinstance(raw_states, list) or not raw_states:
        raise ValueError("states must be a non-empty list")
    export = SimscapeExport(
        schema=SCHEMA_ID,
        provenance=_parse_provenance(_require_key(root, "provenance", "export")),
        topology=_parse_topology(_require_key(root, "topology", "export")),
        bare_club=_parse_inertia(
            _require_key(inertias, "bare_club", "inertias"), "bare_club"
        ),
        welded_hand_inertias=tuple(
            _parse_inertia(item, f"welded_hand_inertias[{k}]")
            for k, item in enumerate(inertias.get("welded_hand_inertias", []))
        ),
        states=tuple(_parse_state(s, k) for k, s in enumerate(raw_states)),
    )
    probe = ModelProvenance("simscape", "validation")
    for state in export.states:
        _state_snapshot(state, export, probe, InputProvenance("validation", "load"))
    return export


def load_simscape_export(path: str | Path) -> SimscapeExport:
    """Read and validate an export file (see :func:`parse_simscape_export`).

    Raises:
        ValueError: If the file is not valid JSON or fails validation.
    """
    text = Path(path).read_text(encoding="utf-8")
    try:
        doc = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path} is not valid JSON: {exc}") from exc
    return parse_simscape_export(doc)


# ---------------------------------------------------------------- snapshots --


def snapshot_from_export(
    export: SimscapeExport,
    index: int,
    *,
    input_provenance: InputProvenance,
    test_only: bool = False,
) -> ClosedLoopSnapshot:
    """Build the validated snapshot of state ``index`` of ``export``.

    Without ``test_only`` this refuses unless ``export.capability()`` is
    ``FULL`` (a ``matlab_export`` from R2025b).  With ``test_only=True`` any
    valid export may be turned into a snapshot for software tests; the model
    provenance version is then suffixed ``test_only:<provenance_kind>`` so the
    snapshot can never be mistaken for acceptance evidence.

    Raises:
        CapabilityUnavailable: Capability is not ``FULL`` and ``test_only`` unset.
        ValueError: ``index`` is out of range.
    """
    if not 0 <= index < len(export.states):
        raise ValueError(
            f"state index {index} out of range 0..{len(export.states) - 1}"
        )
    level = export.capability()
    if level is not CapabilityLevel.FULL and not test_only:
        raise CapabilityUnavailable(
            f"{Capability.CLOSED_LOOP_SNAPSHOT.value} requires CapabilityLevel.FULL; "
            f"simscape export declares {level.name}: {export.capability_reason()}"
        )
    prov = export.provenance
    version = prov.matlab_version or "none"
    if level is not CapabilityLevel.FULL:
        version = f"test_only:{prov.provenance_kind}"
    model = ModelProvenance("simscape", version, prov.model_sha256 or "")
    return _state_snapshot(export.states[index], export, model, input_provenance)


def verify_model_file(export: SimscapeExport, path: str | Path) -> None:
    """Check that the file at ``path`` has the SHA-256 the export recorded.

    Raises:
        ValueError: If the export records no model (synthetic) or the hash
            differs from the file's actual SHA-256.
    """
    expected = export.provenance.model_sha256
    if expected is None:
        raise ValueError("this export records no model file hash to verify")
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != expected:
        raise ValueError(
            f"model_sha256 mismatch: export records {expected}, {path} is {digest}"
        )


# ------------------------------------------------------------------ writing --


def _inertia_dict(item: SpatialInertia) -> dict[str, Any]:
    return {
        "mass": item.mass,
        "com": item.com.tolist(),
        "inertia": item.inertia.tolist(),
    }


def export_to_dict(export: SimscapeExport) -> dict[str, Any]:
    """Serialise ``export`` to the JSON document layout (inverse of parsing)."""
    prov, topo = export.provenance, export.topology
    return {
        "schema": export.schema,
        "provenance": {
            "provenance_kind": prov.provenance_kind,
            "matlab_release": prov.matlab_release,
            "matlab_version": prov.matlab_version,
            "model_file": prov.model_file,
            "model_sha256": prov.model_sha256,
            "exported_at": prov.exported_at,
            "exporter": prov.exporter,
            "note": prov.note,
        },
        "topology": {
            "closure_kind": topo.closure_kind.value,
            "closing_side": topo.closing_side,
            "holding_side": topo.holding_side,
            "coordinate_order": list(topo.coordinate_order),
            "actuated_indices": list(topo.actuated_indices),
            "unactuated_indices": list(topo.unactuated_indices),
            "loop_support_indices": list(topo.loop_support_indices),
        },
        "inertias": {
            "bare_club": _inertia_dict(export.bare_club),
            "welded_hand_inertias": [
                _inertia_dict(i) for i in export.welded_hand_inertias
            ],
        },
        "states": [
            {
                **{key: getattr(s, key).tolist() for key in _STATE_KEYS},
                "grip_frame": {
                    "rotation": s.grip_rotation.tolist(),
                    "origin": s.grip_origin.tolist(),
                },
            }
            for s in export.states
        ],
    }


def export_from_snapshot(
    snapshot: ClosedLoopSnapshot,
    *,
    note: str = "Serialised from the analytic planar golfer; not a MATLAB export.",
) -> SimscapeExport:
    """Wrap a snapshot as a ``synthetic_test_fixture`` export (test-only).

    The result carries no MATLAB identity and always has capability ``NONE``.
    """
    state = SimscapeState(
        q=snapshot.q,
        v=snapshot.v,
        a=snapshot.a,
        M=snapshot.M,
        h=snapshot.h,
        J_c=snapshot.J_c,
        Jdot_v=snapshot.Jdot_v,
        grip_rotation=snapshot.grip_frame.rotation,
        grip_origin=snapshot.grip_frame.origin,
    )
    return SimscapeExport(
        schema=SCHEMA_ID,
        provenance=SimscapeProvenance(
            provenance_kind=_KIND_SYNTHETIC,
            matlab_release=None,
            matlab_version=None,
            model_file=None,
            model_sha256=None,
            exported_at=None,
            exporter="src.shared.python.loop_nullspace.simscape_export",
            note=note,
        ),
        topology=snapshot.topology,
        bare_club=snapshot.bare_club,
        welded_hand_inertias=snapshot.welded_hand_inertias,
        states=(state,),
    )
