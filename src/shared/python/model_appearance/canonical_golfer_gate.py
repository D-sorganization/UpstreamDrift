"""Refuse a non-current golfer model before any simulation (#15659).

Torque histories do not transfer between golfer models (``skinned_hand_wrists``
moves the wrist joints, so the plant changes), so optimization (#15493),
allocation (#15494) and replay run on the registry's current model only; the
pin comes from ``src/config/golfer_model_versions.json``, never a constant.

A :class:`GolferModelLock` splits the model three ways so a mismatch names the
domain that moved: ``plant_hash`` (dynamics: the spec without visual keys,
visual backends or revision record), ``controller_hash`` (PD gains, spline
knots, control settings) and ``solver_hash`` (integrator and timestep).
Identity fields are checked separately. Allocation pins the plant only. A
Solution Bundle without a lock is ``legacy_replay_only`` and cannot seed work.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any, Literal

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.core.canonical_floats import canonical_floats
from src.shared.python.model_appearance.golfer_model_versions import (
    CurrentGolferModel,
    current_golfer_model,
    deprecated_parts,
)
from src.shared.python.model_appearance.schema import physics_spec_sha256
from src.shared.python.optimization.solution_bundle import (
    MODEL_LOCK_FIELDS,
    SolutionBundle,
    validate_model_lock,
)

Scope = Literal["optimization", "allocation", "replay"]
_HASHES = tuple(name for name in MODEL_LOCK_FIELDS if name.endswith("_hash"))
_IDENTITY = tuple(name for name in MODEL_LOCK_FIELDS if name not in _HASHES)
_SCOPES = {"allocation": ("plant_hash",), "optimization": _HASHES, "replay": _HASHES}
_VISUAL_SELECTORS = ("hand_visual_backend", "body_visual_backend")
LEGACY_REPLAY_ONLY = "legacy_replay_only"


class NonCanonicalGolferError(ValueError):
    """A golfer model, or a result's lock, is not the current canonical model.

    ``part`` names what is off: a deprecated value (``capsule``, ``loft``),
    ``spec_revision``, an identity field, a hash domain (``plant_hash``,
    ``controller_hash``, ``solver_hash``) or ``legacy_replay_only``.
    """

    def __init__(self, part: str, message: str) -> None:
        super().__init__(f"{part}: {message}")
        self.part = part


def _revision(spec: Mapping[str, Any]) -> str:
    revision = spec.get("spec_revision")
    if not isinstance(revision, Mapping):
        return ""
    return str(revision.get("configuration", ""))


def require_canonical_golfer(spec: Mapping[str, Any]) -> CurrentGolferModel:
    """Return the registry's current model if ``spec`` is on it.

    Raises:
        NonCanonicalGolferError: ``spec`` resolves to a deprecated hand or
            body backend (``part`` is the deprecated value), or it lacks the
            current spec revision (``part`` is ``spec_revision``).
    """
    current = current_golfer_model()
    for item in deprecated_parts(spec):
        raise NonCanonicalGolferError(
            item.value,
            f"{item.field} {item.value!r} is deprecated; the current golfer "
            f"model {current.version} uses {item.replacement!r}",
        )
    revision = _revision(spec)
    if revision != current.spec_revision_configuration:
        raise NonCanonicalGolferError(
            "spec_revision",
            f"spec revision {revision or 'none'!r} is not "
            f"{current.spec_revision_configuration!r}; make it with "
            f"{current.make_with}",
        )
    return current


def _sha256(payload: Any) -> str:
    text = json.dumps(canonical_floats(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def plant_hash(spec: Mapping[str, Any]) -> str:
    """Hash of the spec's dynamics: visual backends and revision record removed."""
    stripped = copy.deepcopy(dict(spec))
    stripped.pop("spec_revision", None)
    subject = stripped.get("subject")
    if isinstance(subject, dict):
        for key in _VISUAL_SELECTORS:
            subject.pop(key, None)
    return physics_spec_sha256(stripped)


def controller_hash(
    kp: ArrayLike, kd: ArrayLike, knots: ArrayLike, control: Mapping[str, Any]
) -> str:
    """Hash of the PD gains, the spline knots and the control settings."""
    return _sha256(
        {
            "kp": np.asarray(kp, dtype=float).tolist(),
            "kd": np.asarray(kd, dtype=float).tolist(),
            "knots": np.asarray(knots, dtype=float).tolist(),
            "control": dict(control),
        }
    )


def solver_hash(integrator: Mapping[str, Any]) -> str:
    """Hash of the integrator settings.

    Precondition: ``integrator`` names the method (``name``) and the
    timestep (``dt``); ``ValueError`` otherwise.
    """
    if not integrator.get("name") or "dt" not in integrator:
        raise ValueError("integrator settings must give a name and a dt")
    return _sha256(dict(integrator))


@dataclass(frozen=True)
class GolferModelLock:
    """Which golfer model, controller and solver a result was computed on."""

    registry_version: str
    hand_visual_backend: str
    body_visual_backend: str
    spec_revision: str
    plant_hash: str
    controller_hash: str
    solver_hash: str

    def to_dict(self) -> dict[str, str]:
        """Plain mapping for a JSON manifest."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> GolferModelLock:
        """Parse a lock with the bundle codec's field contract (one source)."""
        validate_model_lock(data)
        return cls(**{name: data[name] for name in MODEL_LOCK_FIELDS})


def lock_for_spec(
    spec: Mapping[str, Any], *, controller: str, solver: str
) -> GolferModelLock:
    """Lock for a result computed on ``spec``; refuses a non-canonical spec."""
    current = require_canonical_golfer(spec)
    return GolferModelLock(
        registry_version=current.version,
        hand_visual_backend=current.hand_visual_backend,
        body_visual_backend=current.body_visual_backend,
        spec_revision=_revision(spec),
        plant_hash=plant_hash(spec),
        controller_hash=controller,
        solver_hash=solver,
    )


def verify_lock(
    recorded: GolferModelLock, expected: GolferModelLock, *, scope: Scope
) -> None:
    """Raise unless ``recorded`` matches ``expected`` for ``scope``.

    Identity is always compared; allocation pins ``plant_hash`` only,
    optimization and replay pin all three hashes.

    Raises:
        ValueError: unknown ``scope``.
        NonCanonicalGolferError: ``part`` is the first mismatched field.
    """
    if scope not in _SCOPES:
        raise ValueError(f"unknown scope {scope!r}; use one of {sorted(_SCOPES)}")
    for name in _IDENTITY + _SCOPES[scope]:
        have, want = getattr(recorded, name), getattr(expected, name)
        if have != want:
            raise NonCanonicalGolferError(
                name, f"recorded {have!r} does not match {want!r} ({scope})"
            )


def bundle_lock_status(bundle: SolutionBundle) -> str:
    """``locked`` if the bundle records a model lock, else ``legacy_replay_only``."""
    return "locked" if bundle.model_lock is not None else LEGACY_REPLAY_ONLY


def require_locked_bundle(
    bundle: SolutionBundle, expected: GolferModelLock, *, scope: Scope
) -> GolferModelLock:
    """The bundle's lock, checked against the bundle's own data and ``expected``.

    Raises ``NonCanonicalGolferError`` for no lock (``legacy_replay_only``), a
    lock whose controller/solver hashes do not match the bundle's gains, knots,
    control and integrator, or a lock that differs from ``expected``.
    """
    if bundle.model_lock is None:
        raise NonCanonicalGolferError(
            LEGACY_REPLAY_ONLY,
            f"bundle {bundle.solution_id} has no model lock; it may only be "
            "replayed under its recorded model, not used for new work",
        )
    recorded = GolferModelLock.from_dict(bundle.model_lock)
    own = {
        "controller_hash": controller_hash(
            bundle.arrays["kp"],
            bundle.arrays["kd"],
            bundle.arrays["knots"],
            bundle.control,
        ),
        "solver_hash": solver_hash(bundle.integrator),
    }
    for name, value in own.items():
        if getattr(recorded, name) != value:
            raise NonCanonicalGolferError(
                name, f"bundle {bundle.solution_id} lock does not match its data"
            )
    verify_lock(recorded, expected, scope=scope)
    return recorded
