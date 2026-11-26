"""Versioned, self-contained Solution Bundle schema and codec.

Part of epic #15493 (Slice 2). A bundle stores one optimized swing solution so
it can be replayed in forward dynamics later. Persistence is a directory with
two files:

- ``manifest.json`` -- strict JSON (no ``NaN``): identity, hashes, conventions,
  controller/integrator settings, impact event, metrics and assumptions.
- ``arrays.npz`` -- the numeric arrays (reference, rollout, torques, spline).
  Loaded with ``allow_pickle=False`` and pinned by ``arrays_sha256``.

Markers are never the replay source; they stay a visualisation export. The
codec fails closed on unsupported versions, incompatible models, non-finite
values, non-monotonic times and dimension mismatches. Recorded playback and
forward re-simulation are separate entry points that consume this codec.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.optimization.admission import validate_admission

__all__ = [
    "SCHEMA_VERSION",
    "SUPPORTED_SCHEMA_VERSIONS",
    "SolutionBundle",
    "SolutionBundleError",
    "MODEL_LOCK_FIELDS",
    "validate_model_lock",
    "load_solution_bundle",
    "save_solution_bundle",
]

SCHEMA_VERSION = 1
# Older versions get an explicit migration here before they may be listed.
SUPPORTED_SCHEMA_VERSIONS = frozenset({SCHEMA_VERSION})

MANIFEST_NAME = "manifest.json"
ARRAYS_NAME = "arrays.npz"
UNITS = "SI"
QUATERNION_CONVENTION = "wxyz"

# name -> shape template; "n" = time samples, "j" = joints, "c" = spline
# coefficients, "k" = knots, "s" = controller state size (any, may be zero).
_ARRAY_SHAPES: dict[str, tuple[str, ...]] = {
    "time": ("n",),
    "q_ref": ("n", "j"),
    "qd_ref": ("n", "j"),
    "tau_ff_ref": ("n", "j"),
    "q_rollout": ("n", "j"),
    "qd_rollout": ("n", "j"),
    "tau_cmd": ("n", "j"),
    "tau_applied": ("n", "j"),
    "knots": ("k",),
    "delta_tau_coeffs": ("c", "j"),
    "q0": ("j",),
    "qd0": ("j",),
    "controller_state0": ("s",),
    "kp": ("j",),
    "kd": ("j",),
}


class SolutionBundleError(ValueError):
    """Raised when a bundle is malformed, incompatible or fails integrity."""


@dataclass(frozen=True)
class SolutionBundle:
    """One optimized solution plus everything needed to replay it."""

    solution_id: str
    parent_id: str | None
    git_sha: str
    model_hash: str
    asset_hashes: dict[str, str]
    frame: str
    joint_order: tuple[str, ...]
    control: dict[str, Any]
    integrator: dict[str, Any]
    environment: dict[str, Any]
    optimizer_seed: int
    impact_event: dict[str, Any]
    flight_metrics: dict[str, float]
    feasibility_margins: dict[str, float]
    assumptions: tuple[str, ...]
    arrays: dict[str, np.ndarray] = field(compare=False)
    schema_version: int = SCHEMA_VERSION
    # Golfer model lock (#15659): which model, controller and solver produced
    # the bundle. ``None`` marks a legacy bundle that may only be replayed.
    model_lock: dict[str, str] | None = None
    # #15808 WP-0c. ``None`` is a pre-admission recording.
    admission: dict[str, Any] | None = None


#: Fields a model lock must give as non-empty strings (#15659).
MODEL_LOCK_FIELDS = (
    "registry_version",
    "hand_visual_backend",
    "body_visual_backend",
    "spec_revision",
    "plant_hash",
    "controller_hash",
    "solver_hash",
)


def validate_model_lock(lock: object) -> None:
    """Raise unless ``lock`` is ``None`` or exactly the lock fields as strings."""
    if lock is None:
        return
    if not isinstance(lock, dict):
        raise SolutionBundleError("model_lock must be an object or null")
    bad = [
        k for k in MODEL_LOCK_FIELDS if not isinstance(lock.get(k), str) or not lock[k]
    ]
    extra = sorted(set(lock) - set(MODEL_LOCK_FIELDS))
    if bad or extra:
        raise SolutionBundleError(
            f"model_lock needs non-empty {bad} and no unknown keys {extra}"
        )


def _validate_arrays(bundle: SolutionBundle) -> None:
    """Require the recorded arrays to share one finite time base."""
    arrays = bundle.arrays
    missing = sorted(set(_ARRAY_SHAPES) - set(arrays))
    if missing:
        raise SolutionBundleError(f"missing arrays: {', '.join(missing)}")
    dims: dict[str, int] = {"j": len(bundle.joint_order)}
    for name, template in _ARRAY_SHAPES.items():
        value = np.asarray(arrays[name])
        if value.dtype.kind not in "fiu":
            raise SolutionBundleError(f"{name} must be numeric, got {value.dtype}")
        if value.ndim != len(template):
            raise SolutionBundleError(
                f"{name} has shape {value.shape}, expected axes {template}"
            )
        for axis, symbol in enumerate(template):
            if symbol == "s":
                continue
            expected = dims.setdefault(symbol, value.shape[axis])
            if value.shape[axis] != expected:
                raise SolutionBundleError(
                    f"{name} has shape {value.shape}: axis {axis} ({symbol}) "
                    f"is {value.shape[axis]}, expected {expected}"
                )
        if not np.all(np.isfinite(value)):
            raise SolutionBundleError(f"{name} contains non-finite values")
    # Compare neighbours directly: np.diff wraps for unsigned/extreme integers.
    time, knots = np.asarray(arrays["time"]), np.asarray(arrays["knots"])
    if time.size < 2:
        raise SolutionBundleError("time must contain at least two samples")
    if np.any(time[1:] <= time[:-1]):
        raise SolutionBundleError("time must be strictly increasing")
    if not time[0] <= bundle.impact_event["t"] <= time[-1]:
        raise SolutionBundleError(
            "impact_event.t must lie within the recorded time interval"
        )
    if np.any(knots[1:] < knots[:-1]):
        raise SolutionBundleError("knots must be nondecreasing")


def _is_finite_number(value: object) -> bool:
    """Return whether ``value`` is a finite int or float, excluding bool."""
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and math.isfinite(value)
    )


def _require_finite(value: object, where: str) -> None:
    """Reject any non-finite float nested in ``value`` (NaN, inf, overflow)."""
    if isinstance(value, dict):
        for key, item in value.items():
            _require_finite(item, f"{where}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_finite(item, f"{where}[{index}]")
    elif isinstance(value, float) and not math.isfinite(value):
        raise SolutionBundleError(f"{where} must be finite, got {value}")


def _validate_metadata(bundle: SolutionBundle) -> None:
    """Require identity, impact, and manifest fields to be finite and complete."""
    if bundle.schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SolutionBundleError(f"unsupported schema_version {bundle.schema_version}")
    names = bundle.joint_order
    if not names or len(set(names)) != len(names):
        raise SolutionBundleError("joint_order must be non-empty with unique names")
    for key in ("solution_id", "git_sha", "model_hash", "frame"):
        if not getattr(bundle, key):
            raise SolutionBundleError(f"{key} must be non-empty")
    validate_model_lock(bundle.model_lock)
    event = bundle.impact_event
    if not _is_finite_number(event.get("t")):
        raise SolutionBundleError("impact_event.t must be a finite number")
    for key in ("q", "qd"):
        values = event.get(key)
        if not isinstance(values, (list, tuple)) or len(values) != len(names):
            raise SolutionBundleError(
                f"impact_event.{key} must have {len(names)} entries"
            )
        if not all(_is_finite_number(v) for v in values):
            raise SolutionBundleError(f"impact_event.{key} must be finite numbers")
    # The whole manifest, not a hand-picked subset: every serialised field must
    # satisfy the strict-JSON contract the writer enforces.
    _require_finite(_manifest(bundle, "0" * 64), "manifest")


def _reject_json_constant(name: str) -> object:
    """``json.loads`` hook: refuse the non-standard NaN/Infinity tokens."""
    raise SolutionBundleError(f"{MANIFEST_NAME} contains non-finite constant {name}")


def _manifest(bundle: SolutionBundle, digest: str) -> dict[str, Any]:
    """Build the strict JSON manifest, using ``digest`` as the array hash."""
    return {
        "schema_version": bundle.schema_version,
        "solution_id": bundle.solution_id,
        "parent_id": bundle.parent_id,
        "git_sha": bundle.git_sha,
        "model_hash": bundle.model_hash,
        "asset_hashes": bundle.asset_hashes,
        "units": UNITS,
        "quaternion_convention": QUATERNION_CONVENTION,
        "frame": bundle.frame,
        "joint_order": list(bundle.joint_order),
        "control": bundle.control,
        "integrator": bundle.integrator,
        "environment": bundle.environment,
        "optimizer_seed": bundle.optimizer_seed,
        "impact_event": bundle.impact_event,
        "flight_metrics": bundle.flight_metrics,
        "feasibility_margins": bundle.feasibility_margins,
        "assumptions": list(bundle.assumptions),
        "arrays_sha256": digest,
        "model_lock": bundle.model_lock,
        "admission": bundle.admission,
    }


def _validate_admission_block(bundle: SolutionBundle) -> None:
    """Check the optional WP-0c admission block against the recorded arrays."""
    validate_admission(
        bundle.admission,
        time=np.asarray(bundle.arrays["time"]),
        n_joints=len(bundle.joint_order),
        arrays=bundle.arrays,
    )


def save_solution_bundle(bundle: SolutionBundle, directory: str | Path) -> Path:
    """Validate ``bundle`` and write it to a new ``directory``.

    The recorded timebase must span at least one interval and contain the
    impact time, including endpoints. This establishes temporal support only;
    it does not qualify a contact root or forward-dynamics replay.

    Raises:
        SolutionBundleError: invalid bundle, or ``directory`` already exists.
    """
    target = Path(directory)
    _validate_metadata(bundle)
    _validate_arrays(bundle)
    _validate_admission_block(bundle)
    if target.exists():
        raise SolutionBundleError(f"{target} already exists; bundles are immutable")
    # Serialise the manifest before touching disk so a bad value leaves nothing.
    try:
        text = json.dumps(
            _manifest(bundle, "0" * 64), indent=2, sort_keys=True, allow_nan=False
        )
    except (TypeError, ValueError) as exc:
        raise SolutionBundleError(f"cannot serialise manifest: {exc}") from exc
    target.mkdir(parents=True)
    arrays_path = target / ARRAYS_NAME
    payload: dict[str, Any] = {k: np.asarray(v) for k, v in bundle.arrays.items()}
    np.savez(arrays_path, **payload)
    digest = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    text = text.replace("0" * 64, digest, 1)
    (target / MANIFEST_NAME).write_text(text + "\n", encoding="utf-8")
    return target


def load_solution_bundle(
    directory: str | Path, *, expected_model_hash: str | None = None
) -> SolutionBundle:
    """Read, verify and validate a bundle directory.

    Args:
        directory: Folder holding ``manifest.json`` and ``arrays.npz``.
        expected_model_hash: When given, the bundle must have been recorded
            against this model.

    Raises:
        SolutionBundleError: missing files, unsupported version, model
            mismatch, hash mismatch, or any validation failure.
    """
    root = Path(directory)
    manifest_path, arrays_path = root / MANIFEST_NAME, root / ARRAYS_NAME
    for path in (manifest_path, arrays_path):
        if not path.is_file():
            raise SolutionBundleError(f"missing {path.name} in {root}")
    try:
        manifest = json.loads(
            manifest_path.read_text(encoding="utf-8"),
            parse_constant=_reject_json_constant,
        )
    except ValueError as exc:  # JSONDecodeError, oversized int literals
        if isinstance(exc, SolutionBundleError):
            raise
        raise SolutionBundleError(f"{MANIFEST_NAME} is not valid JSON: {exc}") from exc
    if not isinstance(manifest, dict):
        raise SolutionBundleError(f"{MANIFEST_NAME} must contain a JSON object")
    # Overflowed literals such as 1e999 parse to inf: check before any use.
    _require_finite(manifest, "manifest")
    version = manifest.get("schema_version")
    if version not in SUPPORTED_SCHEMA_VERSIONS:
        raise SolutionBundleError(f"unsupported schema_version {version!r}")
    if manifest.get("units") != UNITS:
        raise SolutionBundleError(f"unsupported units {manifest.get('units')!r}")
    if manifest.get("quaternion_convention") != QUATERNION_CONVENTION:
        raise SolutionBundleError("unsupported quaternion_convention")
    if (
        expected_model_hash is not None
        and manifest.get("model_hash") != expected_model_hash
    ):
        raise SolutionBundleError("model_hash does not match the expected model")
    actual = hashlib.sha256(arrays_path.read_bytes()).hexdigest()
    if actual != manifest.get("arrays_sha256"):
        raise SolutionBundleError("arrays_sha256 mismatch: arrays.npz was modified")
    with np.load(arrays_path, allow_pickle=False) as data:
        arrays = {name: data[name] for name in data.files}
    try:
        bundle = SolutionBundle(
            solution_id=manifest["solution_id"],
            parent_id=manifest["parent_id"],
            git_sha=manifest["git_sha"],
            model_hash=manifest["model_hash"],
            asset_hashes=manifest["asset_hashes"],
            frame=manifest["frame"],
            joint_order=tuple(manifest["joint_order"]),
            control=manifest["control"],
            integrator=manifest["integrator"],
            environment=manifest["environment"],
            optimizer_seed=manifest["optimizer_seed"],
            impact_event=manifest["impact_event"],
            flight_metrics=manifest["flight_metrics"],
            feasibility_margins=manifest["feasibility_margins"],
            assumptions=tuple(manifest["assumptions"]),
            arrays=arrays,
            schema_version=version,
            model_lock=manifest.get("model_lock"),
            admission=manifest.get("admission"),
        )
    except KeyError as exc:
        raise SolutionBundleError(f"manifest is missing field {exc}") from exc
    _validate_metadata(bundle)
    _validate_arrays(bundle)
    _validate_admission_block(bundle)
    return bundle
