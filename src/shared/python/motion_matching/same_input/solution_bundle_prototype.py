"""Research-only JSON/NPZ solution evidence, not an admitted replay format.

Only scalar-coordinate, one-held-input-per-recorded-interval data is supported.
Assets are identified by hash but not embedded. The forward runner and runtime
identity are caller responsibilities, so a matching result is not independent
verification or physical feasibility certification.
"""

from __future__ import annotations

import hashlib
from io import BytesIO
import json
import os
import tempfile
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]
SCHEMA_VERSION = "research-solution-prototype/v3"
_REQUIRED = frozenset(
    {
        "schema_version",
        "research_only",
        "runner",
        "solution_id",
        "parent_id",
        "git_sha",
        "model_sha256",
        "asset_sha256",
        "conventions",
        "controller",
        "integrator",
        "environment",
        "impact",
        "aero",
        "optimizer",
        "flight_metrics",
        "feasibility",
        "assumptions",
    }
)
_ARRAYS = frozenset(
    {
        "reference_time_s",
        "q_ref",
        "qd_ref",
        "tau_ff_ref",
        "spline_knots_s",
        "spline_coefficients",
        "kp",
        "kd",
        "q0",
        "qd0",
        "recorded_time_s",
        "recorded_hold_time_s",
        "recorded_q",
        "recorded_qd",
        "commanded_tau",
        "ideal_input_tau",
        "saturation_lower",
        "saturation_upper",
        "spline_scale",
    }
)


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hash(value: object, length: int) -> bool:
    return (
        isinstance(value, str)
        and len(value) == length
        and all(c in "0123456789abcdef" for c in value)
    )


def _array(value: object, name: str) -> Array:
    raw = np.asarray(value)
    if raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a real numeric array")
    out = np.array(raw, dtype=float, copy=True)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must be finite")
    out.setflags(write=False)
    return out


def _times(value: Array, name: str) -> None:
    with np.errstate(over="ignore", invalid="ignore"):
        delta = np.diff(value) if value.ndim == 1 else np.array([])
    if (
        value.ndim != 1
        or len(value) < 2
        or not np.isfinite(delta).all()
        or np.any(delta <= 0)
    ):
        raise ValueError(f"{name} must be strictly increasing")


@dataclass(frozen=True)
class SolutionBundle:
    """A validated snapshot; metadata remains a caller-provided assertion."""

    manifest: dict[str, Any]
    arrays: dict[str, Array]
    model_bytes: bytes

    def __post_init__(self) -> None:
        try:
            metadata = json.loads(json.dumps(self.manifest, allow_nan=False))
        except (TypeError, ValueError) as exc:
            raise ValueError("manifest must be finite JSON") from exc
        if not isinstance(metadata, dict) or set(metadata) != _REQUIRED:
            raise ValueError("manifest fields are incomplete or unsupported")
        if metadata["schema_version"] != SCHEMA_VERSION:
            raise ValueError("unsupported schema version; explicit migration required")
        if metadata["research_only"] is not True:
            raise ValueError("research-only marker is required")
        if (
            not isinstance(metadata["solution_id"], str)
            or not metadata["solution_id"].strip()
        ):
            raise ValueError("solution_id is required")
        if metadata["parent_id"] is not None and (
            not isinstance(metadata["parent_id"], str)
            or not metadata["parent_id"].strip()
        ):
            raise ValueError("parent_id must be null or a nonempty string")
        if not _hash(metadata["git_sha"], 40) or not _hash(
            metadata["model_sha256"], 64
        ):
            raise ValueError("git and model hashes are required")
        if (
            not isinstance(self.model_bytes, bytes)
            or _digest(self.model_bytes) != metadata["model_sha256"]
        ):
            raise ValueError("model hash does not match embedded model")
        assets = metadata["asset_sha256"]
        if not isinstance(assets, dict) or any(
            not isinstance(k, str) or not k.strip() or not _hash(v, 64)
            for k, v in assets.items()
        ):
            raise ValueError("asset hashes must be named SHA-256 values")
        for key in (
            "conventions",
            "controller",
            "integrator",
            "environment",
            "impact",
            "aero",
            "optimizer",
            "flight_metrics",
            "feasibility",
        ):
            if not isinstance(metadata[key], dict):
                raise ValueError(f"{key} must be an object")
        convention, controller = metadata["conventions"], metadata["controller"]
        if (
            set(convention) != {"units", "frame", "quaternion", "joint_order"}
            or convention["units"] != "SI"
            or not all(
                isinstance(convention[k], str) and convention[k].strip()
                for k in ("frame", "quaternion")
            )
        ):
            raise ValueError("frame, quaternion and SI conventions are required")
        names = convention["joint_order"]
        if (
            not isinstance(names, list)
            or not names
            or any(not isinstance(x, str) or not x.strip() for x in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("joint order must be nonempty and unique")
        if (
            set(controller)
            != {"kind", "hold", "holds_per_step", "saturation", "internal_state"}
            or controller["kind"] != "fixed_pd_bspline"
            or controller["hold"] != "zoh"
            or not isinstance(controller["saturation"], str)
            or not controller["saturation"].strip()
            or not isinstance(controller["internal_state"], dict)
            or isinstance(controller["holds_per_step"], bool)
            or not isinstance(controller["holds_per_step"], int)
            or controller["holds_per_step"] < 1
        ):
            raise ValueError(
                "supported controller, hold, saturation and state required"
            )
        if not all(
            isinstance(metadata["integrator"].get(k), str)
            and metadata["integrator"][k].strip()
            for k in ("kind", "runtime")
        ):
            raise ValueError("integrator and runtime identity are required")
        if not _hash(metadata["integrator"].get("runtime_sha256"), 64):
            raise ValueError("runtime hash is required")
        if metadata["impact"] != {"status": "none"}:
            raise ValueError("prototype only supports explicitly absent impact")
        if any(
            metadata[key] != {"status": "unavailable"}
            for key in ("flight_metrics", "feasibility")
        ):
            raise ValueError(
                "flight and feasibility results must be unavailable in this prototype"
            )
        if not isinstance(metadata["optimizer"].get("seed"), int) or isinstance(
            metadata["optimizer"]["seed"], bool
        ):
            raise ValueError("optimizer seed must be an integer")
        if not isinstance(metadata["assumptions"], list) or any(
            not isinstance(x, str) or not x.strip() for x in metadata["assumptions"]
        ):
            raise ValueError("assumptions must be a list of strings")
        if set(self.arrays) != _ARRAYS:
            raise ValueError("array inventory is incomplete or unsupported")
        a = {key: _array(value, key) for key, value in self.arrays.items()}
        n = len(names)
        _times(a["reference_time_s"], "reference time")
        _times(a["recorded_time_s"], "recorded time")
        tr, ts = len(a["reference_time_s"]), len(a["recorded_time_s"])
        for key in ("q_ref", "qd_ref", "tau_ff_ref"):
            if a[key].shape != (tr, n):
                raise ValueError(f"{key} dimensions do not match reference")
        for key in ("recorded_q", "recorded_qd"):
            if a[key].shape != (ts, n):
                raise ValueError(f"{key} dimensions do not match recorded time")
        holds = controller["holds_per_step"]
        expected_holds = (ts - 1) * holds
        if a["recorded_hold_time_s"].shape != (expected_holds,):
            raise ValueError("recorded hold time grid has the wrong length")
        for key in ("commanded_tau", "ideal_input_tau"):
            if a[key].shape != (expected_holds, n):
                raise ValueError(f"{key} must have one row per hold")
        dt = a["recorded_time_s"][1] - a["recorded_time_s"][0]
        hold_dt = dt / holds
        with np.errstate(over="ignore", invalid="ignore", under="ignore"):
            grid = np.arange(ts, dtype=float) * dt
            hold_grid = np.arange(expected_holds, dtype=float) * hold_dt
        if (
            not np.isfinite(hold_dt)
            or hold_dt <= 0
            or not np.isfinite(hold_grid).all()
            or np.any(np.diff(hold_grid) <= 0)
            or not np.array_equal(a["recorded_time_s"], grid)
            or not np.array_equal(a["recorded_hold_time_s"], hold_grid)
        ):
            raise ValueError("recorded state or hold time grid is incompatible")
        for key in (
            "q0",
            "qd0",
            "kp",
            "kd",
            "saturation_lower",
            "saturation_upper",
            "spline_scale",
        ):
            if a[key].shape != (n,):
                raise ValueError(f"{key} dimensions do not match joint order")
        if np.any(a["kp"] < 0) or np.any(a["kd"] < 0):
            raise ValueError("gains must be nonnegative")
        if np.any(a["saturation_lower"] > a["saturation_upper"]) or np.any(
            a["spline_scale"] <= 0
        ):
            raise ValueError("saturation bounds and spline scale are invalid")
        if not isinstance(metadata["runner"], dict):
            raise ValueError("runner contract must be an object")
        if metadata["runner"].get("mode") not in ("unavailable", "mujoco_ideal_zoh"):
            raise ValueError("unsupported runner mode")
        if metadata["runner"].get("mode") == "unavailable" and metadata["runner"] != {
            "mode": "unavailable"
        }:
            raise ValueError("unavailable runner cannot declare an execution policy")
        if not np.array_equal(a["q0"], a["recorded_q"][0]) or not np.array_equal(
            a["qd0"], a["recorded_qd"][0]
        ):
            raise ValueError("recorded states must begin at q0 and qd0")
        knots = a["spline_knots_s"]
        coeff = a["spline_coefficients"]
        if (
            knots.ndim != 1
            or len(knots) < 8
            or np.any(np.diff(knots) < 0)
            or coeff.shape != (len(knots) - 4, n)
        ):
            raise ValueError("cubic spline knots and coefficients are incompatible")
        if not (
            np.all(knots[:4] == knots[0])
            and np.all(knots[-4:] == knots[-1])
            and knots[0] == a["reference_time_s"][0]
            and knots[-1] == a["reference_time_s"][-1]
            and np.all(np.diff(knots[3:-3]) > 0)
        ):
            raise ValueError("cubic spline must be clamped to the reference horizon")
        if (
            a["recorded_time_s"][0] < a["reference_time_s"][0]
            or a["recorded_time_s"][-1] > a["reference_time_s"][-1]
        ):
            raise ValueError("recorded times must lie inside reference horizon")
        object.__setattr__(self, "manifest", metadata)
        object.__setattr__(self, "arrays", a)


def migrate_research_v2(
    manifest: Mapping[str, Any], arrays: Mapping[str, Array], model_bytes: bytes
) -> SolutionBundle:
    """Explicitly convert authenticated research-v2 one-hold data to v3.

    The caller must authenticate the old manifest and NPZ bytes before passing
    decoded contents here. This does not migrate production solution schemas.
    """
    if (
        not isinstance(manifest, Mapping)
        or manifest.get("schema_version") != "research-solution-prototype/v2"
    ):
        raise ValueError("expected research v2 manifest")
    metadata = json.loads(json.dumps(manifest, allow_nan=False))
    controller = metadata.get("controller")
    if not isinstance(controller, dict) or set(controller) != {
        "kind",
        "hold",
        "saturation",
        "internal_state",
    }:
        raise ValueError("invalid research v2 controller")
    if set(arrays) != _ARRAYS - {"recorded_hold_time_s"}:
        raise ValueError("invalid research v2 array inventory")
    times = _array(arrays["recorded_time_s"], "recorded time")
    _times(times, "recorded time")
    metadata["schema_version"] = SCHEMA_VERSION
    controller["holds_per_step"] = 1
    migrated = dict(arrays)
    migrated["recorded_hold_time_s"] = times[:-1].copy()
    return SolutionBundle(metadata, migrated, model_bytes)


class Playback(NamedTuple):
    time_s: Array
    hold_time_s: Array
    q: Array
    qd: Array
    commanded_tau: Array
    ideal_input_tau: Array


def recorded_playback(bundle: SolutionBundle) -> Playback:
    """Return copies of recorded states; no engine or dynamics is invoked."""
    bundle = SolutionBundle(bundle.manifest, bundle.arrays, bundle.model_bytes)
    a = bundle.arrays
    return Playback(
        *(
            a[k].copy()
            for k in (
                "recorded_time_s",
                "recorded_hold_time_s",
                "recorded_q",
                "recorded_qd",
                "commanded_tau",
                "ideal_input_tau",
            )
        )
    )


@dataclass(frozen=True)
class ReplayComparison:
    passed: bool
    max_absolute_error: dict[str, float]
    independently_verified: bool = False
    runtime_identity_is_caller_asserted: bool = True


def run_mujoco_ideal(bundle: SolutionBundle) -> Playback:
    """Run the restricted MuJoCo ideal-input prototype, not an actuator model."""
    from src.shared.python.motion_matching.same_input.solution_bundle_mujoco_replay import (
        run_mujoco_ideal as run,
    )

    return run(bundle)


def verify_forward_resimulation(
    bundle: SolutionBundle,
    runner: Callable[[SolutionBundle], Playback],
    tolerances: Mapping[str, tuple[float, float]],
    *,
    runtime_sha256: str,
) -> ReplayComparison:
    """Compare an explicitly supplied runner, without claiming its independence."""
    keys = ("q", "qd", "commanded_tau", "ideal_input_tau")
    if set(tolerances) != set(keys) or any(
        len(tolerances[k]) != 2
        or not np.isfinite(tolerances[k]).all()
        or min(tolerances[k]) < 0
        for k in keys
    ):
        raise ValueError("finite nonnegative per-quantity tolerances are required")
    if runtime_sha256 != bundle.manifest["integrator"]["runtime_sha256"]:
        raise ValueError("runtime identity does not match bundle")
    expected, actual = recorded_playback(bundle), runner(bundle)
    if (
        not isinstance(actual, Playback)
        or not np.array_equal(actual.time_s, expected.time_s)
        or not np.array_equal(actual.hold_time_s, expected.hold_time_s)
    ):
        raise ValueError("runner must return Playback on the recorded time grid")
    errors: dict[str, float] = {}
    passed = True
    for key in keys:
        observed, target = _array(getattr(actual, key), key), getattr(expected, key)
        if observed.shape != target.shape:
            raise ValueError(f"{key} replay shape mismatch")
        atol, rtol = tolerances[key]
        with np.errstate(over="ignore", invalid="ignore"):
            absolute_error = np.abs(observed - target)
            allowed = atol + rtol * np.abs(target)
        if not np.isfinite(absolute_error).all() or not np.isfinite(allowed).all():
            raise ValueError(f"{key} comparison arithmetic is nonfinite")
        errors[key] = float(np.max(absolute_error))
        passed = passed and bool(np.all(absolute_error <= allowed))
    return ReplayComparison(passed, errors)


def save_solution(path: Path, bundle: SolutionBundle) -> str:
    """Publish a manifest last, returning its digest for separate trusted storage.

    The manifest is the atomic publication point for process-level visibility.
    An interrupted write can leave an orphaned content-addressed NPZ but not a
    visible partial manifest. Directory crash durability is not guaranteed.
    The directory is assumed trusted and must support same-volume hard links.
    """
    path = Path(path)
    if path.suffix != ".json":
        raise ValueError("manifest path must end in .json")
    bundle = SolutionBundle(bundle.manifest, bundle.arrays, bundle.model_bytes)
    if path.exists():
        raise FileExistsError("solution files already exist")
    buffer = BytesIO()
    np.savez_compressed(
        buffer,
        model_bytes=np.frombuffer(bundle.model_bytes, dtype=np.uint8),
        **bundle.arrays,  # type: ignore[arg-type]  # NumPy's named-array stub
    )
    raw = buffer.getvalue()
    npz_hash = _digest(raw)
    npz = path.with_name(f"{path.stem}.{npz_hash}.npz")
    with npz.open("xb") as stream:
        stream.write(raw)
        stream.flush()
        os.fsync(stream.fileno())
    manifest = {**bundle.manifest, "npz_sha256": npz_hash}
    encoded = json.dumps(manifest, sort_keys=True, allow_nan=False, indent=2).encode(
        "utf-8"
    )
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=".solution-", delete=False
        ) as stream:
            temporary = stream.name
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.link(temporary, path)
    finally:
        if temporary is not None:
            Path(temporary).unlink(missing_ok=True)
    return _digest(encoded)


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate manifest key")
        result[key] = value
    return result


def load_solution(
    path: Path,
    *,
    expected_manifest_sha256: str,
    expected_model_sha256: str,
    expected_runtime_sha256: str,
    expected_assets: Mapping[str, str] | None = None,
) -> SolutionBundle:
    """Reject unknown schema, mismatched runtime assets and corrupt arrays."""
    path = Path(path)
    if path.suffix != ".json":
        raise ValueError("manifest path must end in .json")
    raw_manifest = path.read_bytes()
    if (
        not _hash(expected_manifest_sha256, 64)
        or _digest(raw_manifest) != expected_manifest_sha256
    ):
        raise ValueError("manifest hash mismatch")
    manifest = json.loads(raw_manifest, object_pairs_hook=_unique_object)
    if not isinstance(manifest, dict):
        raise ValueError("manifest must be an object")
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported schema version; explicit migration required")
    if set(manifest) != _REQUIRED | {"npz_sha256"}:
        raise ValueError("manifest fields are incomplete or unsupported")
    npz_hash = manifest.pop("npz_sha256", None)
    if not _hash(npz_hash, 64):
        raise ValueError("NPZ hash is required")
    raw = path.with_name(f"{path.stem}.{npz_hash}.npz").read_bytes()
    if _digest(raw) != npz_hash:
        raise ValueError("NPZ hash mismatch")
    if manifest.get("model_sha256") != expected_model_sha256:
        raise ValueError("model incompatible with runtime")
    if (
        not isinstance(manifest.get("integrator"), dict)
        or manifest["integrator"].get("runtime_sha256") != expected_runtime_sha256
    ):
        raise ValueError("runtime identity does not match bundle")
    if manifest.get("asset_sha256") != dict(expected_assets or {}):
        raise ValueError("assets incompatible with runtime")
    with np.load(BytesIO(raw), allow_pickle=False) as data:
        if (
            len(data.files) != len(_ARRAYS) + 1
            or set(data.files) != _ARRAYS | {"model_bytes"}
            or data["model_bytes"].dtype != np.uint8
            or data["model_bytes"].ndim != 1
        ):
            raise ValueError("unsupported NPZ inventory")
        model = data["model_bytes"].tobytes()
        arrays = {k: data[k] for k in _ARRAYS}
    return SolutionBundle(manifest, arrays, model)
