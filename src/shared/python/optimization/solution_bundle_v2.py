"""Research v2 bundle codec: explicit no-impact and separate reference grid.

This codec preserves recorded values; it does not attest their provenance,
physical feasibility, impact detection, or independent forward replay.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from .solution_bundle import (
    SolutionBundle,
    SolutionBundleError,
    _ARRAY_SHAPES,
    _manifest,
    _require_finite,
    _validate_arrays,
    _validate_metadata,
)

_REFERENCE = frozenset({"q_ref", "qd_ref", "tau_ff_ref"})


def migrate_v1(bundle: SolutionBundle) -> SolutionBundle:
    """Explicitly promote a valid v1 event without fabricating evidence."""
    if type(bundle.schema_version) is not int or bundle.schema_version != 1:
        raise SolutionBundleError("migration requires schema version 1")
    _validate_metadata(bundle)
    _validate_arrays(bundle)
    arrays = {name: np.asarray(value).copy() for name, value in bundle.arrays.items()}
    arrays["reference_time"] = arrays["time"].copy()
    result = replace(bundle, arrays=arrays, schema_version=2)
    _validate_v2(result)
    return result


def _validate_v2(bundle: SolutionBundle) -> None:
    if type(bundle.schema_version) is not int or bundle.schema_version != 2:
        raise SolutionBundleError("unsupported schema version")
    if (
        not bundle.joint_order
        or any(not isinstance(x, str) or not x for x in bundle.joint_order)
        or len(set(bundle.joint_order)) != len(bundle.joint_order)
    ):
        raise SolutionBundleError("joint order must be nonempty and unique")
    if any(
        not getattr(bundle, name)
        for name in ("solution_id", "git_sha", "model_hash", "frame")
    ):
        raise SolutionBundleError("identity fields must be nonempty")
    if not isinstance(bundle.git_sha, str) or not re.fullmatch(
        r"(?:[0-9a-f]{40}|[0-9a-f]{64})", bundle.git_sha
    ):
        raise SolutionBundleError("invalid git SHA")
    if not isinstance(bundle.model_hash, str) or not re.fullmatch(
        r"[0-9a-f]{64}", bundle.model_hash
    ):
        raise SolutionBundleError("invalid model hash")
    if type(bundle.optimizer_seed) is not int:
        raise SolutionBundleError("optimizer seed must be an integer")
    if not isinstance(bundle.asset_hashes, dict) or any(
        not isinstance(k, str)
        or not k
        or not isinstance(v, str)
        or not re.fullmatch(r"[0-9a-f]{64}", v)
        for k, v in bundle.asset_hashes.items()
    ):
        raise SolutionBundleError("invalid asset hashes")
    if any(
        not isinstance(getattr(bundle, k), dict)
        for k in ("control", "integrator", "environment")
    ):
        raise SolutionBundleError("invalid configuration")
    for name in ("flight_metrics", "feasibility_margins"):
        metrics = getattr(bundle, name)
        if not isinstance(metrics, dict) or any(
            not isinstance(k, str)
            or isinstance(v, bool)
            or not isinstance(v, (int, float))
            or not np.isfinite(v)
            for k, v in metrics.items()
        ):
            raise SolutionBundleError(f"invalid {name}")
    arrays = bundle.arrays
    required = set(_ARRAY_SHAPES) | {"reference_time"}
    if missing := required - set(arrays):
        raise SolutionBundleError(f"missing arrays: {sorted(missing)}")
    dims = {"j": len(bundle.joint_order)}
    templates = {**_ARRAY_SHAPES, "reference_time": ("r",)}
    for name in _REFERENCE:
        templates[name] = ("r", "j")
    for name, raw in arrays.items():
        value = np.asarray(raw)
        if value.dtype.kind not in "fiu" or not np.isfinite(value).all():
            raise SolutionBundleError(f"{name} must be finite numeric data")
        if name not in templates:
            continue
        template = templates[name]
        if value.ndim != len(template):
            raise SolutionBundleError(f"{name} has wrong dimensions")
        for size, symbol in zip(value.shape, template, strict=True):
            if symbol != "s" and dims.setdefault(symbol, size) != size:
                raise SolutionBundleError(f"{name} has mismatched {symbol} dimension")
    time, ref, knots = (
        np.asarray(arrays[k]) for k in ("time", "reference_time", "knots")
    )
    if (
        time.size < 2
        or ref.size < 2
        or np.any(time[1:] <= time[:-1])
        or np.any(ref[1:] <= ref[:-1])
    ):
        raise SolutionBundleError(
            "time grids must be strictly increasing with at least two samples"
        )
    if np.any(knots[1:] < knots[:-1]):
        raise SolutionBundleError("knots must be nondecreasing")
    if ref[0] > time[0] or ref[-1] < time[-1]:
        raise SolutionBundleError("reference grid does not cover rollout")
    event = bundle.impact_event
    if event != {"status": "none"}:
        if not isinstance(event, dict) or set(event) != {"t", "q", "qd"}:
            raise SolutionBundleError("invalid impact event")
        t = event["t"]
        if (
            isinstance(t, bool)
            or not isinstance(t, (int, float))
            or not np.isfinite(t)
            or not time[0] <= t <= time[-1]
        ):
            raise SolutionBundleError("impact time must be finite and in rollout")
        for name in ("q", "qd"):
            value = np.asarray(event[name])
            if (
                value.shape != (len(bundle.joint_order),)
                or value.dtype.kind not in "fiu"
                or not np.isfinite(value).all()
            ):
                raise SolutionBundleError(f"invalid impact {name}")
    _require_finite(_manifest(bundle, "0" * 64), "manifest")


def save_bundle_v2(bundle: SolutionBundle, directory: str | Path) -> Path:
    """Write a validated v2 directory; recorded arrays are not acceptance evidence."""
    _validate_v2(bundle)
    path = Path(directory)
    if path.exists():
        raise SolutionBundleError("bundle already exists")
    try:
        json.dumps(_manifest(bundle, "0" * 64), allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise SolutionBundleError("invalid manifest") from exc
    path.mkdir(parents=True)
    archive = path / "arrays.npz"
    np.savez(archive, **{k: np.asarray(v) for k, v in bundle.arrays.items()})
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (path / "manifest.json").write_text(
        json.dumps(_manifest(bundle, digest), sort_keys=True, indent=2, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )
    return path


def _unique(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in pairs:
        if key in out:
            raise SolutionBundleError(f"duplicate manifest key {key}")
        out[key] = value
    return out


def load_bundle_v2(
    directory: str | Path,
    *,
    expected_model_hash: str | None = None,
    expected_manifest_sha256: str | None = None,
) -> SolutionBundle:
    """Load and validate v2, optionally checking externally supplied identities."""
    path = Path(directory)
    try:
        raw_manifest = (path / "manifest.json").read_bytes()
        if (
            expected_manifest_sha256 is not None
            and hashlib.sha256(raw_manifest).hexdigest() != expected_manifest_sha256
        ):
            raise SolutionBundleError("manifest hash mismatch")
        manifest = json.loads(
            raw_manifest.decode("utf-8"),
            object_pairs_hook=_unique,
            parse_constant=lambda _: (_ for _ in ()).throw(
                SolutionBundleError("nonfinite JSON")
            ),
        )
        if (
            not isinstance(manifest, dict)
            or manifest.get("units") != "SI"
            or manifest.get("quaternion_convention") != "wxyz"
        ):
            raise SolutionBundleError("unsupported manifest conventions")
        if (
            expected_model_hash is not None
            and manifest.get("model_hash") != expected_model_hash
        ):
            raise SolutionBundleError("model hash mismatch")
        archive = path / "arrays.npz"
        if (
            hashlib.sha256(archive.read_bytes()).hexdigest()
            != manifest["arrays_sha256"]
        ):
            raise SolutionBundleError("array hash mismatch")
        with np.load(archive, allow_pickle=False) as payload:
            if len(payload.files) != len(set(payload.files)):
                raise SolutionBundleError("duplicate array key")
            arrays = {name: payload[name] for name in payload.files}
        names = (
            "solution_id",
            "parent_id",
            "git_sha",
            "model_hash",
            "asset_hashes",
            "frame",
            "control",
            "integrator",
            "environment",
            "optimizer_seed",
            "impact_event",
            "flight_metrics",
            "feasibility_margins",
        )
        values = {name: manifest[name] for name in names}
        result = SolutionBundle(
            **values,
            joint_order=tuple(manifest["joint_order"]),
            assumptions=tuple(manifest["assumptions"]),
            arrays=arrays,
            schema_version=manifest["schema_version"],
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, SolutionBundleError):
            raise
        raise SolutionBundleError(f"invalid v2 bundle: {exc}") from exc
    _validate_v2(result)
    return result
