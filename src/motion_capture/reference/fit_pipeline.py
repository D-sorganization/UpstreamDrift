"""Reproducible articulated reference fits, consumable by camera comparison."""

from __future__ import annotations

import hashlib
import inspect
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import numpy as np
import scipy

from src.motion_capture.reconstruct.model import (
    ArticulatedModel,
    FitOptions,
    ModelFit,
    fit_trajectory,
)
from src.motion_capture.reconstruct.model.registry import RegisteredModel, get_model
from src.motion_capture.rig.documents import write_document

from .fitting import MarkerProfile, map_markers
from .importers import MotionDraft
from .model import ReferenceMotion
from .storage import ReferenceLibrary
from .fit_seed import reference_initial_state


@dataclass(frozen=True)
class ReferenceFit:
    """A fitted animation and its exact reproduction parameters and diagnostics."""

    asset: ReferenceMotion
    fit: ModelFit
    report: dict[str, Any]
    manifest: dict[str, Any]
    observed_m: np.ndarray


def _quality(
    fit: ModelFit, observed: np.ndarray, model: ArticulatedModel
) -> dict[str, Any]:
    mask = np.isfinite(observed).all(axis=2)
    distances = np.linalg.norm(fit.landmarks_m - observed, axis=2)
    values = distances[mask]
    per_landmark = {}
    for index, name in enumerate(model.landmark_names):
        valid = mask[:, index]
        errors = distances[valid, index]
        per_landmark[name] = {
            "observed_frames": int(valid.sum()),
            "rms_m": float(np.sqrt(np.mean(errors**2))) if errors.size else None,
            "max_m": float(errors.max()) if errors.size else None,
        }
    return {
        "all_observed_rms_m": float(np.sqrt(np.mean(values**2))),
        "all_observed_max_m": float(values.max()),
        "retained_rms_m": fit.rms_m,
        "observed_samples": int(mask.sum()),
        "retained_samples": int(np.count_nonzero(fit.weights)),
        "rejected_samples": len(fit.rejected),
        "unobserved_landmarks": [
            name for i, name in enumerate(model.landmark_names) if not mask[:, i].any()
        ],
        "per_landmark": per_landmark,
        "iterations": fit.iterations,
        "qualification": "kinematic-fit; surface proxies; not biomechanical validation",
    }


def fit_reference(
    draft: MotionDraft,
    profile: MarkerProfile,
    model_name: str | RegisteredModel,
    *,
    options: FitOptions | None = None,
) -> ReferenceFit:
    """Fit one registry model and return a full-tree Z-up reference animation.

    Preconditions: uniform source clock with >=2 frames and at least one mapped
    observation for this model. Postcondition: timestamps are unchanged; all
    model joints and connectivity are retained, not just observed landmarks.
    """
    time = np.asarray(draft.time_s, dtype=float)
    code_paths = [
        Path(inspect.getfile(function))
        for function in (
            fit_trajectory,
            ArticulatedModel,
            reference_initial_state,
            map_markers,
            fit_reference,
        )
    ]
    implementation = {
        path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in code_paths
    }
    if time.size < 2 or not np.isfinite(time).all():
        raise ValueError("Fitting needs at least two finite timestamps")
    delta = np.diff(time)
    if (delta <= 0).any() or not np.allclose(delta, delta[0], rtol=1e-6, atol=1e-9):
        raise ValueError("Fitting needs a uniform, increasing source clock")
    registered = get_model(model_name) if isinstance(model_name, str) else model_name
    model_name = registered.name
    model = ArticulatedModel(registered.spec)
    observed = registered.landmark_map.observed(model, map_markers(draft, profile))
    if not np.isfinite(observed).all(axis=2).any():
        raise ValueError(f"No observed landmarks for {model_name}")
    selected = options or FitOptions()
    if set(selected.fit_lengths) - set(registered.learnable_lengths):
        raise ValueError("Requested lengths are not learnable for this model")
    fit = fit_trajectory(
        model,
        observed,
        1 / delta[0],
        q0=reference_initial_state(model, observed),
        options=selected,
    )
    if not all(np.isfinite(v) and v > 0 for v in fit.lengths_m.values()):
        raise ValueError("Solver returned invalid fitted dimensions")
    world = model.forward(fit.q, fit.lengths_m)
    if not np.isfinite(world).all() or not np.isfinite(fit.q).all():
        raise ValueError("Solver returned non-finite reference geometry")
    # Inverse of canonical_z_up_to_adr0041_world: x, -z, y.
    canonical = world[..., [0, 2, 1]] * np.array([1, -1, 1])
    names = tuple(j.name for j in model.joints)
    edges = tuple(
        (names.index(j.parent), i)
        for i, j in enumerate(model.joints)
        if j.parent is not None
    )
    manifest = {
        "schema_version": "reference-fit/1.0",
        "source": draft.source.model_dump(mode="json"),
        "profile": profile.model_dump(mode="json"),
        "model_name": model_name,
        "model_spec": asdict(registered.spec),
        "landmark_map": asdict(registered.landmark_map),
        "fit_options": asdict(selected),
        "time_s": list(draft.time_s),
        "runtime": {"numpy": np.__version__, "scipy": scipy.__version__},
        "solver": "articulated-continuous/1.0",
        "implementation_sha256": implementation,
        "observations_sha256": hashlib.sha256(
            observed.astype("<f8").tobytes()
        ).hexdigest(),
    }
    identity_parameters = dict(manifest)
    identity_parameters["source"] = {
        "sha256": draft.source.sha256,
        "format": draft.source.format,
    }
    encoded = json.dumps(identity_parameters, sort_keys=True, allow_nan=False)
    identity = str(uuid5(NAMESPACE_URL, encoded))
    asset = ReferenceMotion(
        id=identity,
        title=f"{profile.name}: {model_name}",
        source=draft.source,
        notes=f"Fitted kinematics, not observed joints. {profile.notes}",
        model_identity=registered.spec.name,
        source_units="m",
        source_axes=("+X", "-Z", "+Y"),
        source_names=names,
        joint_names=names,
        edges=edges,
        time_s=draft.time_s,
        points_m=tuple(tuple(tuple(p) for p in frame) for frame in canonical),
    )
    return ReferenceFit(asset, fit, _quality(fit, observed, model), manifest, observed)


def save_reference_fit(result: ReferenceFit, output: Path) -> Path:
    """Write a new fit bundle; refuse to overwrite any existing output folder.

    The manifest is written last and contains file hashes. ReferenceLibrary can
    load the animation directly; q and lengths regenerate all model geometry.
    """
    output.mkdir(parents=True, exist_ok=False)
    ReferenceLibrary(output / "references").save(result.asset)
    np.save(output / "q.npy", result.fit.q, allow_pickle=False)
    np.save(output / "landmarks_m.npy", result.fit.landmarks_m, allow_pickle=False)
    np.save(output / "observed_m.npy", result.observed_m, allow_pickle=False)
    write_document(output / "quality.json", result.report)
    manifest = dict(result.manifest)
    manifest["lengths_m"] = result.fit.lengths_m
    manifest["reference_id"] = result.asset.id
    manifest["files"] = {
        path.relative_to(output).as_posix(): hashlib.sha256(
            path.read_bytes()
        ).hexdigest()
        for path in sorted(output.rglob("*"))
        if path.is_file()
    }
    write_document(output / "manifest.json", manifest)
    return output
