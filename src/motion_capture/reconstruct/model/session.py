"""Fit an articulated model to a session's reconstructed joints (#9713).

Input is what ``reconstruct_session`` wrote: ``reconstruct/joints_3d_m.npy``
``(T, 15, 3)`` in the 15-joint reconstruct order, the fps and the measured
lengths in ``session_reconstruction.json``. A :class:`LandmarkMap` says which
reconstruct joint observes which model landmark; joints the model has no
landmark for (a scapula pivot) are simply unobserved and follow the
constraints. Outputs go to ``<session>/model/``.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import require
from src.shared.python.logging_pkg.logging_config import get_logger

from ..skeleton import JOINT_NAMES
from .fit import FitOptions, ModelFit, fit_to_dict, fit_trajectory
from .kinematics import ArticulatedModel, ModelSpec

Array = npt.NDArray[np.float64]
logger = get_logger(__name__)

MODEL_DIR = "model"
JOINT_ANGLES_FILE = "joint_angles.json"
FIT_REPORT_FILE = "fit_report.json"
LANDMARKS_FILE = "landmarks_fit.npy"


@dataclass(frozen=True)
class LandmarkMap:
    """``model landmark -> reconstruct joint`` plus which lengths the tape fixed."""

    to_reconstruct: Mapping[str, str | tuple[str, ...]]
    length_from_segment: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        unknown = [
            j
            for value in self.to_reconstruct.values()
            for j in ((value,) if isinstance(value, str) else value)
            if j not in JOINT_NAMES
        ]
        require(not unknown, "map targets must be reconstruct joints", unknown)

    @staticmethod
    def _source(joints_m: Array, value: str | tuple[str, ...]) -> Array:
        """One reconstruct joint, or the mean of several (hands = both wrists)."""
        names = (value,) if isinstance(value, str) else value
        return joints_m[:, [JOINT_NAMES.index(n) for n in names]].mean(axis=1)

    def observed(self, model: ArticulatedModel, joints_m: Array) -> Array:
        """``(T, L, 3)`` landmarks in model order; NaN where the model has no source."""
        require(
            joints_m.ndim == 3 and joints_m.shape[1] == len(JOINT_NAMES),
            "joints must be (T, 15, 3)",
            joints_m.shape,
        )
        out = np.full((joints_m.shape[0], len(model.landmark_names), 3), np.nan)
        for k, name in enumerate(model.landmark_names):
            source = self.to_reconstruct.get(name)
            if source is not None:
                out[:, k] = self._source(joints_m, source)
        return out

    def lengths(
        self, measured: Mapping[str, float], spec: ModelSpec
    ) -> dict[str, float]:
        """Model lengths with tape readings substituted where a segment maps."""
        out = dict(spec.lengths_m)
        for length_name, segment in self.length_from_segment.items():
            if segment in measured and length_name in out:
                out[length_name] = float(measured[segment])
        return out


def initial_state(model: ArticulatedModel, observed: Array) -> Array:
    """Root translation from the root landmark; every angle at rest.

    A rest start is deliberate: the continuity prior then pulls every frame
    from the same side, and the robust stages reach the basin without a
    per-frame IK. Postcondition: ``(T, n_dof)`` with finite values.
    """
    q = np.zeros((observed.shape[0], model.n_dof))
    root = observed[:, 0]
    finite = np.isfinite(root).all(axis=1)
    if finite.any():
        filled = root.copy()
        for c in range(3):
            filled[:, c] = np.interp(
                np.arange(root.shape[0]), np.flatnonzero(finite), root[finite, c]
            )
        q[:, :3] = filled
    return q


def fit_session_model(
    session_dir: Path,
    spec: ModelSpec,
    landmark_map: LandmarkMap,
    *,
    options: FitOptions | None = None,
    out_subdir: str | None = None,
) -> tuple[ModelFit, Path]:
    """Fit ``spec`` to the session's reconstruction; write ``<session>/model/``.

    ``out_subdir`` places the outputs in ``model/<out_subdir>/`` instead (used
    when several models are fitted to one take).

    Precondition: ``reconstruct_session`` has run (joints and summary exist).
    Postcondition: joint angles, fit report and fitted landmarks are on disk.
    """
    recon = session_dir / "reconstruct"
    joints_file = recon / "joints_3d_m.npy"
    summary_file = recon / "session_reconstruction.json"
    require(joints_file.is_file(), "no reconstructed joints", str(joints_file))
    require(summary_file.is_file(), "no session reconstruction", str(summary_file))
    summary = json.loads(summary_file.read_text(encoding="utf-8"))
    fps = float(summary.get("fps") or 0.0)
    require(fps > 0, "session reconstruction does not state the fps")
    joints = np.load(joints_file)
    model = ArticulatedModel(spec)
    observed = landmark_map.observed(model, joints)
    lengths = landmark_map.lengths(summary.get("measured_lengths_m") or {}, spec)
    fit = fit_trajectory(
        model,
        observed,
        fps,
        q0=initial_state(model, observed),
        lengths_m=lengths,
        options=options,
    )
    out_dir = session_dir / MODEL_DIR
    if out_subdir:
        out_dir = out_dir / out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = fit_to_dict(fit, fps)
    payload["model"] = spec.name
    payload["landmark_map"] = {
        k: list(v) if isinstance(v, tuple) else v
        for k, v in landmark_map.to_reconstruct.items()
    }
    (out_dir / JOINT_ANGLES_FILE).write_text(json.dumps(payload), encoding="utf-8")
    np.save(out_dir / LANDMARKS_FILE, fit.landmarks_m)
    (out_dir / FIT_REPORT_FILE).write_text(
        json.dumps(_report(fit, model), indent=2), encoding="utf-8"
    )
    logger.info(
        "model fit %s: rms %.1f mm, %d rejected, %d velocity violations",
        spec.name,
        1000 * fit.rms_m,
        len(fit.rejected),
        fit.velocity_violations,
    )
    return fit, out_dir


def _report(fit: ModelFit, model: ArticulatedModel) -> dict[str, Any]:
    per_landmark = {}
    for k, name in enumerate(model.landmark_names):
        col = fit.residual_m[:, k]
        finite = col[np.isfinite(col)]
        per_landmark[name] = {
            "frames": int(finite.size),
            "rms_mm": float(1000 * np.sqrt(np.mean(finite**2)))
            if finite.size
            else None,
            "rejected": int(sum(1 for r in fit.rejected if r.landmark == name)),
        }
    return {
        "schema_version": "model-fit-report/1.0.0",
        "rms_mm": 1000 * fit.rms_m,
        "landmarks": per_landmark,
        "peak_velocity_rad_s": fit.peak_velocity_rad_s,
        "velocity_violations": fit.velocity_violations,
        "lengths_m": fit.lengths_m,
        "iterations": fit.iterations,
    }
