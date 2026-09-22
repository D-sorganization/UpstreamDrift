"""Classical / analytical dynamics baselines (NM-05)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from .types import DynamicsTaskKind

__all__ = ["ClassicalMethod", "ClassicalModel", "fit_classical", "predict_classical"]


class ClassicalMethod(str, Enum):
    """Named classical baselines required by #10620."""

    ANALYTICAL_PENDULUM = "analytical_pendulum"
    RIDGE = "ridge"
    NEAREST_NEIGHBOR = "nearest_neighbor"


@dataclass(frozen=True, slots=True)
class ClassicalModel:
    method: ClassicalMethod
    task: DynamicsTaskKind
    payload: dict[str, Any]


def fit_classical(
    method: ClassicalMethod,
    *,
    task: DynamicsTaskKind,
    features: np.ndarray,
    targets: np.ndarray,
    active_dofs: tuple[int, ...],
    ridge_lambda: float = 1e-3,
) -> ClassicalModel:
    """Fit one classical baseline on train-only matrices."""
    if features.ndim != 2 or targets.ndim != 2:
        raise ValueError("features and targets must be 2-D")
    if features.shape[0] != targets.shape[0]:
        raise ValueError("features/targets row counts must match")
    if not bool(np.all(np.isfinite(features))) or not bool(
        np.all(np.isfinite(targets))
    ):
        raise ValueError("features/targets must be finite")

    if method is ClassicalMethod.RIDGE:
        ones = np.ones((features.shape[0], 1), dtype=np.float64)
        x = np.concatenate([ones, features], axis=1)
        xtx = x.T @ x
        reg = ridge_lambda * np.eye(xtx.shape[0], dtype=np.float64)
        reg[0, 0] = 0.0
        weights = np.linalg.solve(xtx + reg, x.T @ targets)
        return ClassicalModel(
            method=method,
            task=task,
            payload={"weights": weights, "ridge_lambda": float(ridge_lambda)},
        )

    if method is ClassicalMethod.NEAREST_NEIGHBOR:
        return ClassicalModel(
            method=method,
            task=task,
            payload={
                "features": np.asarray(features, dtype=np.float64).copy(),
                "targets": np.asarray(targets, dtype=np.float64).copy(),
            },
        )

    if method is ClassicalMethod.ANALYTICAL_PENDULUM:
        return ClassicalModel(
            method=method,
            task=task,
            payload={"active_dofs": list(active_dofs)},
        )

    raise ValueError(f"unsupported classical method {method!r}")


def predict_classical(model: ClassicalModel, features: np.ndarray) -> np.ndarray:
    """Predict targets for ``features`` under a fitted classical model."""
    x = np.asarray(features, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError("features must be 2-D")
    if not bool(np.all(np.isfinite(x))):
        raise ValueError("features must be finite")

    if model.method is ClassicalMethod.RIDGE:
        weights = np.asarray(model.payload["weights"], dtype=np.float64)
        ones = np.ones((x.shape[0], 1), dtype=np.float64)
        return np.concatenate([ones, x], axis=1) @ weights

    if model.method is ClassicalMethod.NEAREST_NEIGHBOR:
        train_x = np.asarray(model.payload["features"], dtype=np.float64)
        train_y = np.asarray(model.payload["targets"], dtype=np.float64)
        dists = ((x[:, None, :] - train_x[None, :, :]) ** 2).sum(axis=2)
        idx = np.argmin(dists, axis=1)
        return train_y[idx]

    if model.method is ClassicalMethod.ANALYTICAL_PENDULUM:
        return _analytical_predict(model.task, x)

    raise ValueError(f"unsupported classical method {model.method!r}")


def _analytical_predict(task: DynamicsTaskKind, features: np.ndarray) -> np.ndarray:
    """Apply known 1/2-DOF maps used by the software-contract fixtures."""
    n_feat = features.shape[1]
    if n_feat % 3 != 0:
        raise ValueError("analytical pendulum expects 3-block feature layout")
    n_dof = n_feat // 3
    q = features[:, :n_dof]
    v = features[:, n_dof : 2 * n_dof]
    third = features[:, 2 * n_dof :]

    if task is DynamicsTaskKind.FORWARD_ACCELERATION:
        a = np.zeros_like(q)
        a[:, 0] = -np.sin(q[:, 0]) + third[:, 0]
        if n_dof >= 2:
            a[:, 0] = a[:, 0] - 0.1 * v[:, 1]
            a[:, 1] = -0.5 * np.sin(q[:, 1]) + third[:, 1]
        return a

    if task is DynamicsTaskKind.FORWARD_NEXT_STATE:
        dt = 0.05
        a = _analytical_predict(DynamicsTaskKind.FORWARD_ACCELERATION, features)
        return q + v * dt + 0.5 * a * dt * dt

    if task is DynamicsTaskKind.INVERSE_CONTROL:
        a = third
        u = np.zeros_like(q)
        u[:, 0] = a[:, 0] + np.sin(q[:, 0])
        if n_dof >= 2:
            u[:, 0] = a[:, 0] + np.sin(q[:, 0]) + 0.1 * v[:, 1]
            u[:, 1] = a[:, 1] + 0.5 * np.sin(q[:, 1])
        return u

    raise ValueError(f"unsupported task {task!r}")
