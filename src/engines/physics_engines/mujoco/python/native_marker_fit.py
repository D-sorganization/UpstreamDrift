"""Exact-clock, masked native site-marker cost for F03 manifold fitting.

Observations are world-frame metres at native integration boundaries. This
module supplies a cost and its local MuJoCo tangent derivatives; F05d owns
the BoxFDDP solver, nonlinear candidate admission and T01 torque replay.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

import mujoco as mj
import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


@dataclass(frozen=True)
class NativeMarkerTargets:
    """One ordered world-frame site observation series with explicit masks."""

    time_seconds: Array
    site_names: tuple[str, ...]
    position_m: Array
    valid_mask: NDArray[np.bool_]
    site_weights: Array
    frame_id: str

    def __post_init__(self) -> None:
        times = np.asarray(self.time_seconds, dtype=np.float64)
        sites = tuple(self.site_names)
        positions = np.asarray(self.position_m, dtype=np.float64)
        mask = np.asarray(self.valid_mask, dtype=bool)
        weights = np.asarray(self.site_weights, dtype=np.float64)
        if (
            times.ndim != 1
            or times.size < 2
            or not np.isfinite(times).all()
            or times[0] != 0
            or not np.all(np.diff(times) > 0)
            or not sites
            or any(not isinstance(site, str) or not site for site in sites)
            or len(set(sites)) != len(sites)
            or self.frame_id != "world"
        ):
            raise ValueError(
                "native markers require ordered sites, world frame and exact clock"
            )
        if (
            positions.shape != (len(times), len(sites), 3)
            or mask.shape != (len(times), len(sites))
            or not mask.any(axis=1).all()
            or not np.isfinite(positions[mask]).all()
            or np.isinf(positions).any()
            or weights.shape != (len(sites),)
            or not np.isfinite(weights).all()
            or np.any(weights <= 0)
        ):
            raise ValueError("native marker positions, visibility or weights invalid")
        for name, value in (
            ("time_seconds", times),
            ("position_m", positions),
            ("valid_mask", mask),
            ("site_weights", weights),
        ):
            frozen = value.copy()
            frozen.setflags(write=False)
            object.__setattr__(self, name, frozen)
        object.__setattr__(self, "site_names", sites)

    @property
    def identity_sha256(self) -> str:
        """Bind exact clock, ordered sites, frame, mask and visible metres."""
        visible = [
            [
                self.position_m[row, column].tolist() if valid else None
                for column, valid in enumerate(mask_row)
            ]
            for row, mask_row in enumerate(self.valid_mask)
        ]
        payload = {
            "time_seconds": self.time_seconds.tolist(),
            "site_names": self.site_names,
            "position_m": visible,
            "valid_mask": self.valid_mask.tolist(),
            "site_weights": self.site_weights.tolist(),
            "frame_id": self.frame_id,
        }
        encoded = json.dumps(
            payload, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
        return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


class NativeMarkerObservationCost:
    """Site marker residual and Gauss–Newton curvature in the 16D tangent."""

    def __init__(self, model: Any, targets: NativeMarkerTargets) -> None:
        if (model.nq, model.nv, model.nu) != (9, 8, 2):
            raise ValueError("native marker cost requires nq=9/nv=8/nu=2")
        times = np.arange(len(targets.time_seconds)) * float(model.opt.timestep)
        if not np.allclose(times, targets.time_seconds, rtol=0, atol=1e-12):
            raise ValueError("marker observation clock must equal native grid")
        site_ids = tuple(
            mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE, name)
            for name in targets.site_names
        )
        if any(site_id < 0 for site_id in site_ids):
            raise ValueError("marker site name absent from native model")
        self.model = model
        self.targets = targets
        self.site_ids = site_ids

    @property
    def time_seconds(self) -> Array:
        return self.targets.time_seconds

    def _sample(self, physical: Array, index: int) -> Any:
        if (
            index < 0
            or index >= len(self.time_seconds)
            or physical.shape != (self.model.nq + self.model.nv,)
            or not np.isfinite(physical).all()
            or not np.isclose(np.linalg.norm(physical[3:7]), 1, rtol=0, atol=1e-12)
        ):
            raise ValueError("native marker state/index invalid")
        data = mj.MjData(self.model)
        data.qpos[:] = physical[: self.model.nq]
        data.qvel[:] = physical[self.model.nq :]
        mj.mj_forward(self.model, data)
        return data

    def cost(self, physical: Array, index: int) -> float:
        data = self._sample(physical, index)
        active = self.targets.valid_mask[index]
        residual = (
            data.site_xpos[list(self.site_ids)][active]
            - self.targets.position_m[index][active]
        )
        return float(np.sum(self.targets.site_weights[active, None] * residual**2))

    def tangent(self, physical: Array, index: int) -> tuple[Array, Array]:
        data = self._sample(physical, index)
        rows: list[Array] = []
        residuals: list[Array] = []
        weights: list[Array] = []
        for column, site_id in enumerate(self.site_ids):
            if not self.targets.valid_mask[index, column]:
                continue
            jac_pos = np.empty((3, self.model.nv))
            jac_rot = np.empty((3, self.model.nv))
            mj.mj_jacSite(self.model, data, jac_pos, jac_rot, site_id)
            rows.append(np.hstack((jac_pos, np.zeros_like(jac_pos))))
            residuals.append(
                data.site_xpos[site_id] - self.targets.position_m[index, column]
            )
            weights.append(np.full(3, self.targets.site_weights[column]))
        jacobian = np.vstack(rows)
        residual = np.concatenate(residuals)
        weight = np.concatenate(weights)
        gradient = 2 * jacobian.T @ (weight * residual)
        curvature = 2 * jacobian.T @ (weight[:, None] * jacobian)
        return gradient, curvature
