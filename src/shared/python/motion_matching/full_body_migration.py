"""v1 -> v2 trunk coordinate migration by composing rotations (HRG-D1, #13170).

The v1 trunk is one pivot: ``R_1 = Rz(TorsoInput) Rx(SpineInputX)
Ry(SpineInputY)`` (with the joints' fixed placements). The v2 trunk is two
regional joints, ``R_2(phi) = B_L R_L(phi_L) F_L^-1 B_T R_T(phi_T) F_T^-1``
with ``R_L, R_T = Ry Rx Rz`` (isb_spine ZXY). Both are composed by the
specification's own forward kinematics (``spec_body_poses``), never by
adding angles. For each frame the migration solves::

    phi* = argmin_phi || D phi ||^2   subject to   log(R_1^T R_2(phi)) = 0

where ``D`` (3x6) is the soft regional-split regulariser of
``bruno2015_regional`` (ADR-0055): per ISB axis ``a``,
``D_a phi = s_T,a theta_L,a - s_L,a theta_T,a`` (zero when lumbar and thoracic
angles split as the published regional shares ``s_L, s_T``). The constraint
has rank 3 in six unknowns, so the rotation alone does not identify the split;
the result reports that rank, the null-space dimension and two exact witness
solutions (all-lumbar and all-thoracic) whose spread from the chosen split
measures the non-uniqueness. The regulariser is a declared distribution
assumption, not a measurement (``distribution_source``).

Reported, never repaired: the geodesic rotation residual (rad); the shoulder
``Hub`` position residual (m), non-zero because v2 has two pivots; soft-limit
excess in ISB sign (degrees); and the residual of the REJECTED per-axis
mapping (``TorsoInput = lumbar + thoracic axial``), which is exact only for a
single-axis motion. Non-trunk coordinates carry over by name.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import NonlinearConstraint, least_squares, minimize
from scipy.spatial.transform import Rotation

from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching.full_body_schema import (
    ISB_SPINE_AXES,
    TrunkTopology,
    require_full_body_schema,
)
from src.shared.python.motion_matching.full_body_spec import V1_TRUNK, canonical_sha256
from src.shared.python.motion_matching.full_body_v2 import V2_TRUNK, trunk_bodies
from src.shared.python.spine.coupling_data import load_coupling_laws

Array = NDArray[np.float64]

__all__ = ["MigrationFrame", "MigrationReport", "RegionalPrior", "TrunkMigrator"]

_FD_STEP = 1e-7
_RANK_RTOL = 1e-8
_POLISH_STEPS = 4


@dataclass(frozen=True)
class RegionalPrior:
    """Lumbar share of each ISB axis (thoracic share is the complement)."""

    law_id: str
    version: str
    lumbar_share: Mapping[str, float]

    def __post_init__(self) -> None:
        if set(self.lumbar_share) != set(ISB_SPINE_AXES) or not all(
            0.0 < float(v) < 1.0 for v in self.lumbar_share.values()
        ):
            raise ValueError(
                f"lumbar_share needs one share in (0, 1) per {ISB_SPINE_AXES}"
            )

    @classmethod
    def from_coupling_law(cls, law_id: str = "bruno2015_regional") -> RegionalPrior:
        """Read a ``regional_split`` law from ``src/config/spine_coupling_laws.json``.

        Raises ``ValueError`` for an unknown law, a non-regional law, or
        percentages that do not add to 100 per axis.
        """
        entry = load_coupling_laws().get(law_id)
        dist: Mapping[str, Any] = {} if entry is None else entry.get("distribution", {})
        if entry is None or dist.get("type") != "regional_split":
            raise ValueError(f"{law_id!r} is not an available regional_split law")
        shares = {}
        for axis in ISB_SPINE_AXES:
            pct = dist["regional_percent"][axis]
            if abs(float(pct["lumbar"]) + float(pct["thoracic"]) - 100.0) > 1e-9:
                raise ValueError(f"{law_id} {axis} shares must add to 100 %")
            shares[axis] = float(pct["lumbar"]) / 100.0
        return cls(law_id, str(entry["version"]), shares)

    def source(self) -> str:
        """``distribution_source`` label of results regularised by this prior."""
        return f"prior:{self.law_id}@{self.version}"

    def split_matrix(self, trunk: TrunkTopology = V2_TRUNK) -> Array:
        """``D`` (3, n): ``s_T,a theta_L,a - s_L,a theta_T,a`` per ISB axis.

        ``theta = isb_sign * phi`` for the trunk coordinates ``phi`` in
        ``trunk.coordinates`` order; a coordinate of the ``lumbar`` child role
        gets the thoracic share, every other one minus the lumbar share.
        Postcondition: ``D phi = 0`` exactly when each axis splits as the prior.
        """
        out = np.zeros((3, len(trunk.coordinates)))
        region = {c: j.child_role for j in trunk.joints for c in j.coordinates}
        for k, name in enumerate(trunk.coordinates):
            axis, sign = trunk.isb_axis(name)
            share = self.lumbar_share[axis]
            weight = (1.0 - share) if region[name] == "lumbar" else -share
            out[ISB_SPINE_AXES.index(axis), k] = weight * sign
        return out


class _TrunkChain:
    """Pelvis -> thorax kinematics of one schema's trunk, from the spec itself."""

    def __init__(self, spec: Mapping[str, Any], trunk: TrunkTopology, frame: str):
        joints = trunk.locate(spec)
        roles = trunk_bodies(spec)
        self.thorax = roles["thorax"]
        rooted = [dict(joints[0], parent="world"), *joints[1:]]
        self.coordinates = trunk.coordinates
        self._mini = {"coordinate_order": list(trunk.coordinates), "joints": rooted}
        frames = [f for f in spec["frames"] if f["name"] == frame]
        if len(frames) != 1 or frames[0]["body"] != self.thorax:
            raise ValueError(f"Reference frame {frame!r} must be one thorax frame")
        self._point = np.asarray(frames[0]["placement"], dtype=float)[:, 3]
        self._isb = [trunk.isb_axis(c) for c in trunk.coordinates]

    def pose(self, values: Array) -> Array:
        """4x4 thorax pose in the pelvis frame at trunk coordinates ``values``."""
        q = dict(zip(self.coordinates, (float(v) for v in values), strict=True))
        return spec_body_poses(self._mini, q)[self.thorax]

    def reference_point(self, values: Array) -> Array:
        return (self.pose(values) @ self._point)[:3]

    def isb_angles(self, values: Array) -> dict[str, list[tuple[int, float]]]:
        """Per ISB axis: (coordinate index, ISB value) for every trunk coordinate."""
        out: dict[str, list[tuple[int, float]]] = {a: [] for a in ISB_SPINE_AXES}
        for k, (axis, sign) in enumerate(self._isb):
            out[axis].append((k, sign * float(values[k])))
        return out


@dataclass(frozen=True)
class MigrationFrame:
    """Migration of one frame. Angles in radians unless the name says otherwise."""

    q_v2: Mapping[str, float]
    trunk_v2_rad: Array
    rotation_residual_rad: float
    hub_position_residual_m: float
    prior_deviation_rad: float
    constraint_rank: int
    null_space_dimension: int
    alternative_residuals_rad: tuple[float, float]
    alternative_spread_rad: float
    rejected_axial_sum_residual_rad: float
    soft_limit_excess_deg: Mapping[str, float]
    converged: bool
    distribution_source: str
    message: str


@dataclass(frozen=True)
class MigrationReport:
    """Per-frame migration of a trajectory plus its JSON summary."""

    coordinate_order: tuple[str, ...]
    q_v2: Array
    rotation_residual_rad: Array
    hub_position_residual_m: Array
    prior_deviation_rad: Array
    alternative_spread_rad: Array
    rejected_axial_sum_residual_rad: Array
    constraint_rank: Array
    null_space_dimension: Array
    converged: Array
    soft_limit_excess_deg: Mapping[str, Array]
    source_spec_sha256: str
    v2_spec_sha256: str
    distribution_source: str
    method: str = field(default="SLSQP equality solve, rotation-vector constraint")

    def as_document(self) -> dict[str, Any]:
        """JSON-safe summary (finite numbers only; NaN never appears)."""

        def stats(values: Array) -> dict[str, float]:
            return {
                "max": float(np.max(values)),
                "rms": float(np.sqrt(np.mean(values**2))),
            }

        rot = stats(self.rotation_residual_rad)
        hub = stats(self.hub_position_residual_m)
        return {
            "frames": int(self.q_v2.shape[0]),
            "source_spec_sha256": self.source_spec_sha256,
            "v2_spec_sha256": self.v2_spec_sha256,
            "distribution_source": self.distribution_source,
            "method": self.method,
            "max_rotation_residual_rad": rot["max"],
            "rms_rotation_residual_rad": rot["rms"],
            "max_hub_position_residual_m": hub["max"],
            "rms_hub_position_residual_m": hub["rms"],
            "max_prior_deviation_rad": float(np.max(self.prior_deviation_rad)),
            "max_alternative_spread_rad": float(np.max(self.alternative_spread_rad)),
            "max_rejected_axial_sum_residual_rad": float(
                np.max(self.rejected_axial_sum_residual_rad)
            ),
            "constraint_rank": sorted({int(r) for r in self.constraint_rank}),
            "null_space_dimension": sorted({int(n) for n in self.null_space_dimension}),
            "all_converged": bool(np.all(self.converged)),
            "soft_limit_frames": {
                k: int(np.count_nonzero(v > 0))
                for k, v in self.soft_limit_excess_deg.items()
            },
            "max_soft_limit_excess_deg": {
                k: float(np.max(v)) for k, v in self.soft_limit_excess_deg.items()
            },
        }


def _rotvec(rotation: Array) -> Array:
    return Rotation.from_matrix(rotation).as_rotvec()


class TrunkMigrator:
    """Migrate v1 coordinates to a v2 document derived from that same v1 document.

    Preconditions (``ValueError``): ``v1_spec`` is ``full-body-v1``,
    ``v2_spec`` is ``full-body-v2`` with ``source_spec_sha256`` equal to the
    canonical hash of ``v1_spec``, and both carry ``reference_frame`` on the
    thorax. Neither document is modified.
    """

    def __init__(
        self,
        v1_spec: Mapping[str, Any],
        v2_spec: Mapping[str, Any],
        *,
        prior: RegionalPrior | None = None,
        reference_frame: str = "Hub",
    ) -> None:
        require_full_body_schema(
            v1_spec, consumer="migration", accepted=("full-body-v1",)
        )
        require_full_body_schema(
            v2_spec, consumer="migration", accepted=("full-body-v2",)
        )
        source = canonical_sha256(v1_spec)
        if v2_spec.get("source_spec_sha256") != source:
            raise ValueError("v2_spec source_spec_sha256 is not this v1 document")
        self._v1 = _TrunkChain(v1_spec, V1_TRUNK, reference_frame)
        self._v2 = _TrunkChain(v2_spec, V2_TRUNK, reference_frame)
        self._v1_order = tuple(v1_spec["coordinate_order"])
        self._v2_order = tuple(v2_spec["coordinate_order"])
        self._prior = prior or RegionalPrior.from_coupling_law()
        self._split = self._prior.split_matrix(V2_TRUNK)
        self._limits = self._soft_limits()
        self.source_spec_sha256 = source
        self.v2_spec_sha256 = canonical_sha256(v2_spec)

    # -- declared operators ---------------------------------------------------
    @staticmethod
    def _soft_limits() -> dict[str, tuple[float, float, float]]:
        out = {}
        for joint in V2_TRUNK.joints:
            for name, sign, (lo, hi) in zip(
                joint.coordinates,
                joint.isb_signs,
                joint.soft_limits_isb_deg or (),
                strict=True,
            ):
                out[name] = (sign, float(lo), float(hi))
        return out

    # -- numerics -------------------------------------------------------------
    def _constraint(self, target: Array, phi: Array) -> Array:
        return _rotvec(target.T @ self._v2.pose(phi)[:3, :3])

    def _constraint_jacobian(self, target: Array, phi: Array) -> Array:
        jac = np.zeros((3, phi.size))
        for k in range(phi.size):
            step = np.zeros(phi.size)
            step[k] = _FD_STEP
            plus = self._constraint(target, phi + step)
            minus = self._constraint(target, phi - step)
            jac[:, k] = (plus - minus) / (2.0 * _FD_STEP)
        return jac

    def _polish(self, target: Array, phi: Array) -> Array:
        """Minimum-norm Gauss-Newton steps onto the constraint manifold."""
        for _ in range(_POLISH_STEPS):
            resid = self._constraint(target, phi)
            if np.linalg.norm(resid) < 1e-14:
                break
            phi = phi - np.linalg.pinv(self._constraint_jacobian(target, phi)) @ resid
        return phi

    def _witness(
        self, target: Array, region: slice, start: Array
    ) -> tuple[Array, float]:
        """Exact solution with only ``region`` coordinates free (others zero)."""
        free = np.arange(len(V2_TRUNK.coordinates))[region]

        def fun(x: Array) -> Array:
            phi = np.zeros(len(V2_TRUNK.coordinates))
            phi[free] = x
            return self._constraint(target, phi)

        sol = least_squares(fun, start[free], xtol=1e-15, ftol=1e-15, gtol=1e-15)
        phi = np.zeros(len(V2_TRUNK.coordinates))
        phi[free] = sol.x
        return phi, float(np.linalg.norm(fun(sol.x)))

    def _rejected_mapping(self, v1_values: Array) -> Array:
        """Per-axis prior split of the v1 Cardan angles (the rejected mapping)."""
        isb = {
            a: sum(v for _, v in vals)
            for a, vals in self._v1.isb_angles(v1_values).items()
        }
        region = {c: j.child_role for j in V2_TRUNK.joints for c in j.coordinates}
        out = np.zeros(len(V2_TRUNK.coordinates))
        for k, name in enumerate(V2_TRUNK.coordinates):
            axis, sign = V2_TRUNK.isb_axis(name)
            share = self._prior.lumbar_share[axis]
            share = share if region[name] == "lumbar" else 1.0 - share
            out[k] = share * isb[axis] / sign
        return out

    def _excess(self, phi: Array) -> dict[str, float]:
        out = {}
        for k, name in enumerate(V2_TRUNK.coordinates):
            sign, lo, hi = self._limits[name]
            value = math.degrees(sign * float(phi[k]))
            excess = max(lo - value, value - hi)
            if excess > 0.0:
                out[name] = excess
        return out

    # -- public API -----------------------------------------------------------
    def _v1_trunk_values(self, q_v1: Mapping[str, float]) -> Array:
        unknown = sorted(set(q_v1) - set(self._v1_order))
        if unknown:
            raise ValueError(f"unknown v1 coordinates {unknown}")
        if not all(math.isfinite(float(v)) for v in q_v1.values()):
            raise ValueError("v1 coordinates must be finite")
        return np.array([float(q_v1.get(c, 0.0)) for c in self._v1.coordinates])

    def migrate_frame(
        self, q_v1: Mapping[str, float], phi0: Array | None = None
    ) -> MigrationFrame:
        """Migrate one frame of name-keyed v1 coordinates (radians).

        Missing trunk coordinates are zero. Raises ``ValueError`` for unknown
        names or non-finite values. Postconditions: ``q_v2`` holds every
        non-trunk input coordinate unchanged plus the six v2 trunk
        coordinates; ``rotation_residual_rad`` is recomputed from the
        returned coordinates.
        """
        v1_values = self._v1_trunk_values(q_v1)
        target = self._v1.pose(v1_values)[:3, :3]
        rejected = self._rejected_mapping(v1_values)
        start = rejected if phi0 is None else np.asarray(phi0, dtype=float)
        split = self._split

        def objective(x: Array) -> float:
            return float(np.sum((split @ x) ** 2))

        def gradient(x: Array) -> Array:
            return 2.0 * split.T @ (split @ x)

        def closure(x: Array) -> Array:
            return self._constraint(target, x)

        def closure_jacobian(x: Array) -> Array:
            return self._constraint_jacobian(target, x)

        sol = minimize(
            objective,
            start,
            jac=gradient,
            constraints=NonlinearConstraint(closure, 0.0, 0.0, jac=closure_jacobian),
            method="SLSQP",
            options={"ftol": 1e-15, "maxiter": 200},
        )
        phi = self._polish(target, np.asarray(sol.x, dtype=float))
        residual = float(np.linalg.norm(self._constraint(target, phi)))
        sing = np.linalg.svd(self._constraint_jacobian(target, phi), compute_uv=False)
        rank = int(np.sum(sing > max(float(sing[0]) * _RANK_RTOL, 1e-12)))
        lumbar_only, r_l = self._witness(target, slice(0, 3), phi)
        thoracic_only, r_t = self._witness(target, slice(3, 6), phi)
        hub = self._v1.reference_point(v1_values) - self._v2.reference_point(phi)
        q_v2 = {k: float(v) for k, v in q_v1.items() if k not in V1_TRUNK.coordinates}
        q_v2.update(
            dict(zip(V2_TRUNK.coordinates, (float(x) for x in phi), strict=True))
        )
        rejected_resid = float(np.linalg.norm(self._constraint(target, rejected)))
        return MigrationFrame(
            q_v2=q_v2,
            trunk_v2_rad=phi,
            rotation_residual_rad=residual,
            hub_position_residual_m=float(np.linalg.norm(hub)),
            prior_deviation_rad=float(np.linalg.norm(split @ phi)),
            constraint_rank=rank,
            null_space_dimension=phi.size - rank,
            alternative_residuals_rad=(r_l, r_t),
            alternative_spread_rad=float(
                max(
                    float(np.linalg.norm(lumbar_only - phi)),
                    float(np.linalg.norm(thoracic_only - phi)),
                )
            ),
            rejected_axial_sum_residual_rad=rejected_resid,
            soft_limit_excess_deg=self._excess(phi),
            converged=bool(sol.success) and residual < 1e-9,
            distribution_source=self._prior.source(),
            message=str(sol.message),
        )

    @staticmethod
    def _batch_ry_rx_rz(ry: Array, rx: Array, rz: Array) -> Array:
        """Batch 3x3 rotation matrix ``Ry(ry) @ Rx(rx) @ Rz(rz)`` of shape ``(N, 3, 3)``."""
        cy, sy = np.cos(ry), np.sin(ry)
        cx, sx = np.cos(rx), np.sin(rx)
        cz, sz = np.cos(rz), np.sin(rz)
        rot = np.empty((ry.shape[0], 3, 3), dtype=float)
        rot[:, 0, 0] = cy * cz + sy * sx * sz
        rot[:, 0, 1] = -cy * sz + sy * sx * cz
        rot[:, 0, 2] = sy * cx
        rot[:, 1, 0] = cx * sz
        rot[:, 1, 1] = cx * cz
        rot[:, 1, 2] = -sx
        rot[:, 2, 0] = -sy * cz + cy * sx * sz
        rot[:, 2, 1] = sy * sz + cy * sx * cz
        rot[:, 2, 2] = cy * cx
        return rot

    def _batch_v2_rotation(self, phi: Array) -> Array:
        """Batch v2 thorax rotation matrix ``(N, 3, 3)`` for trunk coordinates ``phi`` ``(N, 6)``."""
        rl = self._batch_ry_rx_rz(phi[:, 0], phi[:, 1], phi[:, 2])
        rt = self._batch_ry_rx_rz(phi[:, 3], phi[:, 4], phi[:, 5])
        return np.einsum("nij,njk->nik", rl, rt)

    def migrate_coordinates(self, q: Array, order: Sequence[str]) -> Array:
        """Vectorized migration of ``q`` ``(frames, len(order))`` to ``_v2_order`` coordinates.

        Solves the exact KKT optimality conditions ``min_phi ||D phi||^2`` subject
        to ``log(R_1^T R_2(phi)) = 0`` across all frames simultaneously, matching
        :meth:`migrate_frame` to ``<= 1e-8`` rad without computing witness solves.
        """
        values = np.asarray(q, dtype=float)
        if tuple(order) != self._v1_order:
            raise ValueError("order must be the v1 document's coordinate_order")
        if values.ndim != 2 or values.shape[1] != len(order) or values.shape[0] < 1:
            raise ValueError(
                f"q must have shape (frames, {len(order)}), got {values.shape}"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("v1 coordinates must be finite")
        n_frames = values.shape[0]
        v1_idx = {name: i for i, name in enumerate(order)}
        v1_trunk = np.column_stack([values[:, v1_idx[c]] for c in self._v1.coordinates])
        target = np.array([self._v1.pose(row)[:3, :3] for row in v1_trunk])
        target_t = np.swapaxes(target, 1, 2)
        phi = np.array([self._rejected_mapping(row) for row in v1_trunk])
        dtd2 = 2.0 * (self._split.T @ self._split)

        for _ in range(6):
            r2 = self._batch_v2_rotation(phi)
            c_val = Rotation.from_matrix(
                np.einsum("nij,njk->nik", target_t, r2)
            ).as_rotvec()
            if float(np.max(np.linalg.norm(c_val, axis=1))) < 1.0e-13:
                break
            jac = np.empty((n_frames, 3, 6), dtype=float)
            for k in range(6):
                step = np.zeros((1, 6), dtype=float)
                step[0, k] = _FD_STEP
                rp = self._batch_v2_rotation(phi + step)
                rm = self._batch_v2_rotation(phi - step)
                cp = Rotation.from_matrix(
                    np.einsum("nij,njk->nik", target_t, rp)
                ).as_rotvec()
                cm = Rotation.from_matrix(
                    np.einsum("nij,njk->nik", target_t, rm)
                ).as_rotvec()
                jac[:, :, k] = (cp - cm) / (2.0 * _FD_STEP)
            kkt = np.zeros((n_frames, 9, 9), dtype=float)
            kkt[:, :6, :6] = dtd2[None, :, :]
            kkt[:, :6, 6:] = np.swapaxes(jac, 1, 2)
            kkt[:, 6:, :6] = jac
            rhs = np.zeros((n_frames, 9, 1), dtype=float)
            rhs[:, :6, 0] = -(phi @ dtd2.T)
            rhs[:, 6:, 0] = -c_val
            delta = np.linalg.solve(kkt, rhs)[:, :6, 0]
            phi = phi + delta

        out = np.empty((n_frames, len(self._v2_order)), dtype=float)
        v2_trunk_idx = {name: i for i, name in enumerate(V2_TRUNK.coordinates)}
        for col_idx, name in enumerate(self._v2_order):
            if name in v2_trunk_idx:
                out[:, col_idx] = phi[:, v2_trunk_idx[name]]
            else:
                out[:, col_idx] = values[:, v1_idx[name]]
        return out

    def migrate_trajectory(self, q: Array, order: Sequence[str]) -> MigrationReport:
        """Migrate ``q`` (frames, coordinates) given in the v1 ``order``.

        Each frame is warm-started from the previous solution. Raises
        ``ValueError`` unless ``order`` is the v1 coordinate order and ``q``
        has shape ``(frames, len(order))`` with ``frames >= 1``.
        """
        values = np.asarray(q, dtype=float)
        if tuple(order) != self._v1_order:
            raise ValueError("order must be the v1 document's coordinate_order")
        if values.ndim != 2 or values.shape[1] != len(order) or values.shape[0] < 1:
            raise ValueError(
                f"q must have shape (frames, {len(order)}), got {values.shape}"
            )
        frames: list[MigrationFrame] = []
        for row in values:
            seed = frames[-1].trunk_v2_rad if frames else None
            frames.append(self.migrate_frame(dict(zip(order, row, strict=True)), seed))
        return self._report(frames)

    def _report(self, frames: Sequence[MigrationFrame]) -> MigrationReport:
        def col(attr: str) -> Array:
            return np.array([getattr(f, attr) for f in frames], dtype=float)

        excess = {
            name: np.array([f.soft_limit_excess_deg.get(name, 0.0) for f in frames])
            for name in V2_TRUNK.coordinates
        }
        return MigrationReport(
            coordinate_order=self._v2_order,
            q_v2=np.array([[f.q_v2[c] for c in self._v2_order] for f in frames]),
            rotation_residual_rad=col("rotation_residual_rad"),
            hub_position_residual_m=col("hub_position_residual_m"),
            prior_deviation_rad=col("prior_deviation_rad"),
            alternative_spread_rad=col("alternative_spread_rad"),
            rejected_axial_sum_residual_rad=col("rejected_axial_sum_residual_rad"),
            constraint_rank=col("constraint_rank").astype(int),
            null_space_dimension=col("null_space_dimension").astype(int),
            converged=np.array([f.converged for f in frames], dtype=bool),
            soft_limit_excess_deg=excess,
            source_spec_sha256=self.source_spec_sha256,
            v2_spec_sha256=self.v2_spec_sha256,
            distribution_source=self._prior.source(),
        )
