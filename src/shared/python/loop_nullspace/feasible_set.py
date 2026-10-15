"""FeasibleSet in equality-reduced null-space coordinates (NSX-21, #12564).

Represents the per-frame or trajectory admissible set of torque, hand-contact,
and friction-cone hypotheses in null-space coordinates ``x(z) = x_0 + N z`` so
envelopes, allocation, and probes share a single canonical definition (DRY).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Literal, TypeAlias

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import linprog

FloatArray: TypeAlias = NDArray[np.float64]


def _readonly_1d(value: ArrayLike, name: str) -> FloatArray:
    arr = np.array(value, dtype=np.float64, copy=True)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    arr.setflags(write=False)
    return arr


def _readonly_2d(value: ArrayLike, name: str) -> FloatArray:
    arr = np.array(value, dtype=np.float64, copy=True)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be a 2-D array, got shape {arr.shape}")
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} must contain only finite values")
    arr.setflags(write=False)
    return arr


def _validate_hypothesis_id(value: str, name: str = "hypothesis_id") -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value.strip()


def cone_linearization_error_bound(cone_facets_k: int) -> float:
    """Return the exact radial linearization error bound ``1 - cos(pi / k)`` for ``k >= 4``."""
    if (
        not isinstance(cone_facets_k, int)
        or isinstance(cone_facets_k, bool)
        or cone_facets_k < 4
    ):
        raise ValueError(
            f"cone_facets_k must be an integer >= 4, got {cone_facets_k!r}"
        )
    return float(1.0 - math.cos(math.pi / float(cone_facets_k)))


@dataclass(frozen=True)
class PolyhedralRowBlock:
    """Block of linear inequalities ``A_ub @ z <= b_ub`` traced to a hypothesis ID."""

    A_ub: FloatArray
    b_ub: FloatArray
    hypothesis_id: str
    label: str = ""
    cone_facets_k: int | None = None
    approximation_error_bound: float = 0.0

    def __post_init__(self) -> None:
        a_mat = _readonly_2d(self.A_ub, "A_ub")
        b_vec = _readonly_1d(self.b_ub, "b_ub")
        if b_vec.shape[0] != a_mat.shape[0]:
            raise ValueError(
                f"b_ub length ({b_vec.shape[0]}) must match A_ub rows ({a_mat.shape[0]})"
            )
        hid = _validate_hypothesis_id(self.hypothesis_id)
        object.__setattr__(self, "A_ub", a_mat)
        object.__setattr__(self, "b_ub", b_vec)
        object.__setattr__(self, "hypothesis_id", hid)

        if self.cone_facets_k is not None:
            err_bound = cone_linearization_error_bound(self.cone_facets_k)
            object.__setattr__(self, "approximation_error_bound", err_bound)
        else:
            err = float(self.approximation_error_bound)
            if not math.isfinite(err) or err < 0.0:
                raise ValueError("approximation_error_bound must be finite and >= 0")
            object.__setattr__(self, "approximation_error_bound", err)

    @property
    def row_count(self) -> int:
        """Number of scalar inequality rows in this block."""
        return int(self.A_ub.shape[0])

    @property
    def dim_z(self) -> int:
        """Null-space coordinate dimension."""
        return int(self.A_ub.shape[1])

    @property
    def row_hypothesis_ids(self) -> tuple[str, ...]:
        """Tuple mapping every row in this block to ``hypothesis_id``."""
        return (self.hypothesis_id,) * int(self.A_ub.shape[0])


@dataclass(frozen=True)
class SecondOrderConeBlock:
    """Exact Lorentz cone block ``||u(z)||_2 <= t(z)`` where ``[t(z); u(z)] = b - A @ z``."""

    A: FloatArray
    b: FloatArray
    mu: float
    hypothesis_id: str = "H-GRIP-FRICTION-CONE"
    contact_label: str = ""

    def __post_init__(self) -> None:
        a_mat = _readonly_2d(self.A, "A")
        b_vec = _readonly_1d(self.b, "b")
        if a_mat.shape[0] < 2:
            raise ValueError(
                f"SecondOrderConeBlock requires >= 2 rows, got {a_mat.shape[0]}"
            )
        if b_vec.shape[0] != a_mat.shape[0]:
            raise ValueError(
                f"b length ({b_vec.shape[0]}) must match A rows ({a_mat.shape[0]})"
            )
        mu_val = float(self.mu)
        if not math.isfinite(mu_val) or mu_val <= 0.0:
            raise ValueError(f"mu must be finite and > 0, got {self.mu!r}")
        hid = _validate_hypothesis_id(self.hypothesis_id)
        object.__setattr__(self, "A", a_mat)
        object.__setattr__(self, "b", b_vec)
        object.__setattr__(self, "mu", mu_val)
        object.__setattr__(self, "hypothesis_id", hid)

    @property
    def dim(self) -> int:
        """Cone dimension ``1 + dim(u)``."""
        return int(self.b.size)

    @property
    def num_vars(self) -> int:
        """Null-space variable dimension."""
        return int(self.A.shape[1])

    def normal_value(self, z: ArrayLike) -> float:
        """Evaluate normal force ``F_n(z) = t(z) / mu`` at ``z``."""
        z_vec = _readonly_1d(z, "z")
        t_val = float(self.b[0] - self.A[0] @ z_vec)
        return t_val / self.mu

    def tangent_vector(self, z: ArrayLike) -> FloatArray:
        """Evaluate tangent force vector ``F_t(z) = u(z)`` at ``z``."""
        z_vec = _readonly_1d(z, "z")
        return _readonly_1d(self.b[1:] - self.A[1:] @ z_vec, "tangent_vector")

    def residual(self, z: ArrayLike) -> float:
        """Return ``||F_t(z)||_2 - mu * F_n(z)`` (``<= 0`` inside the cone)."""
        z_vec = _readonly_1d(z, "z")
        s_vec = self.b - self.A @ z_vec
        return float(np.linalg.norm(s_vec[1:]) - s_vec[0])


@dataclass(frozen=True)
class LinearProgramSpec:
    """Polyhedral LP representation of a :class:`FeasibleSet`."""

    A_ub: FloatArray
    b_ub: FloatArray
    A_eq: FloatArray | None = None
    b_eq: FloatArray | None = None
    bounds: tuple[tuple[float | None, float | None], ...] = ()
    row_hypothesis_ids: tuple[str, ...] = ()
    cone_approximation_bounds: dict[str, float] = field(default_factory=dict)
    dim_z: int = 0


@dataclass(frozen=True)
class SecondOrderConeProgramSpec:
    """Exact second-order cone program representation of a :class:`FeasibleSet`."""

    A_ub: FloatArray
    b_ub: FloatArray
    A_eq: FloatArray | None = None
    b_eq: FloatArray | None = None
    soc_blocks: tuple[SecondOrderConeBlock, ...] = ()
    row_hypothesis_ids: tuple[str, ...] = ()
    cone_hypothesis_ids: tuple[str, ...] = ()
    dim_z: int = 0


LPProblemData: TypeAlias = LinearProgramSpec
SOCPProblemData: TypeAlias = SecondOrderConeProgramSpec


def _extract_decomposition_arrays(
    decomposition: Any,
) -> tuple[FloatArray, FloatArray, int | None]:
    """Extract ``(x_particular, null_basis, n_tau)`` from a Decomposition, ForceNullSpace, or pair."""
    if isinstance(decomposition, tuple) and len(decomposition) == 2:
        x_p = _readonly_1d(decomposition[0], "x_particular")
        n_b = _readonly_2d(decomposition[1], "null_basis")
        return x_p, n_b, None
    if isinstance(decomposition, Mapping):
        x_p = _readonly_1d(
            decomposition.get("x_particular", decomposition.get("particular")),
            "x_particular",
        )
        n_b = _readonly_2d(
            decomposition.get("null_basis", decomposition.get("basis")),
            "null_basis",
        )
        dims = decomposition.get("dimensions")
        n_tau = (
            int(dims["n_tau"])
            if isinstance(dims, Mapping) and "n_tau" in dims
            else None
        )
        return x_p, n_b, n_tau

    if hasattr(decomposition, "x_particular") and hasattr(decomposition, "null_basis"):
        x_p = _readonly_1d(decomposition.x_particular, "x_particular")
        n_b = _readonly_2d(decomposition.null_basis, "null_basis")
        dims = getattr(decomposition, "dimensions", None)
        n_tau = (
            int(dims["n_tau"])
            if isinstance(dims, Mapping) and "n_tau" in dims
            else None
        )
        return x_p, n_b, n_tau

    if hasattr(decomposition, "particular") and hasattr(decomposition, "basis"):
        x_p = _readonly_1d(decomposition.particular, "x_particular")
        n_b = _readonly_2d(decomposition.basis, "null_basis")
        return x_p, n_b, None

    raise TypeError(
        "decomposition must provide (x_particular, null_basis) or (particular, basis)"
    )


def _extract_hypothesis_ids_and_custom_blocks(
    hypothesis_set: Any,
    x_p: FloatArray,
    n_b: FloatArray,
) -> tuple[tuple[str, ...], list[PolyhedralRowBlock]]:
    """Resolve hypothesis IDs and any inline polyhedral blocks from ``hypothesis_set``."""
    if hypothesis_set is None:
        return ("H-TORQUE-CAPACITY-BOX",), []
    if hasattr(hypothesis_set, "hypothesis_ids"):
        ids = tuple(
            _validate_hypothesis_id(str(h)) for h in hypothesis_set.hypothesis_ids
        )
        return ids, []

    if isinstance(hypothesis_set, (str, bytes)):
        raise TypeError(
            "hypothesis_set must be a sequence or HypothesisSet, not a string"
        )

    ids_list: list[str] = []
    custom_blocks: list[PolyhedralRowBlock] = []
    for item in hypothesis_set:
        if isinstance(item, PolyhedralRowBlock):
            custom_blocks.append(item)
            ids_list.append(item.hypothesis_id)
        elif isinstance(item, str):
            ids_list.append(_validate_hypothesis_id(item))
        elif hasattr(item, "id"):
            ids_list.append(_validate_hypothesis_id(str(item.id)))
        elif isinstance(item, Mapping):
            hid = _validate_hypothesis_id(str(item.get("id", "H-CUSTOM")))
            ids_list.append(hid)
            blk = _block_from_mapping(item, hid, x_p, n_b)
            if blk is not None:
                custom_blocks.append(blk)
        else:
            raise TypeError(f"Unsupported hypothesis_set entry: {item!r}")

    return tuple(dict.fromkeys(ids_list)), custom_blocks


def _block_from_mapping(
    spec: Mapping[str, Any],
    hid: str,
    x_p: FloatArray,
    n_b: FloatArray,
) -> PolyhedralRowBlock | None:
    """Build a :class:`PolyhedralRowBlock` from a dict specification if constraint arrays exist."""
    if "A_ub" in spec and "b_ub" in spec:
        return PolyhedralRowBlock(
            A_ub=spec["A_ub"],
            b_ub=spec["b_ub"],
            hypothesis_id=hid,
            label=str(spec.get("label", "custom_z")),
        )
    if "A_x" in spec and "b_x" in spec:
        a_x = _readonly_2d(spec["A_x"], "A_x")
        b_x = _readonly_1d(spec["b_x"], "b_x")
        return PolyhedralRowBlock(
            A_ub=a_x @ n_b,
            b_ub=b_x - a_x @ x_p,
            hypothesis_id=hid,
            label=str(spec.get("label", "custom_x")),
        )
    if "z_min" in spec or "z_max" in spec:
        dim_z = int(n_b.shape[1])
        rows_a: list[FloatArray] = []
        rows_b: list[FloatArray] = []
        eye = np.eye(dim_z, dtype=np.float64)
        if "z_min" in spec:
            z_lo = _readonly_1d(spec["z_min"], "z_min")
            rows_a.append(-eye)
            rows_b.append(-z_lo)
        if "z_max" in spec:
            z_hi = _readonly_1d(spec["z_max"], "z_max")
            rows_a.append(eye)
            rows_b.append(z_hi)
        return PolyhedralRowBlock(
            A_ub=np.vstack(rows_a),
            b_ub=np.concatenate(rows_b),
            hypothesis_id=hid,
            label=str(spec.get("label", "z_box")),
        )
    return None


def _build_capacity_block(
    x_p: FloatArray,
    n_b: FloatArray,
    capacity_source: Any,
    hypothesis_id: str,
    default_n_tau: int | None,
) -> tuple[PolyhedralRowBlock, int]:
    """Construct ``tau_min <= tau_0 + N_tau z <= tau_max`` polyhedral rows."""
    if hasattr(capacity_source, "tau_min_nm") and hasattr(
        capacity_source, "tau_max_nm"
    ):
        tau_min = _readonly_1d(capacity_source.tau_min_nm, "tau_min_nm")
        tau_max = _readonly_1d(capacity_source.tau_max_nm, "tau_max_nm")
    elif isinstance(capacity_source, tuple) and len(capacity_source) == 2:
        tau_min = _readonly_1d(capacity_source[0], "tau_min")
        tau_max = _readonly_1d(capacity_source[1], "tau_max")
    elif isinstance(capacity_source, Mapping):
        lo_raw = capacity_source.get("tau_min", capacity_source.get("lower"))
        hi_raw = capacity_source.get("tau_max", capacity_source.get("upper"))
        if lo_raw is None or hi_raw is None:
            raise ValueError(
                "capacity mapping must provide (tau_min, tau_max) or (lower, upper)"
            )
        tau_min = _readonly_1d(lo_raw, "tau_min")
        tau_max = _readonly_1d(hi_raw, "tau_max")
        if "hypothesis_id" in capacity_source:
            hypothesis_id = _validate_hypothesis_id(
                str(capacity_source["hypothesis_id"])
            )
    else:
        raise TypeError(
            "capacity_bounds must be a CapacityVector, (tau_min, tau_max) tuple, or mapping"
        )

    if tau_min.shape != tau_max.shape:
        raise ValueError(
            f"tau_min shape {tau_min.shape} must match tau_max shape {tau_max.shape}"
        )
    n_tau = int(tau_min.shape[0])
    if n_tau > x_p.shape[0]:
        raise ValueError(
            f"capacity dimension ({n_tau}) exceeds state dimension ({x_p.shape[0]})"
        )
    if default_n_tau is not None and n_tau != default_n_tau:
        n_tau = int(tau_min.shape[0])

    tau_0 = x_p[:n_tau]
    n_tau_mat = n_b[:n_tau, :]
    a_ub = np.vstack([-n_tau_mat, n_tau_mat])
    b_ub = np.concatenate([tau_0 - tau_min, tau_max - tau_0])
    block = PolyhedralRowBlock(
        A_ub=a_ub,
        b_ub=b_ub,
        hypothesis_id=hypothesis_id,
        label="torque_capacity",
    )
    return block, n_tau


def _build_cone_blocks(
    x_p: FloatArray,
    n_b: FloatArray,
    cone: Any,
    cone_facets_k: int,
    default_hypothesis_id: str,
) -> tuple[PolyhedralRowBlock, SecondOrderConeBlock]:
    """Construct linearized polyhedral cone facets and exact SOC block for one contact."""
    err_bound = cone_linearization_error_bound(cone_facets_k)
    if isinstance(cone, Mapping):
        indices = tuple(int(i) for i in cone["indices"])
        frame = _readonly_2d(cone.get("frame", np.eye(3)), "cone.frame")
        mu = float(cone["mu"])
        hid = _validate_hypothesis_id(
            str(cone.get("hypothesis_id", default_hypothesis_id))
        )
        preload_min = float(cone.get("preload_min_n", 0.0))
        approx_mode = str(cone.get("approximation", "outer"))
        label = str(cone.get("label", "friction_cone"))
    else:
        indices = tuple(int(i) for i in cone.indices)
        frame = _readonly_2d(cone.frame, "cone.frame")
        mu = float(cone.mu)
        hid = _validate_hypothesis_id(
            str(getattr(cone, "hypothesis_id", default_hypothesis_id))
        )
        preload_min = float(getattr(cone, "preload_min_n", 0.0))
        approx_mode = str(getattr(cone, "approximation", "outer"))
        label = str(getattr(cone, "label", "friction_cone"))

    if len(indices) != 3:
        raise ValueError(f"Friction cone requires 3 indices, got {indices}")
    if approx_mode not in ("outer", "inner"):
        raise ValueError(
            f"Cone approximation must be 'outer' or 'inner', got {approx_mode!r}"
        )

    f_0 = x_p[list(indices)]
    n_f = n_b[list(indices), :]
    t_1, t_2, normal = frame[:, 0], frame[:, 1], frame[:, 2]

    fn_0 = float(normal @ f_0)
    fn_z = normal @ n_f
    ft1_0 = float(t_1 @ f_0)
    ft1_z = t_1 @ n_f
    ft2_0 = float(t_2 @ f_0)
    ft2_z = t_2 @ n_f

    mu_eff = (
        mu * math.cos(math.pi / float(cone_facets_k)) if approx_mode == "inner" else mu
    )
    angles = (
        2.0
        * math.pi
        * np.arange(cone_facets_k, dtype=np.float64)
        / float(cone_facets_k)
    )
    cos_a = np.cos(angles)
    sin_a = np.sin(angles)

    # Facets: (cos_j ft1_z + sin_j ft2_z - mu_eff fn_z) z <= mu_eff fn_0 - (cos_j ft1_0 + sin_j ft2_0)
    facet_a = (
        cos_a[:, None] * ft1_z[None, :]
        + sin_a[:, None] * ft2_z[None, :]
        - mu_eff * fn_z[None, :]
    )
    facet_b = mu_eff * fn_0 - (cos_a * ft1_0 + sin_a * ft2_0)
    # Normal preload row: -fn_z z <= fn_0 - preload_min
    preload_a = -fn_z[None, :]
    preload_b = np.array([fn_0 - preload_min], dtype=np.float64)

    poly_block = PolyhedralRowBlock(
        A_ub=np.vstack([facet_a, preload_a]),
        b_ub=np.concatenate([facet_b, preload_b]),
        hypothesis_id=hid,
        label=label,
        cone_facets_k=cone_facets_k,
        approximation_error_bound=err_bound,
    )
    soc_a = np.vstack([-mu * fn_z[None, :], -ft1_z[None, :], -ft2_z[None, :]])
    soc_b = np.array([mu * fn_0, ft1_0, ft2_0], dtype=np.float64)
    soc_block = SecondOrderConeBlock(
        A=soc_a,
        b=soc_b,
        mu=mu,
        hypothesis_id=hid,
        contact_label=label,
    )
    return poly_block, soc_block


def _elastic_slack_solve(
    a_ub: FloatArray,
    b_ub: FloatArray,
) -> tuple[float, FloatArray]:
    """Solve ``min 1^T s`` s.t. ``A_ub z - s <= b_ub, s >= 0`` using HiGHS."""
    m_rows, dim_z = a_ub.shape
    if m_rows == 0:
        return 0.0, np.zeros(0, dtype=np.float64)
    if dim_z == 0:
        slacks = np.maximum(0.0, -b_ub)
        return float(np.sum(slacks)), slacks

    c_vec = np.concatenate([np.zeros(dim_z), np.ones(m_rows)])
    a_elastic = np.hstack([a_ub, -np.eye(m_rows, dtype=np.float64)])
    bounds = [(None, None)] * dim_z + [(0.0, None)] * m_rows
    res = linprog(c_vec, A_ub=a_elastic, b_ub=b_ub, bounds=bounds, method="highs")
    if not res.success or res.x is None:
        slacks = np.maximum(0.0, -b_ub)
        return float(np.sum(slacks) + 1.0), slacks
    slacks = np.asarray(res.x[dim_z:], dtype=np.float64)
    return float(np.sum(slacks)), slacks


def _rows_infeasible(a_ub: FloatArray, b_ub: FloatArray, tol: float) -> bool:
    """Return True if the subsystem ``a_ub z <= b_ub`` has positive elastic slack > ``tol``."""
    total_slack, _ = _elastic_slack_solve(a_ub, b_ub)
    return bool(total_slack > tol)


@dataclass(frozen=True)
class FeasibleSet:
    """Equality-reduced feasible set ``x(z) = x_particular + null_basis @ z`` (NSX-21)."""

    x_particular: FloatArray
    null_basis: FloatArray
    polyhedral_blocks: tuple[PolyhedralRowBlock, ...] = ()
    soc_blocks: tuple[SecondOrderConeBlock, ...] = ()
    n_tau: int | None = None
    infeasible_reason_hypothesis_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        x_p = _readonly_1d(self.x_particular, "x_particular")
        n_b = _readonly_2d(self.null_basis, "null_basis")
        if n_b.shape[0] != x_p.shape[0]:
            raise ValueError(
                f"null_basis row count ({n_b.shape[0]}) must match x_particular length ({x_p.shape[0]})"
            )
        dim_z = int(n_b.shape[1])
        poly_tuple = tuple(self.polyhedral_blocks)
        for blk in poly_tuple:
            if not isinstance(blk, PolyhedralRowBlock):
                raise TypeError("polyhedral_blocks entries must be PolyhedralRowBlock")
            if blk.dim_z != dim_z:
                raise ValueError(
                    f"PolyhedralRowBlock dim_z ({blk.dim_z}) must match null_basis columns ({dim_z})"
                )
        soc_tuple = tuple(self.soc_blocks)
        for sblk in soc_tuple:
            if not isinstance(sblk, SecondOrderConeBlock):
                raise TypeError("soc_blocks entries must be SecondOrderConeBlock")
            if sblk.num_vars != dim_z:
                raise ValueError(
                    f"SecondOrderConeBlock num_vars ({sblk.num_vars}) must match null_basis columns ({dim_z})"
                )
        if self.n_tau is not None:
            nt = int(self.n_tau)
            if nt < 0 or nt > x_p.shape[0]:
                raise ValueError(f"n_tau ({nt}) must be in [0, {x_p.shape[0]}]")
            object.__setattr__(self, "n_tau", nt)

        reason_ids = tuple(
            _validate_hypothesis_id(h, "infeasible_reason_hypothesis_ids")
            for h in self.infeasible_reason_hypothesis_ids
        )
        object.__setattr__(self, "x_particular", x_p)
        object.__setattr__(self, "null_basis", n_b)
        object.__setattr__(self, "polyhedral_blocks", poly_tuple)
        object.__setattr__(self, "soc_blocks", soc_tuple)
        object.__setattr__(self, "infeasible_reason_hypothesis_ids", reason_ids)

    @classmethod
    def from_hypotheses(
        cls,
        decomposition: Any,
        hypothesis_set: Any = None,
        capacity: Any = None,
        *,
        capacity_bounds: Any = None,
        friction_cones: Sequence[Any] = (),
        cone_facets_k: int = 8,
    ) -> FeasibleSet:
        """Build a :class:`FeasibleSet` from a null-space decomposition and declared hypotheses."""
        cone_linearization_error_bound(cone_facets_k)
        x_p, n_b, decomp_n_tau = _extract_decomposition_arrays(decomposition)
        hyp_ids, custom_blocks = _extract_hypothesis_ids_and_custom_blocks(
            hypothesis_set, x_p, n_b
        )

        poly_blocks: list[PolyhedralRowBlock] = list(custom_blocks)
        soc_blocks: list[SecondOrderConeBlock] = []
        resolved_n_tau = decomp_n_tau

        cap_source = capacity_bounds if capacity_bounds is not None else capacity
        if cap_source is not None:
            cap_hid = (
                "H-TORQUE-CAPACITY-BOX"
                if "H-TORQUE-CAPACITY-BOX" in hyp_ids or not hyp_ids
                else hyp_ids[0]
            )
            cap_blk, resolved_n_tau = _build_capacity_block(
                x_p, n_b, cap_source, cap_hid, resolved_n_tau
            )
            poly_blocks.append(cap_blk)

        cone_default_hid = (
            "H-GRIP-FRICTION-CONE"
            if "H-GRIP-FRICTION-CONE" in hyp_ids or not hyp_ids
            else hyp_ids[-1]
        )
        for cone in friction_cones:
            p_blk, s_blk = _build_cone_blocks(
                x_p, n_b, cone, cone_facets_k, cone_default_hid
            )
            poly_blocks.append(p_blk)
            soc_blocks.append(s_blk)

        return cls(
            x_particular=x_p,
            null_basis=n_b,
            polyhedral_blocks=tuple(poly_blocks),
            soc_blocks=tuple(soc_blocks),
            n_tau=resolved_n_tau,
        )

    @property
    def dim_x(self) -> int:
        """Full primal vector dimension ``dim(x)``."""
        return int(self.x_particular.shape[0])

    @property
    def dim_z(self) -> int:
        """Equality-reduced null-space coordinate dimension ``dim(z)``."""
        return int(self.null_basis.shape[1])

    @property
    def actuated_count(self) -> int:
        """Number of leading actuated torque coordinates ``n_tau`` in ``x``."""
        return self.dim_x if self.n_tau is None else int(self.n_tau)

    @property
    def tau_0(self) -> FloatArray:
        """Particular actuated joint torque vector ``x_particular[:n_tau]``."""
        return self.x_particular[: self.actuated_count]

    @property
    def N_tau(self) -> FloatArray:
        """Actuated joint torque null-space basis ``null_basis[:n_tau, :]``."""
        return self.null_basis[: self.actuated_count, :]

    @property
    def lambda_0(self) -> FloatArray:
        """Particular constraint reaction multiplier vector ``x_particular[n_tau:]``."""
        return self.x_particular[self.actuated_count :]

    @property
    def N_lambda(self) -> FloatArray:
        """Constraint reaction multiplier null-space basis ``null_basis[n_tau:, :]``."""
        return self.null_basis[self.actuated_count :, :]

    @property
    def A_ub(self) -> FloatArray:
        """Stacked polyhedral inequality matrix ``A_ub`` in ``z``-coordinates."""
        if not self.polyhedral_blocks:
            empty = np.zeros((0, self.dim_z), dtype=np.float64)
            empty.setflags(write=False)
            return empty
        return _readonly_2d(
            np.vstack([blk.A_ub for blk in self.polyhedral_blocks]), "A_ub"
        )

    @property
    def b_ub(self) -> FloatArray:
        """Stacked polyhedral inequality right-hand side ``b_ub``."""
        if not self.polyhedral_blocks:
            empty = np.zeros(0, dtype=np.float64)
            empty.setflags(write=False)
            return empty
        return _readonly_1d(
            np.concatenate([blk.b_ub for blk in self.polyhedral_blocks]), "b_ub"
        )

    @property
    def row_hypothesis_ids(self) -> tuple[str, ...]:
        """Provenance tuple mapping every row of ``A_ub`` to its hypothesis ID."""
        ids: list[str] = []
        for blk in self.polyhedral_blocks:
            ids.extend(blk.row_hypothesis_ids)
        return tuple(ids)

    @property
    def hypothesis_ids(self) -> tuple[str, ...]:
        """Unique hypothesis IDs active in this feasible set, in declaration order."""
        all_ids = [
            *self.infeasible_reason_hypothesis_ids,
            *self.row_hypothesis_ids,
            *(s.hypothesis_id for s in self.soc_blocks),
        ]
        return tuple(dict.fromkeys(all_ids))

    def evaluate_x(self, z: ArrayLike) -> FloatArray:
        """Evaluate full primal vector ``x(z) = x_particular + null_basis @ z``."""
        z_vec = _readonly_1d(z, "z")
        return _readonly_1d(self.x_particular + self.null_basis @ z_vec, "x")

    def evaluate_tau(self, z: ArrayLike) -> FloatArray:
        """Evaluate actuated torque vector ``tau(z) = tau_0 + N_tau @ z``."""
        z_vec = _readonly_1d(z, "z")
        return _readonly_1d(self.tau_0 + self.N_tau @ z_vec, "tau")

    def is_empty(self, tol: float = 1e-9) -> bool:
        """Return True iff no ``z`` satisfies ``A_ub @ z <= b_ub`` within ``tol``."""
        if not math.isfinite(tol) or tol < 0.0:
            raise ValueError(f"tol must be finite and >= 0, got {tol}")
        if self.infeasible_reason_hypothesis_ids:
            return True
        return _rows_infeasible(self.A_ub, self.b_ub, tol)

    def conflict_set(self, tol: float = 1e-9) -> tuple[str, ...]:
        """Return an inclusion-minimal IIS of hypothesis IDs via elastic + deletion filtering."""
        if not self.is_empty(tol=tol):
            return ()
        if self.infeasible_reason_hypothesis_ids:
            return self.infeasible_reason_hypothesis_ids

        a_mat = self.A_ub
        b_vec = self.b_ub
        row_hids = self.row_hypothesis_ids
        _, slacks = _elastic_slack_solve(a_mat, b_vec)

        # Order rows so zero-slack rows are tested for deletion before positive-slack rows
        active_rows = list(range(a_mat.shape[0]))
        deletion_order = sorted(active_rows, key=lambda idx: float(slacks[idx]))
        for r_idx in deletion_order:
            candidate = [i for i in active_rows if i != r_idx]
            if candidate and _rows_infeasible(a_mat[candidate], b_vec[candidate], tol):
                active_rows = candidate

        # Map row-level IIS to unique hypothesis IDs and enforce hypothesis-group minimality
        candidate_hids = list(dict.fromkeys(row_hids[i] for i in active_rows))
        for hid in list(candidate_hids):
            remaining_hids = [h for h in candidate_hids if h != hid]
            if not remaining_hids:
                continue
            rem_rows = [
                i for i, row_h in enumerate(row_hids) if row_h in set(remaining_hids)
            ]
            if rem_rows and _rows_infeasible(a_mat[rem_rows], b_vec[rem_rows], tol):
                candidate_hids = remaining_hids

        return tuple(candidate_hids)

    def to_lp(self) -> LinearProgramSpec:
        """Export polyhedral constraints as a :class:`LinearProgramSpec`."""
        cone_bounds = {
            (blk.label or blk.hypothesis_id): blk.approximation_error_bound
            for blk in self.polyhedral_blocks
            if blk.cone_facets_k is not None
        }
        return LinearProgramSpec(
            A_ub=self.A_ub,
            b_ub=self.b_ub,
            bounds=((None, None),) * self.dim_z,
            row_hypothesis_ids=self.row_hypothesis_ids,
            cone_approximation_bounds=cone_bounds,
            dim_z=self.dim_z,
        )

    def to_socp(self) -> SecondOrderConeProgramSpec:
        """Export non-linearized polyhedral rows and exact SOC blocks as a :class:`SecondOrderConeProgramSpec`."""
        non_cone_blocks = [
            blk for blk in self.polyhedral_blocks if blk.cone_facets_k is None
        ]
        # Retain normal preload rows (last row of each cone polyhedral block)
        for blk in self.polyhedral_blocks:
            if blk.cone_facets_k is not None and blk.row_count > blk.cone_facets_k:
                preload_blk = PolyhedralRowBlock(
                    A_ub=blk.A_ub[blk.cone_facets_k :, :],
                    b_ub=blk.b_ub[blk.cone_facets_k :],
                    hypothesis_id=blk.hypothesis_id,
                    label=f"{blk.label}_preload",
                )
                non_cone_blocks.append(preload_blk)

        if non_cone_blocks:
            a_ub = _readonly_2d(
                np.vstack([b.A_ub for b in non_cone_blocks]), "socp_A_ub"
            )
            b_ub = _readonly_1d(
                np.concatenate([b.b_ub for b in non_cone_blocks]), "socp_b_ub"
            )
            row_hids: list[str] = []
            for b in non_cone_blocks:
                row_hids.extend(b.row_hypothesis_ids)
        else:
            a_ub = np.zeros((0, self.dim_z), dtype=np.float64)
            a_ub.setflags(write=False)
            b_ub = np.zeros(0, dtype=np.float64)
            b_ub.setflags(write=False)
            row_hids = []

        return SecondOrderConeProgramSpec(
            A_ub=a_ub,
            b_ub=b_ub,
            soc_blocks=self.soc_blocks,
            row_hypothesis_ids=tuple(row_hids),
            cone_hypothesis_ids=tuple(s.hypothesis_id for s in self.soc_blocks),
            dim_z=self.dim_z,
        )


__all__ = [
    "FeasibleSet",
    "LPProblemData",
    "LinearProgramSpec",
    "PolyhedralRowBlock",
    "SOCPProblemData",
    "SecondOrderConeBlock",
    "SecondOrderConeProgramSpec",
    "cone_linearization_error_bound",
]
