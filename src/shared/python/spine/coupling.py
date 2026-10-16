"""Spinal coupling laws: independent coordinates to per-level ISB angles (SPN-2a).

Implements ADR-0055 item 6. A coupling law maps independent coordinates
``phi`` (length ``n_phi``) to ISB ZXY angles ``theta_i = (FE, LB, AR)`` in
radians for each :class:`SpinalLevel` it covers. Laws prescribe kinematics
only; they are disabled for inverse dynamics, static optimisation and joint
reaction (ADR-0055 item 6). Every law carries an id, a version and a citation.

Law families:

* :class:`LinearCouplingLaw`: ``theta_i = C_i @ phi``, ``C_i`` of shape
  ``(3, n_phi)``; cross-axis terms allowed.
* :class:`ParityCouplingLaw`: ``theta_i = C_odd_i @ phi + C_even_i @ |phi|``;
  declared axes are even in ``phi`` (e.g. flexion coupled to rotation to
  either side), the remaining axes stay odd.
* :class:`StateDependentCouplingLaw`: ``theta_i = C_i(s) @ phi`` with ``C_i``
  piecewise-linear in one named scheduling coordinate ``s = phi[k]`` (clamped
  outside the breakpoints).
* :class:`PolynomialCouplingLaw`: adapter target for MyoFullBody joint
  equalities, produced by :func:`from_joint_coupling`.

``myofullbody`` is never modified and never imports this module.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol, runtime_checkable

import numpy as np

from src.shared.python.contracts import ensure
from src.shared.python.myofullbody.couplings import JointCoupling
from src.shared.python.spine.levels import SpinalLevel
from src.shared.python.spine.segmental import LevelTransform, SegmentalPose

__all__ = [
    "CouplingLaw",
    "LinearCouplingLaw",
    "ParityCouplingLaw",
    "PolynomialCouplingLaw",
    "StateDependentCouplingLaw",
    "from_joint_coupling",
    "level_rotations",
    "regional_fraction",
    "restrict_to_levels",
]

_DENOM_TOL = 1e-12
_POLY_TERMS = 5


@runtime_checkable
class CouplingLaw(Protocol):
    """Contract of a coupling law (angles in radians, ISB ZXY ``(FE, LB, AR)``)."""

    @property
    def law_id(self) -> str:
        """Stable identifier of the law."""

    @property
    def version(self) -> str:
        """Version string of the law."""

    @property
    def citation(self) -> str:
        """Literature citation (DOI plus table or page)."""

    def independent_names(self) -> tuple[str, ...]:
        """Names of the independent coordinates, in ``phi`` order."""

    def expand(self, phi: np.ndarray) -> dict[SpinalLevel, np.ndarray]:
        """Per-level ISB angles, each of shape ``(3,)``."""

    def jacobian(self, phi: np.ndarray) -> dict[SpinalLevel, np.ndarray]:
        """Per-level ``d theta / d phi``, each of shape ``(3, n_phi)``."""


# --------------------------------------------------------------------- helpers


def _text(value: object, name: str) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a str, got {type(value).__name__}")
    if not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _names(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("independent must be a sequence of str")
    names = tuple(_text(item, "independent name") for item in value)
    if not names:
        raise ValueError("independent must name at least one coordinate")
    if len(set(names)) != len(names):
        raise ValueError(f"independent names must be unique, got {names}")
    return names


def _float_array(value: object, name: str) -> np.ndarray:
    try:
        arr = np.array(value, dtype=float, copy=True)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be a numeric array: {exc}") from exc
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


def _check_contiguous(levels: Sequence[SpinalLevel]) -> tuple[SpinalLevel, ...]:
    for level in levels:
        if not isinstance(level, SpinalLevel):
            raise TypeError(
                f"level keys must be SpinalLevel, got {type(level).__name__}"
            )
    ordered = tuple(sorted(set(levels), key=lambda lv: lv.ordinal))
    if not ordered:
        raise ValueError("a coupling law needs at least one level")
    ords = [lv.ordinal for lv in ordered]
    if any(b - a != 1 for a, b in zip(ords, ords[1:], strict=False)):
        raise ValueError(
            f"level set must be contiguous, got {[lv.name for lv in ordered]}"
        )
    return ordered


def _check_phi(phi: object, n_phi: int) -> np.ndarray:
    arr = _float_array(phi, "phi")
    if arr.shape != (n_phi,):
        raise ValueError(f"phi must have shape ({n_phi},), got {arr.shape}")
    return arr


def _frozen_map(items: Mapping[SpinalLevel, np.ndarray]) -> Mapping:
    for arr in items.values():
        arr.setflags(write=False)
    return MappingProxyType(dict(items))


def _checked_matrices(
    matrices: Mapping[SpinalLevel, object], tail: tuple[int, ...], what: str
) -> dict[SpinalLevel, np.ndarray]:
    if not isinstance(matrices, Mapping):
        raise TypeError("matrices must be a Mapping[SpinalLevel, ndarray]")
    ordered = _check_contiguous(list(matrices))
    out: dict[SpinalLevel, np.ndarray] = {}
    for level in ordered:
        arr = _float_array(matrices[level], f"matrix for {level.name}")
        if arr.shape[-len(tail) :] != tail[-len(tail) :] or arr.ndim != len(tail):
            raise ValueError(
                f"matrix for {level.name} must have shape {what}, got {arr.shape}"
            )
        out[level] = arr
    return out


# ------------------------------------------------------------------ linear law


@dataclass(frozen=True)
class LinearCouplingLaw:
    """``theta_i = C_i @ phi`` with ``C_i`` of shape ``(3, n_phi)``.

    Preconditions (``TypeError``/``ValueError``): non-empty ``law_id``,
    ``version`` and ``citation`` (DOI plus table or page); unique non-empty
    ``independent`` names; ``matrices`` keyed by :class:`SpinalLevel` with a
    contiguous level set and each matrix finite of shape ``(3, len(independent))``.

    Postconditions: matrices are read-only copies; :meth:`expand` returns
    ``(3,)`` radians per level and :meth:`jacobian` returns the constant
    ``C_i``.
    """

    law_id: str
    version: str
    matrices: Mapping[SpinalLevel, np.ndarray]
    independent: tuple[str, ...]
    citation: str

    def __post_init__(self) -> None:
        _text(self.law_id, "law_id")
        _text(self.version, "version")
        _text(self.citation, "citation")
        names = _names(self.independent)
        mats = _checked_matrices(self.matrices, (3, len(names)), f"(3, {len(names)})")
        object.__setattr__(self, "independent", names)
        object.__setattr__(self, "matrices", _frozen_map(mats))

    @property
    def levels(self) -> tuple[SpinalLevel, ...]:
        """Covered levels, caudal to cranial."""
        return tuple(sorted(self.matrices, key=lambda lv: lv.ordinal))

    def independent_names(self) -> tuple[str, ...]:
        return self.independent

    def expand(self, phi: np.ndarray) -> dict[SpinalLevel, np.ndarray]:
        vec = _check_phi(phi, len(self.independent))
        return {lv: self.matrices[lv] @ vec for lv in self.levels}

    def jacobian(self, phi: np.ndarray) -> dict[SpinalLevel, np.ndarray]:
        _check_phi(phi, len(self.independent))
        return {lv: self.matrices[lv].copy() for lv in self.levels}


# ------------------------------------------------------------------ parity law


def _check_even_axes(value: object) -> tuple[int, ...]:
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("even_axes must be a sequence of axis indices (0, 1, 2)")
    axes = []
    for item in value:
        if isinstance(item, bool) or not isinstance(item, int | np.integer):
            raise TypeError(f"even_axes entries must be int, got {item!r}")
        if int(item) not in (0, 1, 2):
            raise ValueError(f"axis must be 0, 1 or 2, got {item}")
        axes.append(int(item))
    if not axes:
        raise ValueError("even_axes must declare at least one even axis")
    if len(set(axes)) != len(axes):
        raise ValueError(f"even_axes must be unique, got {tuple(axes)}")
    return tuple(sorted(axes))


@dataclass(frozen=True)
class ParityCouplingLaw:
    """``theta_i = C_odd_i @ phi + C_even_i @ |phi|`` (issue #12821).

    ``even_axes`` lists the axes (0=FE, 1=LB, 2=AR) declared even in ``phi``:
    ``even_matrices`` is non-zero only on those rows and ``matrices`` (the odd
    part) is zero on them, so each axis is purely even or purely odd. The
    absolute value is element-wise on ``phi``.

    The Jacobian is ``C_odd_i + C_even_i * sign(phi)`` (column-wise); at
    ``phi_k == 0`` the even part uses the subgradient ``sign(0) = 0`` (a
    documented choice: the law is not differentiable there).

    Preconditions (``TypeError``/``ValueError``): those of
    :class:`LinearCouplingLaw`; ``even_matrices`` covers exactly the levels of
    ``matrices`` with the same shapes; ``even_axes`` non-empty, unique, in
    0..2; odd rows on even axes and even rows on odd axes are zero.

    Postconditions: both matrix maps are read-only copies.
    """

    law_id: str
    version: str
    matrices: Mapping[SpinalLevel, np.ndarray]
    even_matrices: Mapping[SpinalLevel, np.ndarray]
    even_axes: tuple[int, ...]
    independent: tuple[str, ...]
    citation: str

    def __post_init__(self) -> None:
        _text(self.law_id, "law_id")
        _text(self.version, "version")
        _text(self.citation, "citation")
        names = _names(self.independent)
        axes = _check_even_axes(self.even_axes)
        shape = (3, len(names))
        odd = _checked_matrices(self.matrices, shape, f"(3, {len(names)})")
        even = _checked_matrices(self.even_matrices, shape, f"(3, {len(names)})")
        if set(odd) != set(even):
            raise ValueError("even_matrices must cover the same levels as matrices")
        odd_axes = [a for a in range(3) if a not in axes]
        for level in odd:
            if np.any(odd[level][list(axes)] != 0.0):
                raise ValueError(
                    f"{level.name}: odd matrix must be zero on even axes {axes}"
                )
            if np.any(even[level][odd_axes] != 0.0):
                raise ValueError(
                    f"{level.name}: even matrix must be zero outside even axes {axes}"
                )
        object.__setattr__(self, "independent", names)
        object.__setattr__(self, "even_axes", axes)
        object.__setattr__(self, "matrices", _frozen_map(odd))
        object.__setattr__(self, "even_matrices", _frozen_map(even))

    @property
    def levels(self) -> tuple[SpinalLevel, ...]:
        """Covered levels, caudal to cranial."""
        return tuple(sorted(self.matrices, key=lambda lv: lv.ordinal))

    def independent_names(self) -> tuple[str, ...]:
        return self.independent

    def expand(self, phi: np.ndarray) -> dict[SpinalLevel, np.ndarray]:
        vec = _check_phi(phi, len(self.independent))
        mag = np.abs(vec)
        return {
            lv: self.matrices[lv] @ vec + self.even_matrices[lv] @ mag
            for lv in self.levels
        }

    def jacobian(self, phi: np.ndarray) -> dict[SpinalLevel, np.ndarray]:
        vec = _check_phi(phi, len(self.independent))
        sgn = np.sign(vec)
        return {
            lv: self.matrices[lv] + self.even_matrices[lv] * sgn[None, :]
            for lv in self.levels
        }


# ---------------------------------------------------------- state-dependent law


@dataclass(frozen=True)
class StateDependentCouplingLaw:
    """``theta_i = C_i(s) @ phi``; ``C_i`` piecewise-linear in ``s = phi[k]``.

    ``scheduling`` names the scheduling coordinate (one of ``independent``);
    ``breakpoints`` is a strictly increasing 1-D array (>= 2 values) and
    ``matrices[level]`` has shape ``(n_breakpoints, 3, n_phi)``. Outside the
    breakpoint range ``C_i`` is held constant (``dC/ds = 0``); at an interior
    breakpoint the right-hand slope is used.

    The Jacobian includes the scheduling term::

        d theta_i / d phi = C_i(s) + (dC_i/ds @ phi) e_k^T

    Preconditions are those of :class:`LinearCouplingLaw` plus the breakpoint
    and scheduling checks; all violations raise ``TypeError``/``ValueError``.
    """

    law_id: str
    version: str
    independent: tuple[str, ...]
    scheduling: str
    breakpoints: np.ndarray
    matrices: Mapping[SpinalLevel, np.ndarray]
    citation: str

    def __post_init__(self) -> None:
        _text(self.law_id, "law_id")
        _text(self.version, "version")
        _text(self.citation, "citation")
        names = _names(self.independent)
        if self.scheduling not in names:
            raise ValueError(
                f"scheduling coordinate {self.scheduling!r} not in independent {names}"
            )
        bps = _float_array(self.breakpoints, "breakpoints")
        if bps.ndim != 1 or bps.size < 2:
            raise ValueError("breakpoints must be a 1-D array of at least 2 values")
        if np.any(np.diff(bps) <= 0):
            raise ValueError("breakpoints must be strictly increasing")
        bps.setflags(write=False)
        shape = (bps.size, 3, len(names))
        mats = _checked_matrices(self.matrices, shape, str(shape))
        object.__setattr__(self, "independent", names)
        object.__setattr__(self, "breakpoints", bps)
        object.__setattr__(self, "matrices", _frozen_map(mats))

    @property
    def levels(self) -> tuple[SpinalLevel, ...]:
        """Covered levels, caudal to cranial."""
        return tuple(sorted(self.matrices, key=lambda lv: lv.ordinal))

    def independent_names(self) -> tuple[str, ...]:
        return self.independent

    def _schedule(self, s: float) -> tuple[int, float]:
        """Return segment index and interpolation weight ``w`` in [0, 1]."""
        bps = self.breakpoints
        s_c = min(max(s, float(bps[0])), float(bps[-1]))
        idx = int(np.searchsorted(bps, s_c, side="right")) - 1
        idx = min(max(idx, 0), bps.size - 2)
        return idx, (s_c - float(bps[idx])) / float(bps[idx + 1] - bps[idx])

    def _matrix_and_slope(
        self, level: SpinalLevel, s: float
    ) -> tuple[np.ndarray, np.ndarray]:
        idx, weight = self._schedule(s)
        lo, hi = self.matrices[level][idx], self.matrices[level][idx + 1]
        bps = self.breakpoints
        inside = float(bps[0]) <= s < float(bps[-1])
        slope = (hi - lo) / float(bps[idx + 1] - bps[idx]) if inside else 0.0 * lo
        return lo + weight * (hi - lo), slope

    def expand(self, phi: np.ndarray) -> dict[SpinalLevel, np.ndarray]:
        vec = _check_phi(phi, len(self.independent))
        s = float(vec[self.independent.index(self.scheduling)])
        return {lv: self._matrix_and_slope(lv, s)[0] @ vec for lv in self.levels}

    def jacobian(self, phi: np.ndarray) -> dict[SpinalLevel, np.ndarray]:
        vec = _check_phi(phi, len(self.independent))
        col = self.independent.index(self.scheduling)
        out: dict[SpinalLevel, np.ndarray] = {}
        for lv in self.levels:
            mat, slope = self._matrix_and_slope(lv, float(vec[col]))
            jac = mat.copy()
            jac[:, col] += slope @ vec
            out[lv] = jac
        return out


# ------------------------------------------------- polynomial (MyoFullBody) law


@dataclass(frozen=True)
class PolynomialCouplingLaw:
    """Single-driver polynomial law built by :func:`from_joint_coupling`.

    ``terms`` rows are ``(level, axis, driver_index, offset, x0, coef)`` giving
    ``theta[axis] = offset + sum_k coef[k] (phi[driver_index] - x0)**k``
    (``driver_index < 0``: constant ``offset + coef[0]``). Unlisted axes are
    zero. Same precondition style as the other laws.
    """

    law_id: str
    version: str
    independent: tuple[str, ...]
    terms: tuple[tuple[SpinalLevel, int, int, float, float, np.ndarray], ...]
    citation: str

    def __post_init__(self) -> None:
        _text(self.law_id, "law_id")
        _text(self.version, "version")
        _text(self.citation, "citation")
        names = _names(self.independent)
        _check_contiguous([t[0] for t in self.terms])
        seen: set[tuple[SpinalLevel, int]] = set()
        clean = []
        for level, axis, drv, offset, x0, coef in self.terms:
            if axis not in (0, 1, 2):
                raise ValueError(f"axis must be 0, 1 or 2, got {axis}")
            if (level, axis) in seen:
                raise ValueError(f"duplicate term for {level.name} axis {axis}")
            seen.add((level, axis))
            if not -1 <= drv < len(names):
                raise ValueError(f"driver index {drv} out of range")
            arr = _float_array(coef, "coef")
            if arr.shape != (_POLY_TERMS,):
                raise ValueError(f"coef must have shape ({_POLY_TERMS},)")
            arr.setflags(write=False)
            clean.append((level, int(axis), int(drv), float(offset), float(x0), arr))
        object.__setattr__(self, "independent", names)
        object.__setattr__(self, "terms", tuple(clean))

    @property
    def levels(self) -> tuple[SpinalLevel, ...]:
        """Covered levels, caudal to cranial."""
        return _check_contiguous([t[0] for t in self.terms])

    def independent_names(self) -> tuple[str, ...]:
        return self.independent

    def expand(self, phi: np.ndarray) -> dict[SpinalLevel, np.ndarray]:
        vec = _check_phi(phi, len(self.independent))
        out = {lv: np.zeros(3) for lv in self.levels}
        for level, axis, drv, offset, x0, coef in self.terms:
            x = 0.0 if drv < 0 else float(vec[drv]) - x0
            out[level][axis] = offset + float(np.polynomial.polynomial.polyval(x, coef))
        return out

    def jacobian(self, phi: np.ndarray) -> dict[SpinalLevel, np.ndarray]:
        vec = _check_phi(phi, len(self.independent))
        out = {lv: np.zeros((3, len(self.independent))) for lv in self.levels}
        for level, axis, drv, _offset, x0, coef in self.terms:
            if drv >= 0:
                slope = np.polynomial.polynomial.polyder(coef)
                out[level][axis, drv] = float(
                    np.polynomial.polynomial.polyval(float(vec[drv]) - x0, slope)
                )
        return out


def _driver_name(jc: JointCoupling, level_map: Mapping[int, object], drv: int) -> str:
    if drv not in level_map:
        raise ValueError(f"unmapped driver address {drv} in level_map")
    name = level_map[drv]
    if not isinstance(name, str):
        raise ValueError(
            f"driver address {drv} is a dependent coordinate (chained driver); "
            "only single independent drivers are supported"
        )
    return name


def from_joint_coupling(
    jc: JointCoupling,
    level_map: Mapping[int, str | tuple[SpinalLevel, int]],
    *,
    law_id: str = "myofullbody_joint_coupling",
    version: str = "1",
    citation: str = "MyoFullBody joint equalities (issue #11644)",
) -> PolynomialCouplingLaw:
    """Express single-driver :class:`JointCoupling` equalities as a spine law.

    ``level_map`` maps qpos addresses: a ``str`` marks an independent driver
    coordinate (its name), a ``(SpinalLevel, axis)`` pair marks a dependent
    intervertebral joint (axis 0=FE, 1=LB, 2=AR). Dependent joints absent from
    the map are ignored. The returned law reproduces ``jc.expand`` for every
    mapped dependent joint: ``theta[axis] = qpos0_dep + poly(q_drv - qpos0_drv)``.

    Raises:
        TypeError: if ``jc`` is not a ``JointCoupling``.
        ValueError: on unmapped or non-dependent addresses, a driver that is
            itself dependent, an independent name given to two drivers,
            duplicate ``(level, axis)``, a bad axis or a non-contiguous level set.
    """
    if not isinstance(jc, JointCoupling):
        raise TypeError(f"jc must be a JointCoupling, got {type(jc).__name__}")
    dep_pos = {adr: i for i, adr in enumerate(jc.dependent)}
    names: dict[str, int] = {}
    addr_of_name: dict[int, str] = {}
    for adr, target in level_map.items():
        if isinstance(target, str):
            if adr in dep_pos or not 0 <= adr < jc.n_q:
                raise ValueError(f"unmapped coordinate: address {adr} is not free")
            if target in names:
                raise ValueError(
                    f"independent name {target!r} assigned to two driver "
                    "coordinates (multi-driver)"
                )
            names[target] = adr
            addr_of_name[adr] = target
        elif adr not in dep_pos:
            raise ValueError(f"unmapped coordinate: address {adr} is not dependent")
    order = tuple(names)
    terms = []
    for adr, target in level_map.items():
        if isinstance(target, str):
            continue
        level, axis = target
        drv = int(jc.driver[dep_pos[adr]])
        index = -1 if drv < 0 else order.index(_driver_name(jc, level_map, drv))
        x0 = 0.0 if drv < 0 else float(jc.qpos0[drv])
        terms.append(
            (level, axis, index, float(jc.qpos0[adr]), x0, jc.poly[dep_pos[adr]])
        )
    if not order:
        raise ValueError("unmapped: level_map names no independent driver")
    return PolynomialCouplingLaw(law_id, version, order, tuple(terms), citation)


# ------------------------------------------------------------- restriction


def restrict_to_levels(
    law: CouplingLaw, levels: Sequence[SpinalLevel]
) -> LinearCouplingLaw | ParityCouplingLaw:
    """The sub-law of ``law`` on the contiguous range ``levels`` (SPN-14a).

    Used when a shipped law covers more levels than a model (e.g. a
    full-spine regional law on a lumbar-only fixture).  The retained matrices
    are copied unchanged, so per-level ratios within the range are preserved;
    absolute shares are not (the distribution solve rescales ``phi`` so the
    composed range reaches the target).  This is a repository operation, not
    a published result, and it is recorded in the version
    (``<version>+levels:<first>..<last>``) and the citation.

    Preconditions: ``law`` is a :class:`LinearCouplingLaw` or
    :class:`ParityCouplingLaw` (``TypeError``);
    ``levels`` are :class:`SpinalLevel` (``TypeError``), non-empty, contiguous
    and covered by the law (``ValueError``).

    Postconditions: the result covers exactly ``levels`` with bit-identical,
    read-only matrices; when ``levels`` is the law's whole range the law
    itself is returned.
    """
    if not isinstance(law, LinearCouplingLaw | ParityCouplingLaw):
        raise TypeError(
            "law must be a LinearCouplingLaw or ParityCouplingLaw, "
            f"got {type(law).__name__}"
        )
    if isinstance(levels, str) or not isinstance(levels, Sequence) or not levels:
        raise ValueError("levels must name at least one level")
    wanted = _check_contiguous(list(levels))
    missing = [lv.name for lv in wanted if lv not in law.matrices]
    if missing:
        raise ValueError(f"levels {missing} are not covered by law {law.law_id!r}")
    if wanted == law.levels:
        return law
    span = f"{wanted[0].name}..{wanted[-1].name}"
    common = {
        "law_id": law.law_id,
        "version": f"{law.version}+levels:{span}",
        "matrices": {lv: law.matrices[lv] for lv in wanted},
        "independent": law.independent,
        "citation": f"{law.citation}; restricted to {span} (repository operation)",
    }
    if isinstance(law, ParityCouplingLaw):
        out: LinearCouplingLaw | ParityCouplingLaw = ParityCouplingLaw(
            even_matrices={lv: law.even_matrices[lv] for lv in wanted},
            even_axes=law.even_axes,
            **common,
        )
    else:
        out = LinearCouplingLaw(**common)
    ensure(out.levels == wanted, "restricted law must cover the requested levels")
    return out


# ------------------------------------------------------------ derived quantities


def regional_fraction(
    law: CouplingLaw, levels: Sequence[SpinalLevel], axis: int, phi: np.ndarray
) -> float:
    """Share of the total ``axis`` angle carried by ``levels``.

    ``sum_{levels} theta_i[axis] / sum_{all law levels} theta_i[axis]``.

    Raises:
        ValueError: empty ``levels``, ``axis`` outside 0..2, a level not in the
            law, or a total with magnitude below ``1e-12`` (undefined ratio).
    """
    if axis not in (0, 1, 2):
        raise ValueError(f"axis must be 0 (FE), 1 (LB) or 2 (AR), got {axis!r}")
    chosen = list(levels)
    if not chosen:
        raise ValueError("levels must not be empty")
    angles = law.expand(phi)
    missing = [lv for lv in chosen if lv not in angles]
    if missing:
        raise ValueError(f"levels {[lv.name for lv in missing]} not in law")
    total = sum(float(a[axis]) for a in angles.values())
    if abs(total) < _DENOM_TOL:
        raise ValueError("regional fraction undefined: total denominator is ~0")
    return sum(float(angles[lv][axis]) for lv in set(chosen)) / total


def level_rotations(
    law: CouplingLaw,
    phi: np.ndarray,
    neutral: Mapping[SpinalLevel, np.ndarray] | None = None,
) -> SegmentalPose:
    """Build a :class:`SegmentalPose` from the law's per-level ISB angles.

    ``neutral`` optionally maps levels to (4, 4) neutral offsets (identity for
    levels not listed). Uses :meth:`LevelTransform.from_isb_angles`.

    Raises:
        ValueError: if ``neutral`` names a level the law does not cover.
    """
    angles = law.expand(phi)
    offsets = {} if neutral is None else dict(neutral)
    stray = [lv for lv in offsets if lv not in angles]
    if stray:
        raise ValueError(
            f"neutral given for levels not in law: {[lv.name for lv in stray]}"
        )
    ordered = sorted(angles, key=lambda lv: lv.ordinal)
    pose = SegmentalPose(
        tuple(
            LevelTransform.from_isb_angles(lv, angles[lv], offsets.get(lv))
            for lv in ordered
        )
    )
    ensure(pose.levels == tuple(ordered), "pose must cover the law levels", pose.levels)
    return pose
