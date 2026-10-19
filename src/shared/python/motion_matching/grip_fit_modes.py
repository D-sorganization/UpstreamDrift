"""Grip fitting modes with honest observability (HRG-GR3, #13204).

The capture has no hand or finger markers and the 3-DOF forearm-plus-wrist
chain spans SO(3), so a constant grip rotation is absorbed with no change in
marker residual. ``grip_fit`` therefore only **set-identifies** the grip
through the ``HUMAN_RANGES_DEG`` excess cost: the result is regularised, not
measured. Style, thumb offset, depth and spacing are unobservable and stay
user-declared.

The grip is searched in golf angles ``theta = (alpha, psi, gamma)`` of the
total hand offset. A candidate turns the stack ``relative`` (measured with the
declared grip) by ``E = M(declared)^T M(theta)``. Modes:

``user_fixed``
    No fit; the declared grip is returned, drift exactly zero.
``bounded_prior``
    Minimise ``excess(theta) + lambda ||theta - theta_user||^2`` inside
    ``theta_user +- bound_half_deg``.
``free_fit``
    The same excess over the physical angle ranges, with the weak tie-breaker
    ``lambda ||theta - theta_neutral||^2`` so a zero-cost plateau has one
    answer (the neutral grip is the prior).
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import minimize

from src.shared.python.motion_matching.execution.grip_rotation import (
    GRIP_PROVENANCE_SOURCES,
    apply_grip_config,
    grip_spec_of,
)
from src.shared.python.motion_matching.grip_contract import (
    HandAngles,
    grip_matrix,
    hand_to_euler,
)
from src.shared.python.motion_matching.grip_fit import ExcessObjective
from src.shared.python.motion_matching.grip_report import describe_angles

Array = NDArray[np.float64]
Triple = tuple[float, float, float]

GRIP_FIT_MODES = ("user_fixed", "bounded_prior", "free_fit")
ANGLE_FIELDS = HandAngles.FIELDS
#: A parameter counts as moved by the fit above this many degrees.
MOVED_TOL_DEG = 1e-3
#: Feasible interval: ROM excess within this many degrees of its minimum.
INTERVAL_TOL_DEG = 0.1
#: Default tie-breaker weights, per square degree of distance from the prior.
DEFAULT_LAMBDA = {"bounded_prior": 1e-4, "free_fit": 1e-5}
_DOMAIN = np.array([[-60.0, 60.0], [5.0, 85.0], [-180.0, 180.0]])
_NEUTRAL = np.array([HandAngles().alpha_deg, HandAngles().psi_deg, 0.0])
OBSERVABILITY_NOTE = (
    "set-identified through the joint-range excess cost; regularised, not "
    "measured: no hand markers exist and style, thumb, depth and spacing are "
    "unobservable. The global marker residual is not an angular interval."
)


@dataclass(frozen=True)
class GripFitSettings:
    """Mode and its declared bounds and tie-breaker.

    ``bound_half_deg`` is the half width around the declared angles in
    ``bounded_prior`` (three values or one). ``tie_lambda`` defaults per mode;
    zero removes the tie-breaker. ``ValueError`` for an unknown mode, a
    non-positive bound or a negative or non-finite lambda.
    """

    mode: str = "user_fixed"
    bound_half_deg: tuple[float, float, float] = (20.0, 20.0, 30.0)
    tie_lambda: float | None = None

    def __post_init__(self) -> None:
        if self.mode not in GRIP_FIT_MODES:
            raise ValueError(f"mode must be one of {GRIP_FIT_MODES}, got {self.mode!r}")
        half = np.asarray(self.bound_half_deg, dtype=float)
        if half.shape != (3,) or not np.isfinite(half).all() or (half <= 0).any():
            raise ValueError("bound_half_deg must be three positive finite angles")
        lam = self.tie_lambda
        if lam is not None and not (math.isfinite(lam) and lam >= 0):
            raise ValueError(f"tie_lambda must be finite and >= 0, got {lam!r}")

    @property
    def lam(self) -> float:
        if self.tie_lambda is not None:
            return self.tie_lambda
        return DEFAULT_LAMBDA.get(self.mode, 0.0)


def _triple(hand: HandAngles) -> Array:
    return np.array([hand.alpha_deg, hand.psi_deg, hand.gamma_deg], dtype=float)


def _hand(theta: Array) -> HandAngles:
    return HandAngles(float(theta[0]), float(theta[1]), float(theta[2]))


class HandProblem:
    """One hand's fit problem: the stack measured with ``declared`` in place.

    ``relative`` is ``(frames, 3, 3)``, the forearm-to-hand rotations of the
    document carrying ``declared``. ``ValueError`` for malformed inputs.
    """

    def __init__(
        self,
        relative: Array,
        wrist_base: Array,
        ranges_deg: Sequence[Sequence[float]],
        declared: HandAngles,
        weights: Sequence[float] | None = None,
    ) -> None:
        if not isinstance(declared, HandAngles):
            raise TypeError("declared must be a HandAngles")
        self.objective = ExcessObjective(relative, wrist_base, ranges_deg, weights)
        self.declared = declared
        self.theta0 = _triple(declared)
        self._declared_matrix = grip_matrix(*self.theta0)

    def rotation(self, theta: Array) -> Array:
        """``E = M(declared)^T M(theta)``, the turn applied to ``relative``."""
        return np.asarray(self._declared_matrix.T @ grip_matrix(*theta))

    def rom_cost(self, theta: Array) -> float:
        """RMS excess beyond the ranges, degrees (no tie-breaker)."""
        return self.objective.cost(self.rotation(np.clip(theta, *_DOMAIN.T)))

    def box(self, settings: GripFitSettings) -> Array:
        """``(3, 2)`` search box of a mode, inside the physical domain."""
        if settings.mode == "free_fit":
            return _DOMAIN.copy()
        half = np.asarray(settings.bound_half_deg, dtype=float)
        box = np.stack([self.theta0 - half, self.theta0 + half], axis=1)
        return np.stack(
            [
                np.maximum(box[:, 0], _DOMAIN[:, 0]),
                np.minimum(box[:, 1], _DOMAIN[:, 1]),
            ],
            axis=1,
        )

    def prior(self, settings: GripFitSettings) -> Array:
        return _NEUTRAL if settings.mode == "free_fit" else self.theta0

    def full(self, theta: Array, settings: GripFitSettings, box: Array) -> float:
        clipped = np.clip(theta, box[:, 0], box[:, 1])
        pull = float(np.sum((clipped - self.prior(settings)) ** 2))
        return self.rom_cost(clipped) + settings.lam * pull

    def polish(self, start: Array, settings: GripFitSettings, box: Array) -> Array:
        """Nelder-Mead from ``start`` on the full objective, kept in ``box``."""
        run = minimize(  # type: ignore[call-overload]  # stub lacks bounds for Nelder-Mead
            lambda t: self.full(t, settings, box),
            np.clip(start, box[:, 0], box[:, 1]),
            method="Nelder-Mead",
            bounds=[(float(lo), float(hi)) for lo, hi in box],
            options={"xatol": 1e-3, "fatol": 1e-12, "maxiter": 4000, "adaptive": True},
        )
        return np.clip(np.asarray(run.x, dtype=float), box[:, 0], box[:, 1])

    def seeds(self, settings: GripFitSettings, box: Array) -> list[Array]:
        """Deterministic starts: declared, prior, and the best coarse-grid points."""
        axes = [np.linspace(lo, hi, 7) for lo, hi in box]
        grid = np.array(np.meshgrid(*axes, indexing="ij")).reshape(3, -1).T
        values = [self.full(t, settings, box) for t in grid]
        best = [grid[i] for i in np.argsort(values, kind="stable")[:3]]
        return [self.theta0, self.prior(settings), *best]


def _bound_hits(theta: Any, box: Any) -> dict[str, bool]:
    return {
        name: bool(min(float(t) - float(lo), float(hi) - float(t)) < 1e-6)
        for name, t, (lo, hi) in zip(ANGLE_FIELDS, theta, box, strict=True)
    }


def profile_interval(
    problem: HandProblem,
    settings: GripFitSettings,
    *,
    fitted: HandAngles | None = None,
    n_points: int = 61,
) -> dict[str, dict[str, Any] | None] | None:
    """Per-angle feasible interval from a profile scan, or ``None``.

    For each angle the other two are minimised (pure ROM excess) on a grid of
    ``n_points`` values over the search box (plus the fitted value). The
    interval is the hull of the values whose profile is within
    ``INTERVAL_TOL_DEG`` of the minimum found; its resolution is the grid step.
    ``None`` for ``user_fixed`` (nothing was searched); an angle whose scan has
    no feasible value is ``None``, never a zero-width interval.
    """
    if settings.mode == "user_fixed":
        return None
    box = problem.box(settings)
    centre = _triple(fitted) if fitted is not None else problem.prior(settings)
    scans = []
    for axis in range(3):
        grid = np.union1d(
            np.linspace(box[axis, 0], box[axis, 1], n_points), centre[axis]
        )
        scans.append((grid, [_profile(problem, box, axis, v, centre) for v in grid]))
    floor = min(min(costs) for _, costs in scans)
    out: dict[str, dict[str, Any] | None] = {}
    for axis, (grid, costs) in enumerate(scans):
        ok = grid[np.asarray(costs) <= floor + INTERVAL_TOL_DEG]
        out[ANGLE_FIELDS[axis]] = (
            None
            if ok.size == 0
            else {
                "range_deg": [float(ok.min()), float(ok.max())],
                "tolerance_deg": INTERVAL_TOL_DEG,
                "resolution_deg": float((box[axis, 1] - box[axis, 0]) / (n_points - 1)),
                "min_rom_cost_deg": float(floor),
                "at_search_edge": [
                    bool(ok.min() <= box[axis, 0] + 1e-9),
                    bool(ok.max() >= box[axis, 1] - 1e-9),
                ],
            }
        )
    return out


def _profile(
    problem: HandProblem, box: Array, axis: int, value: float, start: Array
) -> float:
    """Minimum ROM excess with ``theta[axis]`` pinned at ``value``."""
    free = [i for i in range(3) if i != axis]

    def at(x: Array) -> float:
        theta = start.copy()
        theta[axis] = value
        theta[free] = np.clip(x, box[free, 0], box[free, 1])
        return problem.rom_cost(theta)

    run = minimize(
        at,
        start[free],
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-12, "maxiter": 300, "adaptive": True},
    )
    return float(min(run.fun, at(start[free])))


def multistart_sensitivity(
    problem: HandProblem,
    settings: GripFitSettings,
    *,
    n_starts: int = 8,
    seed: int = 0,
) -> dict[str, Any] | None:
    """Spread of the optimum over random starts, or ``None``.

    ``n_starts`` seeded uniform starts in the search box are polished on the
    mode's objective. Starts whose ROM excess is within ``INTERVAL_TOL_DEG`` of
    the best are the equally good set; its per-angle spread (max minus min)
    says how well the mode determines the grip. ``None`` for ``user_fixed``.
    """
    if settings.mode == "user_fixed":
        return None
    if n_starts < 2:
        raise ValueError(f"n_starts must be at least 2, got {n_starts}")
    box = problem.box(settings)
    rng = np.random.default_rng(seed)
    starts = rng.uniform(box[:, 0], box[:, 1], size=(n_starts, 3))
    sols = np.array([problem.polish(s, settings, box) for s in starts])
    roms = np.array([problem.rom_cost(s) for s in sols])
    fulls = np.array([problem.full(s, settings, box) for s in sols])
    keep = roms <= roms.min() + INTERVAL_TOL_DEG
    best = sols[int(np.argmin(fulls))]
    return {
        "n_starts": int(n_starts),
        "seed": int(seed),
        "n_equally_good": int(keep.sum()),
        "best_deg": dict(zip(ANGLE_FIELDS, map(float, best), strict=True)),
        "spread_deg": {
            n: float(np.ptp(sols[keep][:, i])) for i, n in enumerate(ANGLE_FIELDS)
        },
        "rom_cost_range_deg": [float(roms.min()), float(roms.max())],
    }


@dataclass(frozen=True)
class HandGripFit:
    """Result of :func:`fit_hand` for one hand."""

    mode: str
    declared: HandAngles
    fitted: HandAngles
    cost_before: float
    cost_after: float
    bound_hits: dict[str, bool]
    interval: dict[str, dict[str, Any] | None] | None = None
    sensitivity: dict[str, Any] | None = None
    unavailable_reason: str = ""
    fell_back_to_declared: bool = False
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def drift_deg(self) -> Triple:
        d = _triple(self.fitted) - _triple(self.declared)
        return (float(d[0]), float(d[1]), float(d[2]))

    @property
    def rotation_deg(self) -> Triple:
        return hand_to_euler(self.fitted)

    @property
    def moved(self) -> dict[str, bool]:
        return {
            n: abs(v) > MOVED_TOL_DEG
            for n, v in zip(ANGLE_FIELDS, self.drift_deg, strict=True)
        }

    def as_document(self) -> dict[str, Any]:
        """JSON-ready report; unavailable items are ``null`` with a reason."""
        return {
            "mode": self.mode,
            "declared_deg": dict(
                zip(ANGLE_FIELDS, _triple(self.declared).tolist(), strict=True)
            ),
            "fitted_deg": dict(
                zip(ANGLE_FIELDS, _triple(self.fitted).tolist(), strict=True)
            ),
            "drift_deg": dict(zip(ANGLE_FIELDS, self.drift_deg, strict=True)),
            "golf_declared": describe_angles(self.declared),
            "golf_fitted": describe_angles(self.fitted),
            "rotation_deg": list(self.rotation_deg),
            "rom_cost_before_deg": self.cost_before,
            "rom_cost_after_deg": self.cost_after,
            "bound_hits": dict(self.bound_hits),
            "interval": self.interval,
            "sensitivity": self.sensitivity,
            "unavailable_reason": self.unavailable_reason,
            "fell_back_to_declared": self.fell_back_to_declared,
            "observability": OBSERVABILITY_NOTE,
        }


def fit_hand(
    problem: HandProblem,
    settings: GripFitSettings,
    *,
    uncertainty: bool = True,
    n_starts: int = 8,
    seed: int = 0,
) -> HandGripFit:
    """Fit one hand's grip in the mode of ``settings``.

    Postconditions: ``user_fixed`` returns the declared grip with zero drift;
    ``cost_after <= cost_before`` in every mode (a result that would exceed it
    falls back to the declared grip and says so); a ``bounded_prior`` result
    lies inside the declared bounds. With ``uncertainty`` the feasible interval
    and the multistart sensitivity are attached, else they are ``None`` with
    the reason recorded. The result does not depend on ``seed``.
    """
    before = problem.rom_cost(problem.theta0)
    box = problem.box(settings)
    if settings.mode == "user_fixed":
        return HandGripFit(
            "user_fixed",
            problem.declared,
            problem.declared,
            before,
            before,
            _bound_hits(problem.theta0, box),
            unavailable_reason="user_fixed: no fit was run, nothing to bound",
        )
    cands = [problem.polish(s, settings, box) for s in problem.seeds(settings, box)]
    best = min(cands, key=lambda t: problem.full(t, settings, box))
    after = problem.rom_cost(best)
    fell_back = after > before + 1e-12
    if fell_back:
        best, after = np.clip(problem.theta0, box[:, 0], box[:, 1]), before
    fitted = _hand(best)
    interval = sens = None
    reason = "interval and sensitivity not requested"
    if uncertainty:
        interval = profile_interval(problem, settings, fitted=fitted)
        sens = multistart_sensitivity(problem, settings, n_starts=n_starts, seed=seed)
        reason = ""
    return HandGripFit(
        settings.mode,
        problem.declared,
        fitted,
        before,
        after,
        _bound_hits(best, box),
        interval,
        sens,
        reason,
        fell_back,
    )


def grip_mode_of(document: Mapping[str, Any]) -> str:
    """``subject.grip_mode`` of a document; ``user_fixed`` when absent.

    A declared grip never moves unannounced. ``ValueError`` for an unknown mode.
    """
    mode = document.get("subject", {}).get("grip_mode", "user_fixed")
    if mode not in GRIP_FIT_MODES:
        raise ValueError(
            f"subject.grip_mode must be one of {GRIP_FIT_MODES}, got {mode!r}"
        )
    return str(mode)


def fitted_document(
    document: Mapping[str, Any],
    fits: Mapping[str, HandGripFit],
    *,
    mode: str,
) -> dict[str, Any]:
    """New spec revision carrying fitted hands, ``rom_set_identified`` provenance.

    ``fits`` maps ``"lead"``/``"trail"`` to a :class:`HandGripFit`. Only the
    angles a fit moved are stamped ``rom_set_identified`` (a fit that returned
    the declared value identified nothing); all other parameters keep their
    existing provenance. The physical write goes through
    :func:`apply_grip_config`; ``subject.grip_mode`` records ``mode``.
    ``ValueError`` for an unknown mode or hand, or a document without
    ``subject.grip_config``.
    """
    if mode not in GRIP_FIT_MODES:
        raise ValueError(f"mode must be one of {GRIP_FIT_MODES}, got {mode!r}")
    unknown = sorted(set(fits) - {"lead", "trail"})
    if unknown:
        raise ValueError(f"fits keys must be lead/trail, got {unknown}")
    spec = grip_spec_of(document)
    prov = dict(document.get("subject", {}).get("grip_provenance", {}))
    for hand, fit in fits.items():
        spec = (
            replace(spec, lead=fit.fitted)
            if hand == "lead"
            else replace(spec, trail=fit.fitted)
        )
        if mode != "user_fixed":
            for name, moved in fit.moved.items():
                if moved:
                    prov[f"{hand}.{name}"] = GRIP_PROVENANCE_SOURCES[1]
    out = apply_grip_config(document, spec, provenance=prov)
    out = copy.deepcopy(out)
    out["subject"]["grip_mode"] = mode
    return out
