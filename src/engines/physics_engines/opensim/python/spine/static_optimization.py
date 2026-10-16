"""Trunk static optimisation with cost exponent p and reserves (SPN-10b, #12761).

Problem per selected frame (spinal level coordinates only; couplers disabled,
ADR-0055 item 6)::

    min_a  sum_m a_m^p                       p in {2, 3}
    s.t.   sum_m R_mj (P_m + a_m A_m) + r_j = tau_j     for every level DOF j
           0 <= a_m <= 1,   r = the smallest achievable sum_j (r_j / L_j)^2

``tau`` are the ID generalized forces per coordinate
(:func:`inverse_dynamics.spinal_generalized_forces`, muscles excluded from the
ID), ``R_mj`` the native moment arms (``GeometryPath.computeMomentArm``) and
``L_j`` the declared reserve limits.  The reserves are lexicographic: they
carry exactly the remainder the muscles cannot produce (see
``static_opt_core.solve_activations``), so they are zero to solver precision
whenever ``tau`` is reachable, and never trade against activation.

Muscle model (declared approximation).  Each muscle is evaluated natively with
a RIGID tendon (``ignore_tendon_compliance = true`` on the model copy): the
fibre length and velocity follow from the path length and lengthening speed,
and the tendon force is ``F = P + a A`` with ``A = F0 fL fV cos(alpha)`` and
``P`` the passive (and fibre-damping) force from OpenSim's own curves.  ``A``
and ``P`` are read at activations 0.5 and 1 and extrapolated linearly, which
is exact for the rigid-tendon Millard muscle above its minimum activation
(0.01 by default); below it OpenSim clamps, so ``a = 0`` means the affine
model, not a simulated force.  Activation dynamics are not represented
(static optimisation).

Reserves.  A frame is flagged (never hidden) when the reserve share
``sum |r_j| / sum |tau_j|`` exceeds the declared ``max_share``, when any
``|r_j|`` exceeds its limit ``L_j``, or when the optimiser reports failure.

Runtime contract.  Frames are subsampled by named events (e.g. address, top,
impact: nearest sample) plus a stride; ``max_frames`` makes an over-long
selection an error instead of a silent truncation.  CI uses the fixture and
at most five frames.  The native Bruno-model run on the owner workstation
(three events, reserve share reported) is deferred on #12749.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import MappingProxyType

import numpy as np

from src.engines.physics_engines.opensim.python.spine._osim_helpers import (
    muscle_moment_arm,
)
from src.engines.physics_engines.opensim.python.spine.inventory import (
    ModelLevelMap,
    inventory_spine,
)
from src.engines.physics_engines.opensim.python.spine.muscle_model import (
    SWAP_NOTE,
    check_muscle_model_policy,
)
from src.engines.physics_engines.opensim.python.spine.inverse_dynamics import (
    SpinalGeneralizedForces,
    prepare_model_for_id,
)
from src.engines.physics_engines.opensim.python.static_opt_core import (
    SUPPORTED_EXPONENTS,
    participation_ratio,
    solve_activations,
)
from src.shared.python.contracts import ensure

__all__ = [
    "FrameSelection",
    "ReserveSpec",
    "SpineMuscleBasis",
    "SpineSOOptions",
    "SpineStaticOptimizationResult",
    "flag_frame",
    "reserve_share",
    "run_spine_static_optimization",
    "select_frames",
]

_A_LOW = 0.5
_A_HIGH = 1.0
_ZERO_DEMAND_NM = 1e-12
_SCREEN_STEP_RAD = 1e-4
_SCREEN_LENGTH_TOL_M = 1e-13


def _ro(array: np.ndarray, dtype: type = float) -> np.ndarray:
    out: np.ndarray = np.array(array, dtype=dtype, copy=True)
    out.setflags(write=False)
    return out


def _positive(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"{name} must be a number, got {type(value).__name__}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive, got {value!r}")
    return float(value)


# ------------------------------------------------------------------ options


@dataclass(frozen=True)
class ReserveSpec:
    """Declared reserve limits per spinal DOF and the flagging share.

    ``limit_nm``: default limit ``L_j`` [N*m] for every level coordinate;
    ``per_coordinate``: overrides by coordinate name; ``max_share``: largest
    acceptable ``sum |r| / sum |tau|`` per frame.  All values finite and > 0.
    The limits weight the reserve metric ``sum (r_j / L_j)^2`` and flag any
    ``|r_j| > L_j``.
    """

    limit_nm: float = 1.0
    per_coordinate: Mapping[str, float] | None = None
    max_share: float = 0.05

    def __post_init__(self) -> None:
        _positive(self.limit_nm, "limit_nm")
        _positive(self.max_share, "max_share")
        overrides = dict(self.per_coordinate or {})
        for name, value in overrides.items():
            _positive(value, f"per_coordinate[{name!r}]")
        object.__setattr__(self, "per_coordinate", MappingProxyType(overrides))

    def limits(self, names: tuple[str, ...]) -> np.ndarray:
        """``(len(names),)`` limits [N*m] in ``names`` order."""
        table = self.per_coordinate or {}
        return np.array([float(table.get(n, self.limit_nm)) for n in names])

    def describe(self) -> str:
        """One-line provenance text."""
        extra = ", ".join(
            f"{k}={v:g}" for k, v in sorted((self.per_coordinate or {}).items())
        )
        return (
            f"limit {self.limit_nm:g} N*m per DOF"
            + (f" ({extra})" if extra else "")
            + f"; max share {self.max_share:g}"
        )


@dataclass(frozen=True)
class SpineSOOptions:
    """Settings of :func:`run_spine_static_optimization`.

    ``exponent``: cost exponent p (2 or 3; Bruno 2015 uses 3);
    ``reserves``: :class:`ReserveSpec`; ``events``: name -> time [s];
    ``stride``: every ``stride``-th sample; ``max_frames``: refuse a larger
    selection; ``level_map``: explicit level map for the inventory;
    ``muscle_model_policy``: ``native`` (a Thelen model is refused) or
    ``millard_equivalent`` (see ``muscle_model``); ``unlock_coordinates``:
    locked coordinates to unlock.
    """

    exponent: int = 3
    reserves: ReserveSpec = field(default_factory=ReserveSpec)
    events: Mapping[str, float] | None = None
    stride: int | None = None
    max_frames: int | None = None
    level_map: ModelLevelMap | None = None
    muscle_model_policy: str = "native"
    unlock_coordinates: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        check_muscle_model_policy(self.muscle_model_policy)
        if not isinstance(self.exponent, int) or isinstance(self.exponent, bool):
            raise TypeError("exponent must be an int")
        if self.exponent not in SUPPORTED_EXPONENTS:
            raise ValueError(
                f"exponent must be one of {SUPPORTED_EXPONENTS}, got {self.exponent}"
            )
        if not isinstance(self.reserves, ReserveSpec):
            raise TypeError("reserves must be a ReserveSpec")


# --------------------------------------------------------- frame selection


@dataclass(frozen=True)
class FrameSelection:
    """Selected sample indices (ascending, unique) and one label each."""

    indices: np.ndarray
    labels: tuple[str, ...]


def check_optional_count(value: object, name: str) -> int | None:
    """Validate an optional positive frame count (``None`` passes through).

    Raises:
        TypeError: ``value`` is neither ``None`` nor an ``int`` (bools rejected).
        ValueError: ``value`` is an ``int`` below 1.
    """
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{name} must be an int or None")
    if value < 1:
        raise ValueError(f"{name} must be >= 1, got {value}")
    return value


def select_frames(
    time_s: np.ndarray,
    events: Mapping[str, float] | None = None,
    stride: int | None = None,
    max_frames: int | None = None,
) -> FrameSelection:
    """Frames at named events (nearest sample) plus every ``stride``-th sample.

    With neither ``events`` nor ``stride`` every sample is selected.  An event
    label wins over ``"stride"`` on a shared sample.

    Preconditions: ``time_s`` 1-D, strictly increasing; event times within
    ``[time_s[0], time_s[-1]]``; ``stride``/``max_frames`` ints >= 1.
    Postconditions: indices ascending and unique; ``len <= max_frames``.

    Raises:
        TypeError: ``events`` not a mapping, non-int ``stride``/``max_frames``.
        ValueError: an event outside the time range, bad counts, or more frames
            than ``max_frames``.
    """
    times = np.asarray(time_s, dtype=float)
    if times.ndim != 1 or times.size == 0 or np.any(np.diff(times) <= 0):
        raise ValueError("time_s must be 1-D, non-empty and strictly increasing")
    if events is not None and not isinstance(events, Mapping):
        raise TypeError(f"events must be a mapping, got {type(events).__name__}")
    step = check_optional_count(stride, "stride")
    cap = check_optional_count(max_frames, "max_frames")
    labels: dict[int, str] = {}
    if step is not None:
        labels.update({int(i): "stride" for i in range(0, times.size, step)})
    for name, t in (events or {}).items():
        if not times[0] - 1e-12 <= float(t) <= times[-1] + 1e-12:
            raise ValueError(f"event {name!r} at {t} s is outside the motion")
        labels[int(np.argmin(np.abs(times - float(t))))] = str(name)
    if events is None and step is None:
        labels = dict.fromkeys(range(times.size), "all")
    indices = sorted(labels)
    if cap is not None and len(indices) > cap:
        raise ValueError(
            f"{len(indices)} frames selected but max_frames is {cap}; "
            "use events and a larger stride"
        )
    return FrameSelection(
        _ro(np.array(indices), int), tuple(labels[i] for i in indices)
    )


# ----------------------------------------------------------- reserve flags


def reserve_share(tau: np.ndarray, reserves: np.ndarray) -> float:
    """``sum |r| / sum |tau|``; 0 for no reserve, ``inf`` for reserve on no demand."""
    total = float(np.sum(np.abs(tau)))
    used = float(np.sum(np.abs(reserves)))
    if used == 0.0:
        return 0.0
    return used / total if total > _ZERO_DEMAND_NM else math.inf


def flag_frame(
    tau: np.ndarray,
    reserves: np.ndarray,
    limits: np.ndarray,
    max_share: float,
    *,
    success: bool,
) -> tuple[str, ...]:
    """Reasons a frame is flagged (empty tuple: not flagged).

    Flags: reserve share above ``max_share``; any ``|r_j| > limits_j``; the
    optimiser reported failure.
    """
    reasons = []
    share = reserve_share(tau, reserves)
    if share > max_share:
        reasons.append(f"reserve share {share:.4g} > max_share {max_share:g}")
    over = np.flatnonzero(np.abs(reserves) > limits)
    if over.size:
        reasons.append(f"reserve above limit on DOF indices {over.tolist()}")
    if not success:
        reasons.append("solver did not report success")
    return tuple(reasons)


# --------------------------------------------------------- native muscles


def _muscle_list(model: object) -> list:
    muscles = model.getMuscles()  # type: ignore[attr-defined]
    return [muscles.get(i) for i in range(muscles.getSize())]


def _coordinate_objects(model: object, names: tuple[str, ...]) -> list:
    coords = model.getCoordinateSet()  # type: ignore[attr-defined]
    missing = [n for n in names if coords.getIndex(n) < 0]
    if missing:
        raise ValueError(f"muscle model has no coordinates {missing}")
    return [coords.get(n) for n in names]


class SpineMuscleBasis:
    """Native rigid-tendon muscle forces and moment arms at prescribed states.

    Args:
        model_path: muscle-bearing ``.osim``; constraints (couplers) are
            disabled and every muscle is switched to a rigid tendon.
        model_id: level-map key for :func:`inventory_spine`.
        level_map: explicit level map (else the registry entry).

    ``coordinate_names`` are the level coordinates (caudal first, (flex_ext,
    lat_bend, axial_rot) per level); ``muscle_names`` the model muscles.
    """

    def __init__(
        self,
        model_path: str | Path,
        model_id: str,
        *,
        level_map: ModelLevelMap | None = None,
        muscle_model_policy: str = "native",
        unlock_coordinates: tuple[str, ...] = (),
    ) -> None:
        prepared = prepare_model_for_id(
            model_path,
            muscle_policy=check_muscle_model_policy(muscle_model_policy),
            unlock_coordinates=unlock_coordinates,
        )
        self.muscle_model_policy = muscle_model_policy
        self.swapped_muscles = prepared.swapped_muscles
        self.model = prepared.model
        self._muscles = _muscle_list(self.model)
        if not self._muscles:
            raise ValueError(f"model {Path(model_path).name} has no muscles")
        for muscle in self._muscles:
            muscle.set_ignore_tendon_compliance(True)
        self.state = self.model.initSystem()  # type: ignore[attr-defined]
        inventory = inventory_spine(self.model, model_id, level_map=level_map)
        self.levels = inventory.levels
        self.coordinate_names = tuple(
            c for lv in inventory.levels for c in inventory.coordinate_names[lv]
        )
        self.muscle_names = tuple(str(m.getName()) for m in self._muscles)

    def _pose(self, names: tuple[str, ...], q: np.ndarray, qd: np.ndarray) -> None:
        if q.shape != (len(names),) or qd.shape != (len(names),):
            raise ValueError(f"q and qd must have shape ({len(names)},)")
        for coord, value, speed in zip(
            _coordinate_objects(self.model, names), q, qd, strict=True
        ):
            coord.setValue(self.state, float(value), False)
            coord.setSpeedValue(self.state, float(speed))

    def _forces_at(self, activation: float) -> np.ndarray:
        for muscle in self._muscles:
            muscle.setActivation(self.state, float(activation))
        self.model.realizeDynamics(self.state)  # type: ignore[attr-defined]
        return np.array([m.getActuation(self.state) for m in self._muscles])

    def tendon_force(
        self, names: tuple[str, ...], q: np.ndarray, qd: np.ndarray, activation: float
    ) -> np.ndarray:
        """Native tendon forces [N] of every muscle at one uniform activation."""
        self._pose(names, np.asarray(q, float), np.asarray(qd, float))
        return self._forces_at(activation)

    def _path_lengths(self) -> np.ndarray:
        self.model.realizePosition(self.state)  # type: ignore[attr-defined]
        return np.array([m.getLength(self.state) for m in self._muscles])

    def active_pair_mask(
        self, names: tuple[str, ...], q: np.ndarray, qd: np.ndarray
    ) -> np.ndarray:
        """``(nm, c)`` bool: muscle/level-coordinate pairs worth a native arm.

        Native ``computeMomentArm`` costs tens of milliseconds per pair, which
        makes a full-body model (hundreds of muscles, ~50 level coordinates)
        impractical.  A pair is kept when a central-difference step of the
        coordinate changes the muscle path length, and always kept for a
        locked coordinate (a locked value cannot be perturbed).  A pair
        screened out has an exactly unchanged path length, so its arm is
        reported as zero.  The pose is restored.
        """
        self._pose(names, np.asarray(q, float), np.asarray(qd, float))
        mask = np.ones((len(self._muscles), len(self.coordinate_names)), dtype=bool)
        for j, coord in enumerate(
            _coordinate_objects(self.model, self.coordinate_names)
        ):
            if coord.getLocked(self.state):
                continue
            value = float(coord.getValue(self.state))
            coord.setValue(self.state, value + _SCREEN_STEP_RAD, False)
            plus = self._path_lengths()
            coord.setValue(self.state, value - _SCREEN_STEP_RAD, False)
            minus = self._path_lengths()
            coord.setValue(self.state, value, False)
            mask[:, j] = np.abs(plus - minus) > _SCREEN_LENGTH_TOL_M
        return mask

    def evaluate(
        self, names: tuple[str, ...], q: np.ndarray, qd: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """``(active (nm,), passive (nm,), moment (nm, 3 L))`` at ``(q, qd)``.

        ``names`` order ``q``/``qd`` (all model coordinates).  ``active`` >= 0.
        """
        mask = self.active_pair_mask(names, q, qd)
        self._pose(names, np.asarray(q, float), np.asarray(qd, float))
        high = self._forces_at(_A_HIGH)
        low = self._forces_at(_A_LOW)
        active = np.maximum((high - low) / (_A_HIGH - _A_LOW), 0.0)
        passive = high - _A_HIGH * active
        moment = np.zeros(mask.shape)
        for i, muscle in enumerate(self.muscle_names):
            for j in np.flatnonzero(mask[i]):
                moment[i, j] = muscle_moment_arm(
                    self.model, self.state, muscle, self.coordinate_names[j]
                )
        return active, passive, moment


# ------------------------------------------------------------------ result


@dataclass(frozen=True)
class SpineStaticOptimizationResult:
    """Per-frame trunk static optimisation (arrays read-only).

    ``time_s``/``frame_index``/``frame_labels`` (k,) the selected frames;
    ``muscle_names`` (m,); ``coordinate_names`` (c,) level coordinates;
    ``activation``/``force_n`` (k, m) activation and tendon force [N];
    ``required_nm``/``muscle_moment_nm``/``reserve_nm`` (k, c) the ID
    generalized forces, the muscle part and the reserves [N*m] with
    ``muscle + reserve = required``; ``reserve_share`` (k,);
    ``flagged`` (k,) and ``flag_reasons`` per frame; ``success`` (k,);
    ``reserve_limits_nm`` (c,); ``cost_exponent``; ``provenance``.
    """

    time_s: np.ndarray
    frame_index: np.ndarray
    frame_labels: tuple[str, ...]
    levels: tuple
    muscle_names: tuple[str, ...]
    coordinate_names: tuple[str, ...]
    activation: np.ndarray
    force_n: np.ndarray
    required_nm: np.ndarray
    muscle_moment_nm: np.ndarray
    reserve_nm: np.ndarray
    reserve_share: np.ndarray
    flagged: np.ndarray
    flag_reasons: tuple[tuple[str, ...], ...]
    success: np.ndarray
    reserve_limits_nm: np.ndarray
    cost_exponent: int
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        for name in (
            "time_s",
            "activation",
            "force_n",
            "required_nm",
            "muscle_moment_nm",
            "reserve_nm",
            "reserve_share",
            "reserve_limits_nm",
        ):
            object.__setattr__(self, name, _ro(getattr(self, name)))
        object.__setattr__(self, "frame_index", _ro(self.frame_index, int))
        object.__setattr__(self, "flagged", _ro(self.flagged, bool))
        object.__setattr__(self, "success", _ro(self.success, bool))
        object.__setattr__(self, "provenance", MappingProxyType(dict(self.provenance)))

    @property
    def participation(self) -> np.ndarray:
        """(k,) activation participation ratio per frame (1 if all zero)."""
        return np.array(
            [participation_ratio(a) if np.any(a) else 1.0 for a in self.activation]
        )


# ------------------------------------------------------------------ driver


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _provenance(
    path: Path,
    efforts: SpinalGeneralizedForces,
    opts: SpineSOOptions,
    model_id: str,
    extra: Mapping[str, str],
) -> dict[str, str]:
    return {
        "model_id": model_id,
        "model_sha256": _sha256(path),
        "solver": (
            "static optimisation: lexicographic reserves (BVLS) then min sum a^p "
            "(stage-2 solver per frame in stage2_solver)"
        ),
        "cost_exponent": str(opts.exponent),
        "distribution_source": efforts.provenance.get("distribution_source", "n/a"),
        "external_loads": efforts.provenance.get("external_loads", "n/a"),
        "id_model_sha256": efforts.provenance.get("model_sha256", "n/a"),
        "id_muscles": efforts.provenance.get("muscles", "n/a"),
        "muscle_model": (
            "native Millard curves, rigid tendon (ignore_tendon_compliance), "
            f"affine in activation from a={_A_LOW:g} and a={_A_HIGH:g}"
        ),
        "muscle_model_policy": opts.muscle_model_policy,
        "couplers": "disabled",
        **extra,
        "reserves": opts.reserves.describe(),
        "units": "N*m (generalized forces), N (tendon forces)",
    }


def _run_provenance(basis: SpineMuscleBasis, solvers: set[str]) -> dict[str, str]:
    swapped = basis.swapped_muscles
    return {
        "stage2_solver": ",".join(sorted(solvers)),
        "moment_arm_screen": (
            f"path-length central difference, step {_SCREEN_STEP_RAD:g} rad, "
            f"length threshold {_SCREEN_LENGTH_TOL_M:g} m; locked coordinates "
            "always evaluated natively"
        ),
        "muscles_swapped": str(swapped),
        "muscle_swap_note": SWAP_NOTE if swapped else "none",
    }


def _open_basis(
    model_path: str | Path,
    efforts: SpinalGeneralizedForces,
    model_id: str,
    options: SpineSOOptions | None,
) -> tuple[SpineSOOptions, FrameSelection, SpineMuscleBasis]:
    """Validate the inputs and open the muscle basis for the chosen frames."""
    if not isinstance(efforts, SpinalGeneralizedForces):
        raise TypeError(
            f"efforts must be a SpinalGeneralizedForces, got {type(efforts).__name__}"
        )
    opts = SpineSOOptions() if options is None else options
    if not isinstance(opts, SpineSOOptions):
        raise TypeError("options must be a SpineSOOptions or None")
    selection = select_frames(efforts.time_s, opts.events, opts.stride, opts.max_frames)
    basis = SpineMuscleBasis(
        model_path,
        model_id,
        level_map=opts.level_map,
        muscle_model_policy=opts.muscle_model_policy,
        unlock_coordinates=opts.unlock_coordinates,
    )
    if basis.coordinate_names != efforts.level_coordinates:
        raise ValueError(
            f"level coordinates differ: model {basis.coordinate_names} vs "
            f"efforts {efforts.level_coordinates}"
        )
    return opts, selection, basis


def run_spine_static_optimization(
    model_path: str | Path,
    efforts: SpinalGeneralizedForces,
    model_id: str,
    *,
    options: SpineSOOptions | None = None,
) -> SpineStaticOptimizationResult:
    """Resolve trunk muscle redundancy against ID generalized forces.

    Args:
        model_path: muscle-bearing ``.osim`` with the coordinates of
            ``efforts.model_coordinates`` (couplers are disabled here).
        efforts: :func:`spinal_generalized_forces` output (muscles excluded).
        model_id: level-map key (or ``options.level_map``).
        options: :class:`SpineSOOptions`.

    Returns:
        :class:`SpineStaticOptimizationResult` over the selected frames.
        Postconditions: ``muscle_moment + reserve = required`` per frame;
        activations in [0, 1]; flagged frames are reported, never dropped.

    Raises:
        TypeError: ``efforts`` or ``options`` of the wrong type.
        ValueError: a missing file or coordinate, a level mismatch between
            ``efforts`` and the model, or the frame contract is violated.
    """
    opts, selection, basis = _open_basis(model_path, efforts, model_id, options)
    limits = opts.reserves.limits(basis.coordinate_names)
    tau_all = efforts.level_tau()
    k, nm, nc = selection.indices.size, len(basis.muscle_names), limits.size
    act, force = np.zeros((k, nm)), np.zeros((k, nm))
    muscle, reserve = np.zeros((k, nc)), np.zeros((k, nc))
    share, success, reasons = np.zeros(k), np.zeros(k, bool), []
    solvers: set[str] = set()
    for row, frame in enumerate(selection.indices):
        active, passive, arms = basis.evaluate(
            efforts.model_coordinates, efforts.q[frame], efforts.qd[frame]
        )
        sol = solve_activations(
            active,
            passive,
            arms,
            tau_all[frame],
            exponent=opts.exponent,
            reserve_scale=limits,
        )
        act[row], force[row] = sol.activation, passive + sol.activation * active
        solvers.add(sol.solver)
        muscle[row], reserve[row], success[row] = (
            sol.muscle_moment,
            sol.reserve,
            sol.success,
        )
        share[row] = reserve_share(tau_all[frame], sol.reserve)
        reasons.append(
            flag_frame(
                tau_all[frame],
                sol.reserve,
                limits,
                opts.reserves.max_share,
                success=sol.success,
            )
        )
    required = tau_all[selection.indices]
    ensure(
        bool(np.allclose(muscle + reserve, required, rtol=0.0, atol=1e-9)),
        "muscle moment plus reserve must equal the required generalized force",
    )
    return SpineStaticOptimizationResult(
        time_s=efforts.time_s[selection.indices],
        frame_index=selection.indices,
        frame_labels=selection.labels,
        levels=basis.levels,
        muscle_names=basis.muscle_names,
        coordinate_names=basis.coordinate_names,
        activation=act,
        force_n=force,
        required_nm=required,
        muscle_moment_nm=muscle,
        reserve_nm=reserve,
        reserve_share=share,
        flagged=np.array([bool(r) for r in reasons]),
        flag_reasons=tuple(reasons),
        success=success,
        reserve_limits_nm=limits,
        cost_exponent=opts.exponent,
        provenance=_provenance(
            Path(model_path),
            efforts,
            opts,
            model_id,
            _run_provenance(basis, solvers),
        ),
    )
