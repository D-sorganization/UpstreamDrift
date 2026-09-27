"""Differentiable knot trajectory optimisation using MuJoCo MJX (#11049).

Provides the core optimisation and diagnostic rollout functions for refining
tracked motion reference trajectories via Adam gradient descent.
"""

from __future__ import annotations

import json
import math
import xml.etree.ElementTree as ET
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import defusedxml.ElementTree as DET
import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from src.engines.physics_engines.mujoco.python.motion_matching.mjx_tracking_plant import (
    TrackingPlantSpec,
    build_tracking_plant,
    reference_derivatives,
    substep_tables,
)
from src.shared.python.motion_matching.jax_contact import WeldGains
from src.shared.python.motion_matching.knot_gradient_optimiser import (
    AdamSettings,
    adam_minimise,
    horizon_knot_mask,
    knot_basis,
    knot_grid,
)

ARMATURE_KG_M2: float = 5e-3
WELD_STIFFNESS_N_M: float = 2.0e5
WELD_DAMPING_N_S_M: float = 400.0
WELD_ROT_STIFFNESS_N_M_RAD: float = 2.0e3
WELD_ROT_DAMPING_N_M_S: float = 4.0

ROOT_VERTICAL_COORDINATE: str = "TranslationInputZ"


@dataclass(frozen=True)
class MjxPackage:
    """Loaded MJX package components for optimisation."""

    meta: dict[str, Any]
    arrays: dict[str, np.ndarray]
    model: mujoco.MjModel


@dataclass(frozen=True, kw_only=True)
class KnotOptimisationSettings:
    """Settings governing the MJX trajectory knot optimisation."""

    iterations: int = 40
    substeps: int = 6
    knot_spacing_s: float = 0.04
    learning_rate: float = 2e-3
    regularisation: float = 1e-3
    horizon_s: float = 1.65
    weld_stiffness: float = WELD_STIFFNESS_N_M
    weld_damping: float = WELD_DAMPING_N_S_M

    def __post_init__(self) -> None:
        if self.iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {self.iterations}")
        if self.substeps < 1:
            raise ValueError(f"substeps must be >= 1, got {self.substeps}")
        if not math.isfinite(self.regularisation) or self.regularisation < 0.0:
            raise ValueError(
                f"regularisation must be finite and >= 0, got {self.regularisation}"
            )
        if not math.isfinite(self.knot_spacing_s) or self.knot_spacing_s <= 0.0:
            raise ValueError(
                f"knot_spacing_s must be finite and > 0, got {self.knot_spacing_s}"
            )
        if not math.isfinite(self.learning_rate) or self.learning_rate <= 0.0:
            raise ValueError(
                f"learning_rate must be finite and > 0, got {self.learning_rate}"
            )
        if not math.isfinite(self.horizon_s) or self.horizon_s <= 0.0:
            raise ValueError(f"horizon_s must be finite and > 0, got {self.horizon_s}")
        if not math.isfinite(self.weld_stiffness) or self.weld_stiffness <= 0.0:
            raise ValueError(
                f"weld_stiffness must be finite and > 0, got {self.weld_stiffness}"
            )
        if not math.isfinite(self.weld_damping) or self.weld_damping <= 0.0:
            raise ValueError(
                f"weld_damping must be finite and > 0, got {self.weld_damping}"
            )


@dataclass(frozen=True)
class KnotOptimisationResult:
    """Result of knot trajectory optimisation."""

    q_best: np.ndarray
    delta_best: np.ndarray
    history: list[dict[str, Any]]
    port_check_replay_marker_rms_m: float
    best_replay_marker_rms_m: float
    stop_reason: str
    knots: int
    actuated_coordinates: int
    dtype: str


@dataclass(frozen=True)
class DiagnoseResult:
    """Diagnostics from forward rollout of tracking reference.

    ``first_bad_frame`` is -1 when every frame is finite; ``valid`` is the
    (frames x markers) mask of costed samples inside the horizon.
    """

    markers: np.ndarray
    q_sim: np.ndarray
    peak_qvel: np.ndarray
    first_bad_frame: int
    replay_marker_rms_m: float
    valid: np.ndarray


def load_mjx_package(run: Path) -> MjxPackage:
    """Package plus the model with every equality removed: MJX's constraint
    solver is an iterative loop JAX cannot reverse-differentiate, so the grip
    weld is applied here as a stiff spring-damper wrench instead."""
    run_path = Path(run)
    json_path = run_path / "mjx_package.json"
    npz_path = run_path / "mjx_package.npz"
    xml_path = run_path / "mjx_package.xml"

    for p in (json_path, npz_path, xml_path):
        if not p.is_file():
            raise FileNotFoundError(f"Missing required MJX package file: {p}")

    meta = json.loads(json_path.read_text(encoding="utf-8"))
    pkg = dict(np.load(npz_path))
    root = DET.fromstring(xml_path.read_text(encoding="utf-8"))
    for equality in root.findall("equality"):
        root.remove(equality)
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    # The rigid weld of the shared simulator couples the near-massless hand
    # standoff to the club; with a spring weld those dofs need an armature
    # (rotor inertia) floor or they explode. Applied to every dof.
    model.dof_armature[:] = np.maximum(model.dof_armature, ARMATURE_KG_M2)
    return MjxPackage(meta=meta, arrays=pkg, model=model)


def build_reference(
    q_track: jnp.ndarray | np.ndarray,
    act_indices: jnp.ndarray | np.ndarray,
    basis: jnp.ndarray | np.ndarray,
    delta: jnp.ndarray | np.ndarray,
    knot_mask: jnp.ndarray | np.ndarray | None = None,
) -> jnp.ndarray:
    """Tracked reference plus the knot correction; masked knots are frozen."""
    q = jnp.asarray(q_track)
    act = jnp.asarray(act_indices)
    b = jnp.asarray(basis)
    d = (
        jnp.asarray(delta)
        if knot_mask is None
        else jnp.asarray(delta) * jnp.asarray(knot_mask)[:, None]
    )
    return q.at[:, act].add(b @ d)


@dataclass(frozen=True)
class _Problem:
    """Plant, knot basis and costed targets shared by optimise and diagnose."""

    plant: Any
    times: np.ndarray
    q_track: np.ndarray
    targets: jnp.ndarray
    valid: jnp.ndarray
    basis: jnp.ndarray
    knot_mask: jnp.ndarray
    act: np.ndarray
    mass: float
    substeps: int

    @property
    def delta_shape(self) -> tuple[int, int]:
        return (int(self.basis.shape[1]), int(self.act.size))

    def reference(self, delta: jnp.ndarray | np.ndarray) -> jnp.ndarray:
        """Tracked reference plus the knot correction (tail knots frozen)."""
        return build_reference(
            self.q_track, self.act, self.basis, delta, self.knot_mask
        )

    def rollout_tables(
        self, q: jnp.ndarray
    ) -> tuple[Any, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """Initial state and per-substep q, v, a tables for reference ``q``."""
        v, a = reference_derivatives(q, self.times)
        d0 = self.plant.initial_state(q[0], v[0], self.mass)
        return (
            d0,
            substep_tables(q, self.substeps),
            substep_tables(v, self.substeps),
            substep_tables(a, self.substeps),
        )


def _prepare(package: MjxPackage, settings: KnotOptimisationSettings) -> _Problem:
    """Build the tracking plant and knot problem for ``package``.

    Raises ValueError when the package has no ``ROOT_VERTICAL_COORDINATE``.
    """
    meta = package.meta
    pkg = package.arrays
    order = meta["coordinate_order"]
    if ROOT_VERTICAL_COORDINATE not in order:
        raise ValueError(
            f"package coordinate_order has no {ROOT_VERTICAL_COORDINATE!r}"
        )
    weld_gains = WeldGains(
        k=settings.weld_stiffness,
        c=settings.weld_damping,
        rot_k=WELD_ROT_STIFFNESS_N_M_RAD,
        rot_c=WELD_ROT_DAMPING_N_M_S,
    )
    spec = TrackingPlantSpec.from_package(
        meta,
        pkg,
        substeps=settings.substeps,
        root_vertical_index=order.index(ROOT_VERTICAL_COORDINATE),
        weld_gains=weld_gains,
    )
    times = pkg["time_s"]
    in_horizon = (times <= settings.horizon_s)[:, None]
    knots = knot_grid(times, settings.knot_spacing_s)
    return _Problem(
        plant=build_tracking_plant(package.model, spec),
        times=times,
        q_track=pkg["q_track"],
        targets=jnp.asarray(np.nan_to_num(pkg["targets_m"])),
        valid=jnp.asarray(
            pkg["valid"] & np.isfinite(pkg["targets_m"]).all(axis=2) & in_horizon
        ),
        basis=jnp.asarray(knot_basis(times, knots)),
        knot_mask=jnp.asarray(
            horizon_knot_mask(knots, settings.horizon_s), dtype=jnp.float32
        ),
        act=np.asarray(spec.act_indices),
        mass=float(meta["mass_kg"]),
        substeps=settings.substeps,
    )


def _forward_iterate(
    problem: _Problem,
    on_iteration: Callable[[int, np.ndarray, float, float, np.ndarray], None],
) -> Callable[[int, Any, float, float], None]:
    """Adapt the Adam driver's callback to the caller's (k, delta, total, objective, q)."""

    def callback(k: int, x: Any, total: float, objective: float) -> None:
        on_iteration(
            k,
            np.asarray(x),
            float(total),
            float(objective),
            np.asarray(problem.reference(x)),
        )

    return callback


def diagnose_reference(
    package: MjxPackage,
    settings: KnotOptimisationSettings,
) -> DiagnoseResult:
    """Forward rollout diagnostic of the unmodified tracked reference in MJX.

    Writes no files. The replay RMS is taken over finite, valid samples and is
    NaN when there are none.
    """
    problem = _prepare(package, settings)
    q = problem.reference(jnp.zeros(problem.delta_shape))
    m, speed, q_sim = problem.plant.rollout_diagnostic(*problem.rollout_tables(q))
    m, speed, q_sim = np.asarray(m), np.asarray(speed), np.asarray(q_sim)
    finite = np.isfinite(m).all(axis=(1, 2))
    first_bad = int(np.argmin(finite)) if not finite.all() else -1
    err = np.sqrt(np.sum((m - np.asarray(problem.targets)) ** 2, axis=2))
    valid = np.asarray(problem.valid)
    ok = valid & finite[:, None]
    rms = float(np.sqrt(np.mean(err[ok] ** 2))) if ok.any() else float("nan")
    return DiagnoseResult(
        markers=m,
        q_sim=q_sim,
        peak_qvel=speed,
        first_bad_frame=first_bad,
        replay_marker_rms_m=rms,
        valid=valid,
    )


def optimise_reference(
    package: MjxPackage,
    settings: KnotOptimisationSettings,
    *,
    init_delta: jnp.ndarray | np.ndarray | None = None,
    on_iteration: Callable[[int, np.ndarray, float, float, np.ndarray], None]
    | None = None,
) -> KnotOptimisationResult:
    """Optimise knot corrections on the tracked reference with Adam.

    ``init_delta`` must have shape (knots, actuated coordinates) or ValueError
    is raised. ``on_iteration(k, delta, total, objective, q)`` is called once
    per evaluated iterate. Writes no files and does not change JAX config; the
    result records the dtype it ran in. ``history[0]`` is the port check of
    the starting reference.
    """
    problem = _prepare(package, settings)
    expected_shape = problem.delta_shape
    if init_delta is None:
        delta = jnp.zeros(expected_shape)
    else:
        init_arr = np.asarray(init_delta)
        if init_arr.shape != expected_shape:
            raise ValueError(
                f"init_delta shape must be {expected_shape}, got {init_arr.shape}"
            )
        delta = jnp.asarray(init_arr)
    count = float(problem.valid.sum())

    def cost(d: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        m = problem.plant.rollout(*problem.rollout_tables(problem.reference(d)))
        err2 = jnp.sum((m - problem.targets) ** 2, axis=2)
        marker_cost = jnp.sum(jnp.where(problem.valid, err2, 0.0)) / count
        reg = settings.regularisation * jnp.mean(d**2)
        return marker_cost + reg, marker_cost

    value_and_grad = jax.jit(jax.value_and_grad(cost, has_aux=True))

    result = adam_minimise(
        value_and_grad,
        delta,
        AdamSettings(
            learning_rate=settings.learning_rate,
            max_iterations=settings.iterations,
        ),
        xp=jnp,
        on_iteration=(
            None if on_iteration is None else _forward_iterate(problem, on_iteration)
        ),
    )

    history: list[dict[str, Any]] = []
    for row in result.history:
        entry: dict[str, Any] = {
            "iteration": int(row["iteration"]),
            "replay_marker_rms_m": float(np.sqrt(max(0.0, row["objective"]))),
            "total_cost": float(row["total"]),
        }
        if row["iteration"] > 0:
            entry["delta_max_rad"] = float(row["max_abs_x"])
        history.append(entry)

    return KnotOptimisationResult(
        q_best=np.asarray(problem.reference(result.best_x)),
        delta_best=np.asarray(result.best_x),
        history=history,
        port_check_replay_marker_rms_m=history[0]["replay_marker_rms_m"],
        best_replay_marker_rms_m=float(np.sqrt(max(0.0, result.best_objective))),
        stop_reason=result.stop_reason,
        knots=expected_shape[0],
        actuated_coordinates=expected_shape[1],
        dtype=str(jnp.array(1.0).dtype),
    )
