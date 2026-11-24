"""Kinetic match: forces and torques for a kinematic best match (#15685, K1).

A kinematic best match has joint angles but no kinetics. This module computes
one set of joint torques, ground forces and hand-on-club forces that produce
it, so that every new best match can be given forces and torques the same way:

1. reorder the bundle's coordinates into the engine's native order;
2. low-pass the reference (``cutoff_hz``), project every pose onto the
   hand-on-club closure (``closure_tolerance_m``) and differentiate it. An
   open weld gives the loop Jacobian root columns the size of the gap, so the
   grip multiplier becomes a fake external moment that the allocator buys
   instead of the penalised root residual (kN of grip on the current driver);
3. project the velocities and accelerations onto the hand-on-club closure,
   mass-weighted, so ``J v = 0`` and ``J a + Jdot v = 0`` (the matched angles
   do not satisfy the weld exactly, and inverse dynamics of an inconsistent
   state answers with huge internal loop forces);
   then, with ``shaft_source`` ``"response"``, replace the passive
   coordinates (a flexible shaft: no actuator, no closure force on their
   rows) by their response to that prescribed motion, integrated in
   ``passive_substep_s`` steps, so their rows balance exactly. With
   ``"reference"`` the bundle's own bend (fitted to the club markers) is
   kept, and the torque its rows need beyond the spring and damper is
   allocated on them and reported as the model shaft's mismatch with it;
4. allocate torques, ground forces and the closure force by the conic
   allocator (minimum effort, with ``regularisation_grip`` penalising the
   internal loop force), reporting any root residual it could not close;
5. write the torques back into the bundle as efforts, in the bundle's order,
   with a receipt whose plausibility verdict compares the peaks with the
   configured limits.

The arms and club form a closed loop, so infinitely many force sets produce
the same motion. The result is one solution in that null space, chosen by the
objective in the settings; it is never a unique or measured answer. Settings
live in ``src/config/kinetic_match.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
import warnings
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

from src.shared.python.motion_matching.execution.downswing import condition
from src.shared.python.motion_matching.passive_response import (
    passive_indices,
    passive_response,
)

logger = logging.getLogger(__name__)

SETTINGS_PATH = Path(__file__).resolve().parents[3] / "config" / "kinetic_match.json"
METHOD = "inverse_dynamics_allocation"
#: provenance ``replay_mode`` the native export labels (recorded motion, ID efforts)
REPLAY_MODE = "inverse_dynamics"
NULL_SPACE_NOTE = (
    "One minimum-effort solution in the null space of the closed arm loop; "
    "not unique and not measured."
)
MIN_FRAMES = 8
_FLOATS = (
    "cutoff_hz",
    "regularisation_grip",
    "root_penalty_weight",
    "root_residual_limit_n",
    "max_grip_force_n",
    "max_joint_torque_n_m",
    "closure_tolerance_m",
    "passive_substep_s",
)

#: Where the flexible shaft's bend comes from (see the module docstring).
SHAFT_SOURCES = ("response", "reference")

__all__ = ["SHAFT_SOURCES", "KineticMatchSettings", "kinetic_match", "load_settings"]


@dataclass(frozen=True)
class KineticMatchSettings:
    """Validated kinetic-match settings (see ``kinetic_match.json``)."""

    method: str
    engine: str
    cutoff_hz: float
    stride: int
    regularisation_grip: float
    root_penalty_weight: float
    root_residual_limit_n: float
    max_grip_force_n: float
    max_joint_torque_n_m: float
    closure_tolerance_m: float
    passive_substep_s: float
    shaft_source: str

    def __post_init__(self) -> None:
        if self.method != METHOD:
            raise ValueError(f"unsupported method {self.method!r}")
        if self.shaft_source not in SHAFT_SOURCES:
            raise ValueError(f"shaft_source must be one of {SHAFT_SOURCES}")
        if int(self.stride) != self.stride or self.stride < 1:
            raise ValueError("stride must be an integer >= 1")
        for name in _FLOATS:
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} limit must be finite and positive")


def load_settings(path: Path | str = SETTINGS_PATH) -> KineticMatchSettings:
    """Read the settings file; ``ValueError`` on a missing or invalid field."""
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    fields = KineticMatchSettings.__dataclass_fields__
    missing = [name for name in fields if name not in document]
    if missing:
        raise ValueError(f"kinetic match settings miss {missing}")
    return KineticMatchSettings(**{name: document[name] for name in fields})


def _native_order(bundle: Any, adapter: Any) -> list[int]:
    order = list(bundle.coordinate_order)
    native = list(adapter.coordinate_order)
    if sorted(order) != sorted(native):
        raise ValueError("bundle and engine coordinate sets differ")
    return [order.index(name) for name in native]


def _project(adapter: Any, q: np.ndarray, v: np.ndarray, a: np.ndarray) -> None:
    """Mass-weighted projection of ``v`` and ``a`` onto the closure, in place."""
    for k in range(q.shape[0]):
        mass, _ = adapter.compute_mass_and_bias(q[k], v[k])
        jac = adapter.compute_grip_jacobian(q[k])
        m_inv_jt = np.linalg.solve(mass, jac.T)
        gain = m_inv_jt @ np.linalg.pinv(jac @ m_inv_jt, rcond=1e-8)
        v[k] -= gain @ (jac @ v[k])
        a[k] -= gain @ (jac @ a[k] + adapter.compute_closure_drift(q[k], v[k]))


PoseCloser = Callable[[np.ndarray], tuple[np.ndarray, float]]


def _pose_closer(spec_bytes: bytes, tolerance_m: float) -> PoseCloser:
    """Default: Newton projection onto the closure manifold in spec order."""
    from src.shared.python.motion_matching.same_input.closure import (
        project_to_closure,
    )
    from src.shared.python.motion_matching.same_input.plant import VectorPlant

    plant = VectorPlant("mujoco", spec_bytes)

    def close(q: np.ndarray) -> tuple[np.ndarray, float]:
        result = project_to_closure(plant, q, np.zeros_like(q), tolerance=tolerance_m)
        return result.q, result.pose_residual_before

    return close


def _close_poses(close: PoseCloser, q: np.ndarray) -> float:
    """Close every pose of ``q`` in place; the largest gap closed (m)."""
    gap = 0.0
    for k in range(q.shape[0]):
        q[k], before = close(q[k])
        gap = max(gap, float(before))
    return gap


def _allocate(adapter, t, q, v, a, settings: KineticMatchSettings) -> Any:
    """Default allocation: the conic allocator over the engine adapter."""
    from src.shared.python.motion_matching.multi_engine_torque_allocator import (
        MultiEngineTorqueAllocator,
    )

    with warnings.catch_warnings():
        # The shim notice; migrating to loop_nullspace.facade is tracked on #15685.
        warnings.simplefilter("ignore", DeprecationWarning)
        allocator = MultiEngineTorqueAllocator(
            adapter=adapter,
            regularisation_grip=settings.regularisation_grip,
            root_penalty_weight=settings.root_penalty_weight,
        )
    return allocator.allocate_trajectory(t, q, v, a, stride=settings.stride)


class _ShaftActuated:
    """An adapter whose passive (shaft) rows are allocated like actuated ones.

    Their torque is what the measured bend needs beyond the adapter's own
    spring and damper (which stay in its bias), i.e. the shaft's mismatch.
    """

    def __init__(self, adapter: Any, passive: list[int]) -> None:
        self._adapter = adapter
        # Body joints first, in the adapter's order, so their columns keep it.
        self.actuated_indices = (*adapter.actuated_indices, *passive)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._adapter, name)

    def closure_topology(self) -> Any:
        """The wrapped topology with the shaft moved to the actuated side."""
        topology = self._adapter.closure_topology()
        actuated = set(self.actuated_indices)
        return replace(
            topology,
            actuated_indices=self.actuated_indices,
            unactuated_indices=tuple(i for i in range(self.nv) if i not in actuated),
        )


def _shaft_columns(solver: Any, passive: list[int]) -> tuple[list[int], list[int]]:
    """Columns of the allocated torques that are body joints and shaft joints."""
    shaft = set(passive)
    columns = list(enumerate(solver.actuated_indices))
    return [c for c, i in columns if i not in shaft], [
        c for c, i in columns if i in shaft
    ]


def _shaft_mismatch(result: Any, solver: Any, passive: list[int]) -> dict[str, Any]:
    """Peak torque the shaft rows needed beyond their spring (0 for a response)."""
    _, shaft = _shaft_columns(solver, passive)
    if not shaft:
        return {"shaft_mismatch_n_m": 0.0, "shaft_mismatch_coordinate": None}
    tau = np.abs(np.asarray(result.tau_actuated, dtype=float)[:, shaft])
    _, column = np.unravel_index(int(np.argmax(tau)), tau.shape)
    native = solver.actuated_indices[shaft[column]]
    return {
        "shaft_mismatch_n_m": float(tau.max()),
        "shaft_mismatch_coordinate": solver.coordinate_order[native],
    }


def _body_view(result: Any, solver: Any, passive: list[int]) -> Any:
    """``result`` with only the body-joint torque columns (for the peaks)."""
    body, _ = _shaft_columns(solver, passive)
    tau = np.asarray(result.tau_actuated, dtype=float)[:, body]
    fields = {
        name: getattr(result, name)
        for name in ("time_s", "lambda_grip", "delta_tau_root")
    }
    return SimpleNamespace(
        tau_actuated=tau,
        max_equilibrium_residual=result.max_equilibrium_residual,
        **fields,
    )


def _root_window(times: np.ndarray, root_n: np.ndarray, limit: float) -> list[float]:
    bad = np.flatnonzero(root_n > limit)
    end = times[-1] if bad.size == 0 else times[max(bad[0] - 1, 0)]
    return [float(times[0]), float(end)]


def _receipt(result, adapter, settings, bundle) -> dict[str, Any]:
    times = np.asarray(result.time_s, dtype=float)
    tau = np.asarray(result.tau_actuated, dtype=float)
    grip_n = np.linalg.norm(np.asarray(result.lambda_grip)[:, :3], axis=1)
    root_n = np.linalg.norm(np.asarray(result.delta_tau_root)[:, :3], axis=1)
    frame, column = np.unravel_index(int(np.argmax(np.abs(tau))), tau.shape)
    actuated = list(adapter.actuated_indices)
    peaks = {
        "grip_force_n": float(grip_n.max()),
        "grip_force_time_s": float(times[int(np.argmax(grip_n))]),
        "joint_torque_n_m": float(abs(tau[frame, column])),
        "joint_torque_coordinate": adapter.coordinate_order[actuated[column]],
        "joint_torque_time_s": float(times[frame]),
        "root_residual_n": float(root_n.max()),
    }
    reasons = []
    if peaks["grip_force_n"] > settings.max_grip_force_n:
        reasons.append(f"peak grip force {peaks['grip_force_n']:.0f} N over limit")
    if peaks["joint_torque_n_m"] > settings.max_joint_torque_n_m:
        reasons.append(
            f"peak joint torque {peaks['joint_torque_n_m']:.0f} N m on "
            f"{peaks['joint_torque_coordinate']} over limit"
        )
    if peaks["root_residual_n"] > settings.root_residual_limit_n:
        reasons.append(f"root residual {peaks['root_residual_n']:.0f} N over limit")
    return {
        "method": METHOD,
        "settings": asdict(settings),
        "null_space": NULL_SPACE_NOTE,
        "impact_time_s": bundle.provenance.get("impact_time_s"),
        "peaks": peaks,
        "root_residual_window_s": _root_window(
            times, root_n, settings.root_residual_limit_n
        ),
        "max_equilibrium_residual": float(result.max_equilibrium_residual),
        "verdict": "implausible" if reasons else "plausible",
        "reasons": reasons,
    }


def _passive_states(bundle, adapter, perm, q, v) -> dict[str, np.ndarray]:
    """Bundle reference states with the passive columns from the response.

    Keeps the bundle invariant ``reference[0] == q0, v0`` when the bundle
    carries velocities and initial states.
    """
    names = [n for n in ("reference_q", "reference_v") if hasattr(bundle, n)]
    sources = {"reference_q": q, "reference_v": v}
    states = {n: np.array(getattr(bundle, n), dtype=float) for n in names}
    for native_index in passive_indices(adapter):
        for name in names:
            states[name][:, perm[native_index]] = sources[name][:, native_index]
    for name, initial in (("reference_q", "q0"), ("reference_v", "v0")):
        if name in states and hasattr(bundle, initial):
            states[initial] = states[name][0].copy()
    return states


def kinetic_match(
    bundle: Any,
    settings: KineticMatchSettings,
    adapter: Any,
    allocate: Callable[..., Any] = _allocate,
    close_pose: PoseCloser | None = None,
) -> tuple[Any, dict[str, Any]]:
    """Return ``(bundle with efforts, receipt)`` for a kinematic bundle.

    Preconditions: the bundle and the adapter name the same coordinates; the
    bundle has at least ``MIN_FRAMES`` frames. ``ValueError`` otherwise.
    Postconditions: ``efforts`` holds the allocated actuated torques in the
    bundle's coordinate order (zeros on unactuated coordinates), sampled at
    the step times; ``provenance["kinetic_match"]`` is the receipt, with the
    largest closure gap closed (``closure_gap_max_m``) and the number of
    passive coordinates (``passive_coordinates``); the reference angles are
    unchanged except the passive ones (a flexible shaft), which become their
    response to the prescribed motion (with ``q0``/``v0`` kept equal to the
    first reference state). ``close_pose`` maps a bundle-order pose to
    ``(closed pose, gap before)``; by default the MuJoCo closure of the
    bundle's spec.
    """
    perm = _native_order(bundle, adapter)
    frames = bundle.reference_q.shape[0]
    if frames < MIN_FRAMES:
        raise ValueError(f"need at least {MIN_FRAMES} frames, got {frames}")
    times = np.arange(frames) * bundle.dt_s
    closer = close_pose or _pose_closer(bundle.spec_bytes, settings.closure_tolerance_m)
    q = condition(bundle.reference_q, times, settings.cutoff_hz)
    gap = _close_poses(closer, q)
    q = q[:, perm]
    v = np.gradient(q, times, axis=0)
    a = np.gradient(v, times, axis=0)
    _project(adapter, q, v, a)
    passive = list(passive_indices(adapter))
    solver = adapter
    states: dict[str, np.ndarray] = {}
    if settings.shaft_source == "response":
        passive_response(adapter, times, q, v, a, settings.passive_substep_s)
        states = _passive_states(bundle, adapter, perm, q, v)
    elif passive:
        solver = _ShaftActuated(adapter, passive)
    result = allocate(solver, times, q, v, a, settings)
    receipt = {
        **_receipt(_body_view(result, solver, passive), adapter, settings, bundle),
        "closure_gap_max_m": gap,
        "passive_coordinates": len(passive),
        "shaft_source": settings.shaft_source,
        **_shaft_mismatch(result, solver, passive),
    }
    efforts = np.zeros((frames - 1, len(perm)))
    body, _ = _shaft_columns(solver, passive)
    tau = np.asarray(result.tau_actuated, dtype=float)
    for column in body:
        native_index = solver.actuated_indices[column]
        efforts[:, perm[native_index]] = np.interp(
            times[:-1], result.time_s, tau[:, column]
        )
    provenance = {
        **bundle.provenance,
        "kinetic_match": receipt,
        "replay_mode": REPLAY_MODE,
    }
    if hasattr(bundle, "__dataclass_fields__"):
        out = replace(bundle, efforts=efforts, provenance=provenance, **states)
    else:
        out = bundle
        out.efforts, out.provenance = efforts, provenance
        for name, value in states.items():
            setattr(out, name, value)
    return out, receipt


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python3 -m src.shared.python.motion_matching.kinetic_match``."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("bundle", type=Path, help="kinematic input bundle (.npz)")
    parser.add_argument("out", type=Path, help="output bundle with efforts (.npz)")
    parser.add_argument("--settings", type=Path, default=SETTINGS_PATH)
    args = parser.parse_args(argv)
    from src.shared.python.motion_matching.multi_engine_torque_allocator import (
        MujocoForceAdapter,
    )
    from src.shared.python.motion_matching.same_input.bundle import InputBundle

    settings = load_settings(args.settings)
    if settings.engine != "mujoco":
        raise ValueError("only the mujoco engine is wired")
    bundle = InputBundle.load(args.bundle)
    out, receipt = kinetic_match(
        bundle, settings, MujocoForceAdapter(bundle.spec_bytes)
    )
    out.save(args.out)
    receipt_path = args.out.with_suffix(".kinetic_receipt.json")
    receipt_path.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    logging.basicConfig(level=logging.INFO)
    logger.info("kinetic match %s: %s", receipt["verdict"], receipt["reasons"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
