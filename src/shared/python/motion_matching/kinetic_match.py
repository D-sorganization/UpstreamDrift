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
   then replace the passive coordinates (a flexible shaft: no actuator, no
   closure force on their rows) by their response to that prescribed motion,
   integrated in ``passive_substep_s`` steps, so their rows balance exactly;
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

__all__ = ["KineticMatchSettings", "kinetic_match", "load_settings"]


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

    def __post_init__(self) -> None:
        if self.method != METHOD:
            raise ValueError(f"unsupported method {self.method!r}")
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
    response to the prescribed motion. ``close_pose`` maps a bundle-order pose to ``(closed pose,
    gap before)``; by default the MuJoCo closure of the bundle's spec.
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
    passive = passive_response(adapter, times, q, v, a, settings.passive_substep_s)
    result = allocate(adapter, times, q, v, a, settings)
    receipt = {
        **_receipt(result, adapter, settings, bundle),
        "closure_gap_max_m": gap,
        "passive_coordinates": passive,
    }
    reference = np.array(bundle.reference_q, dtype=float)
    for native_index in passive_indices(adapter):
        reference[:, perm[native_index]] = q[:, native_index]
    efforts = np.zeros((frames - 1, len(perm)))
    for column, native_index in enumerate(adapter.actuated_indices):
        efforts[:, perm[native_index]] = np.interp(
            times[:-1], result.time_s, np.asarray(result.tau_actuated)[:, column]
        )
    provenance = {**bundle.provenance, "kinetic_match": receipt}
    if hasattr(bundle, "__dataclass_fields__"):
        out = replace(
            bundle, efforts=efforts, provenance=provenance, reference_q=reference
        )
    else:
        out = bundle
        out.efforts, out.provenance = efforts, provenance
        out.reference_q = reference
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
