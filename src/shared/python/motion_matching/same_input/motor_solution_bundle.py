"""Research bridge: versioned Solution Bundle to constrained motor replay.

Trusted manifest/model/runtime digests must be stored outside the bundle. This
supports no impact, no projection, no passive shaft and one ZOH hold per step.
Numerical stage equality is not a physical or independent-runtime certificate.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

from src.shared.python.optimization import solution_bundle
from src.shared.python.optimization.solution_bundle import load_solution_bundle
from .motor_forward_replay import (
    MotorForwardCheck,
    measured_motor_runtime,
    verify_motor_forward,
)
from .solution_bundle_prototype import _unique_object
from .stage_numeric_archive import MOTOR_SEMANTICS, decode_numeric

_PAYLOADS = ("motor_stage_npz", "motor_xml", "controller_json", "model_json")
_BASE = {
    "time",
    "q_ref",
    "qd_ref",
    "tau_ff_ref",
    "q_rollout",
    "qd_rollout",
    "tau_cmd",
    "tau_applied",
    "knots",
    "delta_tau_coeffs",
    "q0",
    "qd0",
    "controller_state0",
    "kp",
    "kd",
}


def _digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def measured_bundle_motor_runtime() -> str:
    """Fingerprint this bridge and codec as well as the motor runtime subset."""
    if solution_bundle.__file__ is None:
        raise ValueError("bundle codec source file unavailable")
    parts = [
        measured_motor_runtime(),
        _digest(Path(solution_bundle.__file__).read_bytes()),
        _digest(Path(__file__).read_bytes()),
    ]
    return _digest("\n".join(parts).encode())


def verify_motor_solution(
    directory: str | Path,
    *,
    expected_manifest_sha256: str,
    expected_model_sha256: str,
    expected_runtime_sha256: str,
) -> MotorForwardCheck:
    """Load a v2 bundle, validate its numerical evidence and freshly replay.

    The directory is assumed stable while read; externally trusted hashes must
    not be taken from the bundle itself. No independent runtime or physical
    feasibility claim follows from this restricted replay.
    """
    root = Path(directory)
    if measured_bundle_motor_runtime() != expected_runtime_sha256:
        raise ValueError("bundle runtime fingerprint mismatch")
    if (root / "arrays.npz").stat().st_size > 512 * 1024 * 1024:
        raise ValueError("bundle archive exceeds size limit")
    manifest = (root / "manifest.json").read_bytes()
    if _digest(manifest) != expected_manifest_sha256:
        raise ValueError("manifest hash mismatch")
    bundle = load_solution_bundle(root, expected_model_hash=expected_model_sha256)
    if _digest((root / "manifest.json").read_bytes()) != expected_manifest_sha256:
        raise ValueError("manifest changed while loading")
    if bundle.schema_version != 2 or bundle.impact_event != {"status": "none"}:
        raise ValueError("only explicit v2 no-impact evidence is supported")
    if bundle.frame != "world" or bundle.environment or bundle.flight_metrics:
        raise ValueError("unsupported frame, environment or flight")
    if bundle.control != {
        "kind": "fixed_pd_bspline",
        "hold": "zoh",
        "saturation": "mujoco_motor",
        "holds_per_step": 1,
    }:
        raise ValueError("unsupported controller policy")
    if set(bundle.arrays) != _BASE | set(_PAYLOADS):
        raise ValueError("unsupported motor bundle array inventory")
    data = {}
    for name in _PAYLOADS:
        payload = bundle.arrays[name]
        if payload.dtype != np.uint8 or payload.ndim != 1 or not payload.size:
            raise ValueError(f"invalid {name} bytes")
        data[name] = payload.tobytes()
    if _digest(data["model_json"]) != expected_model_sha256:
        raise ValueError("model hash mismatch")
    assets = {name: _digest(data[name]) for name in _PAYLOADS if name != "model_json"}
    if bundle.asset_hashes != assets:
        raise ValueError("motor asset hash mismatch")
    try:
        cfg = json.loads(data["controller_json"], object_pairs_hook=_unique_object)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("invalid controller JSON") from exc
    trace = decode_numeric(data["motor_stage_npz"], expected_semantics=MOTOR_SEMANTICS)
    plan = trace.plan
    if plan is None or plan.effort_holds != 1 or plan.project:
        raise ValueError("unsupported stage plan")
    if tuple(bundle.joint_order) != plan.coordinate_order:
        raise ValueError("coordinate inventory mismatch")
    expected_integrator = {
        "kind": "rk4",
        "dt_s": plan.dt_s,
        "substeps": plan.substeps,
        "project": False,
        "runtime_sha256": expected_runtime_sha256,
    }
    if bundle.integrator != expected_integrator:
        raise ValueError("integrator policy mismatch")
    replay = verify_motor_forward(
        data["motor_stage_npz"],
        data["model_json"],
        data["motor_xml"],
        data["controller_json"],
        expected_archive_sha256=assets["motor_stage_npz"],
        expected_model_sha256=expected_model_sha256,
        expected_motor_sha256=assets["motor_xml"],
        expected_controller_sha256=assets["controller_json"],
        expected_runtime_sha256=measured_motor_runtime(),
    )
    a = bundle.arrays
    times = np.arange(plan.requested_steps + 1, dtype=float) * plan.dt_s
    if cfg["times_s"][0] > times[0] or cfg["times_s"][-1] < times[-1]:
        raise ValueError("reference does not cover rollout")
    q = np.vstack((np.asarray(cfg["q0"]), *(x.post_projection_q for x in trace.steps)))
    v = np.vstack((np.asarray(cfg["v0"]), *(x.post_projection_v for x in trace.steps)))
    command = np.stack([x.stages[0].tau for x in trace.steps])
    drive = np.stack([x.stages[0].applied_drive for x in trace.steps])
    for name, expected in (
        ("time", times),
        ("q0", cfg["q0"]),
        ("qd0", cfg["v0"]),
        ("q_rollout", q),
        ("qd_rollout", v),
        ("tau_cmd", np.vstack((command, command[-1]))),
        ("tau_applied", np.vstack((drive, drive[-1]))),
        ("kp", cfg["kp"]),
        ("kd", cfg["kd"]),
        ("delta_tau_coeffs", cfg["coefficients"]),
    ):
        if not np.array_equal(a[name], np.asarray(expected)):
            raise ValueError(f"{name} boundary/controller mismatch")
    if a["controller_state0"].size:
        raise ValueError("controller state is unsupported")
    points = np.asarray(cfg["breakpoints_s"], dtype=float)
    knots = np.r_[np.repeat(points[0], 4), points[1:-1], np.repeat(points[-1], 4)]
    if not np.array_equal(a["knots"], knots):
        raise ValueError("spline knots mismatch")
    for name in ("q_ref", "qd_ref", "tau_ff_ref"):
        values = np.asarray(cfg[name], dtype=float)
        sampled = np.column_stack(
            [
                np.interp(times, cfg["times_s"], values[:, i])
                for i in range(len(bundle.joint_order))
            ]
        )
        if not np.array_equal(a[name], sampled):
            raise ValueError(f"{name} reference mismatch")
    return replay
