"""Restricted real MuJoCo runner for the research solution prototype.

Only rigid-club, scalar-coordinate, fixed-hold RK4 runs with no impact or a
recorded latched force are supported. A recorded force is replayed without
detecting or certifying physical contact. Clipping is software ideal input.
The runtime fingerprint measures a limited declared subset and does not prove
the full dependency environment or cross-platform determinism.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

from src.shared.python.impact_optimization.option_b_prototype import (
    FixedGainController,
    SplineResidual,
)
from src.shared.python.motion_matching.club_shaft_chain import passive_coordinates
from src.shared.python.motion_matching import club_shaft_chain
from src.shared.python.motion_matching import impact_force
from src.shared.python.impact_parameters import ball_impact
from src.shared.python.motion_matching.same_input import (
    closure,
    integrator,
    plant,
    solution_bundle_prototype,
)
from src.shared.python.motion_matching.same_input.solution_bundle_prototype import (
    Playback,
    SolutionBundle,
)


def measured_runtime_fingerprint() -> str:
    """Hash a declared subset of runtime files; not a complete environment lock."""
    import mujoco
    import mujoco._functions
    import scipy

    from src.engines.physics_engines.mujoco.python import full_body_model, native_mjcf
    from src.shared.python.impact_optimization import option_b_prototype

    modules = (
        integrator,
        plant,
        closure,
        option_b_prototype,
        full_body_model,
        native_mjcf,
        club_shaft_chain,
        solution_bundle_prototype,
        impact_force,
        ball_impact,
    )
    files = {}
    for module in modules:
        if module.__file__ is None:
            raise ValueError("runtime source file is unavailable")
        files[module.__name__] = hashlib.sha256(
            Path(module.__file__).read_bytes()
        ).hexdigest()
    files["runner"] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    files["mujoco_functions"] = hashlib.sha256(
        Path(mujoco._functions.__file__).read_bytes()
    ).hexdigest()
    directory = Path(mujoco.__file__).parent
    candidates = {
        p.resolve()
        for pattern in ("mujoco.dll", "libmujoco.so*", "libmujoco*.dylib")
        for p in directory.glob(pattern)
        if p.is_file()
    }
    if len(candidates) != 1:
        raise ValueError("runtime fingerprint requires exactly one MuJoCo core library")
    core = next(iter(candidates))
    files["mujoco_core"] = hashlib.sha256(core.read_bytes()).hexdigest()
    record = {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "mujoco": mujoco.__version__,
        "files": files,
    }
    return hashlib.sha256(json.dumps(record, sort_keys=True).encode()).hexdigest()


def run_mujoco_ideal(bundle: SolutionBundle) -> Playback:
    """Reintegrate supported ideal-direct input and return actual grid records."""
    b = SolutionBundle(bundle.manifest, bundle.arrays, bundle.model_bytes)
    m, a = b.manifest, b.arrays
    policy = m["runner"]
    required = {"mode", "dt_s", "substeps", "project", "trust_radius", "min_spacing_s"}
    if set(policy) != required or policy.get("mode") != "mujoco_ideal_zoh":
        raise ValueError("runner contract is unavailable or unsupported")
    dt = policy["dt_s"]
    substeps, project = policy["substeps"], policy["project"]
    holds = m["controller"]["holds_per_step"]

    def positive_finite(value: object) -> bool:
        if isinstance(value, bool) or not isinstance(value, int | float):
            return False
        try:
            number = float(value)
        except OverflowError:
            return False
        return bool(np.isfinite(number) and number > 0)

    if not positive_finite(dt):
        raise ValueError("runner dt must be finite and positive")
    if (
        isinstance(substeps, bool)
        or not isinstance(substeps, int)
        or substeps < 1
        or substeps % holds != 0
        or not isinstance(project, bool)
    ):
        raise ValueError("runner integration policy is invalid")
    for name in ("trust_radius", "min_spacing_s"):
        value = policy[name]
        if not positive_finite(value):
            raise ValueError(f"runner {name} must be finite and positive")
    if m["controller"]["internal_state"] != {}:
        raise ValueError("runner only supports a stateless controller")
    if m["environment"] != {} or m["aero"] != {}:
        raise ValueError("runner only supports model-default environment and no flight")
    if (
        m["integrator"]
        != {
            "kind": "rk4_zoh",
            "runtime": "partial_measured_v1",
            "runtime_sha256": m["integrator"]["runtime_sha256"],
        }
        or m["controller"]["saturation"] != "ideal_clip_total"
    ):
        raise ValueError("runner integrator or saturation is unsupported")
    if m["conventions"]["frame"] != "world" or m["conventions"]["quaternion"] != "none":
        raise ValueError("runner requires world frame and scalar coordinates")
    if m["asset_sha256"]:
        raise ValueError("runner cannot verify external assets")
    if measured_runtime_fingerprint() != m["integrator"]["runtime_sha256"]:
        raise ValueError("measured runtime fingerprint mismatch")
    try:
        spec = json.loads(b.model_bytes)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("runner model must be valid JSON") from exc
    names = tuple(m["conventions"]["joint_order"])
    if not isinstance(spec, dict) or spec.get("coordinate_order") != list(names):
        raise ValueError("runner model coordinate inventory mismatch")
    if passive_coordinates(spec):
        raise ValueError("runner requires matching coordinates and a rigid shaft")
    times = a["recorded_time_s"]
    expected_times = np.arange(len(times), dtype=float) * dt
    if not np.array_equal(times, expected_times):
        raise ValueError("recorded time grid must equal fixed runner dt from zero")
    roots = integrator.root_indices(names)
    if np.any(a["saturation_lower"][roots] != 0) or np.any(
        a["saturation_upper"][roots] != 0
    ):
        raise ValueError("unactuated root saturation bounds must both be zero")
    spline = SplineResidual(
        breakpoints_s=a["spline_knots_s"][3:-3],
        coefficients_nm=a["spline_coefficients"],
        scale_nm=a["spline_scale"],
        trust_radius=policy["trust_radius"],
        min_spacing_s=policy["min_spacing_s"],
    )
    controller = FixedGainController(
        times_s=a["reference_time_s"],
        q_ref=a["q_ref"],
        qd_ref=a["qd_ref"],
        tau_ff_ref=a["tau_ff_ref"],
        kp=a["kp"],
        kd=a["kd"],
        lower_nm=a["saturation_lower"],
        upper_nm=a["saturation_upper"],
        residual=spline,
    )
    model = plant.VectorPlant("mujoco", b.model_bytes)
    impact_record = m["impact"]
    impact = None
    events = []
    if impact_record["status"] == "recorded_latched_force":
        if impact_record["frame"] not in model.kinematic_frames(a["q0"]):
            raise ValueError("impact frame is unavailable in model")
        impact = impact_force.ImpactForce(
            frame=impact_record["frame"],
            point_in_frame=np.asarray(impact_record["point_in_frame_m"], dtype=float),
            axis_in_frame=np.asarray(impact_record["axis_in_frame"], dtype=float),
            t_start_s=impact_record["time_s"],
            duration_s=impact_record["duration_s"],
            swing_span_s=(0.0, float(times[-1])),
            force_world=np.asarray(impact_record["force_world_n"], dtype=float),
        )

    def observe_event(time_s, q, v):
        events.append((float(time_s), q.copy(), v.copy()))

    commands, ideal_inputs = [], []

    def effort(_index, time_s, q, v):
        decision = controller.evaluate(time_s, q, v)
        commands.append(decision.commanded_nm.copy())
        ideal_inputs.append(decision.applied_nm.copy())
        return decision.applied_nm

    rollout = integrator.integrate(
        model,
        a["q0"],
        a["qd0"],
        effort,
        steps=len(times) - 1,
        dt_s=float(dt),
        policy=integrator.StepPolicy(
            substeps=substeps, project=project, effort_holds=holds
        ),
        impact=impact,
        impact_event_observer=observe_event if impact is not None else None,
    )
    if rollout.failure is not None or not np.array_equal(
        rollout.efforts, np.asarray(ideal_inputs)
    ):
        raise ValueError("ideal input was modified or rollout failed")
    if len(events) != (0 if impact is None else 1):
        raise ValueError("recorded impact event was not observed exactly once")
    return Playback(
        rollout.time_s,
        np.arange(len(commands), dtype=float) * (float(dt) / holds),
        rollout.q,
        rollout.v,
        np.asarray(commands),
        rollout.efforts,
        None if not events else events[0],
    )
