"""Muscle-redundancy pipeline driven by the matched MuJoCo swing (issue #11617).

Phase 2 replaces the phase-1 inputs (IK states with a degraded tail and an
indeterminate ground reaction) with the forward-dynamics-consistent same-input
bundle: reference ``q``/``v``/efforts at 1 ms, the exact deterministic contact
law, and the spec skeleton.  Per sampled frame the pipeline

1. evaluates open-chain inverse dynamics on the OpenSim skeleton with the exact
   contact forces and compares with the bundle efforts (leg/pelvis/trunk only,
   because the grip-weld reaction acts on the arm coordinates);
2. checks that the muscle model's skeleton poses bodies identically;
3. resolves leg-muscle redundancy (``musculoskeletal_static_opt``) against the
   reference leg efforts; arms and trunk keep the reference torques as
   stand-ins because no upper-limb muscles exist in the lower-limb model.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python import (
    musculoskeletal_static_opt as so,
)
from src.engines.physics_engines.opensim.python.full_body_osim import (
    clean_osim_body_name,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_grf import (
    write_external_loads,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_spec_bundle import (
    SpecBundle,
    load_spec_bundle,
    sha256_of,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_spec_dynamics import (
    SkeletonDynamics,
    foot_contact_points,
    is_loop_coordinate,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_spec_model import (
    build_spec_musculoskeletal_model,
)
from src.shared.python.contracts import require

RECEIPT_SCHEMA = "musculoskeletal-spec-receipt/v2"
ROOT_PREFIXES = ("pelvis_", "Hip", "hip_tx", "hip_ty", "hip_tz")


@dataclass(frozen=True)
class V2Config:
    """Run configuration.  Strides are in 1 ms bundle steps."""

    bundle: Path
    out_dir: Path
    base_model: Path | None = None
    id_stride: int = 10
    so_stride: int = 5
    reserve_weight: float = 1.0

    def __post_init__(self) -> None:
        require(self.id_stride >= 1 and self.so_stride >= 1, "strides must be >= 1")


def _stats(x: np.ndarray) -> dict[str, float]:
    a = np.abs(x)
    return {
        "rms": float(np.sqrt(np.mean(x**2))),
        "peak": float(a.max()),
        "median_abs": float(np.median(a)),
        "p95_abs": float(np.percentile(a, 95)),
    }


def impact_index(dyn: SkeletonDynamics, bundle: SpecBundle, stride: int = 5) -> int:
    """Step index of maximum club-head (clubface body origin) speed."""
    face = clean_osim_body_name(dyn.spec["closure"]["body_b"])
    idx = np.arange(0, bundle.steps, stride)
    pos = np.empty((len(idx), 3))
    for n, k in enumerate(idx):
        dyn.set_state(bundle.q[k], bundle.v[k])
        pos[n] = dyn.body_origins([face])[0]
    speed = np.linalg.norm(np.gradient(pos, idx * bundle.dt_s, axis=0), axis=1)
    return int(idx[int(np.argmax(speed))])


def inverse_dynamics_check(
    dyn: SkeletonDynamics, bundle: SpecBundle, stride: int, impact: int
) -> dict[str, Any]:
    """Compare open-chain ID (exact contact) with ``bundle.efforts``."""
    acc = bundle.step_accelerations()
    qm, vm = bundle.step_midpoint_states()
    loop = np.array([is_loop_coordinate(n) for n in bundle.coordinate_order])
    ks = np.arange(0, bundle.steps, stride)
    err = np.empty((len(ks), bundle.nv))
    for n, k in enumerate(ks):
        effort, _ = dyn.required_efforts(qm[k], vm[k], acc[k])
        err[n] = effort - bundle.efforts[k]
    out: dict[str, Any] = {
        "frames": int(len(ks)),
        "loop_coordinates": [
            n for n, m in zip(bundle.coordinate_order, loop, strict=True) if m
        ],
    }
    for label, mask in (("non_loop", ~loop), ("loop", loop)):
        out[label] = _stats(err[:, mask])
    pre = ks < impact
    out["non_loop_pre_impact"] = _stats(err[pre][:, ~loop])
    out["non_loop_post_impact"] = _stats(err[~pre][:, ~loop])
    worst = np.abs(err[:, ~loop]).max(axis=0)
    names = [n for n, m in zip(bundle.coordinate_order, loop, strict=True) if not m]
    top = np.argsort(worst)[::-1][:5]
    out["worst_non_loop"] = {names[i]: float(worst[i]) for i in top}
    return out


def body_origin_mismatch(
    dyn: SkeletonDynamics, basis: so.MuscleBasis, bundle: SpecBundle, stride: int
) -> dict[str, Any]:
    """Max body-origin distance (mm) between the skeleton and the muscle model."""
    names = [b.getName() for b in dyn.model.getBodySet()]
    msk_set = basis.model.getBodySet()
    worst = 0.0
    where = ""
    for k in range(0, bundle.steps, stride):
        dyn.set_state(bundle.q[k], bundle.v[k])
        ref = dyn.body_origins(names)
        basis.set_state(bundle.q[k], bundle.v[k])
        basis.model.realizePosition(basis.state)
        for i, name in enumerate(names):
            p = msk_set.get(name).getTransformInGround(basis.state).p()
            d = float(np.linalg.norm(np.array([p.get(j) for j in range(3)]) - ref[i]))
            if d > worst:
                worst, where = d, f"{name}@{k}"
    return {"max_mm": worst * 1e3, "at": where, "bodies": len(names)}


def parity_body_origin_mismatch(
    basis: so.MuscleBasis, bundle: SpecBundle, stride: int
) -> dict[str, Any]:
    """Body-origin distance (mm) between the MuJoCo parity model and the muscle model.

    The MuJoCo model is the MJCF export of the same spec the reference was
    produced with, so this is the check that the OpenSim model poses the bodies
    where the reference plant does.  Constant offsets between exporters'
    body-frame conventions (the club body, the grip standoff and the tibia frame)
    are reported per body rather than hidden.
    """
    import mujoco

    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    xml, _ = export_full_body_mjcf(bundle.spec_bytes)
    mj = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(mj)
    address = [mj.joint(n).qposadr[0] for n in bundle.coordinate_order]
    pairs = [
        (i, clean_osim_body_name(mujoco.mj_id2name(mj, mujoco.mjtObj.mjOBJ_BODY, i)))
        for i in range(1, mj.nbody)
    ]
    bodies = basis.model.getBodySet()
    worst = dict.fromkeys((n for _, n in pairs), 0.0)
    for k in range(0, bundle.steps, stride):
        data.qpos[address] = bundle.q[k]
        mujoco.mj_kinematics(mj, data)
        basis.set_state(bundle.q[k], bundle.v[k])
        basis.model.realizePosition(basis.state)
        for i, name in pairs:
            p = bodies.get(name).getTransformInGround(basis.state).p()
            gap = np.array([p.get(j) for j in range(3)]) - data.xpos[i]
            worst[name] = max(worst[name], float(np.linalg.norm(gap)))
    per_body = {n: round(v * 1e3, 3) for n, v in worst.items()}
    excluded = {"Clubhead"}
    return {
        "per_body_max_mm": per_body,
        "max_mm_excluding_club_body": max(
            v for n, v in per_body.items() if n not in excluded
        ),
        "club_body_note": "OpenSim 'Clubhead' origin is the head; MuJoCo's club body"
        " origin is the grip end (a fixed 1.1 m convention offset)",
        "frames": len(range(0, bundle.steps, stride)),
    }


def ground_reaction(
    dyn: SkeletonDynamics, bundle: SpecBundle, stride: int, out_dir: Path
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Exact per-sphere contact forces at the sampled steps; writes ExternalLoads."""
    ks = np.arange(0, bundle.steps, stride)
    qm, vm = bundle.step_midpoint_states()
    forces = np.empty((len(ks), len(dyn.spheres), 3))
    for n, k in enumerate(ks):
        dyn.set_state(qm[k], vm[k])
        forces[n] = dyn.contact_generalised_force()[1]
    times = ks * bundle.dt_s
    write_external_loads(
        out_dir, times, forces, foot_contact_points(dyn.spheres), stem="grf_exact"
    )
    fz = forces[:, :, 2]
    total = fz.sum(axis=1)
    mass = sum(s["mass_kg"] for b in dyn.spec["bodies"] for s in b["solids"])
    weight = mass * 9.80665
    summary = {
        "frames": int(len(ks)),
        "total_mass_kg": float(mass),
        "peak_vertical_n": float(total.max()),
        "peak_vertical_bw": float(total.max() / weight),
        "min_vertical_n": float(total.min()),
        "mean_vertical_bw": float(total.mean() / weight),
        "contact_free_fraction": float(np.mean(total <= 1e-9)),
    }
    return times, forces, summary


def static_optimisation(
    basis: so.MuscleBasis, bundle: SpecBundle, stride: int, reserve_weight: float
) -> dict[str, np.ndarray]:
    """Per-frame bounded least-squares muscle redundancy on the leg coordinates."""
    qm, vm = bundle.step_midpoint_states()
    ks = np.arange(0, bundle.steps, stride)
    cols = [bundle.index(n) for n in basis.coords]
    activation = np.empty((len(ks), len(basis.muscles)))
    reserve = np.empty((len(ks), len(cols)))
    force = np.empty_like(activation)
    for n, k in enumerate(ks):
        active, passive, moment = basis.evaluate(qm[k], vm[k])
        tau = bundle.efforts[k, cols]
        sol = so.solve_frame(
            active, passive, moment, tau, reserve_weight=reserve_weight
        )
        activation[n] = sol.activation
        reserve[n] = sol.reserve
        force[n] = passive + sol.activation * active
    return {
        "steps": ks,
        "activation": activation,
        "reserve": reserve,
        "force": force,
        "tau": bundle.efforts[
            np.ix_(np.asarray(ks, dtype=np.intp), np.asarray(cols, dtype=np.intp))
        ],
    }


def upper_body_summary(bundle: SpecBundle) -> dict[str, dict[str, float]]:
    """RMS and peak of the reference efforts of non-leg, non-root coordinates."""
    leg = set(so.leg_coordinates())
    out = {}
    for i, name in enumerate(bundle.coordinate_order):
        if name in leg or name.startswith(ROOT_PREFIXES):
            continue
        out[name] = _stats(bundle.efforts[:, i])
    return out


def run_v2(config: V2Config) -> dict[str, Any]:
    """Run the complete phase-2 pipeline and return the receipt dictionary."""
    bundle = load_spec_bundle(config.bundle)
    config.out_dir.mkdir(parents=True, exist_ok=True)
    dyn = SkeletonDynamics(bundle.spec_bytes)
    model, info = build_spec_musculoskeletal_model(bundle.spec_bytes, config.base_model)
    basis = so.MuscleBasis(model, bundle.coordinate_order)
    impact = impact_index(dyn, bundle)
    result = static_optimisation(basis, bundle, config.so_stride, config.reserve_weight)
    ks = result["steps"]
    pre = ks < impact
    reserve = result["reserve"]
    receipt: dict[str, Any] = {
        "schema": RECEIPT_SCHEMA,
        "capture": bundle.capture,
        "bundle_sha256": bundle.sha256,
        "base_model": info["base_model"],
        "base_model_sha256": sha256_of(info["base_model"]),
        "dt_s": bundle.dt_s,
        "steps": bundle.steps,
        "impact_step": impact,
        "impact_time_s": impact * bundle.dt_s,
        "model": {k: info[k] for k in ("n_muscles", "n_wrap_objects", "n_coordinates")},
        "inverse_dynamics": inverse_dynamics_check(
            dyn, bundle, config.id_stride, impact
        ),
        "body_origin_mismatch": body_origin_mismatch(
            dyn, basis, bundle, config.id_stride * 5
        ),
        "parity_body_origin_mismatch": parity_body_origin_mismatch(
            basis, bundle, config.id_stride * 5
        ),
    }
    times, forces, grf = ground_reaction(dyn, bundle, config.so_stride, config.out_dir)
    receipt["ground_reaction"] = grf
    receipt["static_optimisation"] = {
        "frames": int(len(ks)),
        "reserve_weight": config.reserve_weight,
        "reserve_all": _stats(reserve),
        "reserve_pre_impact": _stats(reserve[pre]),
        "reserve_post_impact": _stats(reserve[~pre]),
        "torque_rms": float(np.sqrt(np.mean(result["tau"] ** 2))),
        "activation_mean": float(result["activation"].mean()),
        "activation_peak": float(result["activation"].max()),
        "saturated_fraction": float(np.mean(result["activation"] > 0.99)),
        "muscle_force_peak_n": float(result["force"].max()),
    }
    receipt["upper_body_torque_standins"] = upper_body_summary(bundle)
    arrays: dict[str, Any] = {
        "times": times,
        "muscle_names": np.array(basis.muscle_names),
        "coordinates": np.array(basis.coords),
        **result,
    }
    np.savez_compressed(config.out_dir / "static_opt.npz", **arrays)
    digest = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, default=str).encode()
    ).hexdigest()
    receipt["receipt_digest"] = digest
    return receipt


def compare_with_v1(receipt: dict[str, Any], v1: dict[str, Any]) -> dict[str, Any]:
    """Phase-1 versus phase-2 reserve magnitudes (leg coordinates and pelvis).

    ``v1`` is the phase-1 receipt (``results.reserve_torques`` per actuator).  The
    phase-2 pipeline has no pelvis residual at all (the root is driven by the
    exact contact forces), so only the phase-1 pelvis figures are reported.
    """
    rows = v1["results"]["reserve_torques"]
    legs = {k: v for k, v in rows.items() if "pelvis" not in k}
    pelvis = {k: v for k, v in rows.items() if "pelvis" in k}
    v2 = receipt["static_optimisation"]
    return {
        "v1_leg_reserve_peak": max(r["peak"] for r in legs.values()),
        "v1_leg_reserve_rms_max_coordinate": max(r["rms"] for r in legs.values()),
        "v1_pelvis_residual_peak": max(r["peak"] for r in pelvis.values()),
        "v1_pelvis_residual_rms_max": max(r["rms"] for r in pelvis.values()),
        "v2_leg_reserve_peak": v2["reserve_all"]["peak"],
        "v2_leg_reserve_rms_all": v2["reserve_all"]["rms"],
        "v2_pelvis_residual": "none: root driven by exact contact (ID check above)",
    }
