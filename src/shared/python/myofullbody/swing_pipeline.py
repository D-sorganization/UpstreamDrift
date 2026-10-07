"""End-to-end MyoFullBody swing analysis for one bundle (issues #11644, #11645).

Maps the same-input bundle onto MyoFullBody, resolves muscle redundancy per
frame against the bundle's spec efforts, and returns a fail-closed receipt plus
the arrays the renderer needs.  See :mod:`redundancy` for the formulation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.myofullbody import (
    assets,
    attribution,
    mapping,
    mapping_report,
    neck,
    redundancy,
    share_ik,
)

Array = np.ndarray
SCHEMA = "myofullbody-swing-receipt-v1"
SPEC_MASS_KG = 78.0
WEIGHT_SENSITIVITY = (1.0, 10.0, 1000.0)
FORCE_SCALE_SENSITIVITY = (0.95, 1.5)
TOP_MUSCLES = 20
ROM_POLICIES = ("share", "extend")
IMPACT_HALF_WIDTH_STEPS = 10


@dataclass(frozen=True)
class SwingConfig:
    """Inputs of one run."""

    bundle: Path
    stride: int = 5
    reserve_weight: float = 100.0
    cache_root: Path | None = None
    invocation: str = ""
    rom_policy: str = "share"
    neck_actuator: bool = True
    workers: int = 1

    def __post_init__(self) -> None:
        require(self.stride >= 1, "stride must be >= 1")
        require(
            self.rom_policy in ROM_POLICIES, f"rom_policy must be in {ROM_POLICIES}"
        )
        require(self.workers >= 1, "workers must be >= 1")
        require(self.reserve_weight > 0.0, "reserve_weight must be positive")


@dataclass
class SwingResult:
    """Receipt and arrays of one run."""

    receipt: dict[str, Any]
    arrays: dict[str, Any] = field(repr=False)


def frame_steps(steps: int, stride: int, keys: mapping_report.KeyFrames) -> Array:
    """Strided step indices plus the key-frame steps, sorted and unique.

    Raises:
        ValueError: if a key frame lies outside ``[0, steps)``.
    """
    extra = list(keys.as_dict().values())
    require(all(0 <= k < steps for k in extra), "key frame outside the swing")
    return np.unique(np.concatenate([np.arange(0, steps, stride), extra]))


def _solve_all(
    bases: list[redundancy.FrameBasis],
    tau: Array,
    weight: float,
    scale: float = 1.0,
    passive_scale: float = 1.0,
) -> tuple[Array, Array, bool]:
    n, nm = len(bases), bases[0].active.shape[0]
    act, res = np.empty((n, nm)), np.empty_like(tau)
    ok = True
    for i, b in enumerate(bases):
        sol = redundancy.solve_frame(
            scale * b.active,
            scale * passive_scale * b.passive,
            b.moment,
            tau[i],
            reserve_weight=weight,
        )
        act[i], res[i], ok = sol.activation, sol.reserve, ok and sol.success
    return act, res, ok


def _id_check(
    basis: redundancy.MuscleBasis,
    plant: Any,
    bundle: Any,
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Independent checks of the solved torques (see module docstring of redundancy)."""
    ks, qm, vm = ctx["ks"], ctx["qm"], ctx["vm"]
    accel = np.diff(bundle.reference_v, axis=0) / bundle.dt_s
    cols = basis.columns
    closure, err_plain, err_rebuilt, ref = [], [], [], []
    for i, k in enumerate(ks):
        basis.set_state(qm[k], vm[k], ctx["poses"][i])
        tau_m = basis.generalised_force(ctx["act"][i], ctx["phi"][i]) + ctx["neck"][i]
        closure.append(np.abs(tau_m + ctx["res"][i] - ctx["tau"][i]).max())
        tau_full = bundle.efforts[k].copy()
        tau_full[cols] = tau_m + ctx["res"][i]
        a_plain = plant.acceleration(qm[k], vm[k], bundle.efforts[k])
        a_rebuilt = plant.acceleration(qm[k], vm[k], tau_full)
        err_plain.append(a_plain - accel[k])
        err_rebuilt.append(a_rebuilt - a_plain)
        ref.append(accel[k])
    scale = float(np.sqrt(np.mean(np.square(ref))))
    rms = lambda x: float(np.sqrt(np.mean(np.square(x))))  # noqa: E731
    return {
        "torque_closure_max_nm": float(max(closure)),
        "torque_closure_limit_nm": redundancy.TORQUE_CLOSURE_LIMIT_NM,
        "plant_acceleration_rms_error_plain_efforts": rms(err_plain),
        "plant_acceleration_rms_difference_muscle_reserve_vs_plain": rms(err_rebuilt),
        "reference_acceleration_rms": scale,
        "relative_rms_error_muscle_reserve": rms(err_rebuilt) / scale,
        "relative_rms_limit": redundancy.ID_RELATIVE_RMS_LIMIT,
        "frames": int(len(ks)),
        "passed": bool(
            max(closure) <= redundancy.TORQUE_CLOSURE_LIMIT_NM
            and rms(err_rebuilt) / scale <= redundancy.ID_RELATIVE_RMS_LIMIT
        ),
        "definition": (
            "tau_muscle (MuJoCo qfrc_actuator at the solved activations, mapped by "
            "Phi^T) + reserve must equal the spec effort, and the spec plant driven "
            "by that rebuilt effort must give the same acceleration as the plant "
            "driven by the original effort.  plant_acceleration_rms_error_plain_"
            "efforts is the pre-existing plant-versus-bundle residual (midpoint "
            "state versus step mean) and is reported only as a baseline"
        ),
    }


def _sensitivity(
    bases: list[redundancy.FrameBasis],
    tau: Array,
    groups: dict[str, list[int]],
    cols: list[int],
    weight: float,
    run: tuple[int, int | None],
) -> list[dict[str, Any]]:
    """Reserve ratios for other weights, muscle strengths and passive off.

    ``run`` is ``(workers, n_muscles)``; strength scaling applies to muscles only.
    """
    variants = (
        [(f"weight_{w:g}", 1.0, 1.0, w) for w in WEIGHT_SENSITIVITY]
        + [(f"force_x{f:g}", f, 1.0, None) for f in FORCE_SCALE_SENSITIVITY]
        + [("passive_off", 1.0, 0.0, None)]
    )
    workers, n_muscles = run
    return attribution.reserve_ratios(
        bases, tau, groups, cols, variants, weight, workers, n_muscles
    )


def muscle_findings(
    names: list[str], act: Array, ks: Array, keys: mapping_report.KeyFrames, dt: float
) -> dict[str, Any]:
    """Which muscles carry the transition and downswing (top to impact)."""
    window = (ks >= keys.top) & (ks <= keys.impact)
    require(bool(window.any()), "no frames between top and impact")
    mean = act[window].mean(axis=0)
    peak = act[window].max(axis=0)
    order = np.argsort(mean)[::-1][:TOP_MUSCLES]
    return {
        "window_steps": [keys.top, keys.impact],
        "window_frames": int(window.sum()),
        "top_by_mean_activation": [
            {
                "muscle": names[i],
                "mean_activation": float(mean[i]),
                "peak_activation": float(peak[i]),
                "peak_time_to_impact_ms": float(
                    (ks[window][int(np.argmax(act[window][:, i]))] - keys.impact)
                    * dt
                    * 1e3
                ),
            }
            for i in order
        ],
        "muscles_saturated_over_90pct_of_window": int(
            ((act[window] > 0.99).mean(axis=0) > 0.9).sum()
        ),
    }


def _map_poses(
    config: SwingConfig, bundle: Any, model: Any, qm: Array, ks: Array
) -> tuple[mapping.MyoMapper, list[Any], Any]:
    """Mapper, poses and the spec-to-Myo rate map for the configured ROM policy."""
    if config.rom_policy == "share":
        mapper = mapping.MyoMapper(
            bundle.spec_bytes, model, bundle.reference_q[0], rom_policy="clamp"
        )
        poses = share_ik.map_sequence(mapper, qm, ks)
        rate = functools.partial(share_ik.velocity_map, mapper)
        return mapper, poses, rate
    mapper = mapping.MyoMapper(
        bundle.spec_bytes, model, bundle.reference_q[0], rom_policy="extend"
    )
    return mapper, mapping_report.map_sequence(mapper, qm, ks), mapper.velocity_map


def solve_with_neck(
    bases: list[redundancy.FrameBasis],
    tau: Array,
    weight: float,
    n_muscles: int,
) -> tuple[Array, Array, Array, bool]:
    """Solve every frame; returns ``(muscle activation, neck activation, reserve, ok)``.

    ``bases`` may carry torque actuators after the first ``n_muscles`` entries.
    """
    act, res, ok = _solve_all(bases, tau, weight)
    muscle, neck_act = neck.split(act, n_muscles)
    return muscle, neck_act, res, ok


def _neck_receipt(
    bases: list[redundancy.FrameBasis], neck_act: Array, names: list[str]
) -> dict[str, Any]:
    """Capacity, source and saturation of the torque-actuated neck."""
    return {
        "enabled": True,
        "is_muscle": False,
        "source": neck.SOURCE,
        "capacity_nm": neck.CAPACITY_NM,
        "actuators": names,
        "peak_activation": float(neck_act.max()) if neck_act.size else 0.0,
        "frames_saturated_over_99pct": float((neck_act > 0.99).any(axis=1).mean()),
        "statement": (
            "MyoFullBody has no neck joints or muscles; these bounded torque "
            "actuators stand in for the neck.  They are not muscles and no neck "
            "motion is represented in MyoFullBody."
        ),
    }


def run_swing(config: SwingConfig) -> SwingResult:
    """Map, solve and check one bundle; the receipt is fail-closed."""
    from src.shared.python.motion_matching.same_input import InputBundle, VectorPlant

    bundle = InputBundle.load(config.bundle)
    tree = assets.cached_tree(root=config.cache_root)
    if tree is None:
        raise ValueError("MyoFullBody cache missing: run scripts/fetch_myofullbody.py")
    model, data = assets.load_myofullbody(tree)
    qm = 0.5 * (bundle.reference_q[:-1] + bundle.reference_q[1:])
    vm = 0.5 * (bundle.reference_v[:-1] + bundle.reference_v[1:])
    mapper0 = mapping.MyoMapper(
        bundle.spec_bytes, model, bundle.reference_q[0], rom_policy="extend"
    )
    keys = mapping_report.key_frames(mapper0, bundle.reference_q, bundle.dt_s)
    ks = frame_steps(bundle.steps, config.stride, keys)
    mapper, poses, rate = _map_poses(config, bundle, model, qm, ks)
    groups = redundancy.coordinate_groups(bundle.coordinate_order)
    cols = sorted(c for g in groups.values() for c in g)
    basis = redundancy.MuscleBasis(mapper, cols, velocity_map=rate)
    bases = [basis.evaluate(qm[k], vm[k], p) for k, p in zip(ks, poses, strict=True)]
    n_muscles = len(basis.names)
    neck_names: list[str] = []
    if config.neck_actuator:
        order = tuple(bundle.coordinate_order)
        neck_names = neck.actuator_names(order, cols)
        bases = [neck.augment(b, order, cols) for b in bases]
    tau = bundle.efforts[np.ix_(ks, cols)]
    act, neck_act, res, ok = solve_with_neck(
        bases, tau, config.reserve_weight, n_muscles
    )
    neck_tau = [
        neck.torque(b, np.concatenate([a, na]), n_muscles)
        for b, a, na in zip(bases, act, neck_act, strict=True)
    ]
    ctx = {
        "ks": ks, "qm": qm, "vm": vm, "act": act, "res": res, "tau": tau,
        "phi": [b.phi for b in bases], "poses": poses, "neck": neck_tau,
    }  # fmt: skip
    plant = VectorPlant("mujoco", bundle.spec_bytes)
    id_check = _id_check(basis, plant, bundle, ctx)
    metrics = redundancy.group_reserve_metrics(res, tau, groups, cols)
    uncovered = () if config.neck_actuator else redundancy.UNCOVERED_GROUPS
    receipt = _receipt(
        config,
        (bundle, mapper, model, data, tree),
        (keys, ks, poses),
        (metrics, id_check, ok, uncovered),
    )
    receipt["muscle_findings"] = muscle_findings(
        basis.names, act, ks, keys, bundle.dt_s
    )
    run = (config.workers, n_muscles)
    receipt["sensitivity"] = _sensitivity(
        bases, tau, groups, cols, config.reserve_weight, run
    )
    names = [bundle.coordinate_order[c] for c in cols]
    receipt["attribution"] = attribution.attribution_receipt(
        bases,
        tau,
        groups,
        cols,
        names,
        config.reserve_weight,
        config.workers,
        n_muscles,
    )
    receipt["phase_breakdown"] = attribution.phase_breakdown(
        ks, keys.as_dict(), res, tau, groups, cols, IMPACT_HALF_WIDTH_STEPS
    )
    if config.neck_actuator:
        receipt["neck_actuator"] = _neck_receipt(bases, neck_act, neck_names)
    else:
        receipt["neck_actuator"] = {"enabled": False}
    receipt["per_coordinate_reserve"] = {
        bundle.coordinate_order[c]: {
            "rms_nm": float(np.sqrt(np.mean(res[:, i] ** 2))),
            "peak_nm": float(np.abs(res[:, i]).max()),
            "effort_rms_nm": float(np.sqrt(np.mean(tau[:, i] ** 2))),
        }
        for i, c in enumerate(cols)
    }
    force = [
        b.passive[:n_muscles] + a * b.active[:n_muscles]
        for a, b in zip(act, bases, strict=True)
    ]
    arrays = {
        "steps": ks, "time_s": ks * bundle.dt_s, "qpos": np.array([p.qpos for p in poses]),
        "activation": act, "neck_activation": neck_act, "reserve": res, "tau": tau,
        "coordinates": np.array(names),
        "muscles": np.array(basis.names), "key_frames": np.array(list(keys.as_dict().values())),
        "force": np.array(force),
    }  # fmt: skip
    return SwingResult(receipt, arrays)


def _receipt(
    config: SwingConfig,
    context: tuple[Any, mapping.MyoMapper, Any, Any, Path],
    frames: tuple[Any, Array, list[Any]],
    solved: tuple[dict[str, Any], dict[str, Any], bool, tuple[str, ...]],
) -> dict[str, Any]:
    bundle, mapper, model, data, tree = context
    keys, ks, poses = frames
    metrics, id_check, ok, uncovered = solved
    clamp = mapping.MyoMapper(
        bundle.spec_bytes, model, bundle.reference_q[0], rom_policy="clamp"
    )
    qm = 0.5 * (bundle.reference_q[:-1] + bundle.reference_q[1:])
    at_key = {n: int(np.searchsorted(ks, k)) for n, k in keys.as_dict().items()}
    clamp_poses = {
        n: clamp.map_pose(qm[keys.as_dict()[n]], bounded=True) for n in at_key
    }
    q_idx = np.asarray(ks, dtype=int)
    receipt: dict[str, Any] = {
        "schema": SCHEMA,
        "invocation": config.invocation,
        "bundle": {
            "path_name": config.bundle.name,
            "sha256": hashlib.sha256(config.bundle.read_bytes()).hexdigest(),
            "spec_sha256": bundle.spec_sha256,
            "dt_s": bundle.dt_s,
            "steps": bundle.steps,
        },
        "myofullbody": assets.asset_receipt(tree, assets.load_manifest(), model),
        "spec_mass_kg": SPEC_MASS_KG,
        "frames": int(len(ks)),
        "stride": config.stride,
        "key_frames": keys.as_dict(),
        "mapping": {
            "rom_policy_used_for_poses": config.rom_policy,
            "tolerance_deg": {
                "three_dof_segments": mapping_report.TOLERANCE_DEG_THREE_DOF,
                "structurally_1_or_2_dof_segments": (
                    mapping_report.TOLERANCE_DEG_STRUCTURAL
                ),
                "three_dof_set": list(mapping_report.THREE_DOF),
            },
            "errors_deg_all_frames": mapping_report.segment_errors(poses),
            "errors_deg_at_key_frames_primary": {
                n: {s: round(poses[i].error_deg[s], 3) for s in poses[i].error_deg}
                for n, i in at_key.items()
            },
            "errors_deg_at_key_frames_clamp": {
                n: {s: round(p.error_deg[s], 3) for s in p.error_deg}
                for n, p in clamp_poses.items()
            },
            "rom_clamps": (
                mapping_report.rom_exceedance(mapper, poses)
                if config.rom_policy == "extend"
                else "not applicable: share IK keeps every joint inside its range"
            ),
            "hinge_agreement": mapping_report.hinge_agreement(mapper, qm, q_idx, poses),
            "muscle_lengths": mapping_report.length_report(model, data, poses),
        },
        "id_consistency": id_check,
        "reserves_by_group": metrics,
        "solver": {
            "method": "projected Newton on min sum(a^2) + w sum(reserve^2), 0<=a<=1",
            "reserve_weight": config.reserve_weight,
            "all_frames_converged": bool(ok),
            "cross_checked_against": "scipy BVLS (musculoskeletal_static_opt.solve_frame)",
        },
        "qualification": redundancy.qualification(
            metrics, ok, id_check["passed"], uncovered=uncovered
        ),
        "hybrid_statement": (
            "Spec inverse dynamics provides the joint efforts; MyoFullBody provides "
            "only muscle moment arms and force capacity at the mapped pose."
        ),
    }
    digest = hashlib.sha256(
        json.dumps(receipt, sort_keys=True, default=str).encode()
    ).hexdigest()
    receipt["receipt_digest"] = digest
    return receipt
