"""Barbell-on-bushings runs: quasi-static hold, bundle, cross-engine parity.

LIFT-4 (#11744) on the LIFT-5 back-squat same-input bundle (#11745).  Uses
:class:`~src.shared.python.lifting.barbell_bushing.BarbellBushingPlant`.

* :func:`quasi_static_hold` holds one pose in MuJoCo with the pipeline
  computed-torque controller and the bar on both hands, and records the
  shared-law ground reaction, the per-hand bushing forces and deflections.
  It is the physical check: once settled, the vertical ground reaction must
  equal the weight of body plus bar (the bar has no other support).
* :func:`build_barbell_bundle` re-tracks a back-squat bundle's reference with
  the bar attached and returns a same-input bundle whose body arrays are the
  loaded run; the bar's states and parameters travel in the provenance.
* :func:`barbell_plant` rebuilds the coupled plant of such a bundle in any
  same-input engine and :func:`instantaneous_parity` compares, at chosen
  reference states, the per-hand bushing wrenches and the coupled
  accelerations of each engine against MuJoCo.

The controller compensates the measured hand load (it subtracts the bushing
generalized force on the actuated coordinates, the computed-torque law with
a known interaction wrench); the unactuated root is left to the contact.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.lifting.barbell_bushing import (
    BAR_COORDINATES,
    SIDES,
    Barbell,
    BarbellBushingPlant,
)
from src.shared.python.motion_matching.same_input import (
    InputBundle,
    Rollout,
    VectorPlant,
    integrate,
    project_to_closure,
)

Array = NDArray[np.float64]
Controller = Callable[[float, Array, Array], Array]
PROVENANCE_KEY = "barbell"
SAMPLE_EVERY = 10

__all__ = [
    "HoldResult",
    "barbell_plant",
    "build_barbell_bundle",
    "instantaneous_parity",
    "loaded_effort_source",
    "quasi_static_hold",
]


def loaded_effort_source(
    plant: BarbellBushingPlant, controller: Controller
) -> Callable[[int, float, Array, Array], Array]:
    """Effort source: body controller minus the measured hand load; bar zero."""
    nv = plant.nv_body

    def source(_k: int, t: float, q: Array, v: Array) -> Array:
        tau = np.zeros(nv + len(BAR_COORDINATES))
        tau[:nv] = np.asarray(controller(t, q[:nv], v[:nv]), dtype=float)
        tau[:nv] -= plant.hand_generalized_force(q, v)
        return tau

    return source


def _tracking(
    spec_bytes: bytes, times: Array, track: Array, duration_s: float
) -> tuple[Any, Array, Array, int, Any]:
    from src.engines.physics_engines.mujoco.python.full_body_model import (
        NativeMujocoFullBodyModel,
    )
    from src.shared.python.motion_matching import full_body_forward_dynamics as fs
    from src.shared.python.motion_matching.same_input.reference import (
        _tracking_setup,
    )

    controller, q0, v0, steps = _tracking_setup(spec_bytes, times, track, duration_s)
    simulator = fs.FullBodySimulator(NativeMujocoFullBodyModel(spec_bytes))
    return controller, q0, v0, steps, simulator


def _preload_bar(simulator: Any, q: Array, bar_mass_kg: float) -> Array:
    """Sink the feet further so the spheres also carry the bar at rest."""
    adapter = simulator.adapter
    stiffness = float(adapter.contact_parameters.stiffness_n_m)
    gravity = float(np.linalg.norm(simulator.gravity))
    extra = bar_mass_kg * gravity / (stiffness * len(adapter._spheres))  # noqa: SLF001
    normal = np.asarray(adapter.ground_plane.normal, dtype=float)
    out = np.asarray(q, dtype=float).copy()
    out[:3] += np.linalg.solve(simulator.root_translation_axes(out), -extra * normal)
    return out


def _attached_start(
    spec_bytes: bytes, q: Array, v: Array, barbell: Barbell, engine: str = "mujoco"
) -> tuple[BarbellBushingPlant, Array, Array]:
    plant = VectorPlant(engine, spec_bytes)
    start = project_to_closure(plant, q, v)
    loaded = BarbellBushingPlant(plant, spec_bytes, barbell, start.q)
    q_aug, v_aug = loaded.attached_state(start.q, start.v)
    return loaded, q_aug, v_aug


@dataclass(frozen=True)
class HoldResult:
    """Sampled quasi-static hold (every :data:`SAMPLE_EVERY` steps)."""

    time_s: Array
    grf_n: Array
    hand_force_n: dict[str, Array]
    deflection_m: Array
    body_weight_n: float
    bar_weight_n: float
    rollout: Rollout

    def summary(self, settle_window_s: float) -> dict[str, Any]:
        """Means over the last ``settle_window_s`` and whole-run peaks."""
        if not (np.isfinite(settle_window_s) and settle_window_s > 0.0):
            raise ValueError("settle_window_s must be positive")
        late = self.time_s >= self.time_s[-1] - settle_window_s + 1e-12
        total = self.body_weight_n + self.bar_weight_n
        hands = {s: self.hand_force_n[s][late].mean(axis=0) for s in SIDES}
        hand_sum = hands["L"] + hands["R"]
        grf = self.grf_n[late].mean(axis=0)
        return {
            "duration_s": float(self.time_s[-1]),
            "settle_window_s": float(settle_window_s),
            "body_weight_n": self.body_weight_n,
            "bar_weight_n": self.bar_weight_n,
            "mean_grf_n": grf.tolist(),
            "grf_vertical_over_total_weight": float(grf[2] / total),
            "mean_hand_force_on_bar_n": {s: hands[s].tolist() for s in SIDES},
            "hand_vertical_sum_over_bar_weight": float(hand_sum[2] / self.bar_weight_n),
            "hand_vertical_split_left": float(hands["L"][2] / hand_sum[2]),
            "peak_deflection_m": float(self.deflection_m.max()),
            "failure": self.rollout.failure,
        }


def _contact_force(adapter: Any, names: Sequence[str], q: Array, v: Array) -> Array:
    samples = adapter.evaluate_contact_samples(
        dict(zip(names, map(float, q), strict=True)),
        dict(zip(names, map(float, v), strict=True)),
    )
    return np.sum([s.normal_force_n + s.friction_force_n for s in samples.values()], 0)


def quasi_static_hold(
    spec_bytes: bytes, q_hold: Array, barbell: Barbell, *, duration_s: float
) -> HoldResult:
    """Hold ``q_hold`` in MuJoCo with the bar on both hands for ``duration_s``.

    The feet are preloaded for body plus bar; the bar starts undeflected and
    at rest.  Postcondition: samples every :data:`SAMPLE_EVERY` steps.
    """
    pose = np.asarray(q_hold, dtype=float)
    track = np.vstack([pose, pose])
    times = np.array([0.0, float(duration_s)])
    controller, q0, v0, steps, simulator = _tracking(
        spec_bytes, times, track, duration_s
    )
    q0 = _preload_bar(simulator, q0, barbell.load_kg)
    loaded, q_aug, v_aug = _attached_start(spec_bytes, q0, v0, barbell)
    rollout = integrate(
        loaded,
        q_aug,
        v_aug,
        loaded_effort_source(loaded, controller),
        steps=steps,
        dt_s=float(duration_s) / steps,
    )
    rows = range(0, rollout.q.shape[0], SAMPLE_EVERY)
    nv, names = loaded.nv_body, loaded.body_order
    grf, forces, deflection = [], {s: [] for s in SIDES}, []
    for k in rows:
        q, v = rollout.q[k], rollout.v[k]
        grf.append(_contact_force(simulator.adapter, names, q[:nv], v[:nv]))
        sample = loaded.sample(q, v)
        for side in SIDES:
            forces[side].append(sample.wrench[side].force_n)
        deflection.append(sample.max_deflection_m())
    gravity = float(np.linalg.norm(simulator.gravity))
    return HoldResult(
        time_s=rollout.time_s[list(rows)],
        grf_n=np.array(grf),
        hand_force_n={s: np.array(forces[s]) for s in SIDES},
        deflection_m=np.array(deflection),
        body_weight_n=float(simulator.mass_kg) * gravity,
        bar_weight_n=barbell.load_kg * gravity,
        rollout=rollout,
    )


def build_barbell_bundle(source: InputBundle, barbell: Barbell) -> InputBundle:
    """Re-track ``source``'s reference in MuJoCo with the bar on both hands.

    The returned bundle keeps the body coordinates (so it validates as an
    ordinary same-input bundle); the bar's reference states, the bar and
    bushing parameters and the attachment pose are under
    ``provenance["barbell"]``.  Replay it with :func:`barbell_plant`.
    """
    times = np.arange(source.reference_q.shape[0]) * source.dt_s
    duration = source.steps * source.dt_s
    controller, q0, v0, steps, simulator = _tracking(
        source.spec_bytes, times, source.reference_q, duration
    )
    q0 = _preload_bar(simulator, q0, barbell.load_kg)
    loaded, q_aug, v_aug = _attached_start(source.spec_bytes, q0, v0, barbell)
    rollout = integrate(
        loaded,
        q_aug,
        v_aug,
        loaded_effort_source(loaded, controller),
        steps=steps,
        dt_s=source.dt_s,
    )
    nv = loaded.nv_body
    record = {
        **loaded.record(),
        "attach_q": q_aug[:nv].tolist(),
        "reference_bar_q": rollout.q[:, nv:].tolist(),
        "reference_bar_v": rollout.v[:, nv:].tolist(),
        "controller": "pipeline computed-torque (kkt) minus measured hand load",
    }
    provenance = {
        **source.provenance,
        "max_pose_drift_m": float(rollout.pose_drift.max()),
        PROVENANCE_KEY: record,
        "source_bundle_spec_sha256": source.spec_sha256,
    }
    return InputBundle(
        spec_bytes=source.spec_bytes,
        coordinate_order=source.coordinate_order,
        dt_s=source.dt_s,
        q0=rollout.q[0, :nv],
        v0=rollout.v[0, :nv],
        efforts=rollout.efforts[:, :nv],
        reference_q=rollout.q[:, :nv],
        reference_v=rollout.v[:, :nv],
        reference_engine="mujoco",
        provenance=provenance,
    )


def barbell_plant(engine: str, bundle: InputBundle) -> BarbellBushingPlant:
    """The coupled plant of a barbell bundle in ``engine``.

    Raises:
        ValueError: if the bundle carries no barbell record.
    """
    record = bundle.provenance.get(PROVENANCE_KEY)
    if not isinstance(record, Mapping):
        raise ValueError("bundle has no barbell record; build it with the bar")
    barbell = Barbell(**record["barbell"])
    plant = VectorPlant(engine, bundle.spec_bytes)
    return BarbellBushingPlant(
        plant, bundle.spec_bytes, barbell, np.asarray(record["attach_q"], float)
    )


def augmented_reference(bundle: InputBundle) -> tuple[Array, Array]:
    """Reference states ``(q, v)`` with the bar coordinates appended."""
    record = bundle.provenance[PROVENANCE_KEY]
    return (
        np.hstack([bundle.reference_q, np.asarray(record["reference_bar_q"])]),
        np.hstack([bundle.reference_v, np.asarray(record["reference_bar_v"])]),
    )


def _engine_values(
    plant: BarbellBushingPlant, q: Array, v: Array, tau: Array
) -> dict[str, Array]:
    sample = plant.sample(q, v)
    out = {f"hand_force_{s}": sample.wrench[s].force_n for s in SIDES}
    out |= {f"hand_moment_{s}": sample.wrench[s].moment_nm for s in SIDES}
    acceleration = plant.acceleration(q, v, tau)
    out["body_acceleration"] = acceleration[: plant.nv_body]
    out["bar_acceleration"] = acceleration[plant.nv_body :]
    return out


def instantaneous_parity(
    bundle: InputBundle, engines: Sequence[str], indices: Sequence[int]
) -> dict[str, Any]:
    """Largest difference from MuJoCo per quantity, at reference states.

    State ``k`` uses the reference state and the effort applied from it (the
    last state reuses the last effort).  An engine that fails to build is
    reported unavailable with the reason, never as zero difference.
    """
    q_ref, v_ref = augmented_reference(bundle)
    efforts = np.hstack([bundle.efforts, np.zeros((bundle.steps, 6))])

    def values(plant: BarbellBushingPlant) -> list[dict[str, Array]]:
        return [
            _engine_values(plant, q_ref[k], v_ref[k], efforts[min(k, bundle.steps - 1)])
            for k in indices
        ]

    reference = values(barbell_plant("mujoco", bundle))
    report: dict[str, Any] = {
        "indices": list(indices),
        "mujoco_scale": {
            key: float(max(np.abs(row[key]).max() for row in reference))
            for key in reference[0]
        },
        "engines": {},
    }
    for engine in engines:
        try:
            rows = values(barbell_plant(engine, bundle))
        except (ImportError, RuntimeError, ValueError) as exc:
            report["engines"][engine] = {"unavailable": f"{type(exc).__name__}: {exc}"}
            continue
        report["engines"][engine] = {
            key: float(
                max(
                    np.abs(row[key] - ref[key]).max()
                    for row, ref in zip(rows, reference, strict=True)
                )
            )
            for key in reference[0]
        }
    return report
