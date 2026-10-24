"""Compliant palm-thumb pad pair, diagnostic spec revision (HRG-H2, #13190).

Diagnostic only. The pad is the physical ``trail_palm_lead_thumb`` hand-hand
pair of the contact topology: the trail thenar (palm pocket) sphere against the
lead thumb capsule, evaluated with the one shared G2 law
(:func:`~src.shared.python.grip_contact.topology_contact.topology_hand_hand_pair_wrench`,
Hunt-Crossley normal plus Stribeck friction). Re-scoped by HRG-GS-G3 (#13301):
the compliant pad exists only where the topology marks the pair physical, that
is ``mode="separated_v2"`` with separate ``LHand``/``RHand`` bodies
(HRG-GS-G11, #13309). It never uses native collision geometry, so every engine
sees the same force. On a welded document the pair stays an allocated
equivalent (:mod:`hand_hand_allocation`) and the revision here refuses it.

Contract (binding acceptance of H2):

* hand-drift gate: no force is reported unless the prescribed trail-in-lead
  drift is under :data:`HAND_DRIFT_GATE_MM` along and across the grip;
  otherwise the result is NaN plus a reason, never zero;
* an isolated 1 mm separation gives a force <= :data:`SEPARATION_FORCE_TOL_N`;
  the normal force is never negative;
* friction-cone residual ``max(0, F_t - mu_s F_n)`` <= :data:`FRICTION_CONE_TOL_N`;
* normalised action/reaction error <= :data:`ACTION_REACTION_TOL`;
* halving the timestep of the pad impact response changes the peak normal
  force and the impulse by <= :data:`TIMESTEP_SENSITIVITY_TOL`;
* preload sensitivity (``dF_n / d delta`` and its elasticity) is reported.

Conventions: all vectors in world axes. ``normal`` points from the trail palm
(pair site A) toward the lead thumb (site B), so a closing trail hand moves
along ``+normal`` and the pad pushes it back along ``-normal``.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.core.handedness import DEFAULT_HANDEDNESS, Handedness
from src.shared.python.grip_contact.bushing_law import BushingState
from src.shared.python.grip_contact.hand_drift import HandDrift
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    topology_hand_hand_pair_wrench,
)
from src.shared.python.motion_matching.full_body_schema import (
    SchemaFeatureUnavailableError,
)
from src.shared.python.motion_matching.full_body_spec import canonical_sha256

#: The physical hand-hand pair that is the compliant palm-thumb pad.
PALM_THUMB_PAIR = "trail_palm_lead_thumb"
#: The only mode in which the pad is physical (HRG-GS-G11, #13309).
PAD_MODE = "separated_v2"
#: Spec-revision identifier of the diagnostic pad block.
PALM_THUMB_PAD_REVISION = "hrg-h2-palm-thumb-pad-v1"
#: Hand-drift gate [mm]: along and across drift must be strictly under this.
HAND_DRIFT_GATE_MM = 2.0
#: Force bound at an isolated 1 mm separation [N].
SEPARATION_FORCE_TOL_N = 1.0e-8
#: Friction-cone residual bound [N].
FRICTION_CONE_TOL_N = 1.0e-8
#: Normalised action/reaction error bound [-].
ACTION_REACTION_TOL = 1.0e-8
#: Relative change of peak force and impulse when the timestep is halved [-].
TIMESTEP_SENSITIVITY_TOL = 0.01

_TINY = 1.0e-300

__all__ = [
    "ACTION_REACTION_TOL",
    "FRICTION_CONE_TOL_N",
    "HAND_DRIFT_GATE_MM",
    "PAD_MODE",
    "PALM_THUMB_PAD_REVISION",
    "PALM_THUMB_PAIR",
    "SEPARATION_FORCE_TOL_N",
    "TIMESTEP_SENSITIVITY_TOL",
    "DriftGate",
    "ImpactResponse",
    "ImpactSetup",
    "PalmThumbPadForce",
    "hand_drift_gate",
    "palm_thumb_impact_response",
    "palm_thumb_pad_force",
    "palm_thumb_pad_spec_revision",
    "palm_thumb_preload_sensitivity",
    "timestep_halving_sensitivity",
]


def _positive(name: str, value: float) -> float:
    v = float(value)
    if not math.isfinite(v) or v <= 0.0:
        raise ValueError(f"{name} must be positive and finite, got {value!r}")
    return v


@dataclass(frozen=True)
class DriftGate:
    """Outcome of the hand-drift gate; ``reason`` is empty exactly when passed."""

    passed: bool
    along_peak_mm: float
    across_peak_mm: float
    limit_mm: float
    reason: str = ""


def hand_drift_gate(
    drift: HandDrift, limit_mm: float = HAND_DRIFT_GATE_MM
) -> DriftGate:
    """Gate the pad report on the prescribed trail-in-lead drift.

    Passes only when the peak drift along and across the grip are both finite
    and strictly under ``limit_mm``. Preconditions: ``drift`` a
    :class:`HandDrift` (``TypeError``), ``limit_mm`` positive (``ValueError``).
    """
    if not isinstance(drift, HandDrift):
        raise TypeError(f"drift must be HandDrift, got {type(drift).__name__}")
    limit = _positive("limit_mm", limit_mm)
    along = np.abs(np.asarray(drift.along_mm, dtype=float))
    across = np.abs(np.asarray(drift.across_mm, dtype=float))
    if along.size == 0 or not (np.isfinite(along).all() and np.isfinite(across).all()):
        return DriftGate(False, math.nan, math.nan, limit, "hand drift is not finite")
    a_pk, c_pk = float(along.max()), float(across.max())
    if a_pk < limit and c_pk < limit:
        return DriftGate(True, a_pk, c_pk, limit)
    reason = (
        f"hand drift along {a_pk:.3f} mm / across {c_pk:.3f} mm is not under "
        f"the {limit:.1f} mm gate"
    )
    return DriftGate(False, a_pk, c_pk, limit, reason)


_NAN3 = (math.nan, math.nan, math.nan)


@dataclass(frozen=True)
class PalmThumbPadForce:
    """Physical palm-thumb pad force, or NaN plus a reason when unavailable.

    ``force_on_trail_n`` acts on the trail palm, ``force_on_lead_n`` on the lead
    thumb; ``action_reaction_error`` is the larger of the force and moment
    imbalance, each normalised by the pair force (and lever).
    """

    available: bool
    reason: str = ""
    method: str = "physical"
    pair: str = PALM_THUMB_PAIR
    normal: tuple[float, float, float] = _NAN3
    penetration_m: float = math.nan
    normal_force_n: float = math.nan
    tangential_force_n: float = math.nan
    force_on_trail_n: tuple[float, float, float] = _NAN3
    force_on_lead_n: tuple[float, float, float] = _NAN3
    friction_cone_residual_n: float = math.nan
    action_reaction_error: float = math.nan


def _tuple3(v: np.ndarray) -> tuple[float, float, float]:
    return (float(v[0]), float(v[1]), float(v[2]))


def _action_reaction_error(pw: Any) -> float:
    f_a, f_b = pw.force_a_n, pw.force_b_n
    scale = max(float(np.linalg.norm(f_a)), float(np.linalg.norm(f_b)))
    if scale <= _TINY:
        return 0.0
    lever = max(
        float(np.linalg.norm(pw.contact_point_m - pw.point_a_m)),
        float(np.linalg.norm(pw.contact_point_m - pw.point_b_m)),
        _TINY,
    )
    net_m = (
        pw.moment_a_nm
        + np.cross(pw.point_a_m, f_a)
        + pw.moment_b_nm
        + np.cross(pw.point_b_m, f_b)
    )
    return max(
        float(np.linalg.norm(f_a + f_b)) / scale,
        float(np.linalg.norm(net_m)) / (scale * lever),
    )


def _evaluate(
    model: TopologyContactModel,
    lead_frame: BushingState,
    trail_frame: BushingState,
    handedness: Handedness,
) -> Any:
    return topology_hand_hand_pair_wrench(
        model,
        PALM_THUMB_PAIR,
        trail_frame,
        lead_frame,
        mode=PAD_MODE,
        handedness=handedness,
    )


def palm_thumb_pad_force(
    model: TopologyContactModel,
    lead_frame: BushingState,
    trail_frame: BushingState,
    *,
    gate: DriftGate,
    handedness: Handedness = DEFAULT_HANDEDNESS,
) -> PalmThumbPadForce:
    """Evaluate the pad with the shared G2 law behind the hand-drift gate.

    Preconditions: ``gate`` a :class:`DriftGate` and ``model`` a
    :class:`TopologyContactModel` (``TypeError``); frame validation is the
    shared kernel's. Postconditions: a failed gate gives ``available=False``
    with the gate reason and NaN quantities; otherwise ``normal_force_n >= 0``.
    """
    if not isinstance(gate, DriftGate):
        raise TypeError(f"gate must be DriftGate, got {type(gate).__name__}")
    if not isinstance(model, TopologyContactModel):
        raise TypeError(
            f"model must be TopologyContactModel, got {type(model).__name__}"
        )
    if not gate.passed:
        return PalmThumbPadForce(False, f"hand drift gate failed: {gate.reason}")
    pw = _evaluate(model, lead_frame, trail_frame, handedness)
    f_n = float(pw.normal_force_mag_n)
    f_t = float(pw.tangential_force_mag_n)
    mu_s = float(model.law.static_friction)
    return PalmThumbPadForce(
        available=True,
        normal=_tuple3(pw.normal),
        penetration_m=float(pw.penetration_m),
        normal_force_n=f_n,
        tangential_force_n=f_t,
        force_on_trail_n=_tuple3(pw.force_a_n),
        force_on_lead_n=_tuple3(pw.force_b_n),
        friction_cone_residual_n=max(0.0, f_t - mu_s * f_n),
        action_reaction_error=_action_reaction_error(pw),
    )


def _shifted(frame: BushingState, offset: np.ndarray, velocity: np.ndarray):
    return BushingState(
        frame.rotation, frame.position_m + offset, velocity, frame.omega_rad_s
    )


@dataclass(frozen=True)
class ImpactResponse:
    """Normal-force peak and impulse of a trail-hand impact on the pad."""

    dt_s: float
    peak_normal_force_n: float
    impulse_n_s: float


@dataclass(frozen=True)
class ImpactSetup:
    """Trail-hand impact on the pad: ``mass_kg`` closing at ``approach_speed_m_s``.

    The trail hand starts ``initial_gap_m`` short of contact and the run lasts
    ``duration_s``. Preconditions (``ValueError``): positive mass and duration,
    finite non-negative speed and gap.
    """

    mass_kg: float
    approach_speed_m_s: float
    initial_gap_m: float
    duration_s: float

    def __post_init__(self) -> None:
        _positive("mass_kg", self.mass_kg)
        _positive("duration_s", self.duration_s)
        for name in ("approach_speed_m_s", "initial_gap_m"):
            v = float(getattr(self, name))
            if not math.isfinite(v) or v < 0.0:
                raise ValueError(f"{name} must be finite and >= 0, got {v!r}")


def palm_thumb_impact_response(
    model: TopologyContactModel,
    lead_frame: BushingState,
    trail_frame: BushingState,
    setup: ImpactSetup,
    *,
    dt_s: float,
    handedness: Handedness = DEFAULT_HANDEDNESS,
) -> ImpactResponse:
    """Trail-hand impact on the pad, integrated with classical RK4.

    The lead hand is held at ``lead_frame``; the trail hand translates along
    the pad normal of ``trail_frame`` as described by ``setup``. The impulse
    is integrated as an extra RK4 state. Preconditions: ``setup`` an
    :class:`ImpactSetup` (``TypeError``), positive ``dt_s`` (``ValueError``).
    """
    if not isinstance(setup, ImpactSetup):
        raise TypeError(f"setup must be ImpactSetup, got {type(setup).__name__}")
    dt = _positive("dt_s", dt_s)
    mass = float(setup.mass_kg)
    rest = _evaluate(model, lead_frame, trail_frame, handedness)
    n = np.asarray(rest.normal, dtype=float)
    start = -(float(rest.penetration_m) + float(setup.initial_gap_m)) * n

    def rhs(s: float, v: float) -> tuple[float, float, float]:
        pw = _evaluate(
            model, lead_frame, _shifted(trail_frame, start + s * n, v * n), handedness
        )
        return v, float(np.dot(pw.force_a_n, n)) / mass, float(pw.normal_force_mag_n)

    s, v, impulse, peak = 0.0, float(setup.approach_speed_m_s), 0.0, 0.0
    for _ in range(int(round(float(setup.duration_s) / dt))):
        k1 = rhs(s, v)
        k2 = rhs(s + 0.5 * dt * k1[0], v + 0.5 * dt * k1[1])
        k3 = rhs(s + 0.5 * dt * k2[0], v + 0.5 * dt * k2[1])
        k4 = rhs(s + dt * k3[0], v + dt * k3[1])
        s += dt * (k1[0] + 2.0 * k2[0] + 2.0 * k3[0] + k4[0]) / 6.0
        v += dt * (k1[1] + 2.0 * k2[1] + 2.0 * k3[1] + k4[1]) / 6.0
        impulse += dt * (k1[2] + 2.0 * k2[2] + 2.0 * k3[2] + k4[2]) / 6.0
        peak = max(peak, k1[2], k4[2])
    return ImpactResponse(dt, peak, impulse)


def timestep_halving_sensitivity(
    model: TopologyContactModel,
    lead_frame: BushingState,
    trail_frame: BushingState,
    setup: ImpactSetup,
    *,
    dt_s: float,
    handedness: Handedness = DEFAULT_HANDEDNESS,
) -> dict[str, Any]:
    """Run :func:`palm_thumb_impact_response` at ``dt_s`` and ``dt_s / 2``.

    Returns the two responses and the relative changes of peak normal force and
    impulse (relative to the fine run). Raises ``ValueError`` when the trail
    hand never reaches the pad within ``setup.duration_s``.
    """
    dt = _positive("dt_s", dt_s)
    runs = [
        palm_thumb_impact_response(
            model, lead_frame, trail_frame, setup, dt_s=step, handedness=handedness
        )
        for step in (dt, dt / 2.0)
    ]
    coarse, fine = runs
    if fine.peak_normal_force_n <= 0.0 or fine.impulse_n_s <= 0.0:
        raise ValueError("impact never reached the pad; increase duration_s")
    peak_change = abs(coarse.peak_normal_force_n - fine.peak_normal_force_n)
    impulse_change = abs(coarse.impulse_n_s - fine.impulse_n_s)
    return {
        "coarse": coarse,
        "fine": fine,
        "peak_relative_change": peak_change / fine.peak_normal_force_n,
        "impulse_relative_change": impulse_change / fine.impulse_n_s,
    }


def palm_thumb_preload_sensitivity(
    model: TopologyContactModel,
    lead_frame: BushingState,
    trail_frame: BushingState,
    *,
    relative_step: float = 0.1,
    handedness: Handedness = DEFAULT_HANDEDNESS,
) -> dict[str, Any]:
    """Report the pad preload and its sensitivity at the frames' static pose.

    Central difference of the normal force over a penetration change of
    ``+-relative_step`` of the preload. Returns ``pair``,
    ``preload_penetration_m``, ``normal_force_n``, ``stiffness_n_m``
    (``dF_n / d delta``) and ``elasticity`` (``(delta / F_n) dF_n / d delta``).
    Preconditions: ``0 < relative_step < 1`` and a preloaded pad
    (``ValueError``).
    """
    h_rel = _positive("relative_step", relative_step)
    if h_rel >= 1.0:
        raise ValueError(f"relative_step must be < 1, got {relative_step!r}")
    still = _shifted(trail_frame, np.zeros(3), np.zeros(3))
    rest = _evaluate(model, lead_frame, still, handedness)
    delta0, f0 = float(rest.penetration_m), float(rest.normal_force_mag_n)
    if delta0 <= 0.0 or f0 <= 0.0:
        raise ValueError("palm-thumb pad has no preload at this pose")
    n = np.asarray(rest.normal, dtype=float)
    h = h_rel * delta0
    f_plus = _evaluate(
        model, lead_frame, _shifted(still, h * n, np.zeros(3)), handedness
    )
    f_minus = _evaluate(
        model, lead_frame, _shifted(still, -h * n, np.zeros(3)), handedness
    )
    k = (f_plus.normal_force_mag_n - f_minus.normal_force_mag_n) / (2.0 * h)
    return {
        "pair": PALM_THUMB_PAIR,
        "preload_penetration_m": delta0,
        "normal_force_n": f0,
        "stiffness_n_m": float(k),
        "elasticity": float(k * delta0 / f0),
    }


def palm_thumb_pad_spec_revision(spec: Mapping[str, Any]) -> dict[str, Any]:
    """New spec revision carrying the diagnostic palm-thumb pad block.

    ``spec`` must be a separated-hand ``full-body-v2`` document
    (``hand_separation.mode == "separated_v2"``). The result is a deep copy with
    a ``palm_thumb_pad`` block (recording ``parent_spec_sha256``) and its own
    ``spec_sha256`` over every other key. The parent is never modified and the
    physics slice (``upper_body_sha256``) is unchanged: the pad is diagnostic.

    Raises ``TypeError`` for a non-mapping, :class:`SchemaFeatureUnavailableError`
    for a welded document, and ``ValueError`` when the block already exists.
    """
    if not isinstance(spec, Mapping):
        raise TypeError(f"spec must be a Mapping, got {type(spec).__name__}")
    separation = spec.get("hand_separation") or {}
    if not separation.get("enabled") or separation.get("mode") != PAD_MODE:
        raise SchemaFeatureUnavailableError(
            str(spec.get("schema_version", "full-body-v2")),
            consumer="grip_contact.palm_thumb_pad",
            feature="physical palm-thumb pad",
            child="HRG-GS-G11",
            issue=13309,
        )
    if "palm_thumb_pad" in spec:
        raise ValueError("spec already carries a palm_thumb_pad revision")
    parent = {k: v for k, v in spec.items() if k != "spec_sha256"}
    out = copy.deepcopy(parent)
    out["palm_thumb_pad"] = {
        "revision": PALM_THUMB_PAD_REVISION,
        "issue": 13190,
        "pair": PALM_THUMB_PAIR,
        "mode": PAD_MODE,
        "law": "pair_contact G2 sphere-capsule: Hunt-Crossley normal, Stribeck friction",
        "diagnostic_only": True,
        "native_collision": False,
        "hand_drift_gate_mm": HAND_DRIFT_GATE_MM,
        "parent_spec_sha256": canonical_sha256(parent),
    }
    out["spec_sha256"] = canonical_sha256(out)
    return out
