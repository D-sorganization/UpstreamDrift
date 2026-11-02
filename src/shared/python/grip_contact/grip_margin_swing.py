"""Scripted pad-contact swing inputs for :class:`GripMarginReport` (#13885 PR 3).

Replays the shared topology law on recorded hand/club frames from a
:class:`ContactRun`, then applies the pure margin metric.  Pad slip speeds for
:func:`slip_events` use the magnitude of the hand–club roll and axial slip
rates (m/s) broadcast to each pad on that hand — a kinematic receipt proxy, not
a MuJoCo-native tangential velocity.

``grip_force_limit_n`` values are recorded on the receipt and checked against
peak per-hand grip force magnitudes from the run (pad contact does not enforce
the cap; the bushing path does, HRG-HX4).
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np

from src.shared.python.core.handedness import lead_side, subject_handedness, trail_side
from src.shared.python.grip_contact.club_dynamics import ClubDynamics
from src.shared.python.grip_contact.contact_run import ContactRun, SIDES
from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.grip_contact.grip_margin import (
    DEFAULT_NORMAL_FLOOR_N,
    GripMarginReport,
    grip_margin_report,
)
from src.shared.python.grip_contact.pad_contact import PadWrench
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    topology_side_wrench,
)

RECEIPT_DURATION_S: Final[float] = 0.04
RECEIPT_SQUEEZE_PER_HAND_N: Final[float] = 1100.0
RECEIPT_LEAD_GRIP_FORCE_LIMIT_N: Final[float] = 2500.0
RECEIPT_TRAIL_GRIP_FORCE_LIMIT_N: Final[float] = 2500.0


@dataclass(frozen=True)
class SwingReceiptConfig:
    """Frozen scripted-swing receipt parameters (deterministic CI receipt)."""

    duration_s: float = RECEIPT_DURATION_S
    squeeze_per_hand_n: float = RECEIPT_SQUEEZE_PER_HAND_N
    lead_grip_force_limit_n: float = RECEIPT_LEAD_GRIP_FORCE_LIMIT_N
    trail_grip_force_limit_n: float = RECEIPT_TRAIL_GRIP_FORCE_LIMIT_N
    normal_floor_n: float = DEFAULT_NORMAL_FLOOR_N

    def __post_init__(self) -> None:
        for name in (
            "duration_s",
            "squeeze_per_hand_n",
            "lead_grip_force_limit_n",
            "trail_grip_force_limit_n",
            "normal_floor_n",
        ):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be positive and finite, got {value}")


def _role_for_side(spec: Mapping[str, Any], side: str) -> str:
    handedness = subject_handedness(spec)
    if side == lead_side(handedness):
        return "lead"
    if side == trail_side(handedness):
        return "trail"
    raise ValueError(f"unknown side {side!r}")


def pad_wrenches_by_role(
    run: ContactRun,
    model: TopologyContactModel,
    spec: Mapping[str, Any],
) -> dict[str, list[PadWrench]]:
    """Per-frame lead/trail :class:`PadWrench` lists from topology replay."""
    if run.hand_states is None or run.club_states is None:
        raise ValueError(
            "ContactRun must include hand_states and club_states from the recorder"
        )
    roles: dict[str, list[PadWrench]] = {"lead": [], "trail": []}
    n = run.series.time_s.size
    for i in range(n):
        for side in SIDES:
            role = _role_for_side(spec, side)
            side_w = topology_side_wrench(
                model,
                side,
                run.hand_states[side][i],
                run.club_states[side][i],
            )
            roles[role].append(
                PadWrench(
                    side_w.force_n,
                    side_w.moment_nm,
                    side_w.normal_force_n,
                    side_w.tangential_force_n,
                )
            )
    return roles


def club_acceleration_m_s2(run: ContactRun, club_mass_kg: float) -> np.ndarray:
    """World-frame club CoM acceleration from net hand force (free club)."""
    if not (math.isfinite(club_mass_kg) and club_mass_kg > 0.0):
        raise ValueError(
            f"club_mass_kg must be positive and finite, got {club_mass_kg}"
        )
    force_l = np.asarray(run.series.force_on_club_n["L"], float)
    force_r = np.asarray(run.series.force_on_club_n["R"], float)
    return (force_l + force_r) / club_mass_kg


def pad_tangential_speed_m_s(
    run: ContactRun,
    spec: Mapping[str, Any],
    model: TopologyContactModel,
) -> np.ndarray:
    """``(frames, pads)`` slip-speed proxy from roll and axial slip rates."""
    time_s = run.series.time_s
    columns: list[np.ndarray] = []
    for role in ("lead", "trail"):
        side = (
            lead_side(subject_handedness(spec))
            if role == "lead"
            else trail_side(subject_handedness(spec))
        )
        roll = np.asarray(run.roll_slip_rad[side], float)
        axial = np.asarray(run.axial_slip_m[side], float)
        rate = np.hypot(
            np.abs(np.gradient(roll, time_s)),
            np.abs(np.gradient(axial, time_s)),
        )
        pad_count = len(model.physical_grip_pairs(role))
        columns.append(
            np.column_stack([rate] * pad_count) if pad_count else rate[:, None]
        )
    return np.hstack(columns)


def peak_hand_force_n(run: ContactRun, spec: Mapping[str, Any]) -> dict[str, float]:
    """Peak per-hand force magnitude (N) keyed by lead/trail."""
    peaks: dict[str, float] = {}
    for side in SIDES:
        role = _role_for_side(spec, side)
        force = np.asarray(run.series.force_on_club_n[side], float)
        peaks[role] = float(np.max(np.linalg.norm(force, axis=1)))
    return peaks


def assert_grip_force_limits(
    run: ContactRun,
    spec: Mapping[str, Any],
    lead_limit_n: float,
    trail_limit_n: float,
) -> None:
    """Receipt gate: scripted swing stays below declared per-hand force caps."""
    peaks = peak_hand_force_n(run, spec)
    if peaks["lead"] > lead_limit_n + 1e-6:
        raise ValueError(
            f"lead peak force {peaks['lead']:.3f} N exceeds limit {lead_limit_n} N"
        )
    if peaks["trail"] > trail_limit_n + 1e-6:
        raise ValueError(
            f"trail peak force {peaks['trail']:.3f} N exceeds limit {trail_limit_n} N"
        )


def grip_margin_from_contact_run(
    run: ContactRun,
    model: TopologyContactModel,
    spec: Mapping[str, Any],
    friction: float,
    *,
    config: SwingReceiptConfig | None = None,
    mu_bias_bound: float | None = None,
) -> GripMarginReport:
    """Build a :class:`GripMarginReport` from one topology :class:`ContactRun`."""
    cfg = config or SwingReceiptConfig()
    assert_grip_force_limits(
        run, spec, cfg.lead_grip_force_limit_n, cfg.trail_grip_force_limit_n
    )
    dt_s = float(np.mean(np.diff(run.series.time_s)))
    club_mass = ClubDynamics.from_spec(spec).mass_kg
    return grip_margin_report(
        pad_wrenches_by_role(run, model, spec),
        club_acceleration_m_s2(run, club_mass),
        pad_tangential_speed_m_s(run, spec, model),
        dt_s,
        friction,
        cfg.normal_floor_n,
        mu_bias_bound=mu_bias_bound,
    )


def load_mu_bias_bound(calibration_path: Path) -> float | None:
    """Read ``declared_bias`` from the first calibration row, if any."""
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    rows = payload.get("calibrations") or []
    if not rows:
        return None
    return rows[0].get("declared_bias")


def receipt_payload(
    report: GripMarginReport,
    *,
    config: SwingReceiptConfig,
    style: GripStyle,
    friction: float,
    calibration_path: Path,
) -> dict[str, Any]:
    """JSON-serialisable scripted swing receipt."""
    return {
        "issue": 13885,
        "style": style.value,
        "friction": friction,
        "calibration_receipt": calibration_path.name,
        "lead_grip_force_limit_n": config.lead_grip_force_limit_n,
        "trail_grip_force_limit_n": config.trail_grip_force_limit_n,
        "squeeze_per_hand_n": config.squeeze_per_hand_n,
        "duration_s": config.duration_s,
        "min_margin": report.min_margin,
        "slip_event_count": report.slip_event_count,
        "slip_duration_s": report.slip_duration_s,
        "peak_frame": report.peak_frame,
        "mu_bias_bound": report.mu_bias_bound,
        "reproduce": "python3 -m scripts.grip.mujoco_grip_margin_swing_receipt",
    }


def serialise_report(report: GripMarginReport) -> dict[str, Any]:
    """Flatten a report for evidence JSON (margins stay JSON-safe)."""
    body = asdict(report)
    body["pad_margin"] = {
        hand: list(margins) for hand, margins in report.pad_margin.items()
    }
    body["net_wrench"] = report.net_wrench.tolist()
    body["internal_wrench"] = report.internal_wrench.tolist()
    return body
