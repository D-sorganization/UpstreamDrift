"""Address grip roll metric and face-on overlay rendering (SHOW-A1, #14550).

Parent: Epic #14549 (MuJoCo Anatomy Refit).

Defines the golf-language grip roll metric at address:
- Hand roll about the shaft from the golfer's view down the shaft.
- 0° = vertical (top of grip perpendicular to the shaft, pointing toward golfer's chin/eyes).
- Positive (+) = strong / clockwise (rotating the top of hands toward trail side / right shoulder).
- Negative (-) = weak / counter-clockwise (rotating the top of hands toward target).

Mapping to HX Alpha / Gamma Decomposition (#13757):
---------------------------------------------------
The full hand offset matrix M factors uniquely as:
    M = R_y(alpha) @ R_x(psi - 90 deg) @ R_z(gamma)
where:
- y is along the shaft (butt to head).
- alpha rotates the hand about the shaft axis (grip strength: + strong, - weak).
- psi is the palm diagonal (radial to distal in palm plane, nominal 40 deg).
- gamma is the wrist flexion / set angle.

The address grip roll angle theta directly recovers alpha:
    alpha = atan2(M[0, 2], M[2, 2])
invariant to palm diagonal psi and wrist flexion gamma.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.shared.python.contracts import require

ROOT = Path(__file__).resolve().parents[4]
DRIVER_SPEC_PATH = (
    ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)


def _is_so3(m: np.ndarray, tol: float = 1e-4) -> bool:
    """Check if matrix m is a 3x3 proper rotation matrix in SO(3)."""
    if m.shape != (3, 3) or not bool(np.isfinite(m).all()):
        return False
    is_ortho = bool(np.allclose(m @ m.T, np.eye(3), atol=tol))
    is_det1 = bool(np.isclose(np.linalg.det(m), 1.0, atol=tol))
    return is_ortho and is_det1


@dataclass(frozen=True)
class AddressGripRoll:
    """Lead and trail hand grip roll angles (degrees) at address."""

    lead_deg: float
    trail_deg: float

    def __iter__(self) -> Iterator[float]:
        return iter((self.lead_deg, self.trail_deg))

    def __getitem__(self, idx: int) -> float:
        return (self.lead_deg, self.trail_deg)[idx]


@dataclass(frozen=True)
class AddressGripState:
    """Orientation frames for lead hand, trail hand, and club shaft."""

    lead_hand_frame: np.ndarray
    trail_hand_frame: np.ndarray
    shaft_frame: np.ndarray = field(default_factory=lambda: np.eye(3))


def address_grip_roll_deg(
    model_state: AddressGripState | Mapping[str, Any],
) -> AddressGripRoll:
    """Compute lead and trail hand roll about the shaft at address.

    Preconditions:
    - Frames must be proper 3x3 orthonormal rotation matrices in SO(3).
    - Raises ContractViolationError if any frame fails orthonormality or det == 1.

    Returns:
    - AddressGripRoll with lead_deg and trail_deg, unpackable as (lead, trail).
    """
    if isinstance(model_state, Mapping) and "joints" in model_state:
        return flagship_driver_grip_roll(model_state)

    if not isinstance(model_state, AddressGripState):
        require(False, "model_state must be an AddressGripState")
        raise TypeError("model_state must be an AddressGripState")

    lead = np.asarray(model_state.lead_hand_frame, dtype=float)
    trail = np.asarray(model_state.trail_hand_frame, dtype=float)
    shaft = np.asarray(model_state.shaft_frame, dtype=float)

    require(
        _is_so3(lead),
        "lead_hand_frame must be a 3x3 rotation in SO(3)",
    )
    require(
        _is_so3(trail),
        "trail_hand_frame must be a 3x3 rotation in SO(3)",
    )
    require(
        _is_so3(shaft),
        "shaft_frame must be a 3x3 rotation in SO(3)",
    )

    # Hand relative to shaft frame: R_rel = shaft.T @ hand
    r_lead = shaft.T @ lead
    r_trail = shaft.T @ trail

    # In golf/club convention (butt to head along +y):
    # R_rel = R_y(alpha) @ R_x(psi - 90 deg) @ R_z(gamma)
    # alpha = atan2(R_rel[0, 2], R_rel[2, 2])
    alpha_lead = float(np.degrees(np.arctan2(r_lead[0, 2], r_lead[2, 2])))
    alpha_trail = float(np.degrees(np.arctan2(r_trail[0, 2], r_trail[2, 2])))

    return AddressGripRoll(lead_deg=alpha_lead, trail_deg=alpha_trail)


def _extract_wrist_c2f(doc: Mapping[str, Any], suffix: str) -> np.ndarray:
    for joint in doc.get("joints", []):
        if joint.get("child", "").endswith(suffix):
            return np.asarray(joint["child_to_follower"], dtype=float)[:3, :3]
    raise ValueError(f"Joint ending with {suffix} not found in spec")


def flagship_driver_grip_roll(
    spec: Mapping[str, Any] | None = None,
) -> AddressGripRoll:
    """Evaluate address grip roll for the flagship driver anthro model."""
    if spec is None:
        spec = json.loads(DRIVER_SPEC_PATH.read_text(encoding="utf-8"))

    c2f_l = _extract_wrist_c2f(spec, "Clubface Vector")
    c2f_r = _extract_wrist_c2f(spec, "RHandStandoff")

    lead_deg = float(np.degrees(np.arctan2(c2f_l[1, 0], c2f_l[1, 1])))
    trail_deg = float(np.degrees(np.arctan2(c2f_r[1, 2], -c2f_r[1, 0])))

    return AddressGripRoll(lead_deg=lead_deg, trail_deg=trail_deg)


def render_address_grip_overlay_png(
    spec_or_rolls: Mapping[str, Any] | AddressGripRoll | None,
    output_path: str | Path,
) -> Path:
    """Render face-on overlay PNG showing address grip roll and target.

    Creates visual evidence of current vs target grip roll angles.
    """
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(spec_or_rolls, AddressGripRoll):
        rolls = spec_or_rolls
    elif isinstance(spec_or_rolls, Mapping):
        rolls = flagship_driver_grip_roll(spec_or_rolls)
    else:
        rolls = flagship_driver_grip_roll()

    fig, ax_raw = plt.subplots(figsize=(8, 7), subplot_kw={"projection": "polar"})
    ax: Any = ax_raw

    # Polar plot: 0 is top (vertical, pi/2 in standard polar)
    ax.set_theta_zero_location("N")
    # Clockwise is strong (+)
    ax.set_theta_direction(-1)

    # Plot axes
    ax.axvline(
        x=0,
        color="gray",
        linestyle="--",
        linewidth=1.5,
        label="0° Vertical (Neutral Top-of-Grip)",
    )
    ax.axvline(
        x=np.radians(15.0),
        color="green",
        linestyle="-.",
        linewidth=2.0,
        label="Target Grip (~15° Strong, #14551)",
    )

    # Plot current flagship values
    lead_rad = np.radians(rolls.lead_deg)
    trail_rad = np.radians(rolls.trail_deg)

    ax.plot(
        [lead_rad, lead_rad],
        [0, 1.0],
        color="tab:blue",
        linewidth=2.5,
        label=f"Current Lead Hand ({rolls.lead_deg:.1f}° Strong)",
    )
    ax.plot(
        [trail_rad, trail_rad],
        [0, 1.0],
        color="tab:orange",
        linewidth=2.5,
        label=f"Current Trail Hand ({rolls.trail_deg:.1f}° Strong)",
    )

    ax.set_ylim(0, 1.1)
    ax.set_yticks([])
    ax.set_title(
        f"MuJoCo Flagship Driver Address Grip Roll (Face-on View)\n"
        f"Lead: {rolls.lead_deg:.1f}° | Trail: {rolls.trail_deg:.1f}° | Target: ~15° (#14550 / #14551)",
        pad=20,
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right", bbox_to_anchor=(1.35, -0.1), frameon=True)

    fig.tight_layout()
    fig.savefig(output, dpi=150)
    plt.close(fig)

    return output
