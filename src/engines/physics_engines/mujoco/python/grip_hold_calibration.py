"""MuJoCo static-hold mu_crit calibration of the pad contact (#13885 PR 1).

A free block of weight ``m g`` is squeezed between two pad spheres, each
pushed horizontally with ``squeeze_n``, so gravity is the only tangential load
and ``mu_crit = m g / (2 N)`` exactly. The pad-block pairs are the pad path's
own ``<pair>`` and ``<option>`` (:func:`grip_contact_sim.pad_pair_xml`), so the
calibration measures the friction of the contact the grip runs on. A run
holds when the block's mean slide speed over the hold window stays at or
below the grip-margin slip threshold. Bisection on ``mu / mu_crit`` gives the
hold/slip transition per integrator; a transition ratio outside
``1 +- BIAS_BAND`` is a declared friction bias that margin reports carry
(``GripMarginReport.mu_bias_bound``). The topology path is not calibrated
here: its per-site ``solref`` comes from ``ClubInHands.calibrate``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Final

from src.engines.physics_engines.mujoco.python.grip_bushing import _mujoco
from src.engines.physics_engines.mujoco.python.grip_contact_sim import (
    DEFAULT_TIMESTEP_S,
    pad_option_xml,
    pad_pair_xml,
)
from src.shared.python.core.physics_constants import GRAVITY_M_S2
from src.shared.python.grip_contact.grip_margin import DEFAULT_SLIP_SPEED_M_S

BIAS_BAND: Final[float] = 0.05
BISECTION_TOL: Final[float] = 0.005
INTEGRATORS: Final[tuple[str, ...]] = ("Euler", "implicitfast", "implicit", "RK4")
# Euler is the pad path's configured integrator; implicitfast is MuJoCo's
# recommended one, recorded so a switch would not need a new calibration.
RECEIPT_INTEGRATORS: Final[tuple[str, ...]] = ("Euler", "implicitfast")
BLOCK_HALF_M: Final[float] = 0.02
PAD_RADIUS_M: Final[float] = 0.01
PAD_MASS_KG: Final[float] = 0.1
PAD_SOLREF: Final[tuple[float, float]] = (0.02, 1.0)
ROLLING: Final[float] = 1e-5


@dataclass(frozen=True)
class HoldRig:
    """Block-between-two-pads hold; ``settle_s`` then ``hold_s`` of motion."""

    block_mass_kg: float = 1.0
    squeeze_n: float = 50.0
    timestep_s: float = DEFAULT_TIMESTEP_S
    settle_s: float = 0.05
    hold_s: float = 0.1

    def __post_init__(self) -> None:
        for name in ("block_mass_kg", "squeeze_n", "timestep_s", "settle_s", "hold_s"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be positive and finite, got {value}")

    @property
    def mu_crit(self) -> float:
        """Friction coefficient at which both pads just carry the weight."""
        return self.block_mass_kg * float(GRAVITY_M_S2) / (2.0 * self.squeeze_n)

    def mjcf(self, friction: float, integrator: str) -> str:
        """Model with both pads touching the block (no preload gap)."""
        x = BLOCK_HALF_M + PAD_RADIUS_M
        pads = "".join(
            f'<body name="pad{k}" pos="{s * x:.17g} 0 0">'
            f'<joint name="slide{k}" type="slide" axis="1 0 0"/>'
            f'<geom name="pad{k}" type="sphere" size="{PAD_RADIUS_M}" '
            f'mass="{PAD_MASS_KG}" contype="0" conaffinity="0"/></body>'
            for k, s in enumerate((-1.0, 1.0))
        )
        ftime = 2.0 * self.timestep_s  # the pad path's default friction time
        pairs = "".join(
            pad_pair_xml(f"pad{k}", "block", friction, ROLLING, PAD_SOLREF, ftime)
            for k in range(2)
        )
        return (
            '<mujoco model="grip_hold_calibration">'
            + pad_option_xml(
                self.timestep_s, (0.0, 0.0, -float(GRAVITY_M_S2)), integrator
            )
            + '<worldbody><body name="block"><freejoint/>'
            f'<geom name="block" type="box" size="{BLOCK_HALF_M} {BLOCK_HALF_M} '
            f'{BLOCK_HALF_M}" mass="{self.block_mass_kg:.17g}" contype="0" '
            'conaffinity="0"/></body>' + pads + "</worldbody>"
            "<contact>" + pairs + "</contact></mujoco>"
        )


def slide_speed_m_s(rig: HoldRig, friction: float, integrator: str) -> float:
    """Mean downward block speed over the hold window.

    Raises:
        ValueError: on a non-positive ``friction`` or an unknown integrator.
    """
    if not (math.isfinite(friction) and friction > 0.0):
        raise ValueError(f"friction must be positive and finite, got {friction}")
    if integrator not in INTEGRATORS:
        raise ValueError(f"integrator must be one of {INTEGRATORS}, got {integrator!r}")
    mujoco = _mujoco()
    model = mujoco.MjModel.from_xml_string(rig.mjcf(friction, integrator))
    data = mujoco.MjData(model)
    block, pads = model.body("block").id, (model.body("pad0").id, model.body("pad1").id)
    data.xfrc_applied[pads[0], 0], data.xfrc_applied[pads[1], 0] = (
        rig.squeeze_n,
        -rig.squeeze_n,
    )
    settle = round(rig.settle_s / rig.timestep_s)
    hold = round(rig.hold_s / rig.timestep_s)
    mujoco.mj_step(model, data, nstep=settle)
    z0 = float(data.xpos[block, 2])
    mujoco.mj_step(model, data, nstep=hold)
    return (z0 - float(data.xpos[block, 2])) / (hold * rig.timestep_s)


def holds(rig: HoldRig, friction: float, integrator: str) -> bool:
    """Whether the block's mean slide speed stays at or below the slip threshold."""
    return slide_speed_m_s(rig, friction, integrator) <= DEFAULT_SLIP_SPEED_M_S


def declared_bias(ratio: float) -> float | None:
    """``ratio`` when it lies outside ``1 +- BIAS_BAND``, else ``None``."""
    return ratio if abs(ratio - 1.0) > BIAS_BAND + 1e-12 else None


@dataclass(frozen=True)
class HoldCalibration:
    """Hold/slip transition of one integrator; ``transition_mu`` holds."""

    integrator: str
    mu_crit: float
    transition_mu: float
    ratio: float
    declared_bias: float | None


def calibrate(
    rig: HoldRig,
    integrator: str,
    lo: float = 0.5,
    hi: float = 2.0,
    tol: float = BISECTION_TOL,
) -> HoldCalibration:
    """Bisect ``mu / mu_crit`` in ``[lo, hi]`` to ``tol`` for the hold/slip edge.

    Raises:
        ValueError: when ``lo`` holds or ``hi`` slips (no bracket).
    """
    mu_crit = rig.mu_crit
    if holds(rig, lo * mu_crit, integrator) or not holds(rig, hi * mu_crit, integrator):
        raise ValueError(f"[{lo}, {hi}] x mu_crit does not bracket the hold/slip edge")
    while hi - lo > tol:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if holds(rig, mid * mu_crit, integrator) else (mid, hi)
    return HoldCalibration(integrator, mu_crit, hi * mu_crit, hi, declared_bias(hi))
