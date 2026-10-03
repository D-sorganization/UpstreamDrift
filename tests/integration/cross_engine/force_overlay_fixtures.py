"""Shared ``synthetic_`` fixtures for the force/torque overlay parity suite (FTO-21, #11306).

One parameter set defines every scenario. Engine builders (beside the test) turn
it into a model; the expected wrenches below are derived here, once, from
statics in the Z-up world (ADR-0026), so no engine row can disagree by
construction.

Scenarios:
    hanging   one link hanging at rest; reaction is +m*g up at the pivot,
              axial load is +m*g (tension).
    inverted  the same link standing at ``HOLD_ANGLE_RAD`` from vertical, held
              by an actuator; actuator torque balances gravity about the pivot,
              axial load is -m*g*cos(theta) (compression).
    resting   a body resting on the ground; contact forces sum to m*g up.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from src.shared.python.force_overlay import ForceTorqueFrame, WrenchKind

GRAVITY_M_S2 = 9.80665
GRAVITY_WORLD = np.array([0.0, 0.0, -GRAVITY_M_S2])


@dataclass(frozen=True)
class PendulumParams:
    """Single-link pendulum shared by every engine builder."""

    mass_kg: float = 2.0
    length_m: float = 1.0
    pivot_height_m: float = 1.0
    hold_angle_rad: float = math.pi / 6

    def __post_init__(self) -> None:
        for name in ("mass_kg", "length_m", "pivot_height_m"):
            value = getattr(self, name)
            if not (math.isfinite(value) and value > 0.0):
                raise ValueError(f"{name} must be positive and finite")
        if not 0.0 < self.hold_angle_rad < math.pi / 2:
            raise ValueError("hold_angle_rad must be in (0, pi/2)")

    @property
    def weight_n(self) -> float:
        return self.mass_kg * GRAVITY_M_S2

    @property
    def pivot_world(self) -> np.ndarray:
        return np.array([0.0, 0.0, self.pivot_height_m])

    @property
    def com_offset_m(self) -> float:
        """Pivot-to-centre-of-mass distance (uniform link)."""
        return self.length_m / 2.0

    def rod_direction_world(self, angle_rad: float) -> np.ndarray:
        """Unit pivot-to-COM direction; ``angle_rad`` is measured from straight up."""
        return np.array([-math.sin(angle_rad), 0.0, math.cos(angle_rad)])


PENDULUM = PendulumParams()
BALL_MASS_KG = 1.0
BALL_RADIUS_M = 0.1

# --- tolerances (set once; never loosen without a Tolerance-Change-Evidence trailer) ---
# Analytic engines solve statics to solver precision.
ANALYTIC_ATOL_N = 1e-6
# Compliant (penalty / Hunt-Crossley) contact settles to a small penetration, so
# the summed force equals the weight only to ~1% (stiffness 1e6 N/m, 1 s settle).
CONTACT_RTOL = 1e-2
CONTACT_ATOL_N = 1e-2
CONTACT_POINT_ATOL_M = 2e-3


def hanging_expected(p: PendulumParams = PENDULUM) -> dict[str, object]:
    """Reaction on the link from its parent, at the pivot (statics)."""
    return {
        "force_n": (0.0, 0.0, p.weight_n),
        "point_m": tuple(p.pivot_world),
        "axial_n": p.weight_n,
    }


def inverted_expected(p: PendulumParams = PENDULUM) -> dict[str, object]:
    """Actuator torque and axial load holding the link at ``hold_angle_rad``."""
    theta = p.hold_angle_rad
    r_com = p.com_offset_m * p.rod_direction_world(theta)
    gravity_torque = np.cross(r_com, p.mass_kg * GRAVITY_WORLD)
    return {
        "actuator_torque_nm": tuple(-gravity_torque),
        "actuator_torque_magnitude_nm": p.weight_n * p.com_offset_m * math.sin(theta),
        "reaction_force_n": (0.0, 0.0, p.weight_n),
        "point_m": tuple(p.pivot_world),
        "axial_n": -p.weight_n * math.cos(theta),
    }


def assert_wrench(
    frame: ForceTorqueFrame,
    kind: WrenchKind,
    *,
    force: tuple[float, float, float] | None = None,
    torque: tuple[float, float, float] | None = None,
    point: tuple[float, float, float] | None = None,
    atol: float = ANALYTIC_ATOL_N,
    rtol: float = 0.0,
    label_prefix: str | None = None,
) -> None:
    """Assert the single wrench of ``kind`` (optionally by label prefix) matches.

    Only arguments that are given are checked, so a torque-only actuator row never
    pretends to have a force. A given half that the wrench lacks fails loudly.
    """
    found = [
        w
        for w in frame.by_kind(kind)
        if label_prefix is None or w.label.startswith(label_prefix)
    ]
    assert len(found) == 1, (
        f"expected exactly one {kind.value} wrench"
        f"{'' if label_prefix is None else f' with prefix {label_prefix!r}'}, "
        f"got {[w.label for w in frame.wrenches]}"
    )
    wrench = found[0]
    for name, expected, actual in (
        ("force_n", force, wrench.force_n),
        ("torque_nm", torque, wrench.torque_nm),
        ("point_m", point, wrench.point_m),
    ):
        if expected is None:
            continue
        assert actual is not None, f"{wrench.label}: {name} unavailable"
        np.testing.assert_allclose(
            actual, expected, rtol=rtol, atol=atol, err_msg=f"{wrench.label}.{name}"
        )
