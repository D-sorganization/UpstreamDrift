"""Named putter balance presets over :class:`~.types.PutterState` (#14461).

A putter's balance type is where its head CG sits relative to the shaft
axis, plus the head's mass and MOI. In :class:`~.types.PutterState` terms
the shaft attachment is offset from the CG by ``hosel_toe_m`` (heel
negative) and ``hosel_forward_m`` (a CG behind the shaft makes it
positive), which is what decides the face twist of a strike
(:func:`~.collision.strike`).

The four heads are the putting studio's putter types (``PUTTER_SPECS`` in
``src/apps/putting_studio/js/state.js``: ``e_face`` is the CG's toe-ward
offset from the shaft, ``e_back`` its offset behind it, MOI in g cm^2).
They are representative catalogue values, not measurements of a named
model. The counterbalanced head's 75 g butt weight is not modelled: a
:class:`~.types.PutterState` is the head at impact.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.shared.python.contracts import require
from src.shared.python.core.physics_constants import PUTTER_FACE_COR

from .types import TYPICAL_LOFT_DEG, PutterState

__all__ = ["BALANCE_PRESETS", "BalancePreset", "balance_preset"]


@dataclass(frozen=True)
class BalancePreset:
    """A named putter head: mass [kg], MOI [kg m^2] and shaft offsets [m]."""

    key: str
    name: str
    head_mass_kg: float
    moi_kg_m2: float
    hosel_toe_m: float
    hosel_forward_m: float

    def putter(
        self,
        speed_mps: float,
        loft_deg: float = TYPICAL_LOFT_DEG,
        cor: float = float(PUTTER_FACE_COR),
    ) -> PutterState:
        """This head swung at ``speed_mps`` (validated by PutterState)."""
        return PutterState(
            head_mass_kg=self.head_mass_kg,
            moi_kg_m2=self.moi_kg_m2,
            loft_deg=loft_deg,
            speed_mps=speed_mps,
            cor=cor,
            hosel_toe_m=self.hosel_toe_m,
            hosel_forward_m=self.hosel_forward_m,
        )


BALANCE_PRESETS: dict[str, BalancePreset] = {
    p.key: p
    for p in (
        BalancePreset("toe_hang", "Toe-Hang Blade", 0.345, 3.2e-4, -0.015, 0.012),
        BalancePreset(
            "face_balanced", "Face-Balanced Mallet", 0.360, 5.2e-4, 0.0, 0.026
        ),
        BalancePreset(
            "zero_torque", "Zero-Torque (Lie-Angle Balanced)", 0.365, 4.9e-4, 0.0, 0.0
        ),
        BalancePreset(
            "counterbalanced", "Heavy Counterbalanced", 0.385, 5.8e-4, -0.002, 0.018
        ),
    )
}


def balance_preset(key: str) -> BalancePreset:
    """The preset named ``key``; raises ``ValueError`` for an unknown key."""
    require(
        key in BALANCE_PRESETS,
        f"balance type must be one of {sorted(BALANCE_PRESETS)}",
        key,
    )
    return BALANCE_PRESETS[key]
