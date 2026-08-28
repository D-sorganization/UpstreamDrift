"""Stateful tangential-compliance countermodel for distributed grip studies.

This is an elastic--perfectly-plastic engineering interface with a radial
Coulomb return map. It is not finger anatomy, measured tissue behavior,
intentional action, or evidence that a human grip follows this law.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

FloatArray = NDArray[np.float64]


class TangentialRegime(str, Enum):
    """Declared state of one tangential attachment station."""

    OPEN = "open"
    STICK = "elastic_stick"
    SLIP = "coulomb_slip"


@dataclass(frozen=True, slots=True)
class StatefulFrictionConfig:
    """Constitutive constants for an isotropic tangential return map."""

    tangential_stiffness_n_m: float
    friction_coefficient: float

    def __post_init__(self) -> None:
        if (
            not np.isfinite(self.tangential_stiffness_n_m)
            or self.tangential_stiffness_n_m <= 0.0
        ):
            raise ValueError("tangential_stiffness_n_m must be finite and positive")
        if (
            not np.isfinite(self.friction_coefficient)
            or self.friction_coefficient < 0.0
        ):
            raise ValueError("friction_coefficient must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class TangentialState:
    """Retained elastic displacement for one contact station."""

    elastic_displacement_m: FloatArray

    def __post_init__(self) -> None:
        value = np.asarray(self.elastic_displacement_m, dtype=np.float64)
        if value.shape != (3,) or not np.all(np.isfinite(value)):
            raise ValueError("elastic_displacement_m must be one finite 3-vector")
        retained = value.copy()
        retained.setflags(write=False)
        object.__setattr__(self, "elastic_displacement_m", retained)

    @classmethod
    def zero(cls) -> TangentialState:
        """Return an undeformed state."""

        return cls(np.zeros(3, dtype=np.float64))


@dataclass(frozen=True, slots=True)
class StatefulFrictionStep:
    """One return-mapped increment and its exact constitutive energy ledger."""

    state: TangentialState
    regime: TangentialRegime
    force_on_club_n: FloatArray
    friction_limit_n: float
    trial_force_norm_n: float
    yield_margin_n: float
    plastic_slip_increment_m: float
    elastic_energy_before_j: float
    elastic_energy_after_j: float
    elastic_energy_change_j: float
    frictional_dissipation_j: float
    release_dissipation_j: float
    constitutive_work_j: float
    static_stick_modeled: bool = True
    human_or_anatomical_inference: bool = False


def _elastic_energy(displacement: FloatArray, stiffness: float) -> float:
    return float(0.5 * stiffness * (displacement @ displacement))


def advance_stateful_friction(
    state: TangentialState,
    *,
    tangential_displacement_increment_m: FloatArray,
    normal_load_n: float,
    active: bool,
    config: StatefulFrictionConfig,
) -> StatefulFrictionStep:
    """Advance one station with an isotropic Coulomb radial return.

    The displacement increment is split into retained elastic displacement and
    irreversible plastic slip. On opening, retained elastic energy is released
    into an explicit dissipation channel and the state resets to zero.
    """

    if not isinstance(state, TangentialState):
        raise TypeError("state must be a TangentialState")
    if not isinstance(config, StatefulFrictionConfig):
        raise TypeError("config must be a StatefulFrictionConfig")
    increment = np.asarray(tangential_displacement_increment_m, dtype=np.float64)
    if increment.shape != (3,) or not np.all(np.isfinite(increment)):
        raise ValueError(
            "tangential_displacement_increment_m must be one finite 3-vector"
        )
    if not np.isfinite(normal_load_n) or normal_load_n < 0.0:
        raise ValueError("normal_load_n must be finite and nonnegative")
    if not isinstance(active, bool):
        raise TypeError("active must be a bool")
    if not active and normal_load_n != 0.0:
        raise ValueError("inactive contact requires zero normal_load_n")

    stiffness = config.tangential_stiffness_n_m
    before = _elastic_energy(state.elastic_displacement_m, stiffness)
    limit = config.friction_coefficient * normal_load_n
    if not active:
        after = 0.0
        release = before
        return StatefulFrictionStep(
            state=TangentialState.zero(),
            regime=TangentialRegime.OPEN,
            force_on_club_n=np.zeros(3),
            friction_limit_n=0.0,
            trial_force_norm_n=0.0,
            yield_margin_n=0.0,
            plastic_slip_increment_m=0.0,
            elastic_energy_before_j=before,
            elastic_energy_after_j=after,
            elastic_energy_change_j=after - before,
            frictional_dissipation_j=0.0,
            release_dissipation_j=release,
            constitutive_work_j=0.0,
        )

    trial_displacement = state.elastic_displacement_m + increment
    trial_force = stiffness * trial_displacement
    trial_norm = float(np.linalg.norm(trial_force))
    margin = limit - trial_norm
    if trial_norm <= limit:
        retained_displacement = trial_displacement
        force = trial_force
        plastic_increment = 0.0
        dissipation = 0.0
        regime = TangentialRegime.STICK
    else:
        direction = trial_force / trial_norm
        retained_displacement = (limit / stiffness) * direction
        force = limit * direction
        plastic_vector = trial_displacement - retained_displacement
        plastic_increment = float(np.linalg.norm(plastic_vector))
        dissipation = limit * plastic_increment
        regime = TangentialRegime.SLIP
    after = _elastic_energy(retained_displacement, stiffness)
    energy_change = after - before
    work = energy_change + dissipation
    return StatefulFrictionStep(
        state=TangentialState(retained_displacement),
        regime=regime,
        force_on_club_n=np.asarray(force, dtype=np.float64),
        friction_limit_n=limit,
        trial_force_norm_n=trial_norm,
        yield_margin_n=margin,
        plastic_slip_increment_m=plastic_increment,
        elastic_energy_before_j=before,
        elastic_energy_after_j=after,
        elastic_energy_change_j=energy_change,
        frictional_dissipation_j=dissipation,
        release_dissipation_j=0.0,
        constitutive_work_j=work,
    )


__all__ = [
    "StatefulFrictionConfig",
    "StatefulFrictionStep",
    "TangentialRegime",
    "TangentialState",
    "advance_stateful_friction",
]
