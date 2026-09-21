"""Swing-state providers — honest engine sourcing for the swing→flight pipeline.

Fixes issue #8819: the Swing→Flight Pipeline GUI offered a "Physics Engine
Source" combo (mujoco / drake / pinocchio / manual) whose selection only
changed the ``engine_name`` *label* stamped on the result — every option
produced byte-identical numbers from the manually entered GUI parameters.

This module introduces a narrow seam between "where a ``SwingState`` comes
from" and "what the pipeline does with it":

* :class:`SwingStateProvider` — protocol for anything that can produce a
  ``SwingState`` from a :class:`SwingStateConfig`.
* :class:`ManualSwingStateProvider` — builds the state directly from the
  user-entered parameters.
* :class:`MuJoCoSwingStateProvider` — sources the state from a real MuJoCo
  forward-dynamics swing (issue #8975); available whenever ``mujoco`` and
  the in-repo golf MJCF asset are importable.
* :class:`UnimplementedEngineProvider` — placeholder for engine-backed
  sourcing that does not exist yet (drake / pinocchio).  Always unavailable,
  with an honest human-readable reason.  GUIs list it but must disable it.

Design-by-Contract
------------------
* Precondition: ``get_swing_state`` may only be called on an available
  provider, with a physically valid config.
* Postcondition: the returned ``SwingState.engine_name`` MUST equal the
  provider's ``provider_id`` — killing false engine attribution at the seam.

Law of Demeter
--------------
GUIs talk to providers; providers build ``SwingState``; only the pipeline
consumes it.  No GUI code reaches into engine internals.
"""

from __future__ import annotations

import importlib.util
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from src.shared.python.contracts import ensure, require

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.shared.python.physics.swing_ball_flight_pipeline import SwingState

#: Reason string for an engine whose Python package is importable but for
#: which no swing-state extraction path exists yet.
REASON_NOT_IMPLEMENTED = "engine sourcing not yet implemented"

#: Reason string for an engine whose Python package is not installed.
REASON_NOT_INSTALLED = "engine not installed"

#: MJCF asset module the MuJoCo provider needs (checked without importing
#: the heavyweight engine at registry-construction time).
_MUJOCO_MODEL_ASSET_MODULE = (
    "src.engines.physics_engines.mujoco._golf_swing_upper_body_xml"
)


@dataclass(frozen=True)
class SwingStateConfig:
    """User-facing swing parameters a provider turns into a ``SwingState``.

    Attributes:
        clubhead_speed_ms: Clubhead speed at impact [m/s]. Must be > 0.
        loft_deg:          Clubface loft angle [degrees]. Must be in (0, 90).
        clubhead_mass_kg:  Effective clubhead mass [kg]. Must be > 0.
    """

    clubhead_speed_ms: float = 45.0
    loft_deg: float = 10.5
    clubhead_mass_kg: float = 0.200


@runtime_checkable
class SwingStateProvider(Protocol):
    """Anything that can produce a ``SwingState`` for the pipeline."""

    provider_id: str

    def is_available(self) -> bool:
        """Return True if this provider can produce a swing state now."""
        ...

    def availability_reason(self) -> str:
        """Return a human-readable reason when :meth:`is_available` is False."""
        ...

    def get_swing_state(self, config: SwingStateConfig) -> SwingState:
        """Produce a ``SwingState`` whose ``engine_name`` == ``provider_id``."""
        ...


class _BaseSwingStateProvider(ABC):
    """Template base enforcing the provider contract (DRY + DbC).

    Subclasses implement :meth:`_build_swing_state`; the public
    :meth:`get_swing_state` wraps it with the precondition (availability,
    valid config) and postcondition (honest ``engine_name``) checks.
    """

    provider_id: str = "unknown"

    def is_available(self) -> bool:
        """Return True if this provider can produce a swing state now."""
        return True

    def availability_reason(self) -> str:
        """Return why the provider is unavailable ('' when available)."""
        return ""

    def get_swing_state(self, config: SwingStateConfig) -> SwingState:
        """Produce a ``SwingState`` from ``config`` (contract-enforced).

        Preconditions:
            - provider is available;
            - config values are finite and physically valid.

        Postcondition:
            ``result.engine_name == self.provider_id`` — a provider may not
            stamp its output with another engine's name (issue #8819).
        """
        require(
            self.is_available(),
            f"swing-state provider '{self.provider_id}' is not available: "
            f"{self.availability_reason() or 'unknown reason'}",
        )
        require(
            math.isfinite(config.clubhead_speed_ms) and config.clubhead_speed_ms > 0.0,
            "clubhead_speed_ms must be finite and > 0",
            config.clubhead_speed_ms,
        )
        require(
            math.isfinite(config.loft_deg) and 0.0 < config.loft_deg < 90.0,
            "loft_deg must be finite and in (0, 90) degrees",
            config.loft_deg,
        )
        require(
            math.isfinite(config.clubhead_mass_kg) and config.clubhead_mass_kg > 0.0,
            "clubhead_mass_kg must be finite and > 0",
            config.clubhead_mass_kg,
        )
        state = self._build_swing_state(config)
        ensure(
            state.engine_name == self.provider_id,
            f"provider '{self.provider_id}' produced a SwingState claiming "
            f"engine_name={state.engine_name!r} — false engine attribution",
            state.engine_name,
        )
        for name in (
            "clubhead_velocity",
            "clubhead_angular_velocity",
            "clubhead_orientation",
        ):
            vector = np.asarray(getattr(state, name), dtype=float)
            ensure(
                vector.shape == (3,) and bool(np.isfinite(vector).all()),
                f"provider '{self.provider_id}' produced a non-finite or "
                f"mis-shaped {name} (expected finite shape (3,))",
                vector,
            )
        return state

    @abstractmethod
    def _build_swing_state(self, config: SwingStateConfig) -> SwingState:
        """Build the swing state (subclass responsibility)."""


class ManualSwingStateProvider(_BaseSwingStateProvider):
    """Builds a ``SwingState`` directly from user-entered parameters.

    This is the current (and only implemented) behavior of the
    Swing→Flight Pipeline GUI: a straight-line clubhead velocity along +x
    with the configured loft, mass, and zero angular velocity.
    """

    provider_id = "manual"

    def _build_swing_state(self, config: SwingStateConfig) -> SwingState:
        # Lazy import keeps this module importable without the full physics
        # stack (the GUI imports us at construction time).
        from src.shared.python.physics.swing_ball_flight_pipeline import SwingState

        return SwingState(
            clubhead_velocity=np.array([config.clubhead_speed_ms, 0.0, 0.0]),
            clubhead_angular_velocity=np.zeros(3),
            clubhead_orientation=np.array([0.0, 0.0, 1.0]),
            clubhead_mass=config.clubhead_mass_kg,
            clubhead_loft_deg=config.loft_deg,
            engine_name=self.provider_id,
        )


class MuJoCoSwingStateProvider(_BaseSwingStateProvider):
    """Sources a ``SwingState`` from a real MuJoCo forward-dynamics swing.

    A narrow facade (LoD) over :mod:`mujoco_swing_source`: the in-repo
    upper-body golf-swing MJCF is driven by a scripted torque pulse under
    full ``mj_step`` dynamics, with the torque scale calibrated so peak
    clubhead speed approaches ``config.clubhead_speed_ms``.  Clubhead
    velocity, angular velocity, face orientation, mass, and MOI are read
    from the simulation/model — never fabricated.  The achieved speed and
    calibration residual are reported in ``SwingState.metadata``.

    Honest limitations (also recorded in metadata): the control is an
    open-loop script, not a biomechanical controller, and the loft in the
    returned state remains the user-configured value (the MJCF clubface
    geometry does not encode per-club loft).
    """

    provider_id = "mujoco"

    def is_available(self) -> bool:
        """True when the ``mujoco`` package and the MJCF asset are present."""
        return self.availability_reason() == ""

    def availability_reason(self) -> str:
        """Return '' when available, else an honest reason."""
        for module in ("mujoco", _MUJOCO_MODEL_ASSET_MODULE):
            try:
                if importlib.util.find_spec(module) is None:
                    return REASON_NOT_INSTALLED
            except (ImportError, ValueError):
                return REASON_NOT_INSTALLED
        return ""

    def _build_swing_state(self, config: SwingStateConfig) -> SwingState:
        from src.shared.python.physics.mujoco_swing_source import (
            run_reference_swing,
        )
        from src.shared.python.physics.swing_ball_flight_pipeline import SwingState

        kinematics, metadata = run_reference_swing(config.clubhead_speed_ms)
        metadata["loft_source"] = "user config (MJCF clubface encodes no loft)"
        return SwingState(
            clubhead_velocity=kinematics.velocity,
            clubhead_angular_velocity=kinematics.angular_velocity,
            clubhead_orientation=kinematics.face_normal,
            clubhead_mass=kinematics.mass,
            clubhead_loft_deg=config.loft_deg,
            clubhead_moi=float(np.max(kinematics.inertia_diagonal)),
            engine_name=self.provider_id,
            metadata=metadata,
        )


class UnimplementedEngineProvider(_BaseSwingStateProvider):
    """Placeholder for engine-backed swing sourcing that does not exist yet.

    Always unavailable.  ``availability_reason`` distinguishes "the engine's
    Python package is not installed" from "installed, but no swing-state
    extraction path has been implemented" so GUI tooltips stay honest.
    """

    def __init__(self, provider_id: str, module_name: str) -> None:
        require(bool(provider_id), "provider_id must be non-empty", provider_id)
        require(bool(module_name), "module_name must be non-empty", module_name)
        self.provider_id = provider_id
        self._module_name = module_name

    def is_available(self) -> bool:
        """Engine sourcing is not implemented — never available."""
        return False

    def availability_reason(self) -> str:
        """Return an honest reason for the disabled GUI entry."""
        try:
            installed = importlib.util.find_spec(self._module_name) is not None
        except (ImportError, ValueError):
            installed = False
        return REASON_NOT_IMPLEMENTED if installed else REASON_NOT_INSTALLED

    def _build_swing_state(self, config: SwingStateConfig) -> SwingState:
        raise NotImplementedError(  # tracked: #8975
            f"engine '{self.provider_id}' swing-state sourcing is not implemented"
        )


def available_swing_state_providers() -> list[SwingStateProvider]:
    """Return all known swing-state providers (available or not), in GUI order.

    The list always contains one provider per engine choice historically
    offered by the GUI (mujoco, drake, pinocchio, manual); callers must check
    :meth:`SwingStateProvider.is_available` before use.

    Postcondition: provider ids are unique and 'manual' is always present
    and available.
    """
    providers: list[SwingStateProvider] = [
        MuJoCoSwingStateProvider(),
        UnimplementedEngineProvider("drake", "pydrake"),
        UnimplementedEngineProvider("pinocchio", "pinocchio"),
        ManualSwingStateProvider(),
    ]
    ids = [p.provider_id for p in providers]
    ensure(len(ids) == len(set(ids)), "provider ids must be unique", ids)
    ensure("manual" in ids, "manual provider must always be registered", ids)
    return providers
