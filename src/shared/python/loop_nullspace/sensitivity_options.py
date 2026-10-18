"""Declared assumptions of the sensitivity stage (NSX-99, #12728).

The four ``AnalysisOptions`` fields -- ``disturbance_spec``, ``mechanisms``,
``strategy_set`` and ``outcome_priority`` -- are validated and normalised here by
:class:`SensitivitySpec`, and :meth:`SensitivitySpec.as_dict` is the canonical
JSON payload that enters the result-cache key.  Every number that changes what
the stage computes (a covariance entry, a signal gain, a bound) is in that
payload, so a changed assumption can never be served from a stale cache entry.

Declared choices (the specification fixes none):

* ``mechanisms`` must agree with the disturbances: the signal-dependent noise
  mechanism (M1) is declared exactly when some disturbance carries a signal law.
  Silently adding or dropping M1 would change the evidence typing of every
  variance, so the disagreement is refused (``ValueError``).
* ``outcome_priority`` is an *order* (first is the primary outcome, default
  ``face_angle``), not a set of weights; nothing is blended across outcomes.
* ``strategy_set`` defaults to the whole catalogue: a strategy that is not
  implemented is then reported with its catalogue reason and no numbers.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.strategy_spec import StrategyId
from src.shared.python.loop_nullspace.variance_records import (
    Mechanism,
    declared_mechanisms,
)
from src.shared.python.perturbation.disturbance import (
    DisturbanceSpec,
    InjectionStage,
    Spectrum,
)

__all__ = [
    "DEFAULT_OUTCOME_PRIORITY",
    "FACE_OUTCOME",
    "SensitivitySpec",
    "disturbance_dict",
    "disturbance_from_dict",
]

FACE_OUTCOME = "face_angle"
DEFAULT_OUTCOME_PRIORITY: tuple[str, ...] = (FACE_OUTCOME,)
_ALL_STRATEGIES: tuple[StrategyId, ...] = tuple(StrategyId)


def disturbance_dict(spec: DisturbanceSpec) -> dict[str, Any]:
    """Every declared field of ``spec`` as JSON (covariance included).

    Postcondition: :func:`disturbance_from_dict` rebuilds an equal spec.
    """
    if not isinstance(spec, DisturbanceSpec):
        raise TypeError("spec must be a DisturbanceSpec")
    return {
        "name": spec.name,
        "stage": spec.stage.value,
        "channels": list(spec.channels),
        "units": spec.units,
        "covariance": np.asarray(spec.covariance, dtype=float).tolist(),
        "spectrum": None if spec.spectrum is None else spec.spectrum.value,
        "persistent": bool(spec.persistent),
        "bandwidth_hz": spec.bandwidth_hz,
        "band_hz": None if spec.band_hz is None else list(spec.band_hz),
        "signal_gain": None if spec.signal_gain is None else list(spec.signal_gain),
        "signal_bounds": (
            None
            if spec.signal_bounds is None
            else [list(b) for b in spec.signal_bounds]
        ),
    }


def disturbance_from_dict(data: Mapping[str, Any]) -> DisturbanceSpec:
    """Rebuild a :class:`DisturbanceSpec` from :func:`disturbance_dict` output.

    Raises:
        TypeError: ``data`` is not a mapping.
        ValueError: A key is missing, an enum value is unknown, or the spec's own
            preconditions fail.
    """
    if not isinstance(data, Mapping):
        raise TypeError("a disturbance must be a mapping")
    try:
        spectrum = data["spectrum"]
        band = data["band_hz"]
        gain = data["signal_gain"]
        bounds = data["signal_bounds"]
        return DisturbanceSpec(
            name=data["name"],
            stage=InjectionStage(data["stage"]),
            channels=tuple(data["channels"]),
            units=data["units"],
            covariance=np.asarray(data["covariance"], dtype=float),
            spectrum=None if spectrum is None else Spectrum(spectrum),
            persistent=bool(data["persistent"]),
            bandwidth_hz=data["bandwidth_hz"],
            band_hz=None if band is None else (float(band[0]), float(band[1])),
            signal_gain=None if gain is None else tuple(float(g) for g in gain),
            signal_bounds=(
                None
                if bounds is None
                else tuple((float(b[0]), float(b[1])) for b in bounds)
            ),
        )
    except KeyError as exc:
        raise ValueError(f"disturbance lacks {exc.args[0]!r}") from exc


def _disturbances(value: object) -> tuple[DisturbanceSpec, ...]:
    if isinstance(value, DisturbanceSpec):
        return (value,)
    if isinstance(value, str) or not isinstance(value, Sequence):
        raise TypeError("disturbance_spec must be a DisturbanceSpec or a sequence")
    specs = tuple(value)
    if len(specs) == 0:
        raise ValueError("disturbance_spec needs at least one DisturbanceSpec")
    if not all(isinstance(s, DisturbanceSpec) for s in specs):
        raise TypeError("every disturbance must be a DisturbanceSpec")
    names = [s.name for s in specs]
    if len(set(names)) != len(names):
        raise ValueError(f"disturbance names must be unique, got {names}")
    return specs


def _mechanism(item: object) -> Mechanism:
    if isinstance(item, Mechanism):
        return item
    try:
        return Mechanism(item)
    except ValueError:
        known = [m.value for m in Mechanism]
        raise ValueError(f"unknown mechanism {item!r}; known: {known}") from None


def _mechanisms(value: object) -> frozenset[Mechanism]:
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise TypeError("mechanisms must be an iterable of mechanism ids")
    return frozenset(_mechanism(m) for m in value)


def _strategies(value: object) -> tuple[StrategyId, ...]:
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise TypeError("strategy_set must be an iterable of strategy ids")
    out: list[StrategyId] = []
    for item in value:
        try:
            out.append(item if isinstance(item, StrategyId) else StrategyId(item))
        except ValueError:
            known = [s.value for s in StrategyId]
            raise ValueError(f"unknown strategy {item!r}; known: {known}") from None
    if len(set(out)) != len(out):
        raise ValueError("strategy_set must not repeat a strategy")
    return tuple(out)


def _priority(value: object) -> tuple[str, ...]:
    if isinstance(value, str) or not isinstance(value, Iterable):
        raise TypeError("outcome_priority must be a sequence of outcome names")
    names = tuple(value)
    if len(names) == 0:
        raise ValueError("outcome_priority needs at least one outcome")
    if not all(isinstance(n, str) and n.strip() for n in names):
        raise ValueError("outcome_priority names must be non-empty strings")
    if len(set(names)) != len(names):
        raise ValueError("outcome_priority must not repeat an outcome")
    return names


@dataclass(frozen=True)
class SensitivitySpec:
    """The validated, normalised sensitivity declaration.

    Raises:
        TypeError: A field of the wrong kind.
        ValueError: No disturbance, a repeated disturbance / strategy / outcome,
            an unknown mechanism or strategy id, or ``mechanisms`` that disagree
            with the disturbances' signal laws (module docstring).
    """

    disturbances: tuple[DisturbanceSpec, ...]
    mechanisms: frozenset[Mechanism]
    strategies: tuple[StrategyId, ...] = _ALL_STRATEGIES
    outcome_priority: tuple[str, ...] = DEFAULT_OUTCOME_PRIORITY

    def __post_init__(self) -> None:
        specs = _disturbances(self.disturbances)
        mech = _mechanisms(self.mechanisms)
        object.__setattr__(self, "disturbances", specs)
        object.__setattr__(self, "mechanisms", mech)
        object.__setattr__(self, "strategies", _strategies(self.strategies))
        object.__setattr__(self, "outcome_priority", _priority(self.outcome_priority))
        derived = declared_mechanisms(specs, geometric_preload=False)
        has_m1 = Mechanism.SIGNAL_DEPENDENT_NOISE in mech
        if has_m1 != (Mechanism.SIGNAL_DEPENDENT_NOISE in derived):
            raise ValueError(
                "mechanisms and disturbances disagree: "
                f"{Mechanism.SIGNAL_DEPENDENT_NOISE.value} must be declared exactly "
                "when a disturbance carries a signal-dependent law"
            )

    @property
    def primary_outcome(self) -> str:
        """The first (highest-priority) outcome; ``face_angle`` by default."""
        return self.outcome_priority[0]

    def as_dict(self) -> dict[str, Any]:
        """Canonical JSON payload (also the cache-key payload)."""
        return {
            "disturbance_spec": [disturbance_dict(s) for s in self.disturbances],
            "mechanisms": sorted(m.value for m in self.mechanisms),
            "strategy_set": [s.value for s in self.strategies],
            "outcome_priority": list(self.outcome_priority),
        }
