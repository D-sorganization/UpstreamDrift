"""Builds the sensitivity request options from the user's choices (NSX-100, #12729).

Pure data, no PyQt.  Every id a choice names (stage, spectrum, mechanism,
strategy, outcome) must be in the backend catalogue
(:func:`~src.shared.python.loop_nullspace.sensitivity_options.option_catalogue`);
nothing is invented here.  The output is the JSON of the API's
``SensitivityOptions``; the service validates it and keys its cache on it.  The
channel amplitude is the diagonal covariance entry the user types, placed as
typed -- no quantity is derived.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DisturbanceChoice",
    "VariabilityChoices",
    "parse_channels",
    "sensitivity_options",
]


def parse_channels(text: str) -> tuple[tuple[str, float], ...]:
    """``"shoulder=0.004, elbow=0.009"`` -> ``(("shoulder", 0.004), ...)``.

    Each value is the channel's declared covariance (variance) entry.

    Raises:
        ValueError: Empty text, a missing ``=``, a repeated channel, or a value
            that is not a finite non-negative number.
    """
    pairs: list[tuple[str, float]] = []
    for part in (p.strip() for p in text.split(",")):
        name, sep, raw = part.partition("=")
        name = name.strip()
        if not sep or not name:
            raise ValueError(f"expected name=variance, got {part!r}")
        try:
            value = float(raw)
        except ValueError:
            raise ValueError(f"variance of {name!r} is not a number: {raw!r}") from None
        if not math.isfinite(value) or value < 0.0:
            raise ValueError(f"variance of {name!r} must be finite and non-negative")
        pairs.append((name, value))
    if not pairs or len({n for n, _ in pairs}) != len(pairs):
        raise ValueError("name at least one channel, each once")
    return tuple(pairs)


@dataclass(frozen=True)
class DisturbanceChoice:
    """One disturbance as the user declared it."""

    name: str
    stage: str
    spectrum: str
    persistent: bool
    channels: tuple[tuple[str, float], ...]
    units: str
    bandwidth_hz: float | None = None

    def __post_init__(self) -> None:
        if not self.name.strip() or not self.units.strip():
            raise ValueError("a disturbance needs a name and units")
        if not self.channels:
            raise ValueError("a disturbance needs at least one channel")


@dataclass(frozen=True)
class VariabilityChoices:
    """Everything the controls hold; ``strategies=None`` means the whole catalogue."""

    disturbances: tuple[DisturbanceChoice, ...]
    mechanisms: tuple[str, ...]
    strategies: tuple[str, ...] | None
    outcome_priority: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.disturbances:
            raise ValueError("declare at least one disturbance")


def _ids(catalogue: Mapping[str, Any], key: str) -> set[str]:
    return {str(entry["id"]) for entry in catalogue[key]}


def _check(values: Sequence[str], allowed: set[str], what: str) -> None:
    unknown = [v for v in values if v not in allowed]
    if unknown:
        raise ValueError(f"{what} {unknown} not in the backend catalogue")


def _disturbance(choice: DisturbanceChoice, cat: Mapping[str, Any]) -> dict[str, Any]:
    _check([choice.stage], _ids(cat, "stages"), "stage")
    _check([choice.spectrum], _ids(cat, "spectra"), "spectrum")
    n = len(choice.channels)
    cov = [
        [v if i == j else 0.0 for j in range(n)]
        for i, (_, v) in enumerate(choice.channels)
    ]
    return {
        "name": choice.name.strip(),
        "stage": choice.stage,
        "channels": [name for name, _ in choice.channels],
        "units": choice.units.strip(),
        "covariance": cov,
        "spectrum": None if choice.persistent else choice.spectrum,
        "persistent": choice.persistent,
        "bandwidth_hz": choice.bandwidth_hz,
        "band_hz": None,
        "signal_gain": None,
        "signal_bounds": None,
    }


def sensitivity_options(
    choices: VariabilityChoices, catalogue: Mapping[str, Any]
) -> dict[str, Any]:
    """The ``SensitivityOptions`` JSON for ``choices``.

    Raises:
        ValueError: A chosen id is not in ``catalogue``, or the outcome priority
            is empty.
    """
    _check(choices.mechanisms, _ids(catalogue, "mechanisms"), "mechanism")
    if choices.strategies is not None:
        _check(choices.strategies, _ids(catalogue, "strategies"), "strategy")
    if not choices.outcome_priority:
        raise ValueError("name at least one outcome")
    _check(choices.outcome_priority, _ids(catalogue, "outcomes"), "outcome")
    return {
        "disturbance_spec": [_disturbance(d, catalogue) for d in choices.disturbances],
        "mechanisms": list(choices.mechanisms),
        "strategy_set": None
        if choices.strategies is None
        else list(choices.strategies),
        "outcome_priority": list(choices.outcome_priority),
    }
