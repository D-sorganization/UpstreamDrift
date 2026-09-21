"""Engine tier policy metadata and warning helpers."""

from __future__ import annotations

import warnings

ALLOWED_TIERS = frozenset({"core", "extended", "experimental", "archived"})

ENGINE_TIERS: dict[str, str] = {
    "mujoco": "core",
    "drake": "extended",
    "pinocchio": "extended",
    "opensim": "experimental",
    "myosuite": "experimental",
    "putting_green": "core",
}


class ExperimentalTierWarning(UserWarning):
    """Warning emitted when constructing an experimental engine adapter."""


def get_engine_tier(engine_name: str) -> str:
    """Return the declared engine tier.

    Postcondition: returned value is one of ``ALLOWED_TIERS``.
    """
    if not isinstance(engine_name, str):
        raise TypeError("engine_name must be a string")

    normalized_name = engine_name.strip().lower()
    if not normalized_name:
        raise ValueError("engine_name must not be empty")

    try:
        tier = ENGINE_TIERS[normalized_name]
    except KeyError as exc:
        raise ValueError(f"Unknown engine tier metadata for {engine_name!r}") from exc

    if tier not in ALLOWED_TIERS:
        raise ValueError(f"Invalid tier {tier!r} for engine {engine_name!r}")

    return tier


def warn_if_experimental(engine_name: str, display_name: str) -> None:
    """Emit ``ExperimentalTierWarning`` for experimental engines.

    Postcondition: no warning is emitted for non-experimental tiers.
    """
    if not isinstance(display_name, str):
        raise TypeError("display_name must be a string")

    tier = get_engine_tier(engine_name)
    if tier != "experimental":
        return

    warnings.warn(
        f"{display_name} engine is in the EXPERIMENTAL tier. "
        "API may change without notice; not covered by the core support SLA. "
        "See docs/operations/tier-policy.md for tier policy.",
        ExperimentalTierWarning,
        stacklevel=3,
    )
