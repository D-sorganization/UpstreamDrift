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
    "jaxsim": "experimental",
    "simscape": "experimental",
}

ENGINE_TIER_ALIASES: dict[str, str] = {
    "myosim": "myosuite",
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
    normalized_name = ENGINE_TIER_ALIASES.get(normalized_name, normalized_name)
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


# ---------------------------------------------------------------------------
# 3-Tier Cross-Engine Mathematical Equivalency Contract
# ---------------------------------------------------------------------------

ALLOWED_EQUIVALENCY_TIERS = frozenset(
    {
        "tier_1_rigid_body",
        "tier_2_soft_contact",
        "tier_3_musculoskeletal",
    }
)

ENGINE_EQUIVALENCY_TIERS: dict[str, frozenset[str]] = {
    "mujoco": frozenset({"tier_1_rigid_body", "tier_2_soft_contact"}),
    "drake": frozenset({"tier_1_rigid_body"}),
    "pinocchio": frozenset({"tier_1_rigid_body"}),
    "putting_green": frozenset({"tier_2_soft_contact"}),
    "simscape": frozenset({"tier_1_rigid_body", "tier_2_soft_contact"}),
    "opensim": frozenset({"tier_1_rigid_body", "tier_3_musculoskeletal"}),
    "myosuite": frozenset({"tier_1_rigid_body", "tier_3_musculoskeletal"}),
    "jaxsim": frozenset({"tier_1_rigid_body"}),
}

PRIMARY_EQUIVALENCY_TIER: dict[str, str] = {
    "mujoco": "tier_1_rigid_body",
    "drake": "tier_1_rigid_body",
    "pinocchio": "tier_1_rigid_body",
    "putting_green": "tier_2_soft_contact",
    "simscape": "tier_2_soft_contact",
    "opensim": "tier_3_musculoskeletal",
    "myosuite": "tier_3_musculoskeletal",
    "jaxsim": "tier_1_rigid_body",
}


def get_engine_equivalency_tier(engine_name: str) -> str:
    """Return the primary mathematical equivalency tier for an engine.

    Precondition: ``engine_name`` must be a non-empty string naming a known engine.
    Postcondition: returned value is in ``ALLOWED_EQUIVALENCY_TIERS``.
    """
    if not isinstance(engine_name, str):
        raise TypeError(
            f"engine_name must be a string, got {type(engine_name).__name__}"
        )

    normalized_name = engine_name.strip().lower()
    normalized_name = ENGINE_TIER_ALIASES.get(normalized_name, normalized_name)
    if not normalized_name:
        raise ValueError("engine_name must not be empty")

    try:
        tier = PRIMARY_EQUIVALENCY_TIER[normalized_name]
    except KeyError as exc:
        raise ValueError(
            f"Unknown engine equivalency tier metadata for {engine_name!r}"
        ) from exc

    if tier not in ALLOWED_EQUIVALENCY_TIERS:
        raise ValueError(
            f"Invalid equivalency tier {tier!r} for engine {engine_name!r}"
        )

    return tier


def are_engines_tier_compatible(
    engine_a: str,
    engine_b: str,
    required_tier: str | None = None,
) -> bool:
    """Check mathematical equivalency tier compatibility between two engines.

    If ``required_tier`` is specified, returns True iff both engines participate
    in that specific equivalency tier.
    If ``required_tier`` is None, returns True iff the two engines share at least
    one common equivalency tier.

    Preconditions:
    - ``engine_a`` and ``engine_b`` must be strings naming known engines.
    - If ``required_tier`` is given, it must be a string in ``ALLOWED_EQUIVALENCY_TIERS``.
    """
    if not isinstance(engine_a, str):
        raise TypeError(f"engine_a must be a string, got {type(engine_a).__name__}")
    if not isinstance(engine_b, str):
        raise TypeError(f"engine_b must be a string, got {type(engine_b).__name__}")

    norm_a = engine_a.strip().lower()
    norm_a = ENGINE_TIER_ALIASES.get(norm_a, norm_a)
    if not norm_a:
        raise ValueError("engine_a must not be empty")
    if norm_a not in ENGINE_EQUIVALENCY_TIERS:
        raise ValueError(f"Unknown engine: {engine_a!r}")

    norm_b = engine_b.strip().lower()
    norm_b = ENGINE_TIER_ALIASES.get(norm_b, norm_b)
    if not norm_b:
        raise ValueError("engine_b must not be empty")
    if norm_b not in ENGINE_EQUIVALENCY_TIERS:
        raise ValueError(f"Unknown engine: {engine_b!r}")

    if required_tier is not None:
        if not isinstance(required_tier, str):
            raise TypeError(
                f"required_tier must be a string or None, got {type(required_tier).__name__}"
            )
        if required_tier not in ALLOWED_EQUIVALENCY_TIERS:
            raise ValueError(
                f"Unknown equivalency tier: {required_tier!r}; "
                f"allowed tiers: {sorted(ALLOWED_EQUIVALENCY_TIERS)}"
            )
        return (
            required_tier in ENGINE_EQUIVALENCY_TIERS[norm_a]
            and required_tier in ENGINE_EQUIVALENCY_TIERS[norm_b]
        )

    return bool(ENGINE_EQUIVALENCY_TIERS[norm_a] & ENGINE_EQUIVALENCY_TIERS[norm_b])


# Public name for the primary equivalency tier lookup (single implementation).
get_equivalency_tier = get_engine_equivalency_tier

_RIGID_TIER = "tier_1_rigid_body"
_MUSCULOSKELETAL_TIER = "tier_3_musculoskeletal"


def assert_valid_parity_pair(
    engine_a: str,
    engine_b: str,
    *,
    allow_stripped_rigid_proxy: bool = False,
) -> None:
    """Fail closed on cross-tier parity comparisons (#11709 GCV-3).

    Precondition: ``engine_a``/``engine_b`` name known engines; the flag is a bool.
    Postcondition: returns ``None`` only if the pair is same-tier, or Tier 1 vs
    Tier 3 with an explicit ``allow_stripped_rigid_proxy=True`` declaration.

    Raises:
        TypeError: an argument has the wrong type.
        ValueError: an unknown engine, or an undeclared Tier 1 vs Tier 3 pair.
    """
    if not isinstance(allow_stripped_rigid_proxy, bool):
        raise TypeError(
            "allow_stripped_rigid_proxy must be a bool, "
            f"got {type(allow_stripped_rigid_proxy).__name__}"
        )
    tier_a = get_equivalency_tier(engine_a)
    tier_b = get_equivalency_tier(engine_b)
    cross_tier = {tier_a, tier_b} == {_RIGID_TIER, _MUSCULOSKELETAL_TIER}
    if cross_tier and not allow_stripped_rigid_proxy:
        raise ValueError(
            f"Parity comparison {engine_a!r} ({tier_a}) vs {engine_b!r} ({tier_b}) "
            "crosses Tier 1 rigid body and Tier 3 musculoskeletal; pass "
            "allow_stripped_rigid_proxy=True to declare a stripped rigid proxy"
        )
