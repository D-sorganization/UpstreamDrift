"""Unit tests for the 3-Tier Engine Mathematical Equivalency Contract (#11709, #11605).

Verifies:
1. Primary equivalency tier assignment for all registered engines and aliases.
2. Cross-engine equivalency tier compatibility queries and guardrails.
3. SimscapeAdapter.compute_contact_forces() contract: raises SimscapeChannelUnavailableError
   when loaded, ensuring unavailable contact channels never fabricate a zero vector.
"""

from __future__ import annotations

import pytest

from src.engines.simscape._errors import (
    SimscapeChannelUnavailableError,
    SimscapeStateError,
)
from src.engines.simscape._lifecycle import AdapterState
from src.engines.simscape.adapter import SimscapeAdapter
from src.engines.tiers import (
    ALLOWED_EQUIVALENCY_TIERS,
    ALLOWED_TIERS,
    ENGINE_EQUIVALENCY_TIERS,
    ENGINE_TIERS,
    PRIMARY_EQUIVALENCY_TIER,
    are_engines_tier_compatible,
    get_engine_equivalency_tier,
    get_engine_tier,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. Primary Equivalency Tier Verification
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("engine_name", "expected_tier"),
    [
        ("mujoco", "tier_1_rigid_body"),
        ("drake", "tier_1_rigid_body"),
        ("pinocchio", "tier_1_rigid_body"),
        ("putting_green", "tier_2_soft_contact"),
        ("simscape", "tier_2_soft_contact"),
        ("opensim", "tier_3_musculoskeletal"),
        ("myosuite", "tier_3_musculoskeletal"),
        ("jaxsim", "tier_1_rigid_body"),
        ("myosim", "tier_3_musculoskeletal"),  # Alias for myosuite
    ],
)
def test_get_engine_equivalency_tier_registered_engines(
    engine_name: str,
    expected_tier: str,
) -> None:
    """Every registered engine resolves to its canonical primary mathematical tier."""
    assert get_engine_equivalency_tier(engine_name) == expected_tier


def test_get_engine_equivalency_tier_postcondition() -> None:
    """All resolved primary tiers are members of ALLOWED_EQUIVALENCY_TIERS."""
    for engine in PRIMARY_EQUIVALENCY_TIER:
        tier = get_engine_equivalency_tier(engine)
        assert tier in ALLOWED_EQUIVALENCY_TIERS


@pytest.mark.parametrize(
    "invalid_input",
    [123, None, ["mujoco"], {"name": "pinocchio"}],
)
def test_get_engine_equivalency_tier_type_error(invalid_input: object) -> None:
    """Non-string engine names raise TypeError."""
    with pytest.raises(TypeError, match="engine_name must be a string"):
        get_engine_equivalency_tier(invalid_input)  # type: ignore[arg-type]


@pytest.mark.parametrize("empty_input", ["", "   ", "\t\n"])
def test_get_engine_equivalency_tier_empty_value_error(empty_input: str) -> None:
    """Empty or whitespace engine names raise ValueError."""
    with pytest.raises(ValueError, match="engine_name must not be empty"):
        get_engine_equivalency_tier(empty_input)


def test_get_engine_equivalency_tier_unknown_engine() -> None:
    """Unregistered engine names raise ValueError."""
    with pytest.raises(ValueError, match="Unknown engine equivalency tier metadata"):
        get_engine_equivalency_tier("non_existent_engine")


# ---------------------------------------------------------------------------
# 2. Tier Compatibility Contract Verification
# ---------------------------------------------------------------------------


def test_are_engines_tier_compatible_valid_pairs() -> None:
    """Rigid-body pairs are compatible in tier_1_rigid_body."""
    assert (
        are_engines_tier_compatible("mujoco", "pinocchio", "tier_1_rigid_body") is True
    )
    assert are_engines_tier_compatible("drake", "jaxsim", "tier_1_rigid_body") is True
    assert (
        are_engines_tier_compatible("mujoco", "simscape", "tier_1_rigid_body") is True
    )


def test_are_engines_tier_compatible_rejects_disjoint_tiers() -> None:
    """Rigid-body engines cannot claim musculoskeletal parity, preventing invalid claims."""
    assert (
        are_engines_tier_compatible("pinocchio", "opensim", "tier_3_musculoskeletal")
        is False
    )
    assert (
        are_engines_tier_compatible("drake", "myosuite", "tier_3_musculoskeletal")
        is False
    )
    assert (
        are_engines_tier_compatible("putting_green", "pinocchio", "tier_1_rigid_body")
        is False
    )


def test_are_engines_tier_compatible_musculoskeletal_pair() -> None:
    """Musculoskeletal engines are mutually compatible in tier 3."""
    assert (
        are_engines_tier_compatible("opensim", "myosuite", "tier_3_musculoskeletal")
        is True
    )
    assert (
        are_engines_tier_compatible("opensim", "myosim", "tier_3_musculoskeletal")
        is True
    )


def test_are_engines_tier_compatible_without_required_tier() -> None:
    """Without an explicit required_tier, returns True if any common tier exists."""
    assert (
        are_engines_tier_compatible("mujoco", "putting_green") is True
    )  # Share tier 2
    assert (
        are_engines_tier_compatible("pinocchio", "putting_green") is False
    )  # Disjoint
    assert (
        are_engines_tier_compatible("simscape", "putting_green") is True
    )  # Share tier 2


def test_are_engines_tier_compatible_validation() -> None:
    """DbC validation enforces string types, known engines, and valid tiers."""
    with pytest.raises(TypeError, match="engine_a must be a string"):
        are_engines_tier_compatible(123, "mujoco")  # type: ignore[arg-type]

    with pytest.raises(TypeError, match="engine_b must be a string"):
        are_engines_tier_compatible("mujoco", 123)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="engine_a must not be empty"):
        are_engines_tier_compatible("", "mujoco")

    with pytest.raises(ValueError, match="engine_b must not be empty"):
        are_engines_tier_compatible("mujoco", "")

    with pytest.raises(ValueError, match="Unknown engine: 'ghost_engine'"):
        are_engines_tier_compatible("ghost_engine", "mujoco")

    with pytest.raises(ValueError, match="Unknown engine: 'ghost_engine'"):
        are_engines_tier_compatible("mujoco", "ghost_engine")

    with pytest.raises(TypeError, match="required_tier must be a string or None"):
        are_engines_tier_compatible("mujoco", "pinocchio", required_tier=42)  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="Unknown equivalency tier"):
        are_engines_tier_compatible("mujoco", "pinocchio", required_tier="invalid_tier")


# ---------------------------------------------------------------------------
# 3. SimscapeAdapter Contact Forces Contract (GCV-3 / #11709 Step 1)
# ---------------------------------------------------------------------------


def test_simscape_compute_contact_forces_raises_unavailable_when_loaded() -> None:
    """SimscapeAdapter.compute_contact_forces() raises SimscapeChannelUnavailableError when loaded.

    Verifies the contract of #11709 Step 1 that unavailable contact channels never
    fabricate a zero ground-reaction force vector.
    """
    adapter = SimscapeAdapter()
    # Transition to LOADED state to test loaded behavior without external MATLAB dependencies
    adapter._lifecycle.transition(AdapterState.LOADED, operation="test_init")
    assert adapter.model_loaded is True

    with pytest.raises(SimscapeChannelUnavailableError) as exc_info:
        adapter.compute_contact_forces()

    assert "contact_forces" in str(exc_info.value)
    assert "#11709" in str(exc_info.value)


def test_simscape_compute_contact_forces_raises_state_error_when_unloaded() -> None:
    """Unloaded SimscapeAdapter raises SimscapeStateError prior to model load."""
    adapter = SimscapeAdapter()
    assert adapter.model_loaded is False

    with pytest.raises(SimscapeStateError):
        adapter.compute_contact_forces()


# ---------------------------------------------------------------------------
# 4. Backward Compatibility Verification
# ---------------------------------------------------------------------------


def test_backward_compatibility_engine_tiers() -> None:
    """Existing engine tiers and allowed tiers contracts are fully preserved."""
    assert set(ALLOWED_TIERS) == {"core", "extended", "experimental", "archived"}
    for eng in ("mujoco", "drake", "pinocchio", "putting_green", "jaxsim"):
        assert get_engine_tier(eng) == ENGINE_TIERS[eng]

    # Simscape reports experimental tier
    assert get_engine_tier("simscape") == "experimental"
