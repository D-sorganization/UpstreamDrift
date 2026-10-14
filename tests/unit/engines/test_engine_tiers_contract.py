"""Contract tests for engine tier registry and cross-tier parity guard (#11709).

Verifies:
1. ``simscape`` is registered in ``ENGINE_TIERS`` (no fallback special case).
2. ``get_equivalency_tier`` returns the primary equivalency tier per engine.
3. ``assert_valid_parity_pair`` fails closed for Tier 1 vs Tier 3 comparisons
   unless an explicit stripped-rigid-proxy declaration is passed.
4. ``SimscapeAdapter.compute_contact_forces`` never fabricates zeros.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.simscape._errors import SimscapeChannelUnavailableError
from src.engines.simscape._lifecycle import AdapterState
from src.engines.simscape.adapter import SimscapeAdapter
from src.engines.tiers import (
    ALLOWED_EQUIVALENCY_TIERS,
    ENGINE_TIERS,
    assert_valid_parity_pair,
    get_engine_tier,
    get_equivalency_tier,
)

pytestmark = pytest.mark.unit


def test_simscape_registered_in_engine_tiers() -> None:
    assert ENGINE_TIERS["simscape"] == "experimental"
    assert get_engine_tier("simscape") == "experimental"


def test_existing_engine_tiers_preserved() -> None:
    assert ENGINE_TIERS["mujoco"] == "core"
    assert ENGINE_TIERS["drake"] == "extended"


@pytest.mark.parametrize(
    ("engine", "expected"),
    [
        ("mujoco", "tier_1_rigid_body"),
        ("opensim", "tier_3_musculoskeletal"),
        ("myosim", "tier_3_musculoskeletal"),
        ("putting_green", "tier_2_soft_contact"),
    ],
)
def test_get_equivalency_tier_known_engines(engine: str, expected: str) -> None:
    tier = get_equivalency_tier(engine)
    assert tier == expected
    assert tier in ALLOWED_EQUIVALENCY_TIERS


def test_get_equivalency_tier_rejects_unknown_engine() -> None:
    with pytest.raises(ValueError):
        get_equivalency_tier("ghost_engine")


def test_get_equivalency_tier_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        get_equivalency_tier(42)  # type: ignore[arg-type]


def test_parity_tier1_vs_tier3_fails_closed() -> None:
    with pytest.raises(ValueError, match="stripped_rigid_proxy"):
        assert_valid_parity_pair("mujoco", "opensim")


def test_parity_tier1_vs_tier3_allowed_with_explicit_proxy() -> None:
    assert_valid_parity_pair("mujoco", "opensim", allow_stripped_rigid_proxy=True)


def test_parity_same_tier_passes() -> None:
    assert_valid_parity_pair("mujoco", "pinocchio")
    assert_valid_parity_pair("opensim", "myosuite")


def test_parity_rejects_non_string_inputs() -> None:
    with pytest.raises(TypeError):
        assert_valid_parity_pair(1, "mujoco")  # type: ignore[arg-type]


def test_adapter_contact_forces_never_zero_vector() -> None:
    adapter = SimscapeAdapter()
    adapter._lifecycle.transition(AdapterState.LOADED, operation="test_init")
    with pytest.raises(SimscapeChannelUnavailableError):
        result = adapter.compute_contact_forces()
        # Reaching here would mean a fabricated value was returned.
        assert result is None or not np.all(result == 0)
