"""Integration tests for installed provider authority and runtime import provenance.

Validates the explicit contracts of UpstreamDrift issue #10529 (ORG-20 / #9406):
- One declared owner per shared behavior;
- Resolved source identity matches pinned authority;
- Missing or mismatched provider yields an explicit fail-closed error / blocked state;
- No silent local fork fallback;
- Compatibility imports share implementation with the canonical provider.
"""

from __future__ import annotations

import os
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

from src.launchers.workspace_navigation import ALIAS_MAP, PRIMARY_WORKSPACES
from src.shared.python.config.tools_vendor_authority import (
    ProviderUnavailableError,
    ToolsVendorAuthority,
    assert_runtime_provenance_parity,
    clear_tools_vendor_authority_cache,
    inspect_provider_authority,
    inspect_tools_vendor_authority,
    verify_provider_provenance,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.integration


# =============================================================================
# RED Acceptance Cases
# =============================================================================


def test_pytest_and_packaged_app_resolving_different_roots_fail_provenance() -> None:
    """RED case 1: pytest and packaged app resolving different implementation roots fail provenance.

    When one execution context (e.g. pytest in repo checkout) resolves one root
    while a packaged app resolves a divergent root, the provenance contract must
    fail closed with ProviderUnavailableError naming the divergent roots.
    """
    repo_vendor_root = (REPO_ROOT / "vendor" / "ud-tools").resolve()
    divergent_packaged_root = Path(
        "/opt/packaged_app/site-packages/forked_tools"
    ).resolve()

    with pytest.raises(
        ProviderUnavailableError, match="(?i)provenance|divergent|mismatch"
    ):
        assert_runtime_provenance_parity(repo_vendor_root, divergent_packaged_root)

    # Also test verify_provider_provenance rejecting a candidate outside canonical root
    with pytest.raises(ProviderUnavailableError, match="(?i)provenance|mismatch"):
        verify_provider_provenance(
            repo_vendor_root, divergent_packaged_root / "contracts.py"
        )


def test_wrong_pin_produces_blocked_state(tmp_path: Path) -> None:
    """RED case 2a: wrong pin produces correct blocked state without silent fallback.

    When the checkout HEAD or wheel metadata disagrees with the superproject pin,
    authority must report available=False with a descriptive 'pin stale' message,
    and must not silently fall back to an unpinned sibling or local copy.
    """
    fake_repo = tmp_path / "UpstreamDrift"
    fake_vendor = fake_repo / "vendor" / "ud-tools"
    (fake_vendor / "src").mkdir(parents=True)

    expected_sha = "11" * 20
    actual_sha = "22" * 20

    authority = inspect_provider_authority(
        repo_root=fake_repo,
        expected_sha=expected_sha,
        actual_sha=actual_sha,
    )

    assert authority.available is False
    assert authority.reason is not None
    assert (
        "pin stale" in authority.reason.lower()
        or "expected" in authority.reason.lower()
    )


def test_missing_wheel_and_vendor_produces_blocked_state(tmp_path: Path) -> None:
    """RED case 2b: missing wheel / vendor checkout produces correct blocked state.

    When neither the pinned vendor submodule nor an installed wheel distribution
    is present, provider authority must fail closed and refuse unpinned fallbacks.
    """
    empty_repo = tmp_path / "EmptyRepo"
    empty_repo.mkdir()

    authority = inspect_provider_authority(
        repo_root=empty_repo,
        distribution_name="nonexistent-ud-tools-wheel",
    )

    assert authority.available is False
    assert authority.reason is not None
    assert (
        "missing" in authority.reason.lower()
        or "unavailable" in authority.reason.lower()
    )


def test_provider_import_failure_produces_blocked_state() -> None:
    """RED case 2c: provider import failure produces correct blocked state.

    If the underlying provider module fails to import (corrupt wheel, missing
    runtime dependencies), authority must fail closed with explicit reason
    rather than crashing unhandled or substituting an unpinned local fork.
    """
    with patch(
        "importlib.import_module",
        side_effect=ImportError("Corrupted provider wheel C-extension failed to load"),
    ):
        authority = inspect_provider_authority(
            repo_root=REPO_ROOT,
            module_probe="corrupted.provider.module",
        )
        assert authority.available is False
        assert authority.reason is not None
        assert "import" in authority.reason.lower()


# =============================================================================
# GREEN Acceptance Cases
# =============================================================================


def test_sidekick_public_seam_runs_through_intended_authority() -> None:
    """GREEN case 3a: Sidekick public seam runs through intended Tools authority."""
    from src.tools.sidekick._embed_adapter import _SidekickEmbedAdapter

    adapter = _SidekickEmbedAdapter()
    assert adapter.tool_id == "sidekick"
    caps = adapter.embed_capabilities()
    assert caps.supports_embedded is True
    assert caps.prefers_dock is True


def test_movement_optimizer_public_seam_runs_through_intended_authority() -> None:
    """GREEN case 3b: Movement Optimizer public seam delegates to tools_movement_optimizer.

    Asserts that the legacy 'movement_optimizer' identifier delegates to canonical
    'tools_movement_optimizer' in ALIAS_MAP, and is registered under 'optimize_train'.
    """
    assert ALIAS_MAP.get("movement_optimizer") == "tools_movement_optimizer"
    optimize_train = PRIMARY_WORKSPACES["optimize_train"]
    assert "tools_movement_optimizer" in optimize_train.member_tool_ids


def test_pendulum_public_seam_runs_through_intended_authority() -> None:
    """GREEN case 3c: Pendulum public seam runs through intended Tools authority."""
    from src.launchers.adapters.swing_objective_lab_embed import (
        TOOL_ID,
        _SwingObjectiveLabEmbedAdapter,
    )

    assert TOOL_ID == "swing_objective_lab"
    adapter = _SwingObjectiveLabEmbedAdapter()
    caps = adapter.embed_capabilities()
    assert caps.supports_embedded is True
    assert caps.min_size == (1180, 760)

    optimize_train = PRIMARY_WORKSPACES["optimize_train"]
    assert "pendulum_simulator" in optimize_train.member_tool_ids
    assert "swing_objective_lab" in optimize_train.member_tool_ids


def test_old_supported_imports_delegate_correctly() -> None:
    """GREEN case 3d: old supported imports delegate cleanly to canonical implementations."""
    from src.launchers.tools_repo_path import ensure_tools_importable

    resolution = ensure_tools_importable(REPO_ROOT)
    assert resolution is not None

    # Test deprecated alias shim delegates with warning
    sys.modules.pop("upstream_drift_tools", None)
    with pytest.deprecated_call():
        import upstream_drift_tools

        assert upstream_drift_tools is not None
        assert hasattr(upstream_drift_tools, "__file__")
