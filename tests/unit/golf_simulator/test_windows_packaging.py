"""Unit tests for Windows packaging, dependency resilience, and licensing policy (GS-08, #10197).

Follows TDD, DbC, Law of Demeter, and DRY.
"""

from __future__ import annotations

import sys
import pytest

from src.shared.python.golf_simulator.packaging import (
    EnvironmentReadinessReport,
    LicensingPolicyViolationError,
    assert_licensing_policy,
    check_environment_readiness,
)

pytestmark = pytest.mark.unit


def test_check_environment_readiness_runs_without_crashing() -> None:
    report = check_environment_readiness()
    assert isinstance(report, EnvironmentReadinessReport)
    assert report.platform == sys.platform
    assert isinstance(report.optional_packages, dict)
    # Checks that keys exist for known optional packages
    assert "pywin32" in report.optional_packages
    assert "psutil" in report.optional_packages
    assert len(report.diagnostic_messages) > 0


def test_licensing_policy_blocks_auto_eula_acceptance() -> None:
    with pytest.raises(
        LicensingPolicyViolationError,
        match="Automatic acceptance of third-party EULA is forbidden by project policy",
    ):
        assert_licensing_policy(auto_accept_eula=True)


def test_licensing_policy_blocks_vendor_binary_modification() -> None:
    with pytest.raises(
        LicensingPolicyViolationError,
        match="Modification of third-party simulator binaries is forbidden",
    ):
        assert_licensing_policy(modify_vendor_binaries=True)


def test_licensing_policy_allows_compliant_operation() -> None:
    # Compliant invocation should succeed cleanly without error
    assert_licensing_policy(auto_accept_eula=False, modify_vendor_binaries=False)
