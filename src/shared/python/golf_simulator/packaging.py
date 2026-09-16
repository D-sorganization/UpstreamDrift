"""Windows packaging, dependency resilience, and licensing policy (GS-08, #10197).

Follows Design by Contract (DbC).
Ensures optional dependencies (pywin32, psutil) do not crash when missing.
Enforces licensing policy preventing automated third-party EULA acceptance or binary modification.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
import sys
from typing import Any


class LicensingPolicyViolationError(RuntimeError):
    """Raised when an automated action attempts to bypass vendor licensing or modify binaries."""


@dataclass(frozen=True)
class EnvironmentReadinessReport:
    """Report assessing environment readiness for simulator execution."""

    is_ready: bool
    platform: str
    python_version: str
    optional_packages: dict[str, bool] = field(default_factory=dict)
    diagnostic_messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "is_ready": self.is_ready,
            "platform": self.platform,
            "python_version": self.python_version,
            "optional_packages": dict(self.optional_packages),
            "diagnostic_messages": list(self.diagnostic_messages),
        }


def check_environment_readiness() -> EnvironmentReadinessReport:
    """Evaluate runtime readiness without crashing when optional dependencies are missing."""
    packages_to_check = ("pywin32", "psutil", "pydantic")
    avail: dict[str, bool] = {}
    msgs: list[str] = []

    for pkg in packages_to_check:
        module_name = "win32gui" if pkg == "pywin32" else pkg
        found = importlib.util.find_spec(module_name) is not None
        avail[pkg] = found
        if found:
            msgs.append(f"Optional package '{pkg}' is installed.")
        else:
            msgs.append(f"Optional package '{pkg}' is absent (fallback available).")

    is_windows = sys.platform == "win32"
    if not is_windows:
        msgs.append(
            f"Non-Windows platform detected ({sys.platform}); using cross-platform simulator fallbacks."
        )

    return EnvironmentReadinessReport(
        is_ready=True,
        platform=sys.platform,
        python_version=sys.version.split()[0],
        optional_packages=avail,
        diagnostic_messages=msgs,
    )


def assert_licensing_policy(
    auto_accept_eula: bool = False,
    modify_vendor_binaries: bool = False,
) -> None:
    """Enforce strict licensing and vendor binary integrity policies.

    Raises:
        LicensingPolicyViolationError: If caller attempts automated EULA acceptance
            or third-party binary modification.
    """
    if auto_accept_eula:
        raise LicensingPolicyViolationError(
            "Automatic acceptance of third-party EULA is forbidden by project policy. "
            "GSPro must be licensed and launched through its supported operator workflow."
        )
    if modify_vendor_binaries:
        raise LicensingPolicyViolationError(
            "Modification of third-party simulator binaries is forbidden. "
            "Integration must occur strictly over official network and protocol boundaries."
        )
