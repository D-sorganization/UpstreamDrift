#!/usr/bin/env python3
"""Upgrade-guard for the pinned JaxSim dependency (issue #6660).

JaxSim is held at an exact pin (``jaxsim==0.9.0``) while the integration is
gated. This guard fails loudly if the pin drifts in ``pyproject.toml`` or if an
installed ``jaxsim`` reports a different version, so an accidental bump cannot
silently invalidate the parity/forward-sim gates.

The pin contract is also asserted by ``tests/unit/test_jaxsim_optional_dependency.py``;
this script is the CI-step front-end that fails the build on drift. It is
dependency-light (only stdlib + tomllib/tomli) so it runs on any runner before
the heavy optional stack is installed.

Usage:
    python scripts/jaxsim/check_jaxsim_pin.py
"""

from __future__ import annotations

import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 compatibility
    import tomli as tomllib  # type: ignore[no-redef]

#: The single source of truth for the expected pin. Keep in lockstep with the
#: assertion in tests/unit/test_jaxsim_optional_dependency.py.
EXPECTED_JAXSIM_REQUIREMENT = "jaxsim==0.9.0"
EXPECTED_JAXSIM_VERSION = "0.9.0"


def read_declared_requirement(pyproject_path: Path) -> str:
    """Return the sole declared ``jaxsim`` optional requirement.

    Args:
        pyproject_path: Path to ``pyproject.toml``.

    Returns:
        The single requirement string in ``optional-dependencies.jaxsim``.

    Raises:
        ValueError: If the extra is missing or does not contain exactly one
            requirement.
    """

    data = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    optional = data["project"]["optional-dependencies"]
    requirements = optional.get("jaxsim")
    if not requirements or len(requirements) != 1:
        raise ValueError(
            "optional-dependencies.jaxsim must declare exactly one requirement, "
            f"got {requirements!r}"
        )
    return str(requirements[0])


def installed_version() -> str | None:
    """Return the installed ``jaxsim`` version, or ``None`` if not installed."""

    try:
        from importlib import metadata
    except ImportError:  # pragma: no cover - defensive
        return None
    try:
        return metadata.version("jaxsim")
    except metadata.PackageNotFoundError:
        return None


def check(pyproject_path: Path) -> list[str]:
    """Return a list of drift errors (empty when the pin is intact)."""

    errors: list[str] = []
    declared = read_declared_requirement(pyproject_path)
    if declared != EXPECTED_JAXSIM_REQUIREMENT:
        errors.append(
            f"pyproject pin drifted: expected {EXPECTED_JAXSIM_REQUIREMENT!r}, "
            f"found {declared!r}"
        )
    version = installed_version()
    if version is not None and version != EXPECTED_JAXSIM_VERSION:
        errors.append(
            f"installed jaxsim version drifted: expected "
            f"{EXPECTED_JAXSIM_VERSION!r}, found {version!r}"
        )
    return errors


def main() -> int:
    """CLI entry point. Returns a non-zero exit code on pin drift."""

    pyproject_path = Path(__file__).resolve().parents[2] / "pyproject.toml"
    errors = check(pyproject_path)
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)  # noqa: T201
        print(
            "JaxSim is intentionally pinned while issue #6660 keeps the "
            "integration gated. Coordinate any bump with the parity gates.",
            file=sys.stderr,
        )  # noqa: T201
        return 1
    print(f"OK: jaxsim pin intact ({EXPECTED_JAXSIM_REQUIREMENT}).")  # noqa: T201
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
