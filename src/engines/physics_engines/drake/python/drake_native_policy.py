"""Reviewed Drake distribution versions for native torque replay (#13019)."""

from __future__ import annotations

import importlib.metadata
from typing import Final

# Layout reviewed on 1.57.0 and 1.58.0: empty GetConstraintIds, abstract
# parameter count num_joints + 3, boolean factory-default prefix unchanged.
DRAKE_REVIEWED_VERSIONS: Final[tuple[str, ...]] = ("1.57.0", "1.58.0")

# Must match ``project.optional-dependencies.drake`` in pyproject.toml.
DRAKE_OPTIONAL_DEPENDENCY_SPEC: Final[str] = ">=1.57.0,<1.59.0"


def installed_drake_distribution_version() -> str:
    """Return the installed ``drake`` distribution version string."""
    return importlib.metadata.version("drake")


def is_reviewed_native_drake_version(version: str) -> bool:
    """Return whether ``version`` is an explicitly reviewed native replay build."""
    if not version:
        raise ValueError("drake version must be a non-empty string")
    return version in DRAKE_REVIEWED_VERSIONS


def require_reviewed_native_drake_version(version: str | None = None) -> str:
    """Fail closed when the installed Drake build is outside the reviewed set."""
    resolved = (
        version if version is not None else installed_drake_distribution_version()
    )
    if not is_reviewed_native_drake_version(resolved):
        reviewed = ", ".join(DRAKE_REVIEWED_VERSIONS)
        raise ValueError(
            "native abstract parameter policy requires reviewed Drake "
            f"({reviewed}); found {resolved!r}"
        )
    return resolved


def skip_unless_reviewed_native_drake() -> None:
    """Skip pytest when Drake is missing or outside the reviewed native set."""
    import pytest

    pytest.importorskip("pydrake.multibody.plant")
    version = installed_drake_distribution_version()
    if is_reviewed_native_drake_version(version):
        return
    reviewed = ", ".join(DRAKE_REVIEWED_VERSIONS)
    pytest.skip(
        f"Drake {version} is not reviewed for native torque replay; "
        f"reviewed versions: {reviewed}"
    )
