import re

import pytest

import src.shared.python.launcher_embed as launcher_embed
import src.shared.python.pose_interchange as pose_interchange
import src.shared.python.realtime as realtime

# The three core cross-engine packages per ADR-0007 / ADR-0012 / ADR-0013.
# These packages establish canonical representations and integration contracts
# between engines and thus require strict semver discipline for schemas.
CROSS_ENGINE_PACKAGES = [
    launcher_embed,
    pose_interchange,
    realtime,
]

# Basic semantic versioning pattern (e.g., "1.0.0", "0.1.2-alpha", etc.)
SEMVER_REGEX = re.compile(
    r"^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-zA-Z0-9-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-zA-Z0-9-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$"
)


@pytest.mark.parametrize("pkg", CROSS_ENGINE_PACKAGES)
def test_cross_engine_api_has_version_attributes(pkg):
    """
    Assert that cross-engine packages expose __version__ and SCHEMA_VERSION.
    """
    assert hasattr(pkg, "__version__"), f"Package {pkg.__name__} missing __version__"
    assert hasattr(pkg, "SCHEMA_VERSION"), (
        f"Package {pkg.__name__} missing SCHEMA_VERSION"
    )


@pytest.mark.parametrize("pkg", CROSS_ENGINE_PACKAGES)
def test_cross_engine_api_versions_are_valid_semver(pkg):
    """
    Assert that the version attributes conform to semantic versioning.
    """
    version = getattr(pkg, "__version__", None)
    schema_version = getattr(pkg, "SCHEMA_VERSION", None)

    assert isinstance(version, str), f"{pkg.__name__}.__version__ must be a string"
    assert SEMVER_REGEX.match(version), (
        f"{pkg.__name__}.__version__ ('{version}') is not valid semver"
    )

    assert isinstance(schema_version, str), (
        f"{pkg.__name__}.SCHEMA_VERSION must be a string"
    )
    assert SEMVER_REGEX.match(schema_version), (
        f"{pkg.__name__}.SCHEMA_VERSION ('{schema_version}') is not valid semver"
    )


@pytest.mark.parametrize("pkg", CROSS_ENGINE_PACKAGES)
def test_cross_engine_api_versions_in_all(pkg):
    """
    Assert that __version__ and SCHEMA_VERSION are exported in __all__.
    """
    assert hasattr(pkg, "__all__"), f"Package {pkg.__name__} missing __all__"
    assert "__version__" in pkg.__all__, (
        f"'__version__' missing from {pkg.__name__}.__all__"
    )
    assert "SCHEMA_VERSION" in pkg.__all__, (
        f"'SCHEMA_VERSION' missing from {pkg.__name__}.__all__"
    )
