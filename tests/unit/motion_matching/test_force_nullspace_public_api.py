"""Public API contract for the force_nullspace package (NSX-03 / #12552).

The pre-split module exposed exactly the names below. After the split into a
package, the package must expose the same public set through ``__all__`` and
every name must be the same object as the one defined in its new submodule.
"""

import importlib

import pytest

pytestmark = pytest.mark.unit

PACKAGE = "src.shared.python.motion_matching.force_nullspace"

PRE_SPLIT_PUBLIC_NAMES = frozenset(
    {
        "Array",
        "ForceConstraints",
        "ForceNullSpace",
        "FrictionCone",
        "NullSpaceAnalysis",
        "RedistributionResult",
        "TorqueDistributionTradeoff",
        "TradeoffAlternative",
        "explore_torque_tradeoffs",
        "redistribute_forces",
        "redistribute_trajectory",
    }
)

NAME_HOME_MODULES = {
    "Array": "_common",
    "NullSpaceAnalysis": "space",
    "ForceNullSpace": "space",
    "FrictionCone": "constraints",
    "ForceConstraints": "constraints",
    "RedistributionResult": "redistribution",
    "redistribute_forces": "redistribution",
    "redistribute_trajectory": "redistribution",
    "TradeoffAlternative": "tradeoffs",
    "TorqueDistributionTradeoff": "tradeoffs",
    "explore_torque_tradeoffs": "tradeoffs",
}


def _package():
    return importlib.import_module(PACKAGE)


def test_all_equals_pre_split_public_name_set() -> None:
    pkg = _package()
    assert hasattr(pkg, "__all__"), "package must declare __all__"
    assert set(pkg.__all__) == PRE_SPLIT_PUBLIC_NAMES
    assert len(pkg.__all__) == len(set(pkg.__all__)), "__all__ must not repeat"


def test_every_public_name_is_importable_from_package() -> None:
    pkg = _package()
    for name in PRE_SPLIT_PUBLIC_NAMES:
        assert hasattr(pkg, name), f"package lacks public name {name}"


@pytest.mark.parametrize("name", sorted(NAME_HOME_MODULES))
def test_public_name_is_same_object_as_submodule_definition(name: str) -> None:
    pkg = _package()
    home = importlib.import_module(f"{PACKAGE}.{NAME_HOME_MODULES[name]}")
    assert getattr(pkg, name) is getattr(home, name)


def test_home_module_map_covers_exactly_the_public_names() -> None:
    assert set(NAME_HOME_MODULES) == PRE_SPLIT_PUBLIC_NAMES
