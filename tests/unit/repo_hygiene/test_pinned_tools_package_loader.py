"""The pinned-Tools package loader stays outside the shared namespaces (#11994)."""

import sys

import pytest

from src.shared.python._seam_redirect import (
    SeamResolutionError,
    load_pinned_tools_package,
)

pytestmark = pytest.mark.unit


def test_loads_the_pinned_package_under_a_private_name() -> None:
    from sidekick import lab

    before = list(lab.__path__)
    mocap = load_pinned_tools_package("sidekick.lab.mocap")

    assert mocap.__name__.startswith("_pinned_tools__")
    assert hasattr(mocap, "ExperimentReplayBundle")
    assert list(lab.__path__) == before
    assert "sidekick.lab.mocap" not in sys.modules


def test_repeated_loads_return_one_module_object() -> None:
    first = load_pinned_tools_package("sidekick.lab.mocap")
    assert load_pinned_tools_package("sidekick.lab.mocap") is first
    assert sys.modules[first.__name__] is first


@pytest.mark.parametrize("relative", ["", "sidekick..lab", "sidekick/lab", "1abc"])
def test_rejects_a_malformed_package_path(relative: str) -> None:
    with pytest.raises(ValueError, match="dotted package path"):
        load_pinned_tools_package(relative)


def test_a_package_missing_from_the_pinned_tree_fails_loudly() -> None:
    with pytest.raises(SeamResolutionError, match="not in the pinned Tools tree"):
        load_pinned_tools_package("sidekick.lab.no_such_package")
