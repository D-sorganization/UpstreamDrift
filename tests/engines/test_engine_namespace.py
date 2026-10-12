"""Engine test directories must not masquerade as installed bindings."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


@pytest.mark.unit
def test_opensim_test_directory_does_not_shadow_runtime(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A moved test directory cannot satisfy an OpenSim runtime probe."""
    monkeypatch.delitem(sys.modules, "opensim", raising=False)
    spec = importlib.util.find_spec("opensim")
    if spec is None:
        return
    shadow = Path(__file__).resolve().parent / "opensim"
    locations = spec.submodule_search_locations or ()
    assert shadow not in {Path(location).resolve() for location in locations}
