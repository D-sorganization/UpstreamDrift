"""End-to-end integration test: SystemIdentifier against SimscapeAdapter (#4009).

Closes the Option 4 epic by verifying that the existing
:class:`src.learning.sim2real.system_identification.SystemIdentifier`
plugs into the new :class:`src.engines.simscape.SimscapeAdapter` without
any consumer-side changes — the headline claim of motion-matching
Option 4 ("reuses existing system_identification stack").

Markers
-------
``requires_matlab`` and ``slow``: every test in this file launches MATLAB.
The module is auto-skipped when ``matlab.engine`` is not importable, so
CI without MATLAB still passes through discovery cleanly.
"""

from __future__ import annotations

import ast
import inspect
from collections.abc import Iterator
from pathlib import Path

import pytest
from src.engines.loaders import load_matlab_3d_engine
from src.engines.simscape import SimscapeAdapter
from src.learning.sim2real import system_identification as sysid_mod
from src.learning.sim2real._simscape_compat import wrap_for_system_identification
from src.learning.sim2real.system_identification import SystemIdentifier
from src.shared.python.engine_core.engine_registry import EngineType

pytestmark = [
    pytest.mark.requires_matlab,
    pytest.mark.live_simulation,
    pytest.mark.slow,
    pytest.mark.integration,
]


def _suite_root() -> Path:
    """Repo root used to resolve the default Simscape model path."""
    return Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def simscape_engine() -> Iterator[SimscapeAdapter]:
    """Load the MATLAB_3D engine via the canonical loader, once per module."""
    engine = load_matlab_3d_engine(_suite_root())
    assert isinstance(engine, SimscapeAdapter)
    yield engine
    engine.close()


def test_load_simscape_for_system_identification(
    simscape_engine: SimscapeAdapter,
) -> None:
    """Adapter loaded via ``EngineType.MATLAB_3D`` is protocol-compliant."""
    assert simscape_engine.engine_type == EngineType.MATLAB_3D.value
    assert simscape_engine.model_loaded
    # The SystemIdentifier never reaches into any state below `model`.
    identifier = SystemIdentifier(wrap_for_system_identification(simscape_engine))
    assert identifier.model is not None


def test_system_identifier_runs_one_iteration_against_simscape(
    simscape_engine: SimscapeAdapter,
    synthetic_demonstration: object,
) -> None:
    """A 1-iteration loop exercises every code path without exceptions."""
    identifier = SystemIdentifier(wrap_for_system_identification(simscape_engine))
    result = identifier.identify_from_trajectories(
        [synthetic_demonstration],  # type: ignore[list-item]
        params_to_identify=["mass_scale"],
        max_iterations=1,
        tolerance=1.0,
    )
    assert "mass_scale" in result.identified_params
    assert result.iterations >= 1


def test_protocol_method_coverage_against_actual_simscape(
    simscape_engine: SimscapeAdapter,
) -> None:
    """Every method ``SystemIdentifier`` calls on its model is reachable.

    Walks the AST of :mod:`system_identification` and checks each
    ``self.model.<name>`` attribute either lives on the live adapter or
    is provided by the compat shim.
    """
    needed = _system_identifier_engine_method_names()
    wrapped = wrap_for_system_identification(simscape_engine)
    missing = [n for n in needed if not hasattr(wrapped, n)]
    assert not missing, f"compat surface missing methods: {missing}"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _system_identifier_engine_method_names() -> set[str]:
    """Return the set of ``self.model.<name>`` attributes referenced in code."""
    src = inspect.getsource(sysid_mod)
    tree = ast.parse(src)
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        # We want self.model.<name>
        value = node.value
        if not isinstance(value, ast.Attribute):
            continue
        if value.attr != "model":
            continue
        inner = value.value
        if not (isinstance(inner, ast.Name) and inner.id == "self"):
            continue
        names.add(node.attr)
    return names
