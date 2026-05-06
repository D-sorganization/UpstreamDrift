"""Unit tests for the SimscapeAdapter <-> SystemIdentifier compat shim (#4009).

These tests run on every CI host (no MATLAB required). The headline test
is :func:`test_simscape_compat_protocol_introspection_matches_system_identifier_calls`
which uses a Python AST walk to find every ``self.model.<method>`` call
inside :mod:`src.learning.sim2real.system_identification` and asserts
each one is satisfied by either :class:`SimscapeAdapter` directly or the
:class:`SimscapeSystemIdCompat` shim. This catches drift between
SystemIdentifier expectations and SimscapeAdapter implementation
without needing a MATLAB licence.
"""

from __future__ import annotations

import ast
import inspect
import logging
from unittest.mock import MagicMock

import numpy as np
import pytest
from src.engines.simscape.adapter import SimscapeAdapter
from src.learning.sim2real import system_identification as sysid_mod
from src.learning.sim2real._simscape_compat import (
    SimscapeSystemIdCompat,
    wrap_for_system_identification,
)

pytestmark = [pytest.mark.unit]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _system_identifier_engine_method_names() -> set[str]:
    """Return every ``self.model.<name>`` attribute referenced in sysid."""
    src = inspect.getsource(sysid_mod)
    tree = ast.parse(src)
    names: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_simscape_compat_protocol_introspection_matches_system_identifier_calls() -> (
    None
):
    """Every ``self.model.<name>`` reference in sysid is reachable.

    The check resolves names against the union of ``SimscapeAdapter``
    attributes and the compat shim. Both are inspected without ever
    instantiating MATLAB.
    """
    needed = _system_identifier_engine_method_names()
    assert needed, "AST walk found no engine methods — parser is broken"

    adapter_attrs = set(dir(SimscapeAdapter))
    compat_attrs = set(SimscapeSystemIdCompat.COMPAT_METHODS) | {"set_joint_damping"}
    available = adapter_attrs | compat_attrs

    missing = sorted(n for n in needed if n not in available)
    assert not missing, (
        f"sysid references methods absent on SimscapeAdapter and compat shim: "
        f"{missing}. Either implement on the adapter or add to the compat shim."
    )


def test_simscape_compat_set_joint_damping_logs_warning(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The compat ``set_joint_damping`` records the call and warns once."""
    adapter = MagicMock(spec=SimscapeAdapter)
    adapter.dof = 3
    compat = wrap_for_system_identification(adapter)

    damping = np.array([0.1, 0.2, 0.3], dtype=np.float64)
    with caplog.at_level(logging.WARNING):
        compat.set_joint_damping(damping)
        compat.set_joint_damping(damping * 2)

    np.testing.assert_array_equal(compat.damping_history[0], damping)
    np.testing.assert_array_equal(compat.damping_history[1], damping * 2)
    # Warning fires once, not on every call.
    relevant = [r for r in caplog.records if "set_joint_damping" in r.getMessage()]
    assert len(relevant) == 1
    assert "deferred to #4006" in relevant[0].getMessage()


def test_simscape_compat_passthrough_to_adapter() -> None:
    """Non-compat attribute access is forwarded to the underlying adapter."""
    adapter = MagicMock(spec=SimscapeAdapter)
    adapter.dof = 7
    adapter.model_loaded = True
    compat = wrap_for_system_identification(adapter)

    assert compat.dof == 7
    assert compat.model_loaded is True


def test_simscape_compat_friction_setter_is_no_op() -> None:
    """``set_friction_coefficients`` is served by the compat shim."""
    adapter = MagicMock(spec=SimscapeAdapter)
    adapter.dof = 4
    compat = wrap_for_system_identification(adapter)

    # Should not raise even though the underlying adapter has no
    # friction-coefficient surface yet.
    compat.set_friction_coefficients(np.array([1.0, 1.0, 1.0, 1.0]))
    out = compat.get_friction_coefficients()
    assert out.shape == (4,)


def test_simscape_compat_rejects_none_adapter() -> None:
    """Constructor refuses a None adapter (DbC)."""
    with pytest.raises(ValueError, match="adapter must be provided"):
        SimscapeSystemIdCompat(None)  # type: ignore[arg-type]
