"""Probe-idempotence diagnostic (issue #7155 D1).

DbC invariant: probing an engine's availability must not mutate global
interpreter state (``sys.modules``). The autouse ``_protect_engine_modules``
fixture in ``tests/conftest.py`` snapshots and restores engine modules on every
test precisely because some probes used to corrupt this state — which both hides
the root cause and makes results order-dependent under ``pytest -n auto``.

This test characterises the invariant directly: it runs each probe twice and
asserts the set of engine-related ``sys.modules`` keys is unchanged. If a probe
is *not* idempotent this test fails loudly, pinpointing the offending probe
instead of letting the autouse fixture paper over it.
"""

from __future__ import annotations

import sys

import pytest

from src.shared.python.engine_core.engine_availability import (
    is_engine_available,
    reset_engine_status_cache,
)

pytestmark = pytest.mark.unit

# Engines that are pure-Python availability checks safe to probe in CI. Heavy
# C-extension engines (drake/pinocchio) are probed only when importable so the
# diagnostic itself never imports something that is not present.
_PROBE_ENGINES = ["mujoco", "drake", "pinocchio", "opensim", "myosuite"]


def _engine_module_keys() -> set[str]:
    """Snapshot sys.modules keys that an engine probe might touch."""
    prefixes = ("mujoco", "pydrake", "pinocchio", "opensim", "myosuite")
    return {k for k in sys.modules if k.startswith(prefixes)}


@pytest.mark.parametrize("engine", _PROBE_ENGINES)
def test_probe_does_not_mutate_sys_modules(engine: str) -> None:
    """Probing twice leaves the engine-module key set unchanged (idempotent)."""
    # Establish a clean, post-probe baseline so we measure the *delta* of a
    # repeat probe, not the one-time cost of a successful import (a successful
    # import legitimately registers the module; the invariant is that a *repeat*
    # probe does not further mutate state).
    reset_engine_status_cache()
    is_engine_available(engine)
    before = _engine_module_keys()

    reset_engine_status_cache()
    is_engine_available(engine)
    after = _engine_module_keys()

    assert after == before, (
        f"probing {engine!r} a second time mutated sys.modules: "
        f"added={sorted(after - before)} removed={sorted(before - after)}"
    )


def test_probe_does_not_inject_mock_modules() -> None:
    """No probe *newly* injects a MagicMock masquerading as a real engine module.

    Some test infrastructure (the conftest fallback for an absent optional
    engine) legitimately pre-seeds a Mock; this test flags only Mocks that a
    probe adds, by comparing against the pre-probe snapshot.
    """
    from unittest.mock import Mock

    def mock_keys() -> set[str]:
        return {
            k for k in _engine_module_keys() if isinstance(sys.modules.get(k), Mock)
        }

    before_mocks = mock_keys()

    for engine in _PROBE_ENGINES:
        reset_engine_status_cache()
        is_engine_available(engine)

    new_mocks = mock_keys() - before_mocks
    assert not new_mocks, (
        f"probing injected Mock modules into sys.modules: {sorted(new_mocks)}"
    )
