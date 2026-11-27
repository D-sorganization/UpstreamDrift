"""A stale ``upstream_physics`` wheel must fall back to Python (#15443 F11).

PR #15460 added the spin term to ``compute_drag`` without any version signal,
so a wheel built earlier imported fine and silently dropped the term.  The
guard compares the kernel's ``PHYSICS_REVISION`` with ``MIN_PHYSICS_REVISION``.
"""

from __future__ import annotations

import importlib
import logging
import re
import sys
import types
from pathlib import Path

import pytest
from src.shared.python.physics import rust_kernel

pytestmark = pytest.mark.unit

CRATE_LIB = (
    Path(__file__).resolve().parents[3] / "rust_core/upstream-physics/src/lib.rs"
)


def _fake_kernel(revision):
    module = types.ModuleType("upstream_physics")
    if revision is not None:
        module.PHYSICS_REVISION = revision
    return module


@pytest.fixture
def reload_kernel(monkeypatch):
    """Reload ``rust_kernel`` against a fake wheel with ``revision``."""

    def _reload(revision):
        monkeypatch.setitem(sys.modules, "upstream_physics", _fake_kernel(revision))
        return importlib.reload(rust_kernel)

    yield _reload
    monkeypatch.undo()
    importlib.reload(rust_kernel)


@pytest.mark.parametrize("revision", [None, 1])
def test_older_kernel_falls_back_and_warns(reload_kernel, caplog, revision):
    with caplog.at_level(logging.WARNING):
        module = reload_kernel(revision)
    assert module.is_rust_available() is False
    messages = [r.getMessage() for r in caplog.records]
    stale = [m for m in messages if "older than the required" in m]
    assert len(stale) == 1
    assert str(module.MIN_PHYSICS_REVISION) in stale[0]
    assert "maturin develop" in stale[0]


def test_current_kernel_stays_available(reload_kernel):
    module = reload_kernel(rust_kernel.MIN_PHYSICS_REVISION)
    assert module.is_rust_available() is True


def test_newer_kernel_stays_available(reload_kernel):
    assert reload_kernel(99).is_rust_available() is True


def test_non_integer_revision_is_treated_as_stale(reload_kernel):
    assert reload_kernel("2").is_rust_available() is False


def test_requirement_matches_crate_const():
    source = CRATE_LIB.read_text(encoding="utf-8")
    match = re.search(r"const PHYSICS_REVISION: u32 = (\d+);", source)
    assert match is not None, "PHYSICS_REVISION const missing from lib.rs"
    assert int(match.group(1)) == rust_kernel.MIN_PHYSICS_REVISION


def test_aerodynamics_facade_follows_the_same_guard(monkeypatch):
    from src.shared.python.physics.aerodynamics import _rust_facade

    try:
        current = rust_kernel.MIN_PHYSICS_REVISION
        for revision, expected in ((current - 1, False), (current, True)):
            fake = _fake_kernel(revision)
            for name in (
                "AerodynamicsEngine",
                "AeroEngineConfig",
                "AeroBallProperties",
                "AirProperties",
            ):
                setattr(fake, name, object)
            monkeypatch.setitem(sys.modules, "upstream_physics", fake)
            module = importlib.reload(_rust_facade)
            assert module.is_rust_available() is expected
    finally:
        monkeypatch.undo()
        importlib.reload(_rust_facade)
