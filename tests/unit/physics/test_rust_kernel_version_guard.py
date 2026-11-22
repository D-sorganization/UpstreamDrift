"""A stale ``upstream_physics`` wheel must fall back to Python (#15443 F11).

PR #15460 added the spin term to ``compute_drag`` without bumping the crate,
so a wheel built earlier imported fine and silently dropped the term.  The
guard compares the installed version with ``MIN_KERNEL_VERSION``.
"""

from __future__ import annotations

import importlib
import logging
import sys
import tomllib
import types
from pathlib import Path

import pytest
from src.shared.python.physics import rust_kernel

pytestmark = pytest.mark.unit

CRATE_TOML = (
    Path(__file__).resolve().parents[3] / "rust_core/upstream-physics/Cargo.toml"
)


@pytest.fixture
def reload_kernel(monkeypatch):
    """Reload ``rust_kernel`` against a fake wheel reporting ``version``."""

    def _reload(version):
        monkeypatch.setitem(sys.modules, "upstream_physics", types.ModuleType("x"))
        monkeypatch.setattr(
            "importlib.metadata.version", lambda name: version, raising=True
        )
        return importlib.reload(rust_kernel)

    yield _reload
    monkeypatch.undo()
    importlib.reload(rust_kernel)


def test_older_kernel_falls_back_and_warns(reload_kernel, caplog):
    with caplog.at_level(logging.WARNING):
        module = reload_kernel("2.1.3")
    assert module.is_rust_available() is False
    messages = [r.getMessage() for r in caplog.records]
    stale = [m for m in messages if "2.1.3" in m and module.MIN_KERNEL_VERSION in m]
    assert len(stale) == 1
    assert "maturin develop" in stale[0]


def test_current_kernel_stays_available(reload_kernel):
    module = reload_kernel(rust_kernel.MIN_KERNEL_VERSION)
    assert module.is_rust_available() is True


def test_newer_kernel_stays_available(reload_kernel):
    assert reload_kernel("10.0.0").is_rust_available() is True


def test_unknown_version_is_treated_as_stale(reload_kernel):
    assert reload_kernel("not-a-version").is_rust_available() is False


def test_requirement_matches_crate_version():
    with CRATE_TOML.open("rb") as handle:
        crate_version = tomllib.load(handle)["package"]["version"]
    assert crate_version == rust_kernel.MIN_KERNEL_VERSION
