"""Tests for the contracts facade module (sys.modules aliases and __all__)."""

from __future__ import annotations

import sys

import src.shared.python.contracts as contracts_mod


def test_aliases_registered():
    for alias in (
        "contracts",
        "shared.python.contracts",
        "src.shared.python.contracts",
    ):
        assert alias in sys.modules
        assert sys.modules[alias] is contracts_mod


def test_all_exports_exist():
    for name in contracts_mod.__all__:
        assert hasattr(contracts_mod, name), name


def test_canonical_helpers_exported():
    assert callable(contracts_mod.require)
    assert callable(contracts_mod.ensure)
    assert callable(contracts_mod.invariant)
    assert callable(contracts_mod.precondition)
    assert callable(contracts_mod.postcondition)
    assert callable(contracts_mod.contract)
    assert callable(contracts_mod.class_invariant)
