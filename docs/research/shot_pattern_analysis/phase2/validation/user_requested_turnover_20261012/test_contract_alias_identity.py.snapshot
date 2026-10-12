"""Keep standalone Tools contracts imports on one canonical module."""

from __future__ import annotations

import importlib
import sys

import pytest

pytestmark = pytest.mark.unit


def test_standalone_contracts_aliases_keep_provider_identity() -> None:
    """Standalone setup must not leave a second enum under the src spelling."""
    provider = importlib.import_module("shared.python.contracts")
    src_module = importlib.import_module("src.shared.python.contracts")
    short_module = importlib.import_module("contracts")
    from src.shared.python.contracts import ContractLevel, set_contract_level

    assert src_module is provider
    assert short_module is provider
    assert sys.modules["src.shared.python.contracts"] is provider
    assert importlib.import_module("src.shared.python").contracts is provider
    assert ContractLevel is provider.ContractLevel
    assert set_contract_level is provider.set_contract_level

    original = provider.get_contract_level()
    try:
        for level in provider.ContractLevel:
            src_module.set_contract_level(level)
            assert provider.get_contract_level() is level
            assert provider.DBC_LEVEL is level
    finally:
        provider.set_contract_level(original)
