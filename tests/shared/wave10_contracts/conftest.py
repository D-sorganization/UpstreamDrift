"""Local conftest for wave10 DbC contracts tests.

Each test runs with ContractLevel.ENFORCE by default and is restored after.
"""

from __future__ import annotations

import pytest

from src.shared.python._contracts_level import (
    ContractLevel,
    _ContractState,
    set_contract_level,
)


@pytest.fixture(autouse=True)
def _restore_contract_level():
    prior = _ContractState.level
    set_contract_level(ContractLevel.ENFORCE)
    try:
        yield
    finally:
        set_contract_level(prior)
