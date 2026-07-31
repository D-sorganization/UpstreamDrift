import sys
from unittest.mock import MagicMock
import pytest


@pytest.fixture(autouse=True)
def reset_module_mocks():
    """Reset module-level MagicMock state across test cases (issue #8180)."""
    yield
    for module in list(sys.modules.values()):
        if isinstance(module, MagicMock):
            module.reset_mock()
            for attr in dir(module):
                val = getattr(module, attr, None)
                if isinstance(val, MagicMock):
                    val.reset_mock()
