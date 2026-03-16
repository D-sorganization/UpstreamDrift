"""Conftest for humanoid_character_builder in-package tests.

The package is importable via pyproject.toml's pythonpath configuration
which includes src/tools, so no sys.path manipulation is needed.
"""

from __future__ import annotations

import pytest


def _torch_available() -> bool:
    """Return True iff torch can be imported without errors."""
    try:
        import torch  # noqa: F401  # type: ignore[import-untyped]

        return True
    except (ImportError, OSError):
        return False


# Module-level flag — evaluated once at collection time
_TORCH_OK = _torch_available()

#: Test names (node id substrings) that require a working torch install.
#: We use node id matching since @patch wrappers obscure the inner co_names.
_TORCH_DEPENDENT_TESTS: frozenset[str] = frozenset(
    [
        "test_generate_produces_stl_files",
    ]
)


@pytest.fixture(autouse=True)
def _skip_if_torch_broken(request: pytest.FixtureRequest) -> None:
    """Skip test if it requires torch and torch is broken on this machine."""
    if _TORCH_OK:
        return  # torch works just fine — nothing to do

    # Check by test name
    node = request.node
    node_id = node.nodeid
    for name in _TORCH_DEPENDENT_TESTS:
        if name in node_id:
            pytest.skip("torch DLL incompatible with current Python/OS environment")
            return

    # Also attempt bytecode inspection on the unwrapped function, if accessible
    fn = request.function
    # Walk through __wrapped__ to bypass @patch decorators
    while fn is not None:
        code = getattr(fn, "__code__", None)
        if code is not None and "torch" in getattr(code, "co_names", ()):
            pytest.skip("torch DLL incompatible with current Python/OS environment")
            return
        fn = getattr(fn, "__wrapped__", None)
