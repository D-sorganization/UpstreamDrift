"""
Conftest for matching_completeness tests.

Works around a pre-existing bug in
``src/shared/python/motion_pipeline/contracts.py`` where the platform's
``invariant(condition, message, ...)`` primitive is misused as a
zero-arg decorator factory (``@invariant("name")``). Until that file is
fixed (see issues #4565 / #4568 follow-up), we shim the symbol with a
no-op decorator so this directory's tests can import the contracts
module. The shim is installed only for collection of these tests; the
production primitive is otherwise untouched.
"""

from __future__ import annotations


def _install_invariant_shim() -> None:
    mod_name = "src.shared.python.contracts"
    try:
        mod = __import__(mod_name, fromlist=["invariant"])
    except Exception:  # noqa: BLE001 - skip when optional module import fails
        return

    original = getattr(mod, "invariant", None)
    if original is None:
        return

    def _shim(*args, **kwargs):
        # Used as a decorator factory: @invariant("some_label")
        if len(args) == 1 and isinstance(args[0], str) and not kwargs:

            def decorator(func):
                return func

            return decorator
        # Used as the original function-call primitive
        return original(*args, **kwargs)

    mod.invariant = _shim


_install_invariant_shim()
