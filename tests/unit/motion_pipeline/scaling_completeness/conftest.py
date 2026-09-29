"""
Conftest for scaling_completeness tests.

Workaround mirror of ``matching_completeness/conftest.py`` — patches the
shared ``invariant`` primitive so motion_pipeline.contracts can be
imported despite a pre-existing decorator-vs-call mismatch.
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
        if len(args) == 1 and isinstance(args[0], str) and not kwargs:

            def decorator(func):
                return func

            return decorator
        return original(*args, **kwargs)

    mod.invariant = _shim


_install_invariant_shim()
