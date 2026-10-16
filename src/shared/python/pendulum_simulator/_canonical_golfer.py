"""Bind the golfer kinematics and closure to the canonical Tools copy (#12671).

ADR-0056 makes the vendored Tools package
``vendor/ud-tools/src/pendulum_simulator/src/double_pendulum_golf`` the single
source of the golfer forward kinematics and the loop-closure constraint and
Jacobian.  ``golfer_kinematics`` and ``golfer_constraints`` keep their
UpstreamDrift import paths and build their public functions with
:func:`delegate`.

The Tools package is loaded under a private module name by the existing seam
loader (``_seam_redirect.load_pinned_tools_package``), so it is never grafted
onto UpstreamDrift's own ``pendulum_simulator`` namespace.  The Tools
functions check ``isinstance(p, GolferParams)`` against their own dataclass,
so :func:`canonical_params` converts UpstreamDrift parameters field by field.
``tests/unit/pendulum_simulator/test_golfer_canonical_parity.py`` pins the two
dataclasses to the same fields and defaults.
"""

from __future__ import annotations

import dataclasses
import functools
import importlib
from collections.abc import Callable
from types import ModuleType
from typing import Any

from src.shared.python._seam_redirect import load_pinned_tools_package

from .physics_golfer import GolferParams

__all__ = ["CANONICAL_PACKAGE", "canonical_module", "canonical_params", "delegate"]

#: Dotted path of the canonical package under the pinned Tools ``src`` root.
CANONICAL_PACKAGE = "pendulum_simulator.src.double_pendulum_golf"


def canonical_module(name: str) -> ModuleType:
    """Return submodule ``name`` of the pinned Tools golfer package.

    Raises:
        SeamResolutionError: when ``vendor/ud-tools`` is not checked out.
    """
    package = load_pinned_tools_package(CANONICAL_PACKAGE)
    return importlib.import_module(f"{package.__name__}.{name}")


@functools.lru_cache(maxsize=32)
def canonical_params(p: GolferParams) -> Any:
    """Return the Tools ``GolferParams`` equal to ``p`` field by field.

    ``GolferParams`` is a frozen dataclass, so the conversion is cached per
    value.  The Tools ``__post_init__`` re-validates the same contract.
    """
    tools_cls = canonical_module("physics_golfer").GolferParams
    return tools_cls(**{f.name: getattr(p, f.name) for f in dataclasses.fields(p)})


def delegate(module: str, name: str) -> Callable[..., Any]:
    """Return an adapter that calls the Tools function ``module.name``.

    Every ``GolferParams`` argument is converted by :func:`canonical_params`;
    any other argument passes through unchanged, so the Tools contracts
    (``TypeError``/``ValueError``) still reject bad input.  The adapter carries
    the Tools docstring and exposes the Tools function as ``__wrapped__``, which
    the parity test checks by identity.
    """
    target = getattr(canonical_module(module), name)

    def _convert(value: Any) -> Any:
        return canonical_params(value) if isinstance(value, GolferParams) else value

    @functools.wraps(target)
    def adapter(*args: Any, **kwargs: Any) -> Any:
        return target(
            *(_convert(a) for a in args),
            **{key: _convert(value) for key, value in kwargs.items()},
        )

    return adapter
