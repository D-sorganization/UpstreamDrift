"""Shared utilities for counterfactual computations."""

from __future__ import annotations

from collections.abc import Callable, Generator
from contextlib import contextmanager
from typing import TypeVar

T = TypeVar("T")


@contextmanager
def preserve_state(
    save_state: Callable[[], T],
    restore_state: Callable[[T], None],
    *,
    restore_action: Callable[[], None] | None = None,
) -> Generator[None, None, None]:
    """Preserve state around counterfactual calculations."""
    state = save_state()
    try:
        yield
    finally:
        restore_state(state)
        if restore_action is not None:
            restore_action()
