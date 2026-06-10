"""Suite-marker classification helpers (issue #7158, defect 2).

The root ``tests/conftest.py`` uses these helpers in a
``pytest_collection_modifyitems`` hook to report (and, once enforced, fail on)
tests that carry none of the recognized suite markers.  Keeping the logic here
(rather than inline in conftest) makes it unit-testable in isolation.
"""

from __future__ import annotations

import os
from collections.abc import Iterable
from typing import Protocol

#: Markers that classify a test into a CI suite/lane.  A test carrying at least
#: one of these is considered "suite-classified".
SUITE_MARKERS: frozenset[str] = frozenset(
    {
        "unit",
        "integration",
        "e2e",
        "slow",
        "live_simulation",
        "benchmark",
        "perf",
        "smoke",
        "gate",
        "parity",
        "scientific",
        "motion_pipeline",
    }
)

#: Environment variable that flips the hook from report-only to enforcing.
ENFORCE_ENV_VAR = "UD_ENFORCE_SUITE_MARKERS"


class _MarkerLike(Protocol):
    name: str


class _ItemLike(Protocol):
    nodeid: str

    def iter_markers(self) -> Iterable[_MarkerLike]: ...


def suite_markers_enforced() -> bool:
    """Return True when missing suite markers should fail collection.

    Report-only by default; opt in with ``UD_ENFORCE_SUITE_MARKERS`` set to a
    truthy value (``1``/``true``/``yes``/``on``).
    """
    return os.environ.get(ENFORCE_ENV_VAR, "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def item_has_suite_marker(item: _ItemLike) -> bool:
    """Return True iff *item* carries at least one suite marker."""
    return bool(SUITE_MARKERS & {m.name for m in item.iter_markers()})


def find_unmarked(items: Iterable[_ItemLike]) -> list[_ItemLike]:
    """Return the subset of *items* carrying no suite marker."""
    return [item for item in items if not item_has_suite_marker(item)]
