"""Suite-marker classification helpers (issue #7158, defect 2).

The root ``tests/conftest.py`` uses these helpers in a
``pytest_collection_modifyitems`` hook to report (and, once enforced, fail on)
tests that carry none of the recognized suite markers.  Keeping the logic here
(rather than inline in conftest) makes it unit-testable in isolation.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
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

#: Environment variable that allows current baseline debt but rejects drift.
RATCHET_ENV_VAR = "UD_RATCHET_SUITE_MARKERS"

REPO_ROOT = Path(__file__).resolve().parents[2]
BASELINE_PATH = REPO_ROOT / "scripts" / "config" / "suite_marker_baseline.json"


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


def suite_marker_ratchet_enabled() -> bool:
    """Return True when missing suite markers should be checked against baseline."""
    return os.environ.get(RATCHET_ENV_VAR, "").strip().lower() in {
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


def normalize_nodeid(nodeid: str) -> str:
    """Normalize parametrized pytest nodeids to their source-level test id."""
    return nodeid.split("[", 1)[0]


def load_baseline_nodeids(path: Path = BASELINE_PATH) -> frozenset[str]:
    """Load normalized unmarked-test nodeids from the suite-marker baseline."""
    if not path.exists():
        return frozenset()
    with path.open(encoding="utf-8") as fh:
        payload = json.load(fh)
    if not isinstance(payload, dict):
        raise ValueError(f"baseline {path} must be a JSON object")
    nodeids = payload.get("unmarked_nodeids", [])
    if not isinstance(nodeids, list) or not all(isinstance(n, str) for n in nodeids):
        raise ValueError(f"baseline {path} must contain string list unmarked_nodeids")
    return frozenset(normalize_nodeid(nodeid) for nodeid in nodeids)


def find_unmarked_baseline_drift(
    items: Iterable[_ItemLike], baseline_nodeids: Iterable[str]
) -> list[_ItemLike]:
    """Return unmarked items that are not covered by the committed baseline."""
    normalized_baseline = {normalize_nodeid(nodeid) for nodeid in baseline_nodeids}
    return [
        item
        for item in find_unmarked(items)
        if normalize_nodeid(item.nodeid) not in normalized_baseline
    ]
