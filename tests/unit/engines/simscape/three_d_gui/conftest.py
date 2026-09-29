"""Per-directory conftest that pivots ``src`` for the 3D-Golf-Model GUI tests.

These tests import the engine's own ``src.apps.*`` package, whose top-level name
collides with the repo's ``src`` package, so ``sys.modules["src"]`` is rebound
while this directory collects and runs.

The pivot is process-global state, so it is installed through the shared,
fully-restoring :class:`tests.helpers.engine_src_pivot.EngineSrcPivot` -- see
that module for why restoring only ``sys.modules["src"]`` is not enough and
which unrelated suites the leftover eviction used to break.

``pytest_make_collect_report``, ``pytest_runtest_setup`` and
``pytest_runtest_teardown`` are all directory-scoped by pytest itself (a
conftest's hooks are only consulted for collectors/items at or below its own
directory), so the pivot window never covers anything outside this directory,
regardless of how xdist interleaves work across workers.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import pytest

from tests.helpers.engine_src_pivot import EngineSrcPivot

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")

# tests/unit/engines/simscape/three_d_gui/conftest.py -> repo root is parents[5].
_PIVOT = EngineSrcPivot(
    Path(__file__).resolve().parents[5],
    precache=(
        "sidekick.lab.bio.c3d_reader",
        "src.shared.python.qt_utils.wheel_event_filter",
        "src.shared.python.motion_matching.body_skeleton",
    ),
)


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(
    collector: pytest.Collector,  # noqa: ARG001
) -> Generator[None, None, None]:
    """Keep the engine ``src`` shadow active only while collecting this dir."""
    _PIVOT.enter()
    try:
        yield
    finally:
        _PIVOT.exit()


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_setup(item: pytest.Item) -> Generator[None, None, None]:  # noqa: ARG001
    """Enter the pivot ahead of this dir's fixture setup."""
    _PIVOT.enter()
    yield


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_teardown(
    item: pytest.Item,  # noqa: ARG001
    nextitem: pytest.Item | None,  # noqa: ARG001
) -> Generator[None, None, None]:
    """Leave the pivot only after this dir's fixture finalizers have run."""
    yield
    _PIVOT.exit()
