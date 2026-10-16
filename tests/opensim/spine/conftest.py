"""OpenSim-lane guard and SciPy read-only emulation for spine tests (#12752).

With ``UD_REQUIRE_OPENSIM=1`` a spine test or collector that is skipped only
because ``opensim`` is not importable is reported as failed. Other skips are
untouched, and nothing changes outside this directory or outside the lane.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest

from tests.helpers.opensim_lane_guard import (
    failure_message,
    is_within_directory,
    should_fail_skip,
)
from tests.helpers.scipy_read_only import strict_rotation_constructors

_SPINE_DIR = Path(__file__).resolve().parent


@pytest.fixture(autouse=True, scope="module")
def _scipy_read_only_rotation() -> Iterator[None]:
    """Emulate SciPy < 1.15, whose ``Rotation`` rejects read-only arrays.

    The cross-engine lane runs SciPy 1.14; module scope makes module-scoped
    fixtures (the driven models) run under the emulation too.
    """
    with pytest.MonkeyPatch.context() as patch:
        strict_rotation_constructors(patch)
        yield


def _skip_text(longrepr: object) -> str:
    """Return the human skip message from a report's ``longrepr``."""
    if isinstance(longrepr, tuple) and len(longrepr) == 3:
        return str(longrepr[2])
    return str(longrepr)


def _enforce_lane(report, path: Path | str) -> None:
    """Turn an absent-OpenSim skip inside the spine directory into a failure."""
    if not report.skipped or not is_within_directory(path, _SPINE_DIR):
        return
    reason = _skip_text(report.longrepr)
    if not should_fail_skip(reason, os.environ):
        return
    report.outcome = "failed"
    report.longrepr = failure_message(reason)


@pytest.hookimpl(wrapper=True)
def pytest_make_collect_report(collector):
    report = yield
    _enforce_lane(report, collector.path)
    return report


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    report = yield
    _enforce_lane(report, item.path)
    return report
