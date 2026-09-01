"""Shared fixture wiring for the ADR-0046 G0 cross-stack drift gates.

One deterministic synthetic session (``adr0046_cross_stack_session_v1.json``)
feeds both launch-monitor stacks. Both stacks accept a *column-mapped*
``pandas.DataFrame``, so the single physical dataset is expressed once and each
stack receives its own request object naming the same columns. The mapping is:

===================================  ==============================  ==========
fixture column                       UD stack request field          Tools stack
===================================  ==============================  ==========
``start_lie``/``start_context``      ``start.lie_column`` / ...      ``before_lie_column`` / ...
``start_distance_yards`` (``yd``)    ``start.distance_column``       ``before_distance_column``
``finish_lie``/``finish_context``    ``finish.lie_column`` / ...     ``after_lie_column`` / ...
``finish_distance_metres`` (``m``)   ``finish.distance_column``      ``after_distance_column``
``carry_distance_metres`` (``m``)    ``analyze_dispersion(forward=)``  ``DispersionRequest.carry_column``
``lateral_carry_metres`` (``m``)     ``analyze_dispersion(lateral=)``  ``DispersionRequest.lateral_column``
``player_id``/``session_id``         ``AnalysisContextV2`` identities  ``LongitudinalRequest`` columns
``session_order``                    ``OrderEvidenceV2.order_column``  ``session_order_column``
===================================  ==============================  ==========

The start distance is carried in yards and the finish distance in metres on
purpose: that exercises each stack's own unit conversion on the same physical
values inside one analysis.

The expected-strokes baseline is a single JSON block. The UD stack validates it
as ``ExpectedStrokesBaselineV2``; the Tools stack loads the identical bytes off
disk through ``load_strokes_gained_baseline``.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

YARDS_PER_METRE = 1.0936132983377078

FIXTURE_PATH = (
    Path(__file__).resolve().parents[2]
    / "fixtures"
    / "launch_monitor"
    / "adr0046_cross_stack_session_v1.json"
)


def _repo_root() -> Path:
    here = Path(__file__).resolve()
    for parent in (here, *here.parents):
        if (parent / "pyproject.toml").is_file():
            return parent
    raise RuntimeError("Could not locate the repository root")


def require_vendored_tools_stack() -> None:
    """Skip locally, FAIL in CI, when ``vendor/ud-tools`` is absent.

    The distinction is load-bearing (found in G1, #9348): CI's default
    checkout materialises no submodule, so a plain skip let all 28 drift
    gates report green without executing while ``vendor-freshness.yml``
    advanced the pin nightly - the exact silent-pass this suite exists to
    prevent. In CI these gates run inside the
    ``shared-tools-consumer-contracts`` job, which materialises the pin via
    ``fetch-pinned-tools``; any other CI placement now fails loudly instead
    of skipping. Local developer checkouts keep the friendly skip.
    """
    vendored = (
        _repo_root()
        / "vendor"
        / "ud-tools"
        / "src"
        / "rate_of_closure"
        / "launch_monitor_strokes_gained.py"
    )
    if vendored.is_file():
        return
    message = (
        "vendor/ud-tools submodule is not materialised; run "
        "`git submodule update --init vendor/ud-tools`"
    )
    if os.environ.get("GITHUB_ACTIONS") == "true":
        pytest.fail(
            "ADR-0046 G0 drift gates cannot run: " + message + ". In CI this "
            "is a hard failure, never a skip - a skipped drift gate reports "
            "green while guarding nothing. Run this suite in the "
            "shared-tools-consumer-contracts job (it materialises the pin).",
            pytrace=False,
        )
    pytest.skip(message, allow_module_level=True)


@pytest.fixture(scope="session")
def fixture_payload() -> dict[str, Any]:
    """Return the committed cross-stack session fixture."""
    return json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="session")
def session_frame(fixture_payload: dict[str, Any]) -> pd.DataFrame:
    """Return the 160-shot clean session both stacks must ingest identically."""
    return pd.DataFrame.from_records(fixture_payload["records"])


@pytest.fixture(scope="session")
def degenerate_frame(fixture_payload: dict[str, Any]) -> pd.DataFrame:
    """Return the four deliberately malformed rows, one per failure mode."""
    return pd.DataFrame.from_records(fixture_payload["degenerate_records"])


@pytest.fixture(scope="session")
def baseline_document(fixture_payload: dict[str, Any]) -> dict[str, Any]:
    """Return the raw expected-strokes baseline document."""
    return dict(fixture_payload["baseline"])


@pytest.fixture(scope="session")
def baseline_path(
    tmp_path_factory: pytest.TempPathFactory, baseline_document: dict[str, Any]
) -> Path:
    """Materialise the baseline for the Tools loader, which reads from disk."""
    path = tmp_path_factory.mktemp("adr0046_baseline") / "baseline.json"
    path.write_text(json.dumps(baseline_document), encoding="utf-8")
    return path
