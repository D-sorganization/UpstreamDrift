"""Tests for the Data-sources panel (issue #4480).

Runs under ``QT_QPA_PLATFORM=offscreen`` so it works in headless CI.
The body-target loader (#4477 / #4478) and ball-impact extractor (#4479)
may not be on ``main`` yet, so tests that need those use stubs.
"""

from __future__ import annotations

import os

# MUST set platform BEFORE any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from dataclasses import dataclass
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

pytest.importorskip(
    "PyQt6",
    reason="PyQt6 required for source-toggle UI tests",
    exc_type=ImportError,
)
pytest.importorskip(
    "PyQt6.QtWidgets",
    reason="PyQt6.QtWidgets not loadable in this environment",
    exc_type=ImportError,
)

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.shared.python.motion_matching.multi_source_target import (  # noqa: E402
    MultiSourceTarget,
)
from src.tools.starting_pose_matcher.gui_source_panel import (  # noqa: E402
    BODY_FILE_FILTER,
    CLUB_FILE_FILTER,
    DataSourcesPanel,
)
from src.tools.starting_pose_matcher.session_schema import (  # noqa: E402
    DEFAULT_BODY_MARKER_SET,
    SESSION_SCHEMA_VERSION,
    DataSourcesBlock,
    default_data_sources,
    parse_data_sources,
    serialize_data_sources,
)
from tests.unit.motion_matching._fixtures import make_target  # noqa: E402

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# Qt application fixture
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def qapp() -> QApplication:
    """One-shot QApplication for the module."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


# ---------------------------------------------------------------------------
# Stubs for not-yet-landed dependencies (#4477, #4479)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BodyStub:
    time: np.ndarray
    marker_xyz: np.ndarray = None  # type: ignore[assignment]


@dataclass(frozen=True)
class _ClubBallStub:
    """Minimal duck-type for a ClubBallTarget."""

    time: np.ndarray
    club: Any
    ball: object


# ---------------------------------------------------------------------------
# Schema tests (no Qt required for these)
# ---------------------------------------------------------------------------


def test_schema_version_bumped() -> None:
    assert SESSION_SCHEMA_VERSION >= 4


def test_default_data_sources_is_empty() -> None:
    d = default_data_sources()
    assert d.club.enabled is False
    assert d.body.enabled is False
    assert d.club.file_path is None
    assert d.body.marker_set == DEFAULT_BODY_MARKER_SET


def test_parse_data_sources_missing_block_returns_default() -> None:
    assert parse_data_sources(None) == default_data_sources()
    assert parse_data_sources({}) == default_data_sources()


def test_serialize_then_parse_round_trips() -> None:
    src = DataSourcesBlock()
    blob = serialize_data_sources(src)
    assert isinstance(blob, dict)
    assert parse_data_sources(blob) == src


# ---------------------------------------------------------------------------
# Panel-construction tests
# ---------------------------------------------------------------------------


def test_panel_widgets_present_and_labelled(qapp: QApplication) -> None:
    panel = DataSourcesPanel()
    assert panel.title() == "Data sources"
    # Required UI elements (object names asserted to keep the contract stable)
    for name in (
        "cb_club",
        "cb_body",
        "btn_club_browse",
        "btn_body_browse",
        "rb_club_only",
        "rb_club_ball",
        "lbl_club_path",
        "lbl_body_path",
        "combo_marker_set",
        "rb_align_impact",
        "rb_align_address",
        "spin_sample_rate",
        "spin_duration",
    ):
        assert panel.findChild(object, name) is not None, f"missing widget: {name}"


def test_filters_are_generic(qapp: QApplication) -> None:
    """No vendor / lab / person names in dialog filters (per #4480)."""
    for token in ("Wiffle", "ProV1", "TW_", "GW_"):
        assert token not in CLUB_FILE_FILTER
        assert token not in BODY_FILE_FILTER
    assert ".xlsx" in CLUB_FILE_FILTER
    assert ".mat" in CLUB_FILE_FILTER
    assert ".c3d" in CLUB_FILE_FILTER
    assert ".c3d" in BODY_FILE_FILTER


def test_align_options_round_trip(qapp: QApplication) -> None:
    panel = DataSourcesPanel()
    panel.spin_sample_rate.setValue(1500)
    panel.spin_duration.setValue(0.500)
    panel.rb_align_address.setChecked(True)
    opts = panel.align_options()
    assert opts.sample_rate_hz == pytest.approx(1500.0)
    assert opts.simulation_time_s == pytest.approx(0.5)
    assert opts.time_alignment == "address"


def test_snapshot_round_trip(qapp: QApplication) -> None:
    panel = DataSourcesPanel()
    panel.cb_club.setChecked(True)
    panel.rb_club_ball.setChecked(True)
    panel.combo_marker_set.setCurrentText("All markers")
    panel.spin_sample_rate.setValue(800)
    panel.spin_duration.setValue(0.250)
    panel.rb_align_impact.setChecked(True)
    block = panel.snapshot()
    assert block.club.enabled is True
    assert block.club.include_ball is True
    assert block.body.marker_set == "All markers"
    assert block.align.sample_rate_hz == pytest.approx(800.0)
    assert block.align.simulation_time_s == pytest.approx(0.25)
    # restore puts us back in the same state
    panel2 = DataSourcesPanel()
    panel2.restore(block)
    block2 = panel2.snapshot()
    assert block == block2


# ---------------------------------------------------------------------------
# Multi-source target assembly through the panel
# ---------------------------------------------------------------------------


def test_loading_body_plus_club_same_timegrid_yields_multi_source(
    qapp: QApplication,
) -> None:
    """When body + club share a timegrid, the panel emits a valid
    ``MultiSourceTarget`` whose ``shared_time()`` equals the club time
    array."""
    panel = DataSourcesPanel()
    club = make_target(n=128)
    body = _BodyStub(time=club.time.copy())

    received: list[Any] = []
    panel.targets_changed.connect(received.append)

    panel._force_set_club_target(club, "/tmp/example.c3d")
    panel._force_set_body_target(body, "/tmp/example.c3d")

    # The last emitted target should be a MultiSourceTarget with both slots.
    final = received[-1]
    assert isinstance(final, MultiSourceTarget)
    assert final.has_club() and final.has_body()
    assert np.array_equal(final.shared_time(), club.time)


def test_toggling_club_to_club_plus_ball_uses_extractor(
    qapp: QApplication,
) -> None:
    """Toggling Club → Club+Ball should call ``extract_ball_impact_from_clubtarget``
    on the cached ``ClubTarget`` rather than re-loading the file."""
    panel = DataSourcesPanel()
    club = make_target(n=64)
    panel._force_set_club_target(club, "/tmp/example.xlsx")

    fake_club_ball = _ClubBallStub(time=club.time.copy(), club=club, ball=object())

    received: list[Any] = []
    panel.targets_changed.connect(received.append)

    with patch(
        "src.tools.starting_pose_matcher.gui_source_panel._safe_extract_ball_impact",
        return_value=fake_club_ball,
    ) as extractor:
        panel.rb_club_ball.setChecked(True)

    # Extractor was invoked exactly once with the cached ClubTarget.
    extractor.assert_called_once_with(club)
    final = received[-1]
    assert isinstance(final, MultiSourceTarget)
    assert final.is_club_ball() is True
    assert final.club is fake_club_ball


def test_mismatched_timegrids_do_not_crash(qapp: QApplication) -> None:
    """Mismatched body/club timegrids must surface as a warning, not a crash."""
    panel = DataSourcesPanel()
    club = make_target(n=64)
    body = _BodyStub(time=np.linspace(0.0, 0.3, 65))  # off-by-one length
    panel._force_set_club_target(club, "/tmp/club.xlsx")

    # Patch QMessageBox.warning so the test doesn't pop a dialog.
    with patch(
        "src.tools.starting_pose_matcher.gui_source_panel.QMessageBox.warning"
    ) as warn:
        panel._force_set_body_target(body, "/tmp/body.c3d")

    assert warn.called, "Expected a QMessageBox.warning on mismatched timegrids"
    # No latest target — assembly failed cleanly.
    assert panel.current_targets() is None


def test_no_slots_emits_none(qapp: QApplication) -> None:
    panel = DataSourcesPanel()
    received: list[Any] = []
    panel.targets_changed.connect(received.append)
    # Toggle a checkbox with nothing loaded.
    panel.cb_club.setChecked(True)
    panel.cb_club.setChecked(False)
    assert received[-1] is None


# ---------------------------------------------------------------------------
# Adapter test (core.py multi-source dispatch)
# ---------------------------------------------------------------------------


def test_dispatch_cost_inputs_dict_shape() -> None:
    """The core.py dispatch helper exposes the right keys per slot."""
    from src.tools.starting_pose_matcher.core import dispatch_cost_inputs

    club = make_target(n=32)
    mst = MultiSourceTarget(club=club, body=None)
    inputs = dispatch_cost_inputs(mst)
    assert "time" in inputs
    assert "club" in inputs
    assert inputs["has_ball"] is False
    assert "body" not in inputs
