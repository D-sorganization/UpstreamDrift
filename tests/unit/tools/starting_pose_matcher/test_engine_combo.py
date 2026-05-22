"""Tests for the fit-engine combo in the starting-pose-matcher GUI.

Slice 1/3 of issue #4707: only the engine-selection combo is wired here.
The Run-fit QThread (slice 2) and Save-fit serialization (slice 3) are
covered by separate PRs.
"""

from __future__ import annotations

import os

# MUST set platform BEFORE any Qt import.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip(
    "PyQt6",
    reason="PyQt6 required for engine-combo UI tests",
    exc_type=ImportError,
)
pytest.importorskip(
    "PyQt6.QtWidgets",
    reason="PyQt6.QtWidgets not loadable in this environment",
    exc_type=ImportError,
)

from PyQt6.QtWidgets import QApplication  # noqa: E402

from src.shared.python.motion_matching import provider_registry  # noqa: E402
from src.tools.starting_pose_matcher import gui as gui_mod  # noqa: E402

pytestmark = pytest.mark.unit


class _StubProvider:
    def __init__(self, name: str) -> None:
        self.engine_name = name

    def fit_swing(self, target, opts):  # pragma: no cover - never called here
        raise AssertionError("fit_swing must not run in slice 1 tests")


class FakeComboBox:
    def __init__(self):
        self.items = []
        self._current_text = ""

    def blockSignals(self, b):
        pass

    def clear(self):
        self.items.clear()
        self._current_text = ""

    def addItems(self, items):
        self.items.extend(items)
        if items and not self._current_text:
            self._current_text = items[0]

    def setCurrentText(self, text):
        self._current_text = text

    def currentText(self):
        return self._current_text

    def count(self):
        return len(self.items)

    def itemText(self, i):
        return self.items[i]


class _Harness:
    """Tiny harness exercising the same combo + helpers without spinning
    up the full StartingPoseMatcher window.

    The combo helpers live on :class:`MainWidget` (via the
    :class:`_BuildersMixin`) since the Subtask 5 / #4998 refactor; the
    ``StartingPoseMatcher`` shell now delegates its body to
    :class:`MainWidget`.
    """

    _populate_engine_combo = gui_mod.MainWidget._populate_engine_combo
    selected_engine = gui_mod.MainWidget.selected_engine

    def __init__(self) -> None:
        self.combo_fit_engine = FakeComboBox()


@pytest.fixture
def _qapp():
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def _clean_registry():
    provider_registry.clear_registry()
    yield provider_registry
    provider_registry.clear_registry()


def test_populate_reads_canonical_registry(_qapp, _clean_registry):
    _clean_registry.register_provider(_StubProvider("alpha"))
    _clean_registry.register_provider(_StubProvider("mujoco"))
    h = _Harness()

    h._populate_engine_combo()

    items = [h.combo_fit_engine.itemText(i) for i in range(h.combo_fit_engine.count())]
    assert items == _clean_registry.available_engines()


def test_default_prefers_mujoco_when_present(_qapp, _clean_registry):
    _clean_registry.register_provider(_StubProvider("alpha"))
    _clean_registry.register_provider(_StubProvider("mujoco"))
    h = _Harness()

    h._populate_engine_combo()

    assert h.selected_engine == "mujoco"


def test_default_falls_back_to_first_engine(_qapp, _clean_registry):
    _clean_registry.register_provider(_StubProvider("alpha"))
    _clean_registry.register_provider(_StubProvider("zeta"))
    h = _Harness()

    h._populate_engine_combo()

    # available_engines() is sorted; "alpha" comes first.
    assert h.selected_engine == "alpha"


def test_repopulate_picks_up_late_registration(_qapp, _clean_registry):
    _clean_registry.register_provider(_StubProvider("alpha"))
    h = _Harness()
    h._populate_engine_combo()
    assert h.selected_engine == "alpha"

    _clean_registry.register_provider(_StubProvider("mujoco"))
    h._populate_engine_combo()

    items = [h.combo_fit_engine.itemText(i) for i in range(h.combo_fit_engine.count())]
    assert items == ["alpha", "mujoco"]
    assert h.selected_engine == "mujoco"


def test_selected_engine_raises_when_registry_empty(_qapp, _clean_registry):
    h = _Harness()

    h._populate_engine_combo()

    with pytest.raises(RuntimeError, match="empty"):
        _ = h.selected_engine
