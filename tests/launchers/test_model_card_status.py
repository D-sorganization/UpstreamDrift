"""Desktop tile status chip reflects real target resolvability (issue #8855).

A tile whose declared launch target does not resolve must not render the
green "Ready" chip, no matter what optimistic status the YAML declares.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pytest

pytestmark = pytest.mark.unit


@dataclass
class _FakeLauncherMeta:
    status: str = "ready"
    category: str = "tool"
    logo: str = "golf_logo.svg"


@dataclass
class _FakeModel:
    id: str
    name: str = "Fake Tool"
    description: str = "A fake tool for status-chip tests"
    type: str = "special_app"
    path: str = ""
    provider: str | None = None
    source_root: str | None = None
    launcher: _FakeLauncherMeta = field(default_factory=_FakeLauncherMeta)


def _make_card(qapp, model):
    """Build a card shell without running the full widget/theme pipeline.

    ``_get_status_info`` only needs ``model`` and the lazily computed
    ``_target_resolvable`` slot, so the UI construction (which pulls in the
    theme stack) is deliberately bypassed for this unit test.
    """
    from src.launchers.model_card import DraggableModelCard

    card = DraggableModelCard.__new__(DraggableModelCard)
    card.model = model
    card._target_resolvable = None
    return card


class TestStatusChipHonesty:
    def test_dead_path_tile_is_not_ready(self, qapp) -> None:
        model = _FakeModel(
            id="dead_tile",
            path="src/tools/does_not_exist_anywhere/__main__.py",
        )
        card = _make_card(qapp, model)
        text, css_class = card._get_status_info()
        assert text == "Unavailable"
        assert css_class == "error"

    def test_existing_path_tile_keeps_declared_status(self, qapp) -> None:
        model = _FakeModel(
            id="live_tile",
            path="src/tools/pose_studio/__main__.py",
        )
        card = _make_card(qapp, model)
        text, css_class = card._get_status_info()
        assert text == "Ready"
        assert css_class == "success"

    def test_virtual_target_keeps_declared_status(self, qapp) -> None:
        model = _FakeModel(id="virtual_tile", path="virtual/matlab_suite")
        card = _make_card(qapp, model)
        text, _ = card._get_status_info()
        assert text == "Ready"

    def test_pathless_exercise_model_keeps_declared_status(self, qapp) -> None:
        model = _FakeModel(id="gait_tile", type="biomech_exercise", path="")
        card = _make_card(qapp, model)
        text, _ = card._get_status_info()
        assert text == "Ready"
