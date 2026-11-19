"""Procedural and opt-in MyoHand backends coexist after main integration."""

import pytest

from src.shared.python.model_appearance.hand_backends import registry
from src.shared.python.model_appearance.hand_backends import settings
from src.shared.python.model_appearance.hand_backends.myohand import (
    myohand_assets_ready,
)

pytestmark = [pytest.mark.unit, pytest.mark.integration]


def test_builtin_backends_keep_their_distinct_default_enablement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(registry, "_REGISTRY", {})
    monkeypatch.setattr(registry, "_builtins_loaded", False)

    enabled = registry.list_hand_visual_backends()
    all_names = registry.list_hand_visual_backends(include_disabled=True)

    assert "procedural_skin" in enabled
    assert "myohand" not in enabled
    assert "procedural_skin" in all_names
    assert ("myohand" in all_names) == myohand_assets_ready()
    if myohand_assets_ready():
        assert registry.get_hand_visual_backend("myohand").default_enabled is False


def test_both_backend_settings_remain_available() -> None:
    procedural = settings.DEFAULT_HAND_VISUAL_SETTINGS
    myohand = settings.default_myohand_settings()

    assert isinstance(procedural, settings.HandVisualSettings)
    assert isinstance(myohand, settings.MyoHandVisualSettings)
    assert procedural.anthropometry == settings.HandAnthropometryReference()
    assert myohand.research_only is True
    assert myohand.license_gate_issue == 13763
