"""Regression tests for dataset list-features query handling."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from src.api.routes import dataset


@pytest.mark.asyncio
async def test_list_features_accepts_default_none_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`category` defaults to None and must not raise."""
    engine = object()
    expected = [{"name": "feature-a"}]
    captured: dict[str, object] = {}

    class FakeRegistry:
        def __init__(self, registry_engine: object) -> None:
            captured["engine"] = registry_engine

        def list_features(
            self, category: str | None = None, available_only: bool = False
        ) -> list[dict[str, str]]:
            captured["category"] = category
            captured["available_only"] = available_only
            return expected

    monkeypatch.setattr(dataset, "_require_active_engine", lambda _: engine)
    monkeypatch.setattr(
        "src.shared.python.control_features_registry.ControlFeaturesRegistry",
        FakeRegistry,
    )

    result = await dataset.list_features(engine_manager=MagicMock())

    assert result == expected
    assert captured["engine"] is engine
    assert captured["category"] is None
    assert captured["available_only"] is False


@pytest.mark.asyncio
async def test_list_features_accepts_explicit_none_category(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit `category=None` must be passed through safely."""
    captured: dict[str, object] = {}

    class FakeRegistry:
        def __init__(self, _: object) -> None:
            pass

        def list_features(
            self, category: str | None = None, available_only: bool = False
        ) -> list[dict[str, object]]:
            captured["category"] = category
            captured["available_only"] = available_only
            return []

    monkeypatch.setattr(dataset, "_require_active_engine", lambda _: object())
    monkeypatch.setattr(
        "src.shared.python.control_features_registry.ControlFeaturesRegistry",
        FakeRegistry,
    )

    result = await dataset.list_features(
        category=None,
        available_only=True,
        engine_manager=MagicMock(),
    )

    assert result == []
    assert captured["category"] is None
    assert captured["available_only"] is True
