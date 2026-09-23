"""Tests for :func:`src.tools.window_theme.apply_theme_best_effort`."""

from __future__ import annotations

import sys
import types
from unittest import mock

import pytest

from src.tools.window_theme import apply_theme_best_effort

_THEME_MODULE = "src.shared.python.sidekick.theme"

pytestmark = pytest.mark.unit


def _fake_theme(apply: object) -> types.ModuleType:
    module = types.ModuleType(_THEME_MODULE)
    module.apply_theme_to_window = apply  # type: ignore[attr-defined]
    return module


def test_applies_theme_to_window() -> None:
    apply = mock.Mock()
    window = object()
    with mock.patch.dict(sys.modules, {_THEME_MODULE: _fake_theme(apply)}):
        apply_theme_best_effort(window)
    apply.assert_called_once_with(window)


def test_missing_theme_module_is_not_fatal() -> None:
    with mock.patch.dict(sys.modules, {_THEME_MODULE: None}):
        apply_theme_best_effort(object())  # must not raise


@pytest.mark.parametrize("error", [RuntimeError, AttributeError, TypeError, ValueError])
def test_theme_application_errors_are_swallowed(error: type[Exception]) -> None:
    apply = mock.Mock(side_effect=error("boom"))
    with mock.patch.dict(sys.modules, {_THEME_MODULE: _fake_theme(apply)}):
        apply_theme_best_effort(object())  # must not raise
    apply.assert_called_once()


def test_unexpected_errors_propagate() -> None:
    apply = mock.Mock(side_effect=KeyError("bug"))
    with mock.patch.dict(sys.modules, {_THEME_MODULE: _fake_theme(apply)}):
        with pytest.raises(KeyError):
            apply_theme_best_effort(object())
