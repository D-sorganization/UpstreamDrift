"""Tests for the real-Pinocchio guard used by the inverse-dynamics tests.

Evidence class: mocked (fake modules); independent of an installed Pinocchio.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest import mock

import pytest

from tests.unit.motion_pipeline.matching_completeness._pinocchio_guard import (
    BrokenPinocchioError,
    mock_skip_reason,
)


def test_magicmock_module_is_a_typed_skip() -> None:
    assert "MagicMock" in (mock_skip_reason(mock.MagicMock()) or "")


def test_magicmock_model_is_a_typed_skip() -> None:
    fake = SimpleNamespace(Model=mock.MagicMock())
    assert "MagicMock" in (mock_skip_reason(fake) or "")


def test_broken_real_constructor_fails() -> None:
    def _boom() -> None:
        raise OSError("libpinocchio.so: undefined symbol")

    with pytest.raises(BrokenPinocchioError, match="undefined symbol"):
        mock_skip_reason(SimpleNamespace(Model=_boom))


def test_non_int_nq_on_real_module_fails() -> None:
    fake = SimpleNamespace(Model=lambda: SimpleNamespace(nq="7"))
    with pytest.raises(BrokenPinocchioError, match="expected int"):
        mock_skip_reason(fake)


def test_healthy_real_module_passes() -> None:
    assert (
        mock_skip_reason(SimpleNamespace(Model=lambda: SimpleNamespace(nq=0))) is None
    )
