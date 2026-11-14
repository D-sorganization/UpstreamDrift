"""Runbook contract for CI Optional Stack slice 1 — Pinocchio gravity (#14227)."""

from __future__ import annotations

from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[2]
_RUNBOOK = _REPO / "docs" / "ci" / "optional_stack_14227.md"
_MODEL_PROVIDER = _REPO / "src/shared/python/optimization/model_provider.py"
_REGRESSION_TEST = _REPO / "tests/unit/optimization/test_model_provider.py"


def test_optional_stack_14227_runbook_exists() -> None:
    assert _RUNBOOK.is_file(), "docs/ci/optional_stack_14227.md must document slice 1"


@pytest.mark.parametrize(
    "needle",
    [
        "GRAVITY_M_S2",
        "build_pinocchio_model",
        "test_pinocchio_model_uses_the_shared_gravity_constant",
        "model_provider.py",
        "Slice 2",
        "Owner-Blocked",
        "must not",
    ],
)
def test_runbook_names_gravity_fix_and_owner_blocked_slice(needle: str) -> None:
    text = _RUNBOOK.read_text(encoding="utf-8")
    assert needle in text


def test_pinocchio_builder_sets_shared_gravity_on_main() -> None:
    text = _MODEL_PROVIDER.read_text(encoding="utf-8")
    assert "GRAVITY_M_S2" in text
    assert "build_pinocchio_model" in text
    assert "model.gravity.linear = np.array([0.0, 0.0, -float(GRAVITY_M_S2)])" in text


def test_regression_test_symbol_is_present() -> None:
    text = _REGRESSION_TEST.read_text(encoding="utf-8")
    assert "def test_pinocchio_model_uses_the_shared_gravity_constant" in text
    assert "GRAVITY_M_S2" in text
