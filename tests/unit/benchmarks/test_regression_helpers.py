from __future__ import annotations

import json
from pathlib import Path

import pytest

from tests.benchmarks import regression_helpers as mod

pytestmark = pytest.mark.unit


def test_regression_threshold_uses_absolute_floor_for_tiny_baselines(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"measurements": {"tiny_hot_path": 1.6e-6}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "BASELINE_PATH", baseline)

    mod.assert_within_regression_threshold("tiny_hot_path", 8.1e-6)


def test_regression_threshold_still_fails_above_absolute_floor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    baseline = tmp_path / "baseline.json"
    baseline.write_text(
        json.dumps({"measurements": {"tiny_hot_path": 1.6e-6}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(mod, "BASELINE_PATH", baseline)

    with pytest.raises(AssertionError, match="tiny_hot_path"):
        mod.assert_within_regression_threshold("tiny_hot_path", 10.1e-6)
