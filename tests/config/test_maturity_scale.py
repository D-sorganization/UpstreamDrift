"""Tests for the one launcher maturity scale and ``min_maturity`` filter (#15150)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from src.config.launcher_manifest_loader import LauncherManifest
from src.shared.python.config.maturity_scale import (
    default_min_maturity,
    filter_models_by_maturity,
    filter_order,
    maturity_for_status,
    meets_min_maturity,
    resolve_model_maturity,
)
from src.shared.python.config.typed_settings import Settings, get_settings

pytestmark = pytest.mark.unit

RUNGS = ["experimental", "prototype", "beta", "stable"]


def test_scale_is_ordered_with_deprecated_lowest() -> None:
    assert list(filter_order()) == ["deprecated", *RUNGS]


@pytest.mark.parametrize("floor", RUNGS)
def test_rungs_pass_only_at_or_above_floor(floor: str) -> None:
    for tile in RUNGS:
        assert meets_min_maturity(tile, floor) is (
            RUNGS.index(tile) >= RUNGS.index(floor)
        )


def test_deprecated_shown_only_at_default_floor() -> None:
    assert meets_min_maturity("deprecated", "deprecated")
    assert not any(meets_min_maturity("deprecated", f) for f in RUNGS)
    assert all(meets_min_maturity(t, "deprecated") for t in RUNGS)


@pytest.mark.parametrize("args", [("bogus", "beta"), ("beta", "bogus"), ("", "beta")])
def test_unknown_level_raises(args: tuple[str, str]) -> None:
    with pytest.raises(ValueError, match="must be one of"):
        meets_min_maturity(*args)


def test_status_mapping_reconciles_vocabularies() -> None:
    assert maturity_for_status("ready") == "stable"
    assert maturity_for_status("beta") == "beta"
    assert maturity_for_status("experimental") == "experimental"
    assert maturity_for_status("broken") == "stable"  # never hide by guessing
    assert maturity_for_status(None) == "stable"


def test_declared_maturity_wins_over_status() -> None:
    declared = SimpleNamespace(launcher={"maturity": "prototype", "status": "ready"})
    by_status = SimpleNamespace(launcher={"status": "beta"})
    assert resolve_model_maturity(declared) == "prototype"
    assert resolve_model_maturity(by_status) == "beta"


def test_filter_models_by_maturity() -> None:
    models = [
        SimpleNamespace(id=i, launcher={"status": s})
        for i, s in [("a", "experimental"), ("b", "beta"), ("c", "ready")]
    ]
    assert [m.id for m in filter_models_by_maturity(models, "beta")] == ["b", "c"]
    with pytest.raises(ValueError):
        filter_models_by_maturity(models, "bogus")


def test_setting_default_shows_everything(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MIN_MATURITY", raising=False)
    assert get_settings().min_maturity == default_min_maturity() == "deprecated"


def test_setting_reads_env_and_rejects_unknown(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MIN_MATURITY", "beta")
    assert get_settings().min_maturity == "beta"
    monkeypatch.setenv("MIN_MATURITY", "bogus")
    with pytest.raises(ValueError):
        Settings()


def test_loader_default_keeps_every_visible_tile(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MIN_MATURITY", raising=False)
    manifest = LauncherManifest.load()
    assert manifest.visible_tiles_meeting() == manifest.visible_tiles


def test_loader_filter_hides_below_floor() -> None:
    manifest = LauncherManifest.load()
    kept = manifest.visible_tiles_meeting("beta")
    assert kept and len(kept) <= len(manifest.visible_tiles)
    assert {t.maturity for t in kept} <= {"beta", "stable"}
    with pytest.raises(ValueError):
        manifest.visible_tiles_meeting("bogus")
