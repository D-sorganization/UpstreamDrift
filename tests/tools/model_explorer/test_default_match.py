"""Where the Model Explorer finds its default matched swing (#15150)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools.model_explorer.default_match import (
    BUNDLE_ENV,
    CONFIG_ENV,
    DEMO_BUNDLE,
    default_match,
    match_scenarios,
    schedule_default_match,
    user_config_path,
)


@pytest.mark.unit
def test_without_settings_the_demo_bundle_is_offered_and_labelled() -> None:
    env = {CONFIG_ENV: "/nonexistent/matched_swings.json"}
    scenario = default_match(env)
    assert DEMO_BUNDLE.is_file()
    assert scenario is not None and scenario.path == DEMO_BUNDLE
    assert scenario.is_demo and "not the best match" in scenario.label


@pytest.mark.unit
def test_env_bundle_wins_then_config_default_then_scenarios(tmp_path: Path) -> None:
    env_bundle, driver, iron = (tmp_path / f"{n}.npz" for n in ("e", "d", "i"))
    for path in (env_bundle, driver, iron):
        path.write_bytes(b"npz")
    config = tmp_path / "matched_swings.json"
    named = {"Driver": str(driver), "7-Iron": str(iron)}
    config.write_text(json.dumps({"default": str(driver), "scenarios": named}))
    env = {BUNDLE_ENV: str(env_bundle), CONFIG_ENV: str(config)}
    scenarios = match_scenarios(env)
    assert [s.path for s in scenarios] == [env_bundle, driver, iron]
    assert [s.origin for s in scenarios] == ["env", "user config", "user config"]
    assert scenarios[2].label == "7-Iron (user config)"
    assert not any(s.is_demo for s in scenarios)


@pytest.mark.unit
def test_missing_files_and_bad_config_are_skipped(tmp_path: Path) -> None:
    config = tmp_path / "matched_swings.json"
    config.write_text("{not json")
    env = {BUNDLE_ENV: str(tmp_path / "gone.npz"), CONFIG_ENV: str(config)}
    scenarios = match_scenarios(env)
    assert [s.origin for s in scenarios] == ["demo"]


@pytest.mark.unit
def test_user_config_path_follows_xdg_and_scheduling_needs_a_loader(
    tmp_path: Path,
) -> None:
    path = user_config_path({"XDG_CONFIG_HOME": str(tmp_path)})
    assert path == tmp_path / "upstreamdrift" / "matched_swings.json"
    assert schedule_default_match(object()) is False
