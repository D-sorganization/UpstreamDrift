"""Configuration defaults must survive expected file and parser failures."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.config.config_utils import (
    ConfigLoader,
    load_json_config,
    load_yaml_config,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("load", [load_json_config, load_yaml_config])
@pytest.mark.parametrize("failure", ["syntax", "encoding", "directory"])
@pytest.mark.parametrize("custom_default", [False, True])
def test_load_failure_returns_fresh_default_without_changing_inputs(
    tmp_path: Path,
    load: Callable[..., dict[str, Any]],
    failure: str,
    custom_default: bool,
) -> None:
    target = tmp_path / "config"
    if failure == "directory":
        target.mkdir()
        original = None
    else:
        original = b"[broken" if failure == "syntax" else b"\xff"
        target.write_bytes(original)
    default = {"mode": "safe"} if custom_default else None

    result = load(target, default=default)

    assert result == ({"mode": "safe"} if custom_default else {})
    assert result is not default
    result["mode"] = "changed by caller"
    assert default == ({"mode": "safe"} if custom_default else None)
    if original is not None:
        assert target.read_bytes() == original
    else:
        assert target.is_dir()


@pytest.mark.parametrize("load", [load_json_config, load_yaml_config])
def test_failed_loads_do_not_share_empty_fallbacks(
    tmp_path: Path, load: Callable[..., dict[str, Any]]
) -> None:
    target = tmp_path / "invalid"
    target.write_text("[broken", encoding="utf-8")

    first = load(target)
    first["leaked"] = True

    assert load(target) == {}


@pytest.mark.parametrize("format_name", ["json", "yaml"])
def test_config_loader_can_recover_after_invalid_file_is_repaired(
    tmp_path: Path, format_name: str
) -> None:
    target = tmp_path / f"settings.{format_name}"
    target.write_text("[broken", encoding="utf-8")
    loader = ConfigLoader(target, format=format_name)
    default = {"mode": "safe"}

    assert loader.load(default) == default
    default["mode"] = "changed by caller"
    assert loader.load() == {"mode": "safe"}
    target.write_text('{"mode": "loaded"}', encoding="utf-8")

    assert loader.load(use_cache=False) == {"mode": "loaded"}
