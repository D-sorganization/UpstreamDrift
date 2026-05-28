"""TDD coverage for the new ``docker_profile_info`` module.

Today's launcher tier-details panel relies on
``load_docker_profiles()`` + ``format_profile_summary()`` to render the
``Manage Environment`` and ``Settings`` panels. The previous launcher
showed only a vague tier name ("Professional", "Research"); the new
behaviour resolves the full feature list out of ``docker/profiles.yaml``
and joins it with the canonical feature registry so the dialog can
display *exactly* what each tier installs.

These tests are pure unit tests — they neither launch Docker nor
invoke the UI; they use a synthetic YAML on a tmp path so they survive
edits to the real ``profiles.yaml``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from src.launchers import docker_profile_info as dpi
from src.launchers.docker_profile_info import (
    ProfileInfo,
    format_profile_summary,
    load_docker_profiles,
)


# ── _resolve_features ----------------------------------------------------


def test_resolve_features_flat_profile() -> None:
    raw = {"slim": {"features": ["api", "pendulum"]}}
    assert dpi._resolve_features(raw, "slim") == ["api", "pendulum"]


def test_resolve_features_walks_extends_chain() -> None:
    raw = {
        "slim": {"features": ["api"]},
        "standard": {"extends": "slim", "features": ["mujoco"]},
        "research": {"extends": "standard", "features": ["drake"]},
    }
    assert dpi._resolve_features(raw, "research") == ["api", "mujoco", "drake"]


def test_resolve_features_deduplicates() -> None:
    raw = {
        "a": {"features": ["x"]},
        "b": {"extends": "a", "features": ["x", "y"]},
    }
    # 'x' must not appear twice even though both levels list it.
    assert dpi._resolve_features(raw, "b") == ["x", "y"]


def test_resolve_features_rejects_cycles() -> None:
    raw = {
        "a": {"extends": "b", "features": []},
        "b": {"extends": "a", "features": []},
    }
    with pytest.raises(ValueError, match="cycle"):
        dpi._resolve_features(raw, "a")


def test_resolve_features_returns_empty_for_unknown_profile() -> None:
    assert dpi._resolve_features({}, "nope") == []


# ── load_docker_profiles -------------------------------------------------


def _write_yaml(tmp: Path, content: str) -> Path:
    p = tmp / "profiles.yaml"
    p.write_text(content, encoding="utf-8")
    return p


def test_load_real_profiles_yaml_resolves_every_tier() -> None:
    """Smoke: the real shipped YAML loads and every tier resolves."""
    out = load_docker_profiles()
    # Skip the test if the YAML moved (CI fail-safe). The contract is
    # "every tier in the YAML is reachable", not "there are exactly N".
    if not out:
        pytest.skip("docker/profiles.yaml not present in this checkout")
    for name, info in out.items():
        assert isinstance(info, ProfileInfo)
        assert info.name == name
        assert info.max_size_mb > 0
        assert info.feature_names  # extends chain must resolve to >=1 feature


def test_load_docker_profiles_missing_yaml_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(dpi, "_PROFILES_PATH", tmp_path / "does-not-exist.yaml")
    assert load_docker_profiles() == {}


def test_load_docker_profiles_malformed_yaml_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    p = _write_yaml(tmp_path, "this: is: : not: valid: yaml: [\n")
    monkeypatch.setattr(dpi, "_PROFILES_PATH", p)
    assert load_docker_profiles() == {}


def test_load_docker_profiles_non_mapping_top_level_returns_empty(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A YAML whose top-level is a list/string must not crash startup.

    Regression for #6538: removing the ``isinstance(data, dict)`` guard
    let a malformed profiles.yaml propagate ``AttributeError`` through
    ``data.get(...)`` and take down the launcher dialog.
    """
    # Top-level list.
    p = _write_yaml(tmp_path, "- not\n- a\n- mapping\n")
    monkeypatch.setattr(dpi, "_PROFILES_PATH", p)
    assert load_docker_profiles() == {}

    # Top-level scalar string.
    p2 = _write_yaml(tmp_path, "just a string\n")
    monkeypatch.setattr(dpi, "_PROFILES_PATH", p2)
    assert load_docker_profiles() == {}


def test_load_docker_profiles_unknown_feature_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A typo'd feature name must not crash the loader.

    The UI prefers to render a partial profile rather than no profile
    at all, so unknown names are dropped with a logged warning.
    """
    yaml_content = (
        "version: 1\n"
        "profiles:\n"
        "  weird:\n"
        "    description: typo tier\n"
        "    features: [api, this-feature-does-not-exist-xyz]\n"
        "    max_size_mb: 100\n"
    )
    p = _write_yaml(tmp_path, yaml_content)
    monkeypatch.setattr(dpi, "_PROFILES_PATH", p)

    out = load_docker_profiles()
    assert "weird" in out
    weird = out["weird"]
    # The unknown name is preserved in ``feature_names`` (the source of
    # truth for "what the YAML asked for"), but the ``Feature``
    # metadata list omits it (the source of truth for "what we can
    # actually describe to the user").
    assert "this-feature-does-not-exist-xyz" in weird.feature_names
    assert all(f.name != "this-feature-does-not-exist-xyz" for f in weird.features)


def test_load_docker_profiles_cycle_profile_is_skipped(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    yaml_content = (
        "version: 1\n"
        "profiles:\n"
        "  a:\n"
        "    description: A\n"
        "    extends: b\n"
        "    max_size_mb: 100\n"
        "  b:\n"
        "    description: B\n"
        "    extends: a\n"
        "    max_size_mb: 100\n"
        "  ok:\n"
        "    description: standalone\n"
        "    features: [api]\n"
        "    max_size_mb: 100\n"
    )
    p = _write_yaml(tmp_path, yaml_content)
    monkeypatch.setattr(dpi, "_PROFILES_PATH", p)

    out = load_docker_profiles()
    # Cyclic profiles dropped, standalone survives.
    assert "a" not in out and "b" not in out
    assert "ok" in out


# ── format_profile_summary ----------------------------------------------


def _make_info(**overrides: Any) -> ProfileInfo:
    defaults: dict[str, Any] = {
        "name": "standard",
        "description": "Default research build",
        "max_size_mb": 2200,
        "feature_names": ("api", "pendulum"),
        "features": (),
        "approx_total_mb": 0,
    }
    defaults.update(overrides)
    return ProfileInfo(**defaults)


def test_format_profile_summary_includes_title_and_description() -> None:
    summary = format_profile_summary(_make_info())
    assert "Standard" in summary
    assert "Default research build" in summary


def test_format_profile_summary_shows_budget_when_set() -> None:
    summary = format_profile_summary(_make_info(approx_total_mb=450))
    assert "≤ 2200 MB" in summary
    assert "~450 MB" in summary


def test_format_profile_summary_falls_back_to_feature_names_when_no_metadata() -> None:
    summary = format_profile_summary(
        _make_info(feature_names=("api", "pendulum"), features=())
    )
    # No Feature[] metadata available — must still tell the user what's in it.
    assert "api" in summary and "pendulum" in summary


def test_format_profile_summary_no_max_size_omits_budget_line() -> None:
    summary = format_profile_summary(
        _make_info(max_size_mb=0, feature_names=(), features=())
    )
    assert "Budget" not in summary
