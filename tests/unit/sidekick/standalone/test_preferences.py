"""T8 TDD: StandalonePreferences round-trip and DbC preconditions.

Marker: headless_safe — no display or PyQt6 required.
Uses an in-memory FakeSessionStore so nothing is written to ~/.config.
"""

from __future__ import annotations

import pytest

from sidekick.standalone.preferences import (
    DEFAULT_DATA_DIR,
    DEFAULT_LLM_PROVIDER,
    DEFAULT_PROFILE,
    VALID_PROFILES,
    StandalonePreferences,
)
from sidekick.standalone.session_store import InMemorySessionStore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _prefs() -> StandalonePreferences:
    """Return a fresh Preferences backed by an isolated in-memory store."""
    return StandalonePreferences(store=InMemorySessionStore())


# ---------------------------------------------------------------------------
# T8-AC-1: round-trip all four settings through the store
# ---------------------------------------------------------------------------


@pytest.mark.headless_safe
def test_profile_default() -> None:
    p = _prefs()
    assert p.profile() == DEFAULT_PROFILE


@pytest.mark.headless_safe
def test_profile_round_trip() -> None:
    p = _prefs()
    p.set_profile("calc-first")
    assert p.profile() == "calc-first"


@pytest.mark.headless_safe
def test_theme_default() -> None:
    p = _prefs()
    assert isinstance(p.theme(), str) and p.theme()


@pytest.mark.headless_safe
def test_theme_round_trip() -> None:
    p = _prefs()
    p.set_theme("Monokai")
    assert p.theme() == "Monokai"


@pytest.mark.headless_safe
def test_data_dir_default(tmp_path) -> None:
    p = _prefs()
    assert p.data_dir() == DEFAULT_DATA_DIR


@pytest.mark.headless_safe
def test_data_dir_round_trip(tmp_path) -> None:
    p = _prefs()
    p.set_data_dir(str(tmp_path))
    assert p.data_dir() == str(tmp_path)


@pytest.mark.headless_safe
def test_llm_provider_default() -> None:
    p = _prefs()
    assert p.llm_provider() == DEFAULT_LLM_PROVIDER


@pytest.mark.headless_safe
def test_llm_provider_round_trip() -> None:
    p = _prefs()
    p.set_llm_provider("codex")
    assert p.llm_provider() == "codex"


# ---------------------------------------------------------------------------
# T8-AC-2: persistence across store reload (same underlying store)
# ---------------------------------------------------------------------------


@pytest.mark.headless_safe
def test_preferences_survive_reload() -> None:
    store = InMemorySessionStore()
    p1 = StandalonePreferences(store=store)
    p1.set_profile("calc-first")
    p1.set_theme("Dracula")

    # Second instance using the same store simulates a restart
    p2 = StandalonePreferences(store=store)
    assert p2.profile() == "calc-first"
    assert p2.theme() == "Dracula"


# ---------------------------------------------------------------------------
# T8-AC-3: DbC — invalid values raise at save time
# ---------------------------------------------------------------------------


@pytest.mark.headless_safe
def test_invalid_profile_raises() -> None:
    p = _prefs()
    with pytest.raises(ValueError, match="profile"):
        p.set_profile("flying-first")


@pytest.mark.headless_safe
def test_invalid_data_dir_raises(tmp_path) -> None:
    """Non-string data dir must raise TypeError."""
    p = _prefs()
    with pytest.raises(TypeError):
        p.set_data_dir(12345)  # type: ignore[arg-type]


@pytest.mark.headless_safe
def test_empty_llm_provider_raises() -> None:
    p = _prefs()
    with pytest.raises(ValueError, match="provider"):
        p.set_llm_provider("")


# ---------------------------------------------------------------------------
# T8-AC-4: LOD — typed getters, no deep-chain access needed
# ---------------------------------------------------------------------------


@pytest.mark.headless_safe
def test_prefs_exposes_only_typed_getters() -> None:
    """Callers should not need to access _store or _raw directly."""
    p = _prefs()
    # All public access via typed getters
    _ = p.profile()
    _ = p.theme()
    _ = p.data_dir()
    _ = p.llm_provider()


# ---------------------------------------------------------------------------
# T8-AC-5: valid profiles constant is up to date
# ---------------------------------------------------------------------------


@pytest.mark.headless_safe
def test_valid_profiles_contains_required() -> None:
    assert "chat-first" in VALID_PROFILES
    assert "calc-first" in VALID_PROFILES
