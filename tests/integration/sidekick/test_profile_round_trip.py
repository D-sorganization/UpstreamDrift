"""State-profile round-trip between embedded and standalone Sidekick.

Covers UD #5983 (epic #5969). Both sides MUST be able to load each other's
profiles deeply-equal, and a legacy embedded profile (no ``schema_version``)
must emit exactly one ``SchemaMigration`` warning on load.

Uses only the public ``save_profile`` / ``load_profile`` APIs of each side —
no reaching into private writers (LOD).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sidekick.persistence import (
    PROFILE_SCHEMA_VERSION,
    PROFILE_SCHEMA_VERSION_KEY,
    ProfilePayload,
    SchemaMigration,
    current_schema_version,
    unwrap_payload,
    validate,
    wrap_state,
)
from sidekick.standalone.session_store import StandaloneSessionStore
from sidekick.ui.tools_sidebar.state import SidebarState
from sidekick.ui.tools_sidebar.state_profiles import (
    SidekickStateProfileStore,
)

pytestmark = [pytest.mark.integration, pytest.mark.headless_safe]


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _sample_state() -> SidebarState:
    """Return a non-default ``SidebarState`` exercising many fields."""
    return SidebarState(
        dock_area="left",
        floating=True,
        minimized=False,
        width=420,
        height=900,
        active_tab="calculator",
        layout_mode="matlab_home",
        tab_order=["chat", "calculator", "notes"],
        default_visible_tabs=["chat", "calculator"],
        default_hidden_tabs=["jupyter"],
        hidden_tabs=["jupyter"],
        popped_out_tabs=[],
        tab_display_names={"chat": "AI", "calculator": "Calc"},
        tab_settings={"chat": {"font_size": 14}},
        calculator_predictive_text_enabled=True,
    )


@pytest.fixture
def embedded_store(tmp_path: Path) -> SidekickStateProfileStore:
    return SidekickStateProfileStore(tmp_path / "shared")


@pytest.fixture
def standalone_store(tmp_path: Path) -> StandaloneSessionStore:
    return StandaloneSessionStore(tmp_path / "shared")


# ---------------------------------------------------------------------------
# Canonical helper unit-ish coverage (still uses public surface)
# ---------------------------------------------------------------------------


def test_wrap_state_attaches_current_schema_version() -> None:
    state = _sample_state().to_dict()
    payload = wrap_state(state)
    assert payload.schema_version == PROFILE_SCHEMA_VERSION == current_schema_version()
    # Round-trips through dict form.
    rehydrated = ProfilePayload.from_dict(payload.to_dict())
    assert rehydrated.data == payload.data
    assert rehydrated.schema_version == payload.schema_version


def test_validate_rejects_missing_schema_version() -> None:
    with pytest.raises(ValueError, match=r"\$\.schema_version"):
        validate({"dock_area": "right"})


def test_validate_rejects_non_int_schema_version() -> None:
    with pytest.raises(ValueError, match=r"\$\.schema_version"):
        validate({PROFILE_SCHEMA_VERSION_KEY: "v1"})


def test_validate_accepts_unknown_future_version() -> None:
    # Forward-compat: a newer version is loadable; unknown keys preserved.
    validate({PROFILE_SCHEMA_VERSION_KEY: PROFILE_SCHEMA_VERSION + 99})


# ---------------------------------------------------------------------------
# Round-trip: embedded → standalone
# ---------------------------------------------------------------------------


def test_embedded_save_then_standalone_load_deep_equal(
    embedded_store: SidekickStateProfileStore,
    standalone_store: StandaloneSessionStore,
) -> None:
    """A profile saved by embedded mode is loadable by standalone, deep-equal."""
    state = _sample_state()
    saved = embedded_store.save_profile("shared_profile", state)
    assert saved.ok and saved.path is not None

    reloaded = standalone_store.load_profile("shared_profile")

    assert reloaded.data == state.to_dict()
    assert reloaded.schema_version == PROFILE_SCHEMA_VERSION


# ---------------------------------------------------------------------------
# Round-trip: standalone → embedded
# ---------------------------------------------------------------------------


def test_standalone_save_then_embedded_load_deep_equal(
    embedded_store: SidekickStateProfileStore,
    standalone_store: StandaloneSessionStore,
) -> None:
    """A profile saved by standalone is loadable by embedded, deep-equal."""
    state = _sample_state()
    standalone_store.save_profile("shared_profile", wrap_state(state.to_dict()))

    result = embedded_store.load_profile("shared_profile")
    assert result.ok, result.message
    assert result.state is not None
    assert result.state.to_dict() == state.to_dict()


# ---------------------------------------------------------------------------
# Forward-compat: SchemaMigration warning + unknown-key preservation
# ---------------------------------------------------------------------------


def test_legacy_profile_load_emits_exactly_one_schema_migration_warning() -> None:
    legacy = _sample_state().to_dict()  # No schema_version key.
    assert PROFILE_SCHEMA_VERSION_KEY not in legacy

    with pytest.warns(SchemaMigration) as record:
        state_dict, version = unwrap_payload(legacy)

    migrations = [w for w in record.list if issubclass(w.category, SchemaMigration)]
    assert len(migrations) == 1, "must emit exactly one SchemaMigration warning"
    assert version == PROFILE_SCHEMA_VERSION
    # State data preserved verbatim.
    assert state_dict == legacy


def test_unknown_top_level_keys_preserved_on_resave(
    standalone_store: StandaloneSessionStore,
) -> None:
    """A future-side key is preserved through load → resave (no silent drop)."""
    state = _sample_state().to_dict()
    state["future_feature_blob"] = {"flag": True, "tuned": [1, 2, 3]}

    payload = wrap_state(state)
    standalone_store.save_profile("forward_compat", payload)
    reloaded = standalone_store.load_profile("forward_compat")

    assert reloaded.data.get("future_feature_blob") == {
        "flag": True,
        "tuned": [1, 2, 3],
    }


# ---------------------------------------------------------------------------
# Fixture-matrix property-style coverage (no hypothesis dep)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "overrides",
    [
        {"dock_area": "left"},
        {"dock_area": "right", "floating": True},
        {"width": 1024, "height": 768},
        {"active_tab": "notes", "layout_mode": "matlab_home"},
        {"hidden_tabs": ["jupyter", "notes"]},
        {"tab_display_names": {"chat": "Conversation"}},
        {"calculator_predictive_text_enabled": True},
    ],
    ids=[
        "dock-left",
        "dock-right-floating",
        "large-window",
        "notes-active-matlab",
        "hidden-tabs",
        "display-names",
        "predictive-text",
    ],
)
def test_round_trip_stability_across_field_matrix(
    overrides: dict[str, object],
    standalone_store: StandaloneSessionStore,
) -> None:
    state = SidebarState(**overrides)  # type: ignore[arg-type]
    payload = wrap_state(state.to_dict())
    standalone_store.save_profile("matrix", payload)
    reloaded = standalone_store.load_profile("matrix")
    assert reloaded.data == state.to_dict()


# ---------------------------------------------------------------------------
# AC#3: saved profiles always carry schema_version on both sides (T5 #5983)
# ---------------------------------------------------------------------------


def test_standalone_saved_profile_carries_schema_version(
    embedded_store: SidekickStateProfileStore,
    standalone_store: StandaloneSessionStore,
) -> None:
    state = _sample_state()
    standalone_store.save_profile("ac3_check", wrap_state(state.to_dict()))
    path = embedded_store.profiles_dir / "ac3_check.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    assert (
        PROFILE_SCHEMA_VERSION_KEY in raw
    ), "standalone-saved profile is missing schema_version"
    assert raw[PROFILE_SCHEMA_VERSION_KEY] == PROFILE_SCHEMA_VERSION
