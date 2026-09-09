"""Named layout presets store with provenance (#9811)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.tools.capture_rig.layout_model import (
    PRESET_NAMES,
    SCHEMA_VERSION,
    LayoutSpec,
    SourceRef,
    preset,
)
from src.tools.capture_rig.layout_presets import (
    BUILTIN,
    SESSION,
    USER,
    LayoutStore,
    LayoutStoreError,
    default_user_root,
    validate_name,
)

pytestmark = [pytest.mark.unit]


def _store(tmp_path: Path, *, session: bool = True) -> LayoutStore:
    sess = tmp_path / "session"
    if session:
        sess.mkdir()
        (sess / "plan.json").write_text("{}", encoding="utf-8")
    return LayoutStore(user_root=tmp_path / "user", session=sess if session else None)


def _spec(name: str = "pair") -> LayoutSpec:
    return preset(
        "side_by_side",
        [SourceRef(kind="live", view="a"), SourceRef(kind="recorded", view="b")],
    ).renamed(name)


class TestNames:
    @pytest.mark.parametrize("name", ["", "  ", "a/b", "a\\b", "..", ".hidden", "x:y"])
    def test_invalid(self, name: str) -> None:
        with pytest.raises(ValueError, match="name"):
            validate_name(name)

    def test_valid(self) -> None:
        assert validate_name(" Face on + DTL ") == "Face on + DTL"


@pytest.mark.parametrize("scope", [USER, SESSION])
class TestCrud:
    def test_save_load_exists_list_delete(self, tmp_path: Path, scope: str) -> None:
        store = _store(tmp_path)
        spec = _spec()
        assert not store.exists("pair", scope)
        path = store.save("pair", spec, scope)
        assert path.is_file() and path.name == "pair.json"
        expected_root = (
            tmp_path / "user" if scope == USER else tmp_path / "session" / "layouts"
        )
        assert path.parent == expected_root
        assert store.exists("pair", scope)
        assert store.load("pair", scope) == spec
        entries = store.list(scope)
        assert [e.name for e in entries] == ["pair"]
        assert entries[0].scope == scope and not entries[0].builtin
        assert entries[0].path == path and entries[0].error is None
        store.delete("pair", scope)
        assert not store.exists("pair", scope)
        assert not path.exists()
        assert store.list(scope) == []

    def test_save_overwrites_and_rename(self, tmp_path: Path, scope: str) -> None:
        store = _store(tmp_path)
        store.save("pair", _spec(), scope)
        other = preset("single", [SourceRef(kind="live", view="z")])
        store.save("pair", other, scope)
        assert store.load("pair", scope).rows == 1
        store.rename("pair", "solo", scope)
        assert not store.exists("pair", scope)
        assert store.load("solo", scope).name == "solo"
        store.save("blocker", other, scope)
        with pytest.raises(LayoutStoreError, match="exists"):
            store.rename("solo", "blocker", scope)

    def test_missing_name_reported(self, tmp_path: Path, scope: str) -> None:
        store = _store(tmp_path)
        with pytest.raises(LayoutStoreError, match="no layout"):
            store.load("nope", scope)
        with pytest.raises(LayoutStoreError, match="no layout"):
            store.delete("nope", scope)
        with pytest.raises(LayoutStoreError, match="no layout"):
            store.rename("nope", "x", scope)

    def test_provenance_stamped(self, tmp_path: Path, scope: str) -> None:
        store = _store(tmp_path)
        path = store.save("pair", _spec(), scope)
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["schema_version"] == SCHEMA_VERSION
        prov = payload["provenance"]
        assert prov["created_utc"].endswith("Z")
        assert prov["generated_by"]["module"] == "src.tools.capture_rig.layout_presets"
        assert "version" in prov["generated_by"] and "git_sha" in prov["generated_by"]
        assert prov["parameters"] == {"name": "pair", "scope": scope}

    def test_malformed_file_reported_not_crashing(
        self, tmp_path: Path, scope: str
    ) -> None:
        store = _store(tmp_path)
        good = store.save("good", _spec(), scope)
        bad = good.with_name("bad.json")
        bad.write_text("{not json", encoding="utf-8")
        wrong = good.with_name("wrong.json")
        wrong.write_text(
            json.dumps(
                {"schema_version": SCHEMA_VERSION, "name": "w", "cols": 2, "rows": 9}
            )
        )
        entries = {e.name: e for e in store.list(scope)}
        assert set(entries) == {"good", "bad", "wrong"}
        assert entries["good"].error is None
        assert entries["bad"].error and "bad.json" in entries["bad"].error
        assert entries["wrong"].error and "rows" in entries["wrong"].error
        with pytest.raises(LayoutStoreError, match="bad.json"):
            store.load("bad", scope)
        # a broken file can still be deleted or renamed away
        store.delete("bad", scope)
        assert not bad.exists()


class TestScopes:
    def test_session_scope_needs_a_session(self, tmp_path: Path) -> None:
        store = _store(tmp_path, session=False)
        with pytest.raises(LayoutStoreError, match="session"):
            store.save("pair", _spec(), SESSION)
        assert store.list(SESSION) == []
        assert not store.exists("pair", SESSION)

    def test_unknown_scope(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        with pytest.raises(ValueError, match="scope"):
            store.list("global")

    def test_list_all_includes_builtins_first(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save("mine", _spec(), USER)
        store.save("take", _spec(), SESSION)
        entries = store.list()
        names = [(e.scope, e.name) for e in entries]
        assert names[: len(PRESET_NAMES)] == [(BUILTIN, n) for n in PRESET_NAMES]
        assert (USER, "mine") in names and (SESSION, "take") in names
        assert all(e.builtin for e in entries if e.scope == BUILTIN)
        assert all(e.path is None for e in entries if e.builtin)

    def test_builtins_read_only(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        assert store.exists("two_by_two", BUILTIN)
        assert store.load("two_by_two", BUILTIN) == preset("two_by_two")
        assert [e.name for e in store.list(BUILTIN)] == list(PRESET_NAMES)
        with pytest.raises(LayoutStoreError, match="built-in"):
            store.delete("two_by_two", BUILTIN)
        with pytest.raises(LayoutStoreError, match="built-in"):
            store.save("two_by_two", _spec(), BUILTIN)
        with pytest.raises(LayoutStoreError, match="built-in"):
            store.rename("two_by_two", "x", BUILTIN)

    def test_user_name_may_shadow_builtin(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save("single", _spec(), USER)
        assert store.load("single", USER).cols == 2
        assert store.load("single", BUILTIN).cols == 1

    def test_stored_spec_carries_its_name(self, tmp_path: Path) -> None:
        store = _store(tmp_path)
        store.save("renamed", _spec("original"), USER)
        assert store.load("renamed", USER).name == "renamed"


def test_default_user_root_is_under_capture_rig_layouts(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("APPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    root = default_user_root()
    assert root.parts[-3:] == ("UpstreamDrift", "capture_rig", "layouts")
    assert LayoutStore().user_root == root
