"""Tests for StandaloneSessionStore — T3 (#5981).

Covers:
  - save / load round-trip (byte-identical)
  - list_profiles, delete_profile, last_profile / set_last_profile
  - precondition: name must match ^[a-zA-Z0-9_-]+$
  - delete missing → KeyError (not FileNotFoundError)
  - load missing → KeyError
  - malformed JSON → StateError
  - concurrent writes are safe
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import pytest

from core.contracts.exceptions import StateError
from sidekick.standalone.session_store import ProfilePayload, StandaloneSessionStore


@pytest.fixture
def store(tmp_path: Path) -> StandaloneSessionStore:
    return StandaloneSessionStore(tmp_path)


@pytest.fixture
def sample_payload() -> ProfilePayload:
    return ProfilePayload(data={"active_tab": "chat", "width": 400})


# ---------------------------------------------------------------------------
# Happy paths
# ---------------------------------------------------------------------------


class TestSaveAndLoad:
    def test_round_trip(
        self, store: StandaloneSessionStore, sample_payload: ProfilePayload
    ) -> None:
        store.save_profile("myprofile", sample_payload)
        loaded = store.load_profile("myprofile")
        assert loaded.data == sample_payload.data

    def test_byte_identical(
        self, store: StandaloneSessionStore, sample_payload: ProfilePayload
    ) -> None:
        store.save_profile("exact", sample_payload)
        loaded = store.load_profile("exact")
        assert json.dumps(loaded.data, sort_keys=True) == json.dumps(
            sample_payload.data, sort_keys=True
        )

    def test_schema_version_preserved(self, store: StandaloneSessionStore) -> None:
        p = ProfilePayload(data={"x": 1}, schema_version=1)
        store.save_profile("versioned", p)
        loaded = store.load_profile("versioned")
        assert loaded.schema_version == 1

    def test_list_profiles(
        self, store: StandaloneSessionStore, sample_payload: ProfilePayload
    ) -> None:
        store.save_profile("alpha", sample_payload)
        store.save_profile("beta", sample_payload)
        assert set(store.list_profiles()) == {"alpha", "beta"}

    def test_delete_profile(
        self, store: StandaloneSessionStore, sample_payload: ProfilePayload
    ) -> None:
        store.save_profile("to-delete", sample_payload)
        store.delete_profile("to-delete")
        assert "to-delete" not in store.list_profiles()

    def test_last_profile_roundtrip(
        self, store: StandaloneSessionStore, sample_payload: ProfilePayload
    ) -> None:
        store.save_profile("alpha", sample_payload)
        store.set_last_profile("alpha")
        assert store.last_profile() == "alpha"

    def test_last_profile_none_when_unset(self, store: StandaloneSessionStore) -> None:
        assert store.last_profile() is None

    def test_overwrite_existing_profile(self, store: StandaloneSessionStore) -> None:
        store.save_profile("p", ProfilePayload(data={"v": 1}))
        store.save_profile("p", ProfilePayload(data={"v": 2}))
        loaded = store.load_profile("p")
        assert loaded.data["v"] == 2


# ---------------------------------------------------------------------------
# Precondition / name validation
# ---------------------------------------------------------------------------


class TestNameValidation:
    def test_invalid_name_with_space_raises(
        self, store: StandaloneSessionStore, sample_payload: ProfilePayload
    ) -> None:
        with pytest.raises(ValueError):
            store.save_profile("invalid name!", sample_payload)

    def test_empty_name_raises(
        self, store: StandaloneSessionStore, sample_payload: ProfilePayload
    ) -> None:
        with pytest.raises(ValueError):
            store.save_profile("", sample_payload)

    def test_valid_names_with_underscores_and_hyphens(
        self, store: StandaloneSessionStore, sample_payload: ProfilePayload
    ) -> None:
        store.save_profile("my_profile-v2", sample_payload)
        assert "my_profile-v2" in store.list_profiles()


# ---------------------------------------------------------------------------
# Error cases
# ---------------------------------------------------------------------------


class TestErrorCases:
    def test_load_missing_raises_key_error(self, store: StandaloneSessionStore) -> None:
        with pytest.raises(KeyError):
            store.load_profile("nonexistent")

    def test_delete_missing_raises_key_error_not_file_not_found(
        self, store: StandaloneSessionStore
    ) -> None:
        with pytest.raises(KeyError):
            store.delete_profile("does-not-exist")

    def test_load_malformed_json_raises_state_error(
        self, store: StandaloneSessionStore, tmp_path: Path
    ) -> None:
        profile_path = tmp_path / "profiles" / "bad.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text("{ bad json ]", encoding="utf-8")
        with pytest.raises(StateError):
            store.load_profile("bad")

    def test_load_non_object_json_raises_state_error(
        self, store: StandaloneSessionStore, tmp_path: Path
    ) -> None:
        profile_path = tmp_path / "profiles" / "list.json"
        profile_path.parent.mkdir(parents=True, exist_ok=True)
        profile_path.write_text("[1, 2, 3]", encoding="utf-8")
        with pytest.raises(StateError):
            store.load_profile("list")


# ---------------------------------------------------------------------------
# Concurrency
# ---------------------------------------------------------------------------


class TestConcurrency:
    def test_concurrent_writes_are_safe(self, store: StandaloneSessionStore) -> None:
        errors: list[Exception] = []
        lock = threading.Lock()

        def write_profile(n: int) -> None:
            try:
                payload = ProfilePayload(data={"thread": n})
                store.save_profile("shared", payload)
            except Exception as exc:  # noqa: BLE001
                with lock:
                    errors.append(exc)

        threads = [threading.Thread(target=write_profile, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors, f"Concurrent write errors: {errors}"
        # Last writer wins — profile must be loadable
        loaded = store.load_profile("shared")
        assert "thread" in loaded.data


# ---------------------------------------------------------------------------
# ProfilePayload unit tests
# ---------------------------------------------------------------------------


class TestProfilePayload:
    def test_to_dict_includes_schema_version(self) -> None:
        p = ProfilePayload(data={"key": "val"})
        d = p.to_dict()
        assert "schema_version" in d

    def test_from_dict_excludes_schema_version_from_data(self) -> None:
        raw = {"active_tab": "chat", "schema_version": 1}
        p = ProfilePayload.from_dict(raw)
        assert "schema_version" not in p.data
        assert p.data == {"active_tab": "chat"}

    def test_invalid_data_type_raises(self) -> None:
        with pytest.raises(TypeError):
            ProfilePayload(data="not a dict")  # type: ignore[arg-type]
