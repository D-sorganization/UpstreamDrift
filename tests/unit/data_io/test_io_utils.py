"""Tests for src.shared.python.data_io.io_utils (Issues #1949, #1744)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from src.shared.python.core.error_utils import FileNotFoundIOError, FileParseError
from src.shared.python.data_io.io_utils import (
    ensure_directory,
    file_exists,
    get_file_size,
    load_json,
    read_text,
    save_json,
    save_yaml,
    write_text,
)

pytestmark = pytest.mark.unit

# ---------------------------------------------------------------------------
# ensure_directory
# ---------------------------------------------------------------------------


class TestEnsureDirectory:
    def test_creates_directory(self, tmp_path: Path) -> None:
        new_dir = tmp_path / "new" / "nested"
        result = ensure_directory(new_dir)
        assert result.is_dir()
        assert result == new_dir

    def test_existing_directory_ok(self, tmp_path: Path) -> None:
        result = ensure_directory(tmp_path)
        assert result == tmp_path

    def test_string_input_accepted(self, tmp_path: Path) -> None:
        new_dir = str(tmp_path / "str_dir")
        result = ensure_directory(new_dir)
        assert result.is_dir()


# ---------------------------------------------------------------------------
# load_json / save_json
# ---------------------------------------------------------------------------


class TestLoadJson:
    def test_loads_dict(self, tmp_path: Path) -> None:
        data = {"key": "value", "num": 42}
        f = tmp_path / "test.json"
        f.write_text(json.dumps(data))
        assert load_json(f) == data

    def test_missing_strict_raises(self) -> None:
        with pytest.raises(FileNotFoundIOError):
            load_json("/nonexistent/path/file.json")

    def test_missing_non_strict_returns_default(self) -> None:
        result = load_json("/nonexistent/path/file.json", strict=False, default={})
        assert result == {}

    def test_invalid_json_raises_parse_error(self, tmp_path: Path) -> None:
        f = tmp_path / "bad.json"
        f.write_text("{not valid json")
        with pytest.raises(FileParseError):
            load_json(f)

    def test_loads_list(self, tmp_path: Path) -> None:
        data = [1, 2, 3]
        f = tmp_path / "list.json"
        f.write_text(json.dumps(data))
        assert load_json(f) == data


class TestSaveJson:
    @pytest.mark.parametrize("kind", ["value", "key", "circular"])
    def test_serialization_failure_preserves_existing_file(
        self, tmp_path: Path, kind: str
    ) -> None:
        payload: dict[object, object] = {"bad": object()}
        if kind == "key":
            payload = {("bad",): 1}
        elif kind == "circular":
            payload = {}
            payload["self"] = payload
        target = tmp_path / "config.json"
        original = b'{"keep": true}\r\n'
        target.write_bytes(original)

        error = ValueError if kind == "circular" else TypeError
        with pytest.raises(error):
            save_json(target, payload)
        assert target.read_bytes() == original

    def test_serialization_failure_does_not_create_parent(self, tmp_path: Path) -> None:
        target = tmp_path / "new" / "config.json"
        with pytest.raises(TypeError):
            save_json(target, {"bad": object()})
        assert not target.parent.exists()

    def test_serialization_options_preserved(self, tmp_path: Path) -> None:
        target = tmp_path / "config.json"
        assert (
            save_json(
                target,
                {"z": "café", "a": 1},
                indent=4,
                ensure_ascii=False,
                sort_keys=True,
            )
            == target
        )
        assert (
            target.read_text(encoding="utf-8") == '{\n    "a": 1,\n    "z": "café"\n}'
        )

    def test_saves_and_reloads(self, tmp_path: Path) -> None:
        data = {"x": 1, "y": [2, 3]}
        f = tmp_path / "out.json"
        result = save_json(f, data)
        assert result == f
        assert load_json(f) == data

    def test_io_utils_creates_parent_dirs(self, tmp_path: Path) -> None:
        f = tmp_path / "deep" / "nested" / "file.json"
        save_json(f, {"a": 1})
        assert f.exists()

    def test_sort_keys(self, tmp_path: Path) -> None:
        data = {"z": 1, "a": 2}
        f = tmp_path / "sorted.json"
        save_json(f, data, sort_keys=True)
        text = f.read_text()
        assert text.index('"a"') < text.index('"z"')

    def test_returns_path_object(self, tmp_path: Path) -> None:
        f = tmp_path / "ret.json"
        result = save_json(f, {})
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
class TestSaveYaml:
    def test_serialization_failure_preserves_existing_file(
        self, tmp_path: Path
    ) -> None:
        yaml = pytest.importorskip("yaml")
        target = tmp_path / "config.yaml"
        original = b"keep: true\r\n"
        target.write_bytes(original)
        with pytest.raises(yaml.representer.RepresenterError):
            save_yaml(target, {"bad": object()})
        assert target.read_bytes() == original

    def test_serialization_failure_does_not_create_parent(self, tmp_path: Path) -> None:
        yaml = pytest.importorskip("yaml")
        target = tmp_path / "new" / "config.yaml"
        with pytest.raises(yaml.representer.RepresenterError):
            save_yaml(target, {"bad": object()})
        assert not target.parent.exists()

    def test_successful_write_preserves_options(self, tmp_path: Path) -> None:
        yaml = pytest.importorskip("yaml")
        target = tmp_path / "config.yaml"
        payload = {"z": "café", "a": 1}
        assert (
            save_yaml(target, payload, default_flow_style=True, sort_keys=True)
            == target
        )
        text = target.read_text(encoding="utf-8")
        assert text.startswith("{a: 1,")
        assert "café" in text
        assert yaml.safe_load(text) == payload


# read_text / write_text
# ---------------------------------------------------------------------------


class TestReadWriteText:
    def test_io_utils_roundtrip(self, tmp_path: Path) -> None:
        f = tmp_path / "test.txt"
        content = "hello\nworld"
        write_text(f, content)
        assert read_text(f) == content

    def test_write_creates_parents(self, tmp_path: Path) -> None:
        f = tmp_path / "sub" / "dir" / "file.txt"
        write_text(f, "data")
        assert f.exists()

    def test_read_missing_raises(self) -> None:
        with pytest.raises(FileNotFoundIOError):
            read_text("/nonexistent/file.txt")

    def test_write_returns_path(self, tmp_path: Path) -> None:
        f = tmp_path / "out.txt"
        result = write_text(f, "content")
        assert isinstance(result, Path)


# ---------------------------------------------------------------------------
# file_exists / get_file_size
# ---------------------------------------------------------------------------


class TestFileExists:
    def test_existing_file(self, tmp_path: Path) -> None:
        f = tmp_path / "exists.txt"
        f.write_text("data")
        assert file_exists(f) is True

    def test_missing_file(self) -> None:
        assert file_exists("/nonexistent/file.txt") is False

    def test_directory_is_true(self, tmp_path: Path) -> None:
        # file_exists checks Path.exists(), so directories also return True
        assert file_exists(tmp_path) is True


class TestGetFileSize:
    def test_returns_correct_size(self, tmp_path: Path) -> None:
        f = tmp_path / "size.txt"
        content = "hello"
        f.write_bytes(content.encode())
        assert get_file_size(f) == len(content)

    def test_io_utils_missing_file_raises(self) -> None:
        with pytest.raises(FileNotFoundIOError):
            get_file_size("/nonexistent/path.txt")
