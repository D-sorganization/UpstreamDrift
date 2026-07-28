"""Contract tests to verify Tools consumer provider-path resolution.

This ensures that we can strictly validate whether shared modules are loaded
from the local UpstreamDrift `src/shared/python` or the vendored
`vendor/ud-tools/src/shared/python` directory depending on `--tools-mode`.
"""

from __future__ import annotations

import importlib.util
import sys
from functools import lru_cache
from pathlib import Path
from types import ModuleType

import pytest

_ROOT_CONFTST = Path(__file__).resolve().parents[1] / "conftest.py"


@lru_cache(maxsize=1)
def _load_root_conftest() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "upstreamdrift_tests_root_conftest",
        _ROOT_CONFTST,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class _ConfigStub:
    def __init__(self, tools_mode: str) -> None:
        self._tools_mode = tools_mode

    def getoption(self, name: str) -> str:
        assert name == "--tools-mode"
        return self._tools_mode


class _ParserStub:
    def __init__(self) -> None:
        self.options: dict[str, dict[str, object]] = {}

    def addoption(self, name: str, **kwargs: object) -> None:
        self.options[name] = kwargs


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _local_tools_path() -> str:
    return str((_repo_root() / "src/shared/python").resolve())


def _vendored_tools_path() -> str:
    return str((_repo_root() / "vendor/ud-tools/src/shared/python").resolve())


def _vendored_tools_root() -> Path:
    return (_repo_root() / "vendor/ud-tools").resolve()


def _tools_python_paths(tools_root: Path) -> list[str]:
    return [
        str((tools_root / "src/shared/python").resolve()),
        str((tools_root / "src").resolve()),
        str((tools_root / "src/python/src").resolve()),
    ]


def _seed_sys_path() -> list[str]:
    repo_root = str(_repo_root().resolve())
    fixtures_dir = str((Path(__file__).resolve().parents[1] / "fixtures").resolve())
    local_path = _local_tools_path()
    vendored_path = _vendored_tools_path()
    return [
        repo_root,
        fixtures_dir,
        vendored_path,
        local_path,
        vendored_path,
        local_path,
    ]


def _configure_tools_mode(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    *,
    alias_contracts: bool = False,
    existing_paths: set[str] | None = None,
) -> None:
    root_conftest = _load_root_conftest()
    local_path = _local_tools_path()
    vendored_path = _vendored_tools_path()
    default_existing = set(_tools_python_paths(_vendored_tools_root()))
    real_exists = root_conftest.os.path.exists

    def _fake_exists(path: object) -> bool:
        return (
            str(path) in {local_path, vendored_path}
            or str(path) in default_existing
            or str(path) in (existing_paths or set())
            or real_exists(path)
        )

    monkeypatch.setattr(root_conftest.os.path, "exists", _fake_exists)
    if alias_contracts:
        canonical_module = ModuleType("shared.python.contracts")
        real_import_module = importlib.import_module

        def _fake_import_module(name: str, package: str | None = None) -> ModuleType:
            if name == "shared.python.contracts":
                sys.modules[name] = canonical_module
                return canonical_module
            return real_import_module(name, package)

        monkeypatch.setattr(importlib, "import_module", _fake_import_module)

    root_conftest.pytest_configure(_ConfigStub(mode))


def _assert_precedence(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    expected_first: str,
    expected_second: str,
) -> None:
    _configure_tools_mode(monkeypatch, mode)

    assert sys.path[0] == expected_first
    assert sys.path.index(expected_first) < sys.path.index(expected_second)
    assert sys.path.count(_local_tools_path()) == 1
    assert sys.path.count(_vendored_tools_path()) == 1


def test_tools_mode_local_prefers_repo_shared_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "path", _seed_sys_path())

    _assert_precedence(
        monkeypatch=monkeypatch,
        mode="local",
        expected_first=_local_tools_path(),
        expected_second=_vendored_tools_path(),
    )


def test_tools_mode_vendored_prefers_vendor_shared_python(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "path", _seed_sys_path())

    _assert_precedence(
        monkeypatch=monkeypatch,
        mode="vendored",
        expected_first=_vendored_tools_path(),
        expected_second=_local_tools_path(),
    )


def test_tools_mode_editable_is_a_documented_parser_choice() -> None:
    parser = _ParserStub()

    _load_root_conftest().pytest_addoption(parser)  # type: ignore[arg-type]

    tools_mode = parser.options["--tools-mode"]
    assert tools_mode["choices"] == ["local", "vendored", "editable"]
    assert "TOOLS_REPO_ROOT" in str(tools_mode["help"])


def test_tools_mode_editable_prefers_explicit_tools_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    tools_root = tmp_path / "Tools"
    for path in _tools_python_paths(tools_root):
        Path(path).mkdir(parents=True, exist_ok=True)
    monkeypatch.setenv("TOOLS_REPO_ROOT", str(tools_root))
    monkeypatch.setattr(sys, "path", _seed_sys_path())

    _configure_tools_mode(
        monkeypatch=monkeypatch,
        mode="editable",
        existing_paths=set(_tools_python_paths(tools_root)),
    )

    assert sys.path[:3] == _tools_python_paths(tools_root)
    assert sys.path.index(_tools_python_paths(tools_root)[0]) < sys.path.index(
        _local_tools_path()
    )


def test_tools_mode_editable_fails_loudly_when_checkout_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    root_conftest = _load_root_conftest()
    local_path = _local_tools_path()
    vendored_path = _vendored_tools_path()

    def _fake_exists(path: object) -> bool:
        return str(path) in {local_path, vendored_path}

    monkeypatch.delenv("TOOLS_REPO_PATH", raising=False)
    monkeypatch.delenv("TOOLS_REPO_ROOT", raising=False)
    monkeypatch.setattr(root_conftest.os.path, "exists", _fake_exists)
    monkeypatch.setattr(root_conftest.Path, "is_dir", lambda self: False)

    with pytest.raises(RuntimeError, match="--tools-mode editable requires"):
        root_conftest.pytest_configure(_ConfigStub("editable"))


def test_tools_mode_aliases_contract_modules_to_one_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(sys, "path", _seed_sys_path())
    for module_name in (
        "contracts",
        "shared.python.contracts",
        "src.shared.python.contracts",
    ):
        monkeypatch.delitem(sys.modules, module_name, raising=False)

    _configure_tools_mode(monkeypatch, "vendored", alias_contracts=True)

    canonical = sys.modules["shared.python.contracts"]
    assert sys.modules["contracts"] is canonical
