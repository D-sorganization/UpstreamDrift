import builtins
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit


def test_src_package_installs_parent_shared_import_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy src.shared imports must resolve to canonical Tools modules."""
    calls: list[str] = []
    shared = ModuleType("shared")
    shared.__path__ = []  # type: ignore[attr-defined]
    shared_python = ModuleType("shared.python")
    shared_python.__path__ = []  # type: ignore[attr-defined]
    aliases = ModuleType("shared.python.import_aliases")
    aliases.install_shared_import_aliases = lambda: calls.append("installed")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "shared", shared)
    monkeypatch.setitem(sys.modules, "shared.python", shared_python)
    monkeypatch.setitem(sys.modules, "shared.python.import_aliases", aliases)

    repo_root = Path(__file__).resolve().parents[2]
    module_globals = runpy.run_path(str(repo_root / "src" / "__init__.py"))

    assert calls == ["installed"]
    assert module_globals["_PARENT_SHARED_ALIASES_INSTALLED"] is True


def test_missing_parent_aliases_do_not_poison_shared_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-bootstrap source import must not cache a partial `shared` package."""
    monkeypatch.delitem(sys.modules, "shared", raising=False)
    monkeypatch.delitem(sys.modules, "shared.python", raising=False)
    real_import = builtins.__import__

    def unavailable_aliases(name, *args, **kwargs):  # noqa: ANN001, ANN202
        if name == "shared.python.import_aliases":
            partial_shared = ModuleType("shared")
            partial_shared.__path__ = []  # type: ignore[attr-defined]
            sys.modules["shared"] = partial_shared
            raise ModuleNotFoundError(
                "canonical Tools path is not bootstrapped",
                name="shared",
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable_aliases)
    repo_root = Path(__file__).resolve().parents[2]

    module_globals = runpy.run_path(str(repo_root / "src" / "__init__.py"))

    assert module_globals["_PARENT_SHARED_ALIASES_INSTALLED"] is False
    assert "shared" not in sys.modules


def test_noncanonical_module_error_propagates_and_restores_partial_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A broken dependency inside Tools must not masquerade as missing Tools."""
    monkeypatch.delitem(sys.modules, "shared", raising=False)
    monkeypatch.delitem(sys.modules, "shared.python", raising=False)
    real_import = builtins.__import__

    def broken_aliases(name, *args, **kwargs):  # noqa: ANN001, ANN202
        if name == "shared.python.import_aliases":
            partial_shared = ModuleType("shared")
            partial_shared.__path__ = []  # type: ignore[attr-defined]
            sys.modules["shared"] = partial_shared
            raise ModuleNotFoundError(
                "No module named 'broken_dependency'",
                name="broken_dependency",
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_aliases)
    repo_root = Path(__file__).resolve().parents[2]

    with pytest.raises(ModuleNotFoundError, match="broken_dependency"):
        runpy.run_path(str(repo_root / "src" / "__init__.py"))

    assert "shared" not in sys.modules


def test_plain_import_error_propagates_instead_of_disabling_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed canonical module is a real defect, not an optional absence."""
    shared = ModuleType("shared")
    shared.__path__ = []  # type: ignore[attr-defined]
    shared_python = ModuleType("shared.python")
    shared_python.__path__ = []  # type: ignore[attr-defined]
    aliases = ModuleType("shared.python.import_aliases")
    monkeypatch.setitem(sys.modules, "shared", shared)
    monkeypatch.setitem(sys.modules, "shared.python", shared_python)
    monkeypatch.setitem(sys.modules, "shared.python.import_aliases", aliases)
    repo_root = Path(__file__).resolve().parents[2]

    with pytest.raises(ImportError, match="install_shared_import_aliases"):
        runpy.run_path(str(repo_root / "src" / "__init__.py"))


def test_installer_failure_restores_every_partial_module_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alias installation is atomic even when the installer itself fails."""
    shared = ModuleType("shared")
    shared.__path__ = []  # type: ignore[attr-defined]
    shared_python = ModuleType("shared.python")
    shared_python.__path__ = []  # type: ignore[attr-defined]
    aliases = ModuleType("shared.python.import_aliases")

    def fail_after_partial_install() -> None:
        sys.modules["shared.python.partial_alias"] = ModuleType(
            "shared.python.partial_alias"
        )
        raise RuntimeError("installer failed")

    aliases.install_shared_import_aliases = fail_after_partial_install  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "shared", shared)
    monkeypatch.setitem(sys.modules, "shared.python", shared_python)
    monkeypatch.setitem(sys.modules, "shared.python.import_aliases", aliases)
    monkeypatch.delitem(sys.modules, "shared.python.partial_alias", raising=False)
    repo_root = Path(__file__).resolve().parents[2]

    with pytest.raises(RuntimeError, match="installer failed"):
        runpy.run_path(str(repo_root / "src" / "__init__.py"))

    assert sys.modules["shared"] is shared
    assert sys.modules["shared.python"] is shared_python
    assert sys.modules["shared.python.import_aliases"] is aliases
    assert "shared.python.partial_alias" not in sys.modules


def test_installer_missing_module_error_is_never_treated_as_optional_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only importing the installer may classify its canonical module as absent."""
    shared = ModuleType("shared")
    shared.__path__ = []  # type: ignore[attr-defined]
    shared_python = ModuleType("shared.python")
    shared_python.__path__ = []  # type: ignore[attr-defined]
    aliases = ModuleType("shared.python.import_aliases")

    def broken_installer() -> None:
        sys.modules["shared.python.partial_alias"] = ModuleType(
            "shared.python.partial_alias"
        )
        raise ModuleNotFoundError(
            "installer dependency is broken",
            name="shared",
        )

    aliases.install_shared_import_aliases = broken_installer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "shared", shared)
    monkeypatch.setitem(sys.modules, "shared.python", shared_python)
    monkeypatch.setitem(sys.modules, "shared.python.import_aliases", aliases)
    monkeypatch.delitem(sys.modules, "shared.python.partial_alias", raising=False)
    repo_root = Path(__file__).resolve().parents[2]

    with pytest.raises(ModuleNotFoundError, match="installer dependency is broken"):
        runpy.run_path(str(repo_root / "src" / "__init__.py"))

    assert "shared.python.partial_alias" not in sys.modules
