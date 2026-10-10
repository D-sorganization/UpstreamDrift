import builtins
import importlib
import runpy
import sys
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.unit

_POLICY_MODULE = "src.shared.python.ud_import_alias_policy"


def test_src_package_installs_parent_shared_import_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Legacy src.shared imports must resolve to canonical Tools modules."""
    calls: list[str] = []
    policy = ModuleType(_POLICY_MODULE)
    policy.install_ud_canonical_shared_import_aliases = (  # type: ignore[attr-defined]
        lambda: calls.append("installed")
    )
    monkeypatch.setitem(sys.modules, _POLICY_MODULE, policy)
    real_import_module = importlib.import_module

    def track_downstream_namespace(name: str, package: str | None = None):
        calls.append(f"loaded:{name}")
        return real_import_module(name, package)

    monkeypatch.setattr(importlib, "import_module", track_downstream_namespace)

    repo_root = Path(__file__).resolve().parents[2]
    module_globals = runpy.run_path(str(repo_root / "src" / "__init__.py"))

    assert calls == ["loaded:src.shared.python", "installed"]
    assert module_globals["_PARENT_SHARED_ALIASES_INSTALLED"] is True


def test_missing_parent_aliases_do_not_poison_shared_namespace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A pre-bootstrap source import must not cache a partial `shared` package."""
    monkeypatch.delitem(sys.modules, "shared", raising=False)
    monkeypatch.delitem(sys.modules, "shared.python", raising=False)
    real_import = builtins.__import__

    def unavailable_policy(name, *args, **kwargs):  # noqa: ANN001, ANN202
        if name == _POLICY_MODULE:
            partial_shared = ModuleType("shared")
            partial_shared.__path__ = []  # type: ignore[attr-defined]
            sys.modules["shared"] = partial_shared
            raise ModuleNotFoundError(
                "canonical Tools path is not bootstrapped",
                name="shared.python.import_aliases",
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", unavailable_policy)
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

    def broken_policy(name, *args, **kwargs):  # noqa: ANN001, ANN202
        if name == _POLICY_MODULE:
            partial_shared = ModuleType("shared")
            partial_shared.__path__ = []  # type: ignore[attr-defined]
            sys.modules["shared"] = partial_shared
            raise ModuleNotFoundError(
                "No module named 'broken_dependency'",
                name="broken_dependency",
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", broken_policy)
    repo_root = Path(__file__).resolve().parents[2]

    with pytest.raises(ModuleNotFoundError, match="broken_dependency"):
        runpy.run_path(str(repo_root / "src" / "__init__.py"))

    assert "shared" not in sys.modules


def test_plain_import_error_propagates_instead_of_disabling_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A malformed canonical module is a real defect, not an optional absence."""
    policy = ModuleType(_POLICY_MODULE)
    monkeypatch.setitem(sys.modules, _POLICY_MODULE, policy)
    repo_root = Path(__file__).resolve().parents[2]

    with pytest.raises(ImportError, match="install_ud_canonical_shared_import_aliases"):
        runpy.run_path(str(repo_root / "src" / "__init__.py"))


def test_installer_failure_restores_every_partial_module_and_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Alias installation is atomic even when the installer itself fails."""
    shared = ModuleType("shared")
    shared.__path__ = []  # type: ignore[attr-defined]
    shared_python = ModuleType("shared.python")
    shared_python.__path__ = []  # type: ignore[attr-defined]
    policy = ModuleType(_POLICY_MODULE)

    def fail_after_partial_install() -> None:
        sys.modules["shared.python.partial_alias"] = ModuleType(
            "shared.python.partial_alias"
        )
        raise RuntimeError("installer failed")

    policy.install_ud_canonical_shared_import_aliases = (  # type: ignore[attr-defined]
        fail_after_partial_install
    )
    monkeypatch.setitem(sys.modules, "shared", shared)
    monkeypatch.setitem(sys.modules, "shared.python", shared_python)
    monkeypatch.setitem(sys.modules, _POLICY_MODULE, policy)
    monkeypatch.delitem(sys.modules, "shared.python.partial_alias", raising=False)
    repo_root = Path(__file__).resolve().parents[2]

    with pytest.raises(RuntimeError, match="installer failed"):
        runpy.run_path(str(repo_root / "src" / "__init__.py"))

    assert sys.modules["shared"] is shared
    assert sys.modules["shared.python"] is shared_python
    assert "shared.python.partial_alias" not in sys.modules


def test_installer_missing_module_error_is_never_treated_as_optional_absence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Only importing the installer may classify its canonical module as absent."""
    shared = ModuleType("shared")
    shared.__path__ = []  # type: ignore[attr-defined]
    shared_python = ModuleType("shared.python")
    shared_python.__path__ = []  # type: ignore[attr-defined]
    policy = ModuleType(_POLICY_MODULE)

    def broken_installer() -> None:
        sys.modules["shared.python.partial_alias"] = ModuleType(
            "shared.python.partial_alias"
        )
        raise ModuleNotFoundError(
            "installer dependency is broken",
            name="broken_dependency",
        )

    policy.install_ud_canonical_shared_import_aliases = broken_installer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "shared", shared)
    monkeypatch.setitem(sys.modules, "shared.python", shared_python)
    monkeypatch.setitem(sys.modules, _POLICY_MODULE, policy)
    monkeypatch.delitem(sys.modules, "shared.python.partial_alias", raising=False)
    repo_root = Path(__file__).resolve().parents[2]

    with pytest.raises(ModuleNotFoundError, match="installer dependency is broken"):
        runpy.run_path(str(repo_root / "src" / "__init__.py"))

    assert "shared.python.partial_alias" not in sys.modules
