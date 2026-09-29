"""Coverage for src/engines/__init__.py routing (get_engine_catalog, is_fit_capable)."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

import src.engines as engines_pkg


def _make_engine_dir(root: Path, name: str, tier_content: str | None = None) -> Path:
    eng = root / name
    eng.mkdir()
    if tier_content is not None:
        (eng / "_tier.py").write_text(tier_content)
    return eng


class TestGetEngineCatalog:
    def test_returns_empty_when_dir_missing(self, tmp_path: Path) -> None:
        fake_init = tmp_path / "engines" / "__init__.py"
        fake_init.parent.mkdir()
        fake_init.write_text("")
        with patch.object(engines_pkg, "__file__", str(fake_init)):
            assert engines_pkg.get_engine_catalog() == {}

    def test_catalog_lists_engines_skips_underscored_and_tests(
        self, tmp_path: Path
    ) -> None:
        engines_root = tmp_path / "engines"
        physics_root = engines_root / "physics_engines"
        physics_root.mkdir(parents=True)
        _make_engine_dir(physics_root, "fake_alpha")
        _make_engine_dir(physics_root, "fake_beta")
        # excluded directories
        _make_engine_dir(physics_root, "_private")
        _make_engine_dir(physics_root, "tests")
        # files (not directories) should also be ignored
        (physics_root / "stray.txt").write_text("x")

        fake_init = engines_root / "__init__.py"
        fake_init.write_text("")

        with patch.object(engines_pkg, "__file__", str(fake_init)):
            catalog = engines_pkg.get_engine_catalog()

        assert set(catalog.keys()) == {"fake_alpha", "fake_beta"}
        # No _tier.py and no provider module → default fit_capable=True
        for entry in catalog.values():
            assert entry["fit_capable"] is True

    def test_engine_marked_fit_incapable_via_tier(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engines_root = tmp_path / "engines"
        physics_root = engines_root / "physics_engines"
        physics_root.mkdir(parents=True)
        _make_engine_dir(physics_root, "fake_incapable", tier_content="FAKE=1")

        fake_init = engines_root / "__init__.py"
        fake_init.write_text("")

        fake_tier_mod = ModuleType("src.engines.physics_engines.fake_incapable._tier")
        fake_tier_mod.FIT_INCAPABLE = True  # type: ignore[attr-defined]
        monkeypatch.setitem(
            sys.modules,
            "src.engines.physics_engines.fake_incapable._tier",
            fake_tier_mod,
        )

        with patch.object(engines_pkg, "__file__", str(fake_init)):
            catalog = engines_pkg.get_engine_catalog()

        assert catalog == {"fake_incapable": {"fit_capable": False}}

    def test_tier_module_without_flag_remains_capable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engines_root = tmp_path / "engines"
        physics_root = engines_root / "physics_engines"
        physics_root.mkdir(parents=True)
        _make_engine_dir(physics_root, "fake_capable", tier_content="X=1")

        fake_init = engines_root / "__init__.py"
        fake_init.write_text("")

        fake_tier_mod = ModuleType("src.engines.physics_engines.fake_capable._tier")
        # No FIT_INCAPABLE attribute → defaults to False → still capable
        monkeypatch.setitem(
            sys.modules,
            "src.engines.physics_engines.fake_capable._tier",
            fake_tier_mod,
        )

        with patch.object(engines_pkg, "__file__", str(fake_init)):
            catalog = engines_pkg.get_engine_catalog()

        assert catalog["fake_capable"]["fit_capable"] is True

    def test_tier_import_error_swallowed(self, tmp_path: Path) -> None:
        engines_root = tmp_path / "engines"
        physics_root = engines_root / "physics_engines"
        physics_root.mkdir(parents=True)
        # _tier.py present but the module path does not exist in sys.modules
        # and is not importable — ImportError is suppressed.
        _make_engine_dir(physics_root, "fake_missingtier", tier_content="FAKE=1")

        fake_init = engines_root / "__init__.py"
        fake_init.write_text("")

        with patch.object(engines_pkg, "__file__", str(fake_init)):
            catalog = engines_pkg.get_engine_catalog()

        # ImportError suppressed; defaults to capable
        assert catalog["fake_missingtier"]["fit_capable"] is True

    def test_provider_import_error_swallowed(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Provider import failure must not bubble up."""
        engines_root = tmp_path / "engines"
        physics_root = engines_root / "physics_engines"
        physics_root.mkdir(parents=True)
        _make_engine_dir(physics_root, "fake_noprovider")

        fake_init = engines_root / "__init__.py"
        fake_init.write_text("")

        with patch.object(engines_pkg, "__file__", str(fake_init)):
            # Provider import attempted but no such module exists →
            # ImportError suppressed, entry still present.
            catalog = engines_pkg.get_engine_catalog()
        assert "fake_noprovider" in catalog

    def test_provider_imported_when_capable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engines_root = tmp_path / "engines"
        physics_root = engines_root / "physics_engines"
        physics_root.mkdir(parents=True)
        _make_engine_dir(physics_root, "fake_provider")

        fake_init = engines_root / "__init__.py"
        fake_init.write_text("")

        provider_path = (
            "src.engines.physics_engines.fake_provider.python.motion_matching.provider"
        )
        provider_mod = ModuleType(provider_path)
        provider_mod.registered = True  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, provider_path, provider_mod)

        with patch.object(engines_pkg, "__file__", str(fake_init)):
            catalog = engines_pkg.get_engine_catalog()
        assert catalog["fake_provider"]["fit_capable"] is True


class TestIsFitCapable:
    def test_unknown_engine_returns_false(self, tmp_path: Path) -> None:
        fake_init = tmp_path / "engines" / "__init__.py"
        fake_init.parent.mkdir()
        fake_init.write_text("")
        with patch.object(engines_pkg, "__file__", str(fake_init)):
            assert engines_pkg.is_fit_capable("does_not_exist") is False

    def test_capable_engine_returns_true(self, tmp_path: Path) -> None:
        engines_root = tmp_path / "engines"
        physics_root = engines_root / "physics_engines"
        physics_root.mkdir(parents=True)
        _make_engine_dir(physics_root, "alpha")
        fake_init = engines_root / "__init__.py"
        fake_init.write_text("")

        with patch.object(engines_pkg, "__file__", str(fake_init)):
            assert engines_pkg.is_fit_capable("alpha") is True

    def test_incapable_engine_returns_false(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        engines_root = tmp_path / "engines"
        physics_root = engines_root / "physics_engines"
        physics_root.mkdir(parents=True)
        _make_engine_dir(physics_root, "beta", tier_content="X=1")
        fake_init = engines_root / "__init__.py"
        fake_init.write_text("")

        fake_tier_mod = ModuleType("src.engines.physics_engines.beta._tier")
        fake_tier_mod.FIT_INCAPABLE = True  # type: ignore[attr-defined]
        monkeypatch.setitem(
            sys.modules, "src.engines.physics_engines.beta._tier", fake_tier_mod
        )

        with patch.object(engines_pkg, "__file__", str(fake_init)):
            assert engines_pkg.is_fit_capable("beta") is False


class TestPublicAPI:
    def test_module_exports(self) -> None:
        assert "get_engine_catalog" in engines_pkg.__all__
        assert "is_fit_capable" in engines_pkg.__all__
