"""The retired src.shared.python roots resolve to the pinned Tools tree (UD #9406)."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_SHARED = _REPO_ROOT / "vendor" / "ud-tools" / "src" / "shared" / "python"
_UD_SHARED = _REPO_ROOT / "src" / "shared" / "python"

if not _VENDOR_SHARED.is_dir():  # pragma: no cover - CI always has the submodule
    pytest.skip("vendor/ud-tools not materialised", allow_module_level=True)

from src.shared.python import _seam_redirect  # noqa: E402


def test_finder_installed_once() -> None:
    first = _seam_redirect.install()
    second = _seam_redirect.install()
    assert first is second
    # The module can be executed twice (``shared.python`` and ``src.shared.python``
    # spellings), so compare by class name rather than identity.
    assert sum(type(f).__name__ == "SeamRedirectFinder" for f in sys.meta_path) == 1


@pytest.mark.parametrize("root", sorted(_seam_redirect.REDIRECTED_ROOTS))
def test_legacy_spelling_is_the_canonical_module(root: str) -> None:
    legacy = importlib.import_module(f"src.shared.python.{root}")
    canonical = importlib.import_module(f"shared.python.{root}")
    assert legacy is canonical
    origin = Path(legacy.__file__).resolve()
    assert _VENDOR_SHARED in origin.parents, origin


@pytest.mark.parametrize("root", sorted(_seam_redirect.REDIRECTED_ROOTS))
def test_no_ud_copy_overlaps_the_tools_copy(root: str) -> None:
    ud_pkg = _UD_SHARED / root
    tools_pkg = _VENDOR_SHARED / root
    if ud_pkg.is_dir():
        overlap = {p.relative_to(ud_pkg).as_posix() for p in ud_pkg.rglob("*.py")} & {
            p.relative_to(tools_pkg).as_posix() for p in tools_pkg.rglob("*.py")
        }
        assert overlap == set(), overlap
    else:
        assert not (_UD_SHARED / f"{root}.py").exists()


def test_submodules_are_one_object_under_both_spellings() -> None:
    legacy = importlib.import_module("src.shared.python.logging_pkg.logger_utils")
    canonical = importlib.import_module("shared.python.logging_pkg.logger_utils")
    assert legacy is canonical


def test_split_root_keeps_ud_only_modules_reachable() -> None:
    theme = importlib.import_module("src.shared.python.theme")
    legacy = importlib.import_module("src.shared.python.theme.layout_metrics")
    # The UD-only directory is attached lazily on the first miss.
    assert str(_UD_SHARED / "theme") in list(theme.__path__)
    canonical = importlib.import_module("shared.python.theme.layout_metrics")
    assert legacy is canonical
    assert Path(legacy.__file__).resolve().parent == (_UD_SHARED / "theme").resolve()


def test_theme_exports_fix_8894() -> None:
    from src.shared.python.theme import Colors, Sizes, Weights, get_qfont  # noqa: F401

    from src.shared.python.ui import toast

    assert toast.THEME_AVAILABLE is True


def test_unknown_submodule_still_raises_module_not_found() -> None:
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module("src.shared.python.logging_pkg.does_not_exist")


def test_redirected_roots_match_cleaned_rulings() -> None:
    import json

    rulings = json.loads(
        (_REPO_ROOT / "docs" / "shared_tools" / "seam_rulings.v1.json").read_text(
            encoding="utf-8"
        )
    )["rulings"]
    cleaned = {
        name.removesuffix(".py")
        for name, entry in rulings.items()
        if entry["status"] == "cleaned"
        and entry["ruling"] in {"tools-canonical", "split"}
        and name not in {"README.md", "tests"}
    }
    assert cleaned == set(_seam_redirect.REDIRECTED_ROOTS)
