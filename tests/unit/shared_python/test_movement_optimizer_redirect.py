"""The retired UD Movement Optimizer copy resolves to the pinned Tools package (#12267).

Tools owns ``movement_optimizer`` as a *top-level* package
(``vendor/ud-tools/src/movement_optimizer``), so the ``shared.python`` seam
guards never saw the drifted copy under ``src/shared/python``. These tests
pin the retirement: the legacy spelling is the vendored module object, it
warns, and the copy cannot quietly come back.
"""

from __future__ import annotations

import importlib
import sys
import warnings
from pathlib import Path

import pytest

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_VENDOR_PKG = _REPO_ROOT / "vendor" / "ud-tools" / "src" / "movement_optimizer"
_UD_COPY = _REPO_ROOT / "src" / "shared" / "python" / "movement_optimizer"
_LEGACY = "src.shared.python.movement_optimizer"

if not _VENDOR_PKG.is_dir():  # pragma: no cover - CI always has the submodule
    pytest.skip("vendor/ud-tools not materialised", allow_module_level=True)

from src.shared.python import _seam_redirect  # noqa: E402


def _forget_legacy_spelling() -> None:
    for name in [n for n in sys.modules if n == _LEGACY or n.startswith(_LEGACY + ".")]:
        del sys.modules[name]


def test_ud_copy_is_gone() -> None:
    assert not _UD_COPY.exists(), (
        "src/shared/python/movement_optimizer is a retired copy of the Tools "
        "package; edit Tools and bump vendor/ud-tools instead (#12267)."
    )
    assert not (_UD_COPY.parent / "movement_optimizer.py").exists()


def test_movement_optimizer_is_a_top_level_redirect() -> None:
    assert _seam_redirect.TOP_LEVEL_REDIRECTS["movement_optimizer"] == (
        "movement_optimizer"
    )
    assert "movement_optimizer" not in _seam_redirect.REDIRECTED_ROOTS


def test_legacy_spelling_is_the_vendored_module_and_warns() -> None:
    _forget_legacy_spelling()
    with pytest.warns(DeprecationWarning, match="#12267"):
        legacy = importlib.import_module(_LEGACY)
    canonical = importlib.import_module("movement_optimizer")
    assert legacy is canonical
    assert _VENDOR_PKG.resolve() in Path(legacy.__file__).resolve().parents


def test_legacy_submodules_are_one_object() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        legacy = importlib.import_module(f"{_LEGACY}.models")
        legacy_traj = importlib.import_module(f"{_LEGACY}.trajectory")
    assert legacy is importlib.import_module("movement_optimizer.models")
    assert legacy_traj is importlib.import_module("movement_optimizer.trajectory")


def test_unknown_legacy_submodule_raises_module_not_found() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ModuleNotFoundError):
            importlib.import_module(f"{_LEGACY}.does_not_exist")
