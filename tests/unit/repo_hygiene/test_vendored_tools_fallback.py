"""The pinned Tools tree answers for retired child copies, and only for those.

Every ``tools-canonical`` ruling in ``docs/shared_tools/seam_rulings.v1.json``
says "delete UpstreamDrift's copy and let the pinned Tools tree answer". All 36
actionable rulings sat at ``pending-cleanup`` because nothing put that tree on
the import path at runtime: deleting a child copy produced
``ModuleNotFoundError``, so no ruling could be executed (UpstreamDrift#9406).

``src/__init__.py`` now registers the vendored tree as a *fallback*. These tests
pin the two properties that make the cleanup safe to continue:

1. a child copy that still exists is still what imports resolve to, so retiring
   modules one at a time changes nothing until each is actually deleted; and
2. a module with no child copy resolves to the pinned tree instead of failing.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

import src as ud_src

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_UD_SHARED = _REPO_ROOT / "src" / "shared" / "python"
_VENDORED_SHARED = _REPO_ROOT / "vendor" / "ud-tools" / "src" / "shared" / "python"

_requires_vendor = pytest.mark.skipif(
    not _VENDORED_SHARED.is_dir(),
    reason="pinned Tools tree not materialised (git submodule update --init vendor/ud-tools)",
)


@_requires_vendor
def test_fallback_is_appended_so_a_present_child_copy_still_wins() -> None:
    """A module UpstreamDrift still owns must not start resolving upstream.

    This is the property that makes #9406 incremental: the fallback may only
    answer imports that would otherwise fail. If it were prepended, resolution
    would silently flip for the modules that still diverge -- the exact
    ambiguity the epic exists to remove.
    """
    still_owned = "src.shared.python.import_aliases"
    assert (_UD_SHARED / "import_aliases.py").is_file(), (
        "fixture assumption: this module is still an UpstreamDrift child copy"
    )
    assert (_VENDORED_SHARED / "import_aliases.py").is_file(), (
        "fixture assumption: the pinned tree also carries it, so both could match"
    )

    spec = importlib.util.find_spec(still_owned)

    assert spec is not None and spec.origin is not None
    assert Path(spec.origin).resolve() == (_UD_SHARED / "import_aliases.py").resolve()


@_requires_vendor
def test_a_retired_child_copy_resolves_to_the_pinned_tree() -> None:
    """``deprecation`` was retired under its tools-canonical ruling.

    Its child copy is deleted, so this import can only succeed through the
    fallback. It failing means the seam cleanup has regressed and no further
    ruling can be executed.
    """
    assert not (_UD_SHARED / "deprecation.py").exists(), (
        "deprecation.py is retired; a reappearing child copy needs a new ruling"
    )

    spec = importlib.util.find_spec("src.shared.python.deprecation")

    assert spec is not None and spec.origin is not None, (
        "retired child copy no longer resolves; the vendored fallback is broken"
    )
    assert (
        Path(spec.origin).resolve() == (_VENDORED_SHARED / "deprecation.py").resolve()
    )


@_requires_vendor
def test_the_fallback_never_reorders_the_search_path() -> None:
    """The vendored location is appended to the package path, never inserted."""
    assert ud_src._VENDORED_TOOLS_FALLBACK_REGISTERED is True

    import src.shared.python as shared_python

    search_path = [Path(entry).resolve() for entry in list(shared_python.__path__)]

    assert search_path[0] == _UD_SHARED.resolve(), (
        "UpstreamDrift's own directory must stay first on the search path"
    )
    assert _VENDORED_SHARED.resolve() in search_path


def test_absent_module_still_raises_rather_than_resolving_to_nothing() -> None:
    """The fallback must not turn a genuine typo into a silent success."""
    assert importlib.util.find_spec("src.shared.python.definitely_not_a_module") is None
