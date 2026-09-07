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
def test_the_fallback_finder_is_consulted_last() -> None:
    """The finder is appended to ``sys.meta_path``, never inserted.

    Being last is the whole safety argument: it is only reached once the normal
    machinery has failed to find a module, so it cannot pre-empt a child copy
    that still exists.
    """
    import sys

    assert ud_src._VENDORED_TOOLS_FALLBACK_FINDER_INSTALLED is True

    positions = [
        index
        for index, finder in enumerate(sys.meta_path)
        if isinstance(finder, ud_src._VendoredToolsFallbackFinder)
    ]

    assert len(positions) == 1, "the fallback finder must be installed exactly once"
    assert positions[0] == len(sys.meta_path) - 1, (
        "the fallback finder must be last on sys.meta_path"
    )


@_requires_vendor
def test_the_fallback_finder_declines_unrelated_modules() -> None:
    """It answers only for the shared namespace, and only when the file exists."""
    finder = ud_src._VendoredToolsFallbackFinder()

    assert finder.find_spec("numpy") is None
    assert finder.find_spec("src.engines.something") is None
    assert finder.find_spec("src.shared.python.definitely_not_a_module") is None


def test_absent_module_still_raises_rather_than_resolving_to_nothing() -> None:
    """The fallback must not turn a genuine typo into a silent success."""
    assert importlib.util.find_spec("src.shared.python.definitely_not_a_module") is None


@_requires_vendor
@pytest.mark.parametrize(
    "module_name",
    [
        "src.shared.python.safe_eval",
        "src.shared.python.safe_pandas_eval",
        "src.shared.python.logging_pkg.logging_config",
        "src.shared.python.chat_contracts.models",
        "src.shared.python.file_watcher",
        "src.shared.python.scripting",
        "src.shared.python.compatibility",
        "src.shared.python.codemap.api",
        "src.shared.python.programmatic_pid.generator",
        "src.shared.python.plot_engine.trendline",
        "src.shared.python.cors",
        "src.shared.python.rotation_transforms.rotation",
    ],
)
def test_retired_clusters_resolve_to_the_pinned_tree(module_name: str) -> None:
    """Each retired cluster must still import, from Tools rather than from here.

    ``upstream_drift_tools`` is deliberately absent from this list. It is a
    deprecated *alias* for ``sidekick``, and ``sidekick`` is still an
    UpstreamDrift-owned cluster, so it correctly resolves here rather than
    upstream -- retiring the alias shim does not move its target.

    These were deleted under their ``tools-canonical`` rulings after every
    tracked file was confirmed byte-identical to the pinned tree, so the
    fallback returns the same bytes that were removed. A failure here means a
    retired module became unreachable, not merely relocated.
    """
    spec = importlib.util.find_spec(module_name)

    assert spec is not None and spec.origin is not None, (
        f"{module_name} was retired but no longer resolves at all"
    )
    assert Path(spec.origin).resolve().is_relative_to(_VENDORED_SHARED.resolve())
