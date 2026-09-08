"""The pinned Tools alias predicate must gate on install layout (Tools#5048).

``_external_src_package_is_available()`` decides which roots
``SharedImportAliasFinder`` may alias for the ``src.shared.python.*`` spelling.
Its repo-relative test describes a *repository* layout: this module at
``<repo>/src/shared/python`` with a downstream ``src`` outside that repo. In a
flattened install (``<site-packages>/shared/python``) the pre-Tools#5049
predicate read every installed package -- a downstream ``src`` included -- as
internal, so the finder rewrote every ``src.shared.python.<root>`` into the
Tools tree. UpstreamDrift's 33-symbol ``config`` then resolved into Tools'
unrelated 5-symbol one and the v2.1.2 wheel could not import
``get_database_pool_pre_ping`` (D-sorganization/UpstreamDrift#9631).

This suite pins the Tools#5048 contract against the exact ``vendor/ud-tools``
revision this repository vendors, in both layouts:

- repository layout: an external ``src`` is available, so downstream alias
  roots (``sidekick``, ``chat``, ...) alias to the Tools tree while clusters
  the downstream owns (``config``) never do;
- flattened install: a consumer ``src`` flattened beside ``shared/python``
  must still read as external, so the consumer keeps its own ``config``.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest

from tests.unit.repo_hygiene.test_vendored_tools_fallback import (
    _REPO_ROOT,
    _check_vendored_tools,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_VENDORED_SHARED = _REPO_ROOT / "vendor" / "ud-tools" / "src" / "shared" / "python"

_DOWNSTREAM_ONLY_ROOT = "config"  # UpstreamDrift-owned; Tools' copy is unrelated
_DOWNSTREAM_ALIAS_ROOT = "sidekick"  # ruled downstream-aliasable


@pytest.fixture(autouse=True)
def _require_pinned_tree() -> None:
    """Skip when the pinned tree is absent locally; fail closed in CI."""
    _check_vendored_tools()


def _load_vendored_module(unique_name: str, module_path: Path):
    """Load a fresh copy of ``import_aliases`` from ``module_path``."""
    spec = importlib.util.spec_from_file_location(unique_name, module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _present_fake_src_package(monkeypatch: pytest.MonkeyPatch, src_root: Path) -> None:
    """Present ``src`` to the finder as a package rooted at ``src_root``."""
    fake_src = types.ModuleType("src")
    fake_src.__path__ = (str(src_root),)  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "src", fake_src)


def test_repo_layout_keeps_a_consumer_config_out_of_the_tools_tree(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """In a repository layout an external ``src`` exists and ``config`` stays ours."""
    module = _load_vendored_module(
        "_pinned_import_aliases_repo_layout", _VENDORED_SHARED / "import_aliases.py"
    )
    _present_fake_src_package(monkeypatch, _REPO_ROOT / "src")

    assert module._external_src_package_is_available() is True
    finder = module.SharedImportAliasFinder()
    assert finder._parse(f"src.shared.python.{_DOWNSTREAM_ONLY_ROOT}") == (None, "")
    assert finder._parse(f"src.shared.python.{_DOWNSTREAM_ALIAS_ROOT}") == (
        _DOWNSTREAM_ALIAS_ROOT,
        "",
    )


def test_flattened_install_keeps_a_consumer_config_out_of_the_tools_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A consumer ``src`` flattened beside ``shared/python`` stays external.

    This is the exact shape that broke the v2.1.2 wheel: with the pre-#5049
    predicate the consumer's ``src`` read as internal, every shared root was
    aliased, and ``src.shared.python.config`` resolved into the Tools copy.
    """
    install_root = tmp_path / "site-packages"
    flattened_shared = install_root / "shared" / "python"
    flattened_shared.mkdir(parents=True)
    (flattened_shared / "import_aliases.py").write_text(
        (_VENDORED_SHARED / "import_aliases.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    consumer_src = install_root / "src"

    module = _load_vendored_module(
        "_pinned_import_aliases_flattened", flattened_shared / "import_aliases.py"
    )
    # The flattened install reports a non-``src`` Tools root; the consumer's
    # ``src`` sits beside ``shared/python`` inside the same install root.
    assert install_root.resolve() == module._TOOLS_SRC_ROOT
    _present_fake_src_package(monkeypatch, consumer_src)

    assert module._external_src_package_is_available() is True
    finder = module.SharedImportAliasFinder()
    assert finder._parse(f"src.shared.python.{_DOWNSTREAM_ONLY_ROOT}") == (None, "")
    assert finder._parse(f"src.shared.python.{_DOWNSTREAM_ALIAS_ROOT}") == (
        _DOWNSTREAM_ALIAS_ROOT,
        "",
    )
