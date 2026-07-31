"""Focused coverage for Sidekick feature catalog search and discovery."""

from __future__ import annotations

import ast
import types

import pytest
from sidekick.agent import feature_catalog as catalog_mod
from sidekick.agent import feature_discovery as discovery
from sidekick.agent.feature_types import FeatureEntry, FeatureKind

pytestmark = pytest.mark.unit


def _entry(feature_id: str, title: str = "Alpha Tool") -> FeatureEntry:
    kind = feature_id.split(".", maxsplit=1)[0]
    return FeatureEntry(
        feature_id=feature_id,
        kind=kind,
        title=title,
        summary="Alpha beta summary",
        module="sidekick",
    )


def test_feature_entry_validates_kind_namespace_and_required_fields() -> None:
    with pytest.raises(ValueError, match="namespace"):
        FeatureEntry(
            feature_id="calculator.bad",
            kind=FeatureKind.SUBTAB.value,
            title="Bad",
            summary="Bad",
            module="sidekick",
        )


def test_build_feature_catalog_sorts_caches_and_skips_failed_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = {"ok": 0}

    def ok_source() -> list[FeatureEntry]:
        calls["ok"] += 1
        return [_entry("subtab.zeta"), _entry("calculator.alpha")]

    def broken_source() -> list[FeatureEntry]:
        raise RuntimeError("broken import")

    monkeypatch.setattr(catalog_mod, "_CATALOG_CACHE", None)
    monkeypatch.setattr(
        catalog_mod, "discover_sources", lambda: (broken_source, ok_source)
    )

    first = catalog_mod.build_feature_catalog(force_refresh=True)
    second = catalog_mod.build_feature_catalog()

    assert list(first) == ["calculator.alpha", "subtab.zeta"]
    assert first is second
    assert calls["ok"] == 1


def test_lookup_and_search_validate_inputs_and_rank_matches(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    entries = {
        "calculator.alpha": _entry("calculator.alpha", "Alpha Calculator"),
        "subtab.workspace": _entry("subtab.workspace", "Workspace Browser"),
    }
    monkeypatch.setattr(catalog_mod, "build_feature_catalog", lambda: entries)

    assert catalog_mod.lookup_feature("calculator.alpha").title == "Alpha Calculator"
    assert (
        catalog_mod.search_features("workspace", limit=1)[0].feature_id
        == "subtab.workspace"
    )
    with pytest.raises(ValueError, match="non-blank"):
        catalog_mod.search_features(" ")
    with pytest.raises(KeyError, match="calculator.alpha"):
        catalog_mod.lookup_feature("calculator.alph")


def test_discovery_helpers_extract_metadata_and_walk_fake_package(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    tree = ast.parse(
        "DEFAULT_SIDEBAR_TAB_HELP = {"
        "'calc': HelpMeta('Calc', 'Run calc', source='pkg.child')"
        "}"
    )
    dict_node = next(
        node.value for node in ast.walk(tree) if isinstance(node, ast.Assign)
    )

    assert discovery._extract_help_dict(dict_node) == {  # noqa: SLF001
        "calc": {"title": "Calc", "summary": "Run calc", "source": "pkg.child"}
    }

    package = types.SimpleNamespace(__path__=["fake"])
    module = types.SimpleNamespace(__doc__="Fake Module - does useful work.")
    imports = {
        "sidekick.calculators": package,
        "sidekick.calculators.fake_module": module,
    }
    monkeypatch.setattr(
        discovery.importlib, "import_module", lambda name: imports[name]
    )
    monkeypatch.setattr(
        discovery.pkgutil,
        "iter_modules",
        lambda path: iter(
            [
                types.SimpleNamespace(name="_private"),
                types.SimpleNamespace(name="fake_module"),
            ]
        ),
    )

    walked = list(
        discovery._walk_package("sidekick.calculators", "calculator")
    )  # noqa: SLF001

    assert walked[0].feature_id == "calculator.fake_module"
    assert walked[0].title == "Fake Module"


def test_workflow_and_importability_discovery(monkeypatch: pytest.MonkeyPatch) -> None:
    workflow = types.SimpleNamespace(name="Build", description="Build things")
    monkeypatch.setattr(
        discovery.importlib,
        "import_module",
        lambda name: types.SimpleNamespace(WORKFLOWS={"build": workflow}),
    )

    workflows = discovery._discover_workflows()  # noqa: SLF001

    assert workflows[0].feature_id == "workflow.build"
    assert (
        discovery._discover_theme()[0].feature_id == "theme.sidekick_tokens"
    )  # noqa: SLF001
    assert tuple(src.__name__ for src in discovery.discover_sources()) == (
        "_discover_calculators",
        "_discover_process_calculators",
        "_discover_theme",
        "_discover_workflows",
        "_discover_subtabs",
    )


def test_iter_named_supports_sequences() -> None:
    items = [types.SimpleNamespace(id="one"), types.SimpleNamespace(name="two")]

    assert list(discovery._iter_named(items)) == [  # noqa: SLF001
        ("one", items[0]),
        ("two", items[1]),
    ]
