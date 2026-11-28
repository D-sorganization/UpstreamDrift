"""Tests for SRP-7 registry SOPs, semantic glossary, and list/describe CLI (#15734)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from src.tools.swing_report_package import registry_docs
from src.tools.swing_report_package.catalog import CATALOG_PATH, load_catalog
from src.tools.swing_report_package.registry_docs import (
    GlossaryTerm,
    ItemDescription,
    RegistryDocumentationBundle,
    build_registry_documentation_bundle,
    describe_registry_item,
    generate_registry_docs_json,
    generate_registry_docs_markdown,
    list_registry_items,
    main,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def test_list_registry_items_matches_catalog_and_custom_additions(
    tmp_path: Path,
) -> None:
    """list_registry_items mirrors load_catalog() dynamically without hardcoding."""
    catalog = load_catalog()
    listed = list_registry_items(catalog)
    assert listed == tuple(catalog.items.keys())
    assert len(listed) == len(set(listed))

    doc = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
    doc["items"]["custom_probe"] = {
        "kind": "video",
        "title": "Custom Probe Render",
        "args": {"ztcf_zvcf_view": "total"},
    }
    custom_path = tmp_path / "catalog.json"
    custom_path.write_text(json.dumps(doc), encoding="utf-8")
    assert "custom_probe" in list_registry_items(load_catalog(custom_path))


def test_describe_registry_item_returns_sops_caveats_and_fails_closed() -> None:
    """describe_registry_item returns SOPs/caveats and fails closed on unknown IDs."""
    catalog = load_catalog()
    desc = describe_registry_item("midhands_total", catalog)
    assert isinstance(desc, ItemDescription)
    assert (
        desc.kind == "video" and desc.title == catalog.items["midhands_total"]["title"]
    )
    assert {"default", "midhands", "full"} <= set(desc.packages)
    assert "src.tools.swing_report_package" in desc.cli_sop
    assert "plan_package" in desc.python_sop
    ast.parse(desc.python_sop)
    assert desc.caveats and desc.effective_args["ztcf_zvcf_hands"] == "midhands"

    for bad_id in ("nonexistent_item_id", ""):
        with pytest.raises((KeyError, ValueError)):
            describe_registry_item(bad_id, catalog)


def test_generate_registry_docs_covers_all_items_packages_and_glossary() -> None:
    """Generated Markdown/JSON include all items, packages, and semantic glossary."""
    catalog = load_catalog()
    bundle = build_registry_documentation_bundle(catalog)
    assert isinstance(bundle, RegistryDocumentationBundle)
    assert all(isinstance(term, GlossaryTerm) for term in bundle.glossary)

    md = generate_registry_docs_markdown(catalog)
    assert md == generate_registry_docs_markdown(catalog)
    assert all(item_id in md for item_id in catalog.items)
    assert all(pkg in md for pkg in ("default", *catalog.packages))
    for keyword in ("ZTCF", "ZVCF", "face_on", "down_the_line", "tension", "caveat"):
        assert keyword.lower() in md.lower()

    raw_json = generate_registry_docs_json(catalog)
    assert raw_json == generate_registry_docs_json(catalog)
    parsed = json.loads(raw_json)
    assert [e["item_id"] for e in parsed["items"]] == list(catalog.items.keys())
    assert set(parsed["packages"]) == {"default", *catalog.packages}
    cats = {t["category"] for t in parsed["glossary"]}
    assert {"coordinate_frame", "sign_convention", "physical_caveat"} <= cats


def test_cli_list_describe_docs_and_no_print_calls(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CLI list/describe/docs output deterministic JSON/Markdown with zero print()."""
    catalog = load_catalog()
    assert main(["list"]) == 0
    first_list = capsys.readouterr().out
    assert main(["list"]) == 0 and capsys.readouterr().out == first_list
    assert json.loads(first_list)["items"] == list(catalog.items.keys())

    assert main(["list", "--format", "markdown"]) == 0
    list_md = capsys.readouterr().out
    assert all(i in list_md for i in catalog.items)

    assert main(["describe", "case_grid"]) == 0
    desc = json.loads(capsys.readouterr().out)
    assert desc["item_id"] == "case_grid"
    assert desc["needs"] == ["case_ztcf", "case_zvcf", "case_total"]

    assert main(["describe", "case_grid", "--format", "markdown"]) == 0
    assert "case_grid" in capsys.readouterr().out
    assert main(["docs", "--format", "json"]) == 0
    assert "glossary" in json.loads(capsys.readouterr().out)

    assert main(["describe", "unknown_item"]) == 2
    assert "unknown" in capsys.readouterr().err.lower()

    tree = ast.parse(Path(registry_docs.__file__).read_text(encoding="utf-8"))
    calls = [
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert "print" not in calls
