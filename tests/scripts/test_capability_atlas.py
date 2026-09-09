"""The capability atlas follows real registries and rejects stale/invalid graphs."""

from copy import deepcopy
from pathlib import Path

import pytest

from scripts.capability_atlas.model import build, validate_graph
from scripts.generate_capability_atlas import outputs
from scripts.capability_atlas.render import catalog, source_link

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]


def test_complete_registry_coverage_and_workflow_branches():
    graph = build(ROOT)
    assert len(graph["features"]) >= 42
    assert len(graph["tiles"]) >= 57
    edges = {(e["source"], e["target"]) for e in graph["edges"]}
    assert ("step.detect", "step.analyze_2d") in edges
    assert ("step.intrinsics", "step.reconstruct") in edges
    assert ("step.annotate", "step.review") in edges
    assert any(
        f["id"] == "tools.capture_rig" and f["status"] == "exempt"
        for f in graph["features"]
    )


@pytest.mark.parametrize("defect", ["duplicate", "endpoint", "evidence", "status"])
def test_invalid_graph_fails(defect):
    graph = deepcopy(build(ROOT))
    if defect == "duplicate":
        graph["nodes"].append(graph["nodes"][0])
    elif defect == "endpoint":
        graph["edges"][0]["target"] = "invented"
    elif defect == "evidence":
        graph["edges"][0]["evidence"] = "../outside.py"
    else:
        graph["features"][0]["status"] = "pretend-ready"
    with pytest.raises(ValueError):
        validate_graph(graph, ROOT)


def test_generated_files_are_fresh_and_deterministic():
    generated = outputs(ROOT)
    assert generated == outputs(ROOT)
    for path, content in generated.items():
        assert path.read_text(encoding="utf-8") == content, (
            f"Stale {path.name}: python3 -m scripts.generate_capability_atlas"
        )


def test_browser_artifact_has_text_alternative_and_safe_data():
    generated = outputs(ROOT)
    html = next(value for path, value in generated.items() if path.suffix == ".html")
    assert 'id="search"' in html
    assert 'aria-live="polite"' in html
    assert "Capability Connections" in html
    assert "<svg" in html and "<title" in html
    assert "https://cdn" not in html


def test_source_links_and_html_escape():
    graph = build(ROOT)
    graph["features"][0]["title"] = '<img src=x onerror="alert(1)">'
    rendered = catalog(graph)
    assert "<img src=x" not in rendered
    assert "&lt;img" in rendered
    assert "/Tools/blob/main/" in source_link("vendor/ud-tools/src/example.py")
    assert all((ROOT / tile["path"]).exists() for tile in graph["tiles"])


def test_atlas_is_reachable_from_existing_launcher_and_project_map():
    launcher = (ROOT / "ui/src/components/simulation/LauncherDashboard.tsx").read_text(
        encoding="utf-8"
    )
    guide = (ROOT / "docs/architecture/PROJECT_MAP.md").read_text(encoding="utf-8")
    assert 'href="/capability-atlas/index.html"' in launcher
    assert "CAPABILITY_ATLAS.md" in guide
