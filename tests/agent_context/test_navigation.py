"""Real navigation contracts, independent of installed simulation engines."""

from pathlib import Path

from agent_context.catalog import load_catalog
from agent_context.service import ContextService

ROOT = Path(__file__).resolve().parents[2]


def test_motion_context_points_to_api_and_real_boundary() -> None:
    context = ContextService(ROOT).context("motion-pipeline")
    assert "motion-api" in context["consumers"]
    assert any("orchestrator.py" in item["path"] for item in context["sources"])
    assert "HTTP" in " ".join(item["excerpt"] for item in context["sources"])


def test_existing_atlas_and_model_parameters_remain_authorities() -> None:
    catalog = load_catalog(ROOT)
    assert any(
        item["path"] == "src/config/feature_parity.json" for item in catalog.inventories
    )
    matches = ContextService(ROOT).search("model parameters")
    assert matches["matches"][0]["id"] == "model-parameters"
