from pathlib import Path

import pytest
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


pytestmark = pytest.mark.live_simulation


def test_nightly_cross_engine_workflow_uses_real_validator_suite() -> None:
    """This legacy heavy-integration placeholder must not be the nightly target."""
    repo_root = Path(__file__).resolve().parents[2]
    workflow = (
        repo_root / ".github" / "workflows" / "nightly-cross-engine.yml"
    ).read_text(encoding="utf-8")

    assert "tests/heavy_integration/test_cross_engine_integration.py" not in workflow
    assert "tests/integration/cross_engine/test_conformance_harness.py" in workflow
