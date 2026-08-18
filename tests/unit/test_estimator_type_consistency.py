"""API-validated estimator types must match implemented ones (#8392).

Previously the API accepted ``movenet`` (which then 500'd inside
``VideoPosePipeline._load_estimator``) while the web UI offered
``blazepose`` (which the API rejected with a 400). This suite pins the
two Python surfaces to a single set; the UI list is pinned by
``ui/src/pages/VideoAnalyzer.test.tsx``.
"""

from __future__ import annotations

from src.api.config import VALID_ESTIMATOR_TYPES
from src.shared.python.pose_estimation.interface import IMPLEMENTED_ESTIMATOR_TYPES


def test_api_estimator_set_matches_pipeline_implementations() -> None:
    assert set(IMPLEMENTED_ESTIMATOR_TYPES) == VALID_ESTIMATOR_TYPES


def test_no_phantom_estimators() -> None:
    """Estimators that never had an implementation must not be advertised."""
    assert "movenet" not in VALID_ESTIMATOR_TYPES
    assert "blazepose" not in VALID_ESTIMATOR_TYPES
