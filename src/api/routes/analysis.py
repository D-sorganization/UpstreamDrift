"""Analysis routes."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from ..models.requests import AnalysisRequest
from ..models.responses import AnalysisResponse
from ..services.analysis_service import AnalysisService

router = APIRouter()

_analysis_service: AnalysisService | None = None
_logger: Any = None


def configure(analysis_service: AnalysisService | None, logger: Any) -> None:
    """Configure dependencies for analysis routes."""
    global _analysis_service, _logger
    _analysis_service = analysis_service
    _logger = logger


@router.post("/analyze/biomechanics", response_model=AnalysisResponse)
async def analyze_biomechanics(request: AnalysisRequest) -> AnalysisResponse:
    """Perform biomechanical analysis on simulation data."""
    if not _analysis_service:
        raise HTTPException(status_code=500, detail="Analysis service not initialized")

    try:
        result = await _analysis_service.analyze_biomechanics(request)
        return result
    except Exception as exc:
        if _logger:
            _logger.error("Analysis error: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {str(exc)}"
        ) from exc
