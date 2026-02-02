"""Analysis routes.

Provides endpoints for biomechanical analysis.
Uses FastAPI's Depends() for dependency injection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException

from ..dependencies import get_analysis_service, get_logger
from ..models.requests import AnalysisRequest
from ..models.responses import AnalysisResponse

if TYPE_CHECKING:
    from ..services.analysis_service import AnalysisService

router = APIRouter()

# Legacy globals for backward compatibility during migration
_analysis_service: AnalysisService | None = None
_logger: Any = None


def configure(analysis_service: AnalysisService | None, logger: Any) -> None:
    """Configure dependencies for analysis routes (legacy).

    Note: This function is deprecated. New code should use Depends() instead.
    """
    global _analysis_service, _logger
    _analysis_service = analysis_service
    _logger = logger


@router.post("/analyze/biomechanics", response_model=AnalysisResponse)
async def analyze_biomechanics(
    request: AnalysisRequest,
    service: AnalysisService = Depends(get_analysis_service),
    logger: Any = Depends(get_logger),
) -> AnalysisResponse:
    """Perform biomechanical analysis on simulation data.

    Args:
        request: Analysis parameters.
        service: Injected analysis service.
        logger: Injected logger.

    Returns:
        Analysis results.

    Raises:
        HTTPException: On analysis failure.
    """
    try:
        result = await service.analyze_biomechanics(request)
        return result
    except Exception as exc:
        if logger:
            logger.error("Analysis error: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Analysis failed: {str(exc)}"
        ) from exc
