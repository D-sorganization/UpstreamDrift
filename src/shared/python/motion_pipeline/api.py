"""
FastAPI API endpoint for MotionPipeline.

Part of issue #4569. Provides REST API for motion capture pipeline.

Usage:
    from motion_pipeline.api import create_app

    app = create_app()
    # Run with: uvicorn motion_pipeline.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, HTTPException, UploadFile, status, Form
from pydantic import BaseModel, Field

from .contracts import MotionMatchingResult
from .orchestrator import MotionPipeline, PipelineConfig, AdapterOverride

logger = logging.getLogger(__name__)


# =============================================================================
# API Models
# =============================================================================


class PipelineRequest(BaseModel):
    """
    Pydantic-validated request model for motion pipeline API.

    Matches the PipelineConfig from orchestrator with additional file handling.
    """

    # Adapter configuration
    source_format: str = Field(
        ..., description="Source format (c3d, trc, bvh, json, mat, fbx)"
    )
    adapter_options: dict[str, Any] = Field(
        default_factory=dict, description="Format-specific adapter options"
    )

    # Preprocessing steps
    preprocessing: list[dict[str, Any]] = Field(
        default_factory=list, description="Ordered list of preprocessing steps"
    )

    # Scaling configuration
    scaling: dict[str, Any] = Field(
        default_factory=dict, description="Scaling marker map and options"
    )

    # Backend selection
    ik_backend: str = Field(
        default="mujoco", description="IK backend (mujoco, drake, pinocchio, opensim)"
    )
    matching_backend: str = Field(
        default="mujoco", description="Motion matching backend"
    )

    # Cost weights
    cost_weights: dict[str, float] = Field(
        default_factory=dict, description="Cost function weights"
    )

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True

    def to_pipeline_config(self) -> PipelineConfig:
        """Convert to PipelineConfig for orchestrator."""
        return PipelineConfig(
            adapter=AdapterOverride(
                format=self.source_format, options=self.adapter_options
            ),
            preprocessing=[
                {
                    "name": step.get("name", ""),
                    "enabled": step.get("enabled", True),
                    "params": step.get("params", {}),
                }
                for step in self.preprocessing
            ],
            scaling=self.scaling,
            ik_backend=self.ik_backend,
            matching_backend=self.matching_backend,
            cost_weights=self.cost_weights,
        )


class PipelineResponse(BaseModel):
    """
    Response model for motion pipeline API.

    Wraps MotionMatchingResult with additional metadata.
    """

    request_id: str = Field(..., description="Associated request identifier")
    success: bool = Field(..., description="Whether processing succeeded")
    result: dict[str, Any] | None = Field(
        default=None, description="Matched trajectory and metrics (if success)"
    )
    error: str | None = Field(default=None, description="Error message (if failed)")
    audit_log: list[dict[str, Any]] = Field(
        default_factory=list, description="Per-stage audit log for provenance"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict, description="Additional metadata"
    )

    @classmethod
    def from_result(
        cls, result: MotionMatchingResult, audit_log: list[dict[str, Any]]
    ) -> PipelineResponse:
        """Create response from MotionMatchingResult."""
        return cls(
            request_id=result.request_id,
            success=result.success,
            result=result.model_dump() if result.success else None,
            error=result.message if not result.success else None,
            audit_log=audit_log,
            metadata=result.metadata,
        )

    @classmethod
    def from_error(cls, request_id: str, error: str) -> PipelineResponse:
        """Create error response."""
        return cls(request_id=request_id, success=False, error=error, audit_log=[])


# =============================================================================
# FastAPI Application
# =============================================================================


def create_app() -> FastAPI:
    """
    Create FastAPI application with motion pipeline endpoints.

    Returns:
        FastAPI application
    """
    app = FastAPI(
        title="Motion Capture Pipeline API",
        description="REST API for motion capture processing",
        version="1.0.0",
    )

    @app.get("/health")
    async def health_check() -> dict[str, str]:
        """Health check endpoint."""
        return {"status": "healthy"}

    @app.post(
        "/api/v1/motion-pipeline/run",
        response_model=PipelineResponse,
        status_code=status.HTTP_200_OK,
        summary="Run motion pipeline",
        description="""
Process motion capture data through the full pipeline:
adapter → preprocessing → scaling → IK → motion-matching

Accepts file uploads in supported formats (C3D, TRC, BVH, JSON, MAT, FBX).
Returns MotionMatchingResult with matched trajectory and error metrics.
        """,
        responses={
            200: {"description": "Processing completed", "model": PipelineResponse},
            400: {"description": "Invalid input or configuration"},
            422: {"description": "Validation error"},
            500: {"description": "Internal server error"},
        },
    )
    async def run_pipeline(
        file: UploadFile = File(..., description="Motion capture file"),
        source_format: str = Form(..., description="Source format"),
        ik_backend: str = Form(default="mujoco", description="IK backend"),
        matching_backend: str = Form(default="mujoco", description="Matching backend"),
    ) -> PipelineResponse:
        """
        Run motion pipeline on uploaded file.

        Args:
            file: Uploaded motion capture file
            source_format: Format of the source file
            ik_backend: IK backend to use
            matching_backend: Motion matching backend to use

        Returns:
            PipelineResponse with results or error
        """
        import uuid

        request_id = f"req-{uuid.uuid4().hex[:12]}"
        logger.info(f"Received pipeline request {request_id} for {file.filename}")

        # Validate file
        if not file.filename:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST, detail="No filename provided"
            )

        # Save uploaded file temporarily
        try:
            with tempfile.NamedTemporaryFile(
                delete=False, suffix=Path(file.filename).suffix
            ) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = Path(tmp.name)

            # Configure pipeline
            config = PipelineConfig(
                adapter=AdapterOverride(format=source_format),
                ik_backend=ik_backend,
                matching_backend=matching_backend,
            )

            # Create and run pipeline
            pipeline = MotionPipeline(config)

            # Add logging hook
            def log_hook(payload) -> None:
                logger.info(
                    f"Request {request_id}: Stage {payload.stage.value} completed"
                )

            from .orchestrator import Stage

            for stage in Stage:
                pipeline.add_hook(stage, log_hook)

            # Run pipeline
            result = pipeline.run(tmp_path)
            audit_log = pipeline.get_audit_log()

            # Clean up temp file
            tmp_path.unlink(missing_ok=True)

            logger.info(
                f"Request {request_id}: Pipeline completed success={result.success}"
            )

            return PipelineResponse.from_result(result, audit_log)

        except RuntimeError as e:
            logger.error(f"Request {request_id}: Pipeline error: {e}")
            return PipelineResponse.from_error(request_id, str(e))
        except Exception as e:
            logger.exception(f"Request {request_id}: Unexpected error: {e}")
            return PipelineResponse.from_error(request_id, f"Internal error: {e}")

    @app.post(
        "/api/v1/motion-pipeline/run-config",
        response_model=PipelineResponse,
        status_code=status.HTTP_200_OK,
        summary="Run motion pipeline with full config",
        description="""
Run motion pipeline with full configuration control.
Accepts JSON config body plus file upload.
        """,
    )
    async def run_pipeline_with_config(
        file: UploadFile = File(..., description="Motion capture file"),
        config: str = Form(..., description="Pipeline configuration (JSON string)"),
    ) -> PipelineResponse:
        """
        Run motion pipeline with full configuration.

        Args:
            file: Uploaded motion capture file
            config: Pipeline configuration JSON string

        Returns:
            PipelineResponse with results or error
        """
        import uuid

        request_id = f"req-{uuid.uuid4().hex[:12]}"
        logger.info(f"Received pipeline request {request_id} with config")

        try:
            # Parse config from JSON string
            parsed_config = PipelineRequest.model_validate_json(config)

            # Save uploaded file temporarily
            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=Path(file.filename).suffix if file.filename else ".tmp",
            ) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = Path(tmp.name)

            # Convert config and run pipeline
            pipeline_config = parsed_config.to_pipeline_config()
            pipeline = MotionPipeline(pipeline_config)

            result = pipeline.run(tmp_path)
            audit_log = pipeline.get_audit_log()

            tmp_path.unlink(missing_ok=True)

            return PipelineResponse.from_result(result, audit_log)

        except RuntimeError as e:
            return PipelineResponse.from_error(request_id, str(e))
        except Exception as e:
            return PipelineResponse.from_error(request_id, f"Internal error: {e}")

    return app


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    import uvicorn

    app = create_app()
    uvicorn.run(app, host="0.0.0.0", port=8000)
