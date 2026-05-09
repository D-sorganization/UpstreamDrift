"""
MotionPipeline Orchestrator with per-stage hooks.

Part of issue #4569. Composes adapter → preprocessing → scaling → IK → motion-matching,
fully driven by the canonical contracts.

Usage:
    # Python API
    from motion_pipeline.orchestrator import MotionPipeline, PipelineConfig

    config = PipelineConfig(
        source_format="c3d",
        ik_backend="mujoco",
        matching_backend="mujoco",
    )
    pipeline = MotionPipeline(config)
    result = pipeline.run(Path("capture.c3d"))

    # CLI
    python -m motion_pipeline run <input> --engine mujoco --output result.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field

from .contracts import (
    Calibration,
    KeypointSequence,
    MarkerTrajectory,
    MotionMatchingRequest,
    MotionMatchingResult,
    MotionTrajectory,
    SkeletonRig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Stage Definitions
# =============================================================================


class Stage(str, Enum):
    """Pipeline stages."""

    ADAPTER = "adapter"
    PREPROCESSING = "preprocessing"
    SCALING = "scaling"
    INVERSE_KINEMATICS = "inverse_kinematics"
    MOTION_MATCHING = "motion_matching"


class HookPayload(BaseModel):
    """Payload delivered to per-stage hook callbacks.

    Fields:
        stage: The pipeline ``Stage`` that just completed.
        data: Stage output (canonical contract object or tuple, type varies by stage).
        metadata: Free-form metadata describing the stage execution.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    stage: Stage
    data: Any = None
    metadata: dict[str, Any] = Field(default_factory=dict)


@dataclass
class StageResult:
    """Result of a pipeline stage."""

    success: bool
    data: Any
    metadata: dict[str, Any]
    error: str | None = None


# =============================================================================
# Pipeline Configuration
# =============================================================================


class AdapterOverride(BaseModel):
    """Adapter override configuration."""

    format: str = Field(..., description="Source format (c3d, trc, bvh, etc.)")
    options: dict[str, Any] = Field(
        default_factory=dict, description="Format-specific options"
    )


class PreprocessingStep(BaseModel):
    """Preprocessing step configuration."""

    name: str = Field(..., description="Step name")
    enabled: bool = Field(default=True, description="Whether step is enabled")
    params: dict[str, Any] = Field(default_factory=dict, description="Step parameters")


class PipelineConfig(BaseModel):
    """
    Pipeline configuration.

    Drives adapter → preprocessing → scaling → IK → motion-matching.
    """

    # Adapter configuration
    adapter: AdapterOverride = Field(..., description="Source adapter configuration")

    # Preprocessing steps
    preprocessing: list[PreprocessingStep] = Field(
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

    # Cost weights for IK/matching
    cost_weights: dict[str, float] = Field(
        default_factory=dict, description="Cost function weights"
    )

    # Output configuration
    output_format: str = Field(default="json", description="Output format")

    class Config:
        """Pydantic config."""

        arbitrary_types_allowed = True


# =============================================================================
# MotionPipeline Orchestrator
# =============================================================================


class MotionPipeline:
    """
    Single entry point for motion capture pipeline.

    Composes: adapter → preprocessing → scaling → IK → motion-matching
    Fully driven by canonical contracts.

    Features:
    - Hooks for observability (UI/GUI progress)
    - Audit logging per stage
    - Pydantic-validated configuration
    - CLI and API entry points
    """

    def __init__(self, config: PipelineConfig):
        """
        Initialize pipeline with configuration.

        Args:
            config: Pipeline configuration
        """
        self.config = config
        self._hooks: dict[Stage, list[Callable]] = {stage: [] for stage in Stage}
        self._audit_log: list[dict[str, Any]] = []
        self._source_hash: str | None = None
        self._config_hash: str | None = None

    def add_hook(self, stage: Stage, fn: Callable[[HookPayload], None]) -> None:
        """
        Add a hook to be called after a stage completes.

        Hooks let UI/GUI subscribe to per-stage progress and intermediate artifacts.

        Args:
            stage: Pipeline stage to hook into
            fn: Callback receiving a ``HookPayload`` Pydantic model with
                ``stage`` (Stage), ``data`` (stage output), and
                ``metadata`` (dict[str, Any]) fields.
        """
        self._hooks[stage].append(fn)
        logger.debug(f"Added hook for stage {stage.value}")

    def _compute_hash(self, data: bytes | str) -> str:
        """Compute SHA256 hash for provenance."""
        if isinstance(data, str):
            data = data.encode()
        return hashlib.sha256(data).hexdigest()

    def _log_audit(self, stage: Stage, result: StageResult, source_hash: str) -> None:
        """Log audit entry for a stage."""
        entry = {
            "stage": stage.value,
            "timestamp": datetime.now().isoformat(),
            "success": result.success,
            "source_hash": source_hash,
            "config_hash": self._config_hash,
            "software_version": self._get_version(),
            "metadata": result.metadata,
        }
        self._audit_log.append(entry)
        logger.info(f"Audit logged for stage {stage.value}")

    def _get_version(self) -> str:
        """Get software version."""
        # Try to get version from package
        try:
            import importlib.metadata

            return importlib.metadata.version("upstream-drift")
        except (ImportError, importlib.metadata.PackageNotFoundError):
            return "dev"

    def _fire_hooks(self, stage: Stage, data: Any, metadata: dict[str, Any]) -> None:
        """Fire all hooks for a stage."""
        payload = HookPayload(stage=stage, data=data, metadata=metadata)
        for hook in self._hooks[stage]:
            try:
                hook(payload)
            except Exception as e:
                logger.warning(f"Hook for {stage.value} failed: {e}")

    # -------------------------------------------------------------------------
    # Stage Implementations
    # -------------------------------------------------------------------------

    def _run_adapter(
        self, source: Path | KeypointSequence | MarkerTrajectory
    ) -> StageResult:
        """
        Run adapter stage: convert source to canonical format.

        Adapters convert raw formats (C3D, TRC, BVH, JSON) to CIR.
        """
        from .sources.loader import load_source

        try:
            if isinstance(source, (KeypointSequence, MarkerTrajectory)):
                # Already in canonical format
                return StageResult(
                    success=True,
                    data=source,
                    metadata={
                        "source_type": type(source).__name__,
                        "adapter": "passthrough",
                    },
                )

            if isinstance(source, Path):
                # Load from file
                data = load_source(source, format_hint=self.config.adapter.format)
                self._source_hash = self._compute_hash(source.read_bytes())
                return StageResult(
                    success=True,
                    data=data,
                    metadata={
                        "source_type": type(data).__name__,
                        "adapter": self.config.adapter.format,
                        "source_path": str(source),
                    },
                )

            return StageResult(
                success=False,
                data=None,
                metadata={},
                error=f"Unknown source type: {type(source)}",
            )

        except Exception as e:
            return StageResult(
                success=False, data=None, metadata={}, error=f"Adapter failed: {e}"
            )

    def _run_preprocessing(self, data: Any) -> StageResult:
        """
        Run preprocessing stage: clean, filter, interpolate.

        Preprocessing steps are configured in PipelineConfig.
        """
        from .preprocessing import apply_preprocessing

        try:
            steps = [step for step in self.config.preprocessing if step.enabled]

            processed = apply_preprocessing(data, steps)

            return StageResult(
                success=True,
                data=processed,
                metadata={
                    "steps_applied": [s.name for s in steps],
                    "num_frames": getattr(processed, "num_frames", None),
                },
            )

        except Exception as e:
            return StageResult(
                success=False,
                data=None,
                metadata={},
                error=f"Preprocessing failed: {e}",
            )

    def _run_scaling(self, data: Any, skeleton: SkeletonRig) -> StageResult:
        """
        Run scaling stage: scale skeleton to subject.

        Uses marker map from config to scale the skeleton.
        """
        from .scaling import scale_skeleton

        try:
            scaled_skeleton = scale_skeleton(skeleton, data, **self.config.scaling)

            return StageResult(
                success=True,
                data=(data, scaled_skeleton),
                metadata={
                    "scale_factor": getattr(scaled_skeleton, "scale", 1.0),
                    "markers_used": len(getattr(data, "marker_names", [])),
                },
            )

        except Exception as e:
            return StageResult(
                success=False, data=None, metadata={}, error=f"Scaling failed: {e}"
            )

    def _run_inverse_kinematics(self, data: Any, skeleton: SkeletonRig) -> StageResult:
        """
        Run IK stage: compute joint angles from markers/keypoints.
        """
        backend = self.config.ik_backend

        try:
            # Import backend dynamically (LoD: no direct imports)
            if backend == "mujoco":
                from .ik.mujoco_backend import run_ik
            elif backend == "drake":
                from .ik.drake_backend import run_ik
            elif backend == "pinocchio":
                from .ik.pinocchio_backend import run_ik
            elif backend == "opensim":
                from .ik.opensim_backend import run_ik
            else:
                return StageResult(
                    success=False,
                    data=None,
                    metadata={},
                    error=f"Unknown IK backend: {backend}",
                )

            trajectory = run_ik(data, skeleton, weights=self.config.cost_weights)

            return StageResult(
                success=True,
                data=trajectory,
                metadata={
                    "backend": backend,
                    "num_frames": trajectory.num_frames if trajectory else 0,
                    "num_dofs": trajectory.num_dofs if trajectory else 0,
                },
            )

        except ImportError as e:
            return StageResult(
                success=False,
                data=None,
                metadata={},
                error=f"IK backend not available: {e}",
            )
        except Exception as e:
            return StageResult(
                success=False, data=None, metadata={}, error=f"IK failed: {e}"
            )

    def _run_motion_matching(
        self, trajectory: MotionTrajectory, skeleton: SkeletonRig
    ) -> StageResult:
        """
        Run motion matching stage: refine trajectory via optimization.
        """
        backend = self.config.matching_backend

        try:
            # Import backend dynamically
            if backend == "mujoco":
                from .matching.mujoco_backend import run_matching
            elif backend == "drake":
                from .matching.drake_backend import run_matching
            elif backend == "pinocchio":
                from .matching.pinocchio_backend import run_matching
            else:
                return StageResult(
                    success=False,
                    data=None,
                    metadata={},
                    error=f"Unknown matching backend: {backend}",
                )

            # Build request
            request = MotionMatchingRequest(
                id=f"mm-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                target_trajectory=trajectory,
                skeleton=skeleton,
                constraints=self.config.scaling,
                solver_config=self.config.cost_weights,
            )

            result = run_matching(request)

            return StageResult(
                success=result.success,
                data=result,
                metadata={
                    "backend": backend,
                    "iterations": result.iterations,
                    "solve_time": result.solve_time,
                    "error_metrics": result.error_metrics,
                },
            )

        except ImportError as e:
            return StageResult(
                success=False,
                data=None,
                metadata={},
                error=f"Matching backend not available: {e}",
            )
        except Exception as e:
            return StageResult(
                success=False,
                data=None,
                metadata={},
                error=f"Motion matching failed: {e}",
            )

    # -------------------------------------------------------------------------
    # Main Entry Point
    # -------------------------------------------------------------------------

    def run(
        self, source: Path | KeypointSequence | MarkerTrajectory
    ) -> MotionMatchingResult:
        """
        Run the full pipeline.

        Composes: adapter → preprocessing → scaling → IK → motion-matching

        Args:
            source: Input source (file path or canonical data)

        Returns:
            MotionMatchingResult with matched trajectory and metrics

        Raises:
            ValueError: If source is invalid
            RuntimeError: If pipeline stage fails
        """
        # Compute config hash for provenance
        self._config_hash = self._compute_hash(self.config.model_dump_json())

        logger.info(f"Starting pipeline run for source: {source}")

        # Stage 1: Adapter
        adapter_result = self._run_adapter(source)
        if not adapter_result.success:
            raise RuntimeError(f"Adapter stage failed: {adapter_result.error}")

        self._fire_hooks(Stage.ADAPTER, adapter_result.data, adapter_result.metadata)
        self._log_audit(Stage.ADAPTER, adapter_result, self._source_hash or "")

        data = adapter_result.data

        # Stage 2: Preprocessing
        preprocess_result = self._run_preprocessing(data)
        if not preprocess_result.success:
            raise RuntimeError(f"Preprocessing failed: {preprocess_result.error}")

        self._fire_hooks(
            Stage.PREPROCESSING, preprocess_result.data, preprocess_result.metadata
        )
        self._log_audit(Stage.PREPROCESSING, preprocess_result, self._source_hash or "")

        data = preprocess_result.data

        # Stage 3: Scaling (requires skeleton)
        # For now, use default skeleton from adapter result
        skeleton = getattr(data, "skeleton", self._get_default_skeleton())

        scaling_result = self._run_scaling(data, skeleton)
        if not scaling_result.success:
            raise RuntimeError(f"Scaling failed: {scaling_result.error}")

        self._fire_hooks(Stage.SCALING, scaling_result.data, scaling_result.metadata)
        self._log_audit(Stage.SCALING, scaling_result, self._source_hash or "")

        data, scaled_skeleton = scaling_result.data

        # Stage 4: Inverse Kinematics
        ik_result = self._run_inverse_kinematics(data, scaled_skeleton)
        if not ik_result.success:
            raise RuntimeError(f"IK failed: {ik_result.error}")

        self._fire_hooks(Stage.INVERSE_KINEMATICS, ik_result.data, ik_result.metadata)
        self._log_audit(Stage.INVERSE_KINEMATICS, ik_result, self._source_hash or "")

        # Build motion trajectory
        trajectory = MotionTrajectory(
            id=f"motion-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            skeleton=scaled_skeleton,
            trajectory=ik_result.data,
            source_provenance={
                "source_hash": self._source_hash,
                "config_hash": self._config_hash,
                "software_version": self._get_version(),
            },
        )

        # Stage 5: Motion Matching
        matching_result = self._run_motion_matching(trajectory, scaled_skeleton)
        if not matching_result.success:
            raise RuntimeError(f"Motion matching failed: {matching_result.error}")

        self._fire_hooks(
            Stage.MOTION_MATCHING, matching_result.data, matching_result.metadata
        )
        self._log_audit(Stage.MOTION_MATCHING, matching_result, self._source_hash or "")

        result = matching_result.data
        assert isinstance(result, MotionMatchingResult)

        # Add audit log to result metadata
        result.metadata["audit_log"] = self._audit_log
        result.metadata["config"] = self.config.model_dump()

        logger.info(f"Pipeline run completed: success={result.success}")

        return result

    def _get_default_skeleton(self) -> SkeletonRig:
        """Get default skeleton for pipeline."""
        # This would load a default skeleton based on config
        # For now, raise error - caller should provide skeleton
        raise RuntimeError("No skeleton provided. Use adapter that provides one.")

    def get_audit_log(self) -> list[dict[str, Any]]:
        """Get audit log for provenance."""
        return self._audit_log.copy()


# =============================================================================
# CLI Entry Point
# =============================================================================


def cli_run(source: str, output: str, engine: str, **kwargs) -> None:
    """
    CLI entry point for pipeline.

    Usage:
        python -m motion_pipeline run <input> --engine mujoco --output result.json
    """
    source_path = Path(source)

    if not source_path.exists():
        sys.stderr.write(f"Error: Source file not found: {source_path}\n")
        sys.exit(1)

    # Configure pipeline
    config = PipelineConfig(
        adapter=AdapterOverride(format=_detect_format(source_path)),
        ik_backend=engine,
        matching_backend=engine,
        cost_weights=kwargs.get("weights", {}),
    )

    pipeline = MotionPipeline(config)

    # Add progress hook for CLI
    def progress_hook(payload: HookPayload) -> None:
        sys.stdout.write(f"  [{payload.stage.value}] completed\n")

    for stage in Stage:
        pipeline.add_hook(stage, progress_hook)

    # Run pipeline
    try:
        sys.stdout.write(f"Processing: {source_path}\n")
        result = pipeline.run(source_path)

        if result.success:
            # Save result
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(result.model_dump_json(indent=2))
            sys.stdout.write(f"Result saved to: {output_path}\n")
            sys.exit(0)
        else:
            sys.stderr.write(f"Pipeline failed: {result.message}\n")
            sys.exit(1)

    except RuntimeError as e:
        sys.stderr.write(f"Pipeline error: {e}\n")
        sys.exit(1)
    except Exception as e:
        sys.stderr.write(f"Unexpected error: {e}\n")
        sys.exit(1)


def _detect_format(path: Path) -> str:
    """Detect source format from file extension."""
    suffix = path.suffix.lower()
    format_map = {
        ".c3d": "c3d",
        ".trc": "trc",
        ".bvh": "bvh",
        ".json": "json",
        ".mat": "mat",
        ".fbx": "fbx",
    }
    return format_map.get(suffix, "unknown")


# =============================================================================
# Main
# =============================================================================


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Motion Capture Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m motion_pipeline run capture.c3d --engine mujoco --output result.json
  python -m motion_pipeline run pose.json --engine drake --output drake_result.json
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run pipeline on source")
    run_parser.add_argument("source", type=str, help="Source file path")
    run_parser.add_argument(
        "--engine",
        type=str,
        default="mujoco",
        choices=["mujoco", "drake", "pinocchio", "opensim"],
        help="Backend engine",
    )
    run_parser.add_argument(
        "--output", type=str, default="result.json", help="Output file path"
    )
    run_parser.add_argument(
        "--weights", type=str, default="{}", help="Cost weights as JSON string"
    )

    args = parser.parse_args()

    if args.command == "run":
        cli_run(
            source=args.source,
            output=args.output,
            engine=args.engine,
            weights=json.loads(args.weights),
        )
    else:
        parser.print_help()
        sys.exit(1)
