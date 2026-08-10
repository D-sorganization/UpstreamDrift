"""
MotionPipeline Orchestrator with per-stage hooks.

Part of issue #4569. Composes adapter → preprocessing → scaling → IK → motion-matching,
fully driven by the canonical contracts.

Usage:
    # Python API
    from motion_pipeline.orchestrator import MotionPipeline, PipelineConfig

    config = PipelineConfig(
        source_format="c3d",
        ik_backend="geometric",
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
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .contracts import (
    KeypointSequence,
    MarkerTrajectory,
    MotionMatchingResult,
    MotionTrajectory,
    SkeletonRig,
)
from .matching.base import MatchingBackendType

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


class InvalidInputError(ValueError):
    """Raised when a contract precondition on caller-supplied input fails.

    Distinguishes client-side contract violations (bad source format,
    unknown backend, unsupported source type) from internal pipeline
    faults so the API layer can map them to 4xx vs 5xx (issue #6932).
    """


class HookExecutionError(RuntimeError):
    """Raised when a per-stage hook fails in strict hook mode."""

    def __init__(
        self,
        *,
        stage: Stage,
        hook_name: str,
        original: BaseException,
    ) -> None:
        self.stage = stage
        self.hook_name = hook_name
        self.original = original
        super().__init__(
            f"Hook {hook_name!r} for stage {stage.value!r} failed: "
            f"{type(original).__name__}: {original}"
        )


@dataclass
class StageResult:
    """Result of a pipeline stage.

    ``error_kind`` classifies a failure as ``"invalid_input"`` (a caller
    contract violation -> HTTP 4xx) or ``"internal"`` (a server-side fault
    -> HTTP 5xx); see issue #6932.
    """

    success: bool
    data: Any
    metadata: dict[str, Any]
    error: str | None = None
    error_kind: str = "internal"


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
        default="geometric",
        description="IK backend (geometric, mujoco, drake, pinocchio, opensim)",
    )
    matching_backend: str = Field(
        default="mujoco", description="Motion matching backend"
    )
    matching_model_urdf: str | None = Field(
        default=None,
        description="Production URDF path for matching backends that require one",
    )

    # Cost weights for IK/matching
    cost_weights: dict[str, float] = Field(
        default_factory=dict, description="Cost function weights"
    )

    # Output configuration
    output_format: str = Field(default="json", description="Output format")

    # Hook execution policy
    strict_hooks: bool = Field(
        default=False,
        description=(
            "When true, per-stage hook failures raise HookExecutionError instead "
            "of being logged and skipped."
        ),
    )

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
        self._hooks: dict[Stage, list[Callable[[HookPayload], None]]] = {
            stage: [] for stage in Stage
        }
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
            except (
                AssertionError,
                LookupError,
                OSError,
                RuntimeError,
                TypeError,
                ValueError,
            ) as exc:
                hook_name = getattr(hook, "__qualname__", repr(hook))
                error = HookExecutionError(
                    stage=stage,
                    hook_name=hook_name,
                    original=exc,
                )
                if self.config.strict_hooks:
                    raise error from exc
                logger.exception("%s", error)

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
        from .sources.base import UnsupportedFormatError
        from .sources.loader import load_source

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

        if not isinstance(source, Path):
            return StageResult(
                success=False,
                data=None,
                metadata={},
                error=f"Unknown source type: {type(source)}",
                error_kind="invalid_input",
            )

        try:
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
        except (ValueError, FileNotFoundError, UnsupportedFormatError) as e:
            # Caller-supplied contract violations: bad format hint,
            # missing/undetectable source file (issue #6932).
            return StageResult(
                success=False,
                data=None,
                metadata={},
                error=f"Adapter failed: {e}",
                error_kind="invalid_input",
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("Adapter failed: %s", e)
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

        except Exception as e:  # noqa: BLE001
            logger.exception("Preprocessing failed: %s", e)
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

        except Exception as e:  # noqa: BLE001
            logger.exception("Scaling failed: %s", e)
            return StageResult(
                success=False, data=None, metadata={}, error=f"Scaling failed: {e}"
            )

    def _run_inverse_kinematics(self, data: Any, skeleton: SkeletonRig) -> StageResult:
        """
        Run IK stage: compute joint angles from markers/keypoints.

        Uses the ``make_ik_solver`` factory rather than a per-backend
        ``run_ik`` function (the latter was removed in #4566 but the
        orchestrator was not updated; tracked under #5911 follow-ups).
        """
        backend = self.config.ik_backend
        if backend not in {"mujoco", "drake", "pinocchio", "opensim", "geometric"}:
            return StageResult(
                success=False,
                data=None,
                metadata={},
                error=f"Unknown IK backend: {backend}",
                error_kind="invalid_input",
            )

        try:
            from .ik.base import MarkerWeights, make_ik_solver

            solver = make_ik_solver(backend)
            weights = MarkerWeights(
                marker_weights=dict(self.config.cost_weights),
            )
            trajectory = solver.solve(data, skeleton, weights=weights)

            num_frames = len(trajectory.frames) if trajectory else 0
            num_dofs = (
                len(trajectory.frames[0].q) if trajectory and trajectory.frames else 0
            )
            return StageResult(
                success=True,
                data=trajectory,
                metadata={
                    "backend": backend,
                    "num_frames": num_frames,
                    "num_dofs": num_dofs,
                },
            )

        except ImportError as e:
            return StageResult(
                success=False,
                data=None,
                metadata={},
                error=f"IK backend not available: {e}",
            )
        except Exception as e:  # noqa: BLE001
            logger.exception("IK failed: %s", e)
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

        # Map the orchestrator's coarse engine name to the concrete matching
        # solver backend. The real solver API is ``make_matching_solver`` +
        # ``.match()`` (the per-backend ``run_matching`` functions never
        # existed - #7047).
        backend_map = {
            "mujoco": MatchingBackendType.TORQUE_MUJOCO,
            "drake": MatchingBackendType.TRAJOPT_DRAKE,
            "pinocchio": MatchingBackendType.INVERSE_DYN_PINOCCHIO,
        }
        solver_backend = backend_map.get(backend)
        if solver_backend is None:
            return StageResult(
                success=False,
                data=None,
                metadata={},
                error=f"Unknown matching backend: {backend}",
                error_kind="invalid_input",
            )

        try:
            from .matching.base import make_matching_solver

            solver = make_matching_solver(
                solver_backend,
                urdf_path=self.config.matching_model_urdf,
            )
            # The matching solvers track a kinematic JointTrajectory.
            result = solver.match(trajectory.trajectory, skeleton)

            # Convert the internal dataclass result to the canonical
            # contract result the rest of the pipeline expects.
            contract_result = result.to_contract()
            solver_metadata = dict(contract_result.metadata or {})
            error_metrics = dict(contract_result.error_metrics or {})
            stage_metadata = {
                "backend": backend,
                "solver_backend": solver_backend.value,
                "solve_time": contract_result.solve_time,
                "error_metrics": error_metrics,
                "solver_metadata": solver_metadata,
            }

            if not contract_result.success:
                error_kind = "internal"
                if solver_metadata.get(
                    "production_ready"
                ) is False or self._is_unavailable_mujoco_matching_backend(
                    solver_backend, solver_metadata, error_metrics
                ):
                    error_kind = "invalid_input"
                return StageResult(
                    success=False,
                    data=None,
                    metadata=stage_metadata,
                    error=contract_result.message
                    or f"{backend} motion matching did not produce a valid solve",
                    error_kind=error_kind,
                )

            return StageResult(
                success=True,
                data=contract_result,
                metadata=stage_metadata,
            )

        except ImportError as e:
            return StageResult(
                success=False,
                data=None,
                metadata={},
                error=f"Matching backend not available: {e}",
            )
        except (RuntimeError, ValueError) as e:
            return StageResult(
                success=False,
                data=None,
                metadata={},
                error=f"Motion matching failed: {e}",
                error_kind="invalid_input",
            )

    @staticmethod
    def _is_unavailable_mujoco_matching_backend(
        solver_backend: MatchingBackendType,
        solver_metadata: dict[str, Any],
        error_metrics: dict[str, Any],
    ) -> bool:
        """Return True for known caller-actionable MuJoCo matching gaps."""
        if solver_backend is not MatchingBackendType.TORQUE_MUJOCO:
            return False
        if solver_metadata.get("mujoco_available") is False:
            return True
        max_torque = error_metrics.get("max_torque")
        try:
            return max_torque is not None and float(max_torque) == 0.0
        except (TypeError, ValueError):
            return False

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
            self._raise_stage_failure("Adapter stage", adapter_result)

        self._fire_hooks(Stage.ADAPTER, adapter_result.data, adapter_result.metadata)
        self._log_audit(Stage.ADAPTER, adapter_result, self._source_hash or "")

        data = adapter_result.data

        # Stage 2: Preprocessing
        preprocess_result = self._run_preprocessing(data)
        if not preprocess_result.success:
            self._raise_stage_failure("Preprocessing", preprocess_result)

        self._fire_hooks(
            Stage.PREPROCESSING, preprocess_result.data, preprocess_result.metadata
        )
        self._log_audit(Stage.PREPROCESSING, preprocess_result, self._source_hash or "")

        data = preprocess_result.data

        # Stage 3: Scaling (requires skeleton)
        # Use an explicit sentinel so adapter-provided skeletons do not trigger
        # the default loader/error path before getattr can return.
        skeleton = getattr(data, "skeleton", None)
        if skeleton is None:
            skeleton = self._get_default_skeleton()

        scaling_result = self._run_scaling(data, skeleton)
        if not scaling_result.success:
            self._raise_stage_failure("Scaling", scaling_result)

        self._fire_hooks(Stage.SCALING, scaling_result.data, scaling_result.metadata)
        self._log_audit(Stage.SCALING, scaling_result, self._source_hash or "")

        data, scaled_skeleton = scaling_result.data

        # Stage 4: Inverse Kinematics
        ik_result = self._run_inverse_kinematics(data, scaled_skeleton)
        if not ik_result.success:
            self._raise_stage_failure("IK", ik_result)

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
            self._raise_stage_failure("Motion matching", matching_result)

        self._fire_hooks(
            Stage.MOTION_MATCHING, matching_result.data, matching_result.metadata
        )
        self._log_audit(Stage.MOTION_MATCHING, matching_result, self._source_hash or "")

        result = matching_result.data
        assert isinstance(result, MotionMatchingResult)

        if result.matched_trajectory is not None:
            result.matched_trajectory.source_provenance = {
                **trajectory.source_provenance,
                **result.matched_trajectory.source_provenance,
            }

        # Add audit log to result metadata
        result.metadata["audit_log"] = self._audit_log
        result.metadata["config"] = self.config.model_dump()

        logger.info(f"Pipeline run completed: success={result.success}")

        return result

    @staticmethod
    def _raise_stage_failure(stage_label: str, result: StageResult) -> None:
        """Raise the failure for a stage, preserving its 4xx/5xx kind.

        ``InvalidInputError`` for caller contract violations, ``RuntimeError``
        for internal faults (issue #6932). Always raises.
        """
        message = f"{stage_label} failed: {result.error}"
        if result.error_kind == "invalid_input":
            raise InvalidInputError(message)
        raise RuntimeError(message)

    def _get_default_skeleton(self) -> SkeletonRig:
        """Get default skeleton for pipeline."""
        # This would load a default skeleton based on config
        # For now, raise error - caller should provide skeleton
        raise InvalidInputError("No skeleton provided. Use adapter that provides one.")

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
    except Exception as e:  # noqa: BLE001
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
