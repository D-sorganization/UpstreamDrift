"""Analysis tools and simulation control routes.

Provides endpoints for:
- Real-time analysis metrics and statistics (#1203)
- Dataset export to CSV/JSON (#1203)
- Body positioning and measurement tools (#1179)

See issue #1203, #1179

All dependencies are injected via FastAPI's Depends() mechanism.
No module-level mutable state.
"""

from __future__ import annotations

import csv
import io
import json
import math
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from src.shared.python.core.contracts import precondition
from src.shared.python.logging_pkg.logging_config import get_logger

from ..dependencies import get_engine_manager, get_logger as get_request_logger
from ..models.requests import (
    BodyPositionUpdateRequest,
    DataExportRequest,
    MeasurementRequest,
)
from ..models.responses import (
    AnalysisMetricsSummary,
    AnalysisStatisticsResponse,
    BodyPositionResponse,
    JointAngleDisplay,
    MeasurementResult,
    MeasurementToolsResponse,
)

if TYPE_CHECKING:
    from src.shared.python.engine_core.engine_manager import EngineManager

router = APIRouter()

logger = get_logger(__name__)


def _check_position_support(engine: Any) -> None:
    """Raise HTTPException 501 when the engine supports neither body-position setter.

    Args:
        engine: Active physics engine instance.

    Raises:
        HTTPException: 501 if the engine lacks both set_body_position and
            set_body_rotation, so callers never silently succeed without moving anything.
    """
    if not hasattr(engine, "set_body_position") and not hasattr(
        engine, "set_body_rotation"
    ):
        raise HTTPException(
            status_code=501,
            detail=(
                "Engine does not support body positioning. "
                "set_body_position and set_body_rotation are not available."
            ),
        )


def _get_body_position_vectors(
    engine: Any, body_a: str, body_b: str
) -> tuple[list[float], list[float]]:
    """Return the 3-D positions of two named bodies.

    Args:
        engine: Active physics engine instance.
        body_a: Name of the first body.
        body_b: Name of the second body.

    Returns:
        Tuple of (pos_a, pos_b) as plain Python float lists.

    Raises:
        HTTPException: 501 if the engine lacks get_body_position.
        HTTPException: 400 if either position cannot be retrieved (returns None).
    """
    if not hasattr(engine, "get_body_position"):
        raise HTTPException(
            status_code=501,
            detail=(
                "Engine does not support body position queries. "
                "get_body_position is not available."
            ),
        )

    pa = engine.get_body_position(body_a)
    if pa is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not retrieve position for body '{body_a}'.",
        )

    pb = engine.get_body_position(body_b)
    if pb is None:
        raise HTTPException(
            status_code=400,
            detail=f"Could not retrieve position for body '{body_b}'.",
        )

    pos_a: list[float] = pa.tolist() if hasattr(pa, "tolist") else list(pa)
    pos_b: list[float] = pb.tolist() if hasattr(pb, "tolist") else list(pb)
    return pos_a, pos_b


def _collect_metrics(engine_manager: EngineManager) -> dict[str, Any]:
    """Collect current metrics from the active engine.

    Args:
        engine_manager: Engine manager instance.

    Returns:
        Dictionary of metric name to current value.
    """
    engine = engine_manager.get_active_physics_engine()
    if engine is None:
        return {}

    metrics: dict[str, Any] = {}

    sim_time = getattr(engine, "time", 0.0)
    metrics["sim_time"] = sim_time

    try:
        q, v = engine.get_state()
        metrics["joint_positions"] = q.tolist() if hasattr(q, "tolist") else list(q)
        metrics["joint_velocities"] = v.tolist() if hasattr(v, "tolist") else list(v)

        # Derived metrics
        if len(v) > 0:
            import numpy as np

            metrics["max_velocity"] = float(np.max(np.abs(v)))
            # ⚡ Bolt: np.vdot is ~2x faster than np.mean(v**2) and avoids temporary array allocations
            metrics["rms_velocity"] = float(np.sqrt(np.vdot(v, v) / v.size))
    except ImportError as exc:
        logger.debug("numpy unavailable for velocity metrics: %s", exc)

    # Energy
    try:
        M = engine.compute_mass_matrix()
        if M is not None:
            import numpy as np

            q, v = engine.get_state()
            metrics["kinetic_energy"] = float(0.5 * v @ M @ v)
    except ImportError as exc:
        logger.debug("numpy unavailable for kinetic energy metric: %s", exc)

    # Club head speed
    try:
        jac = engine.compute_jacobian("club_head")
        if jac is not None and "linear" in jac:
            import numpy as np

            _, v = engine.get_state()
            linear_vel = jac["linear"] @ v
            metrics["club_head_speed"] = float(np.linalg.norm(linear_vel))
    except ImportError as exc:
        logger.debug("numpy unavailable for club head speed metric: %s", exc)

    return metrics


def _get_metric_history(engine_manager: EngineManager) -> list[dict[str, Any]]:
    """Get stored metric history from engine manager.

    Returns:
        List of metric snapshots over time.
    """
    return getattr(engine_manager, "_metric_history", [])


def _store_metric_snapshot(
    engine_manager: EngineManager, metrics: dict[str, Any]
) -> None:
    """Store a metric snapshot in the engine manager history.

    Keeps a bounded buffer of metrics for statistics computation.

    Args:
        engine_manager: Engine manager instance.
        metrics: Current metrics snapshot.
    """
    if not (engine_manager is not None):
        raise ValueError("engine_manager must be provided")
    max_history = 500
    if not hasattr(engine_manager, "_metric_history"):
        engine_manager._metric_history = []  # type: ignore[attr-defined]

    engine_manager._metric_history.append(metrics)  # type: ignore[attr-defined]
    if len(engine_manager._metric_history) > max_history:  # type: ignore[attr-defined]
        engine_manager._metric_history = engine_manager._metric_history[-max_history:]  # type: ignore[attr-defined]


# ──────────────────────────────────────────────────────────────
#  Analysis Metrics (See issue #1203)
# ──────────────────────────────────────────────────────────────


@router.get("/analysis/metrics")
async def get_analysis_metrics(
    engine_manager: EngineManager = Depends(get_engine_manager),
    logger: Any = Depends(get_request_logger),
) -> dict[str, Any]:
    """Get current real-time analysis metrics.

    Collects current biomechanics metrics from the active engine
    and stores a snapshot for historical statistics.

    Returns:
        Current metrics dictionary.

    Raises:
        HTTPException: If no engine is loaded.
    """
    engine = engine_manager.get_active_physics_engine()
    if engine is None:
        raise HTTPException(
            status_code=400,
            detail="No physics engine loaded. Load an engine first.",
        )

    try:
        metrics = _collect_metrics(engine_manager)
        _store_metric_snapshot(engine_manager, metrics)
        return {"status": "ok", "metrics": metrics}
    except (RuntimeError, TypeError, AttributeError) as exc:
        if logger:
            logger.error("Metrics collection error: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Metrics collection failed: {str(exc)}"
        ) from exc


@router.get("/analysis/statistics", response_model=AnalysisStatisticsResponse)
async def get_analysis_statistics(  # noqa: C901
    engine_manager: EngineManager = Depends(get_engine_manager),
    logger: Any = Depends(get_request_logger),
) -> AnalysisStatisticsResponse:
    """Get statistical summary of analysis metrics over time.

    Computes min, max, mean, and std_dev for each scalar metric
    from the stored metric history.

    Returns:
        Statistical summaries.

    Raises:
        HTTPException: If no engine is loaded.
    """
    engine = engine_manager.get_active_physics_engine()
    if engine is None:
        raise HTTPException(
            status_code=400,
            detail="No physics engine loaded. Load an engine first.",
        )

    try:
        history = _get_metric_history(engine_manager)
        sim_time = getattr(engine, "time", 0.0)

        # Compute statistics for scalar metrics
        metric_summaries: list[AnalysisMetricsSummary] = []
        time_series: dict[str, list[float]] = {}

        # Identify scalar metric keys
        scalar_keys: set[str] = set()
        for snapshot in history:
            for key, value in snapshot.items():
                if isinstance(value, int | float) and not math.isnan(value):
                    scalar_keys.add(key)

        for key in sorted(scalar_keys):
            values = []
            for snapshot in history:
                val = snapshot.get(key)
                if isinstance(val, int | float) and not math.isnan(val):
                    values.append(float(val))

            if not values:
                continue

            import numpy as np

            arr = np.array(values)
            metric_summaries.append(
                AnalysisMetricsSummary(
                    metric_name=key,
                    current=values[-1],
                    minimum=float(np.min(arr)),
                    maximum=float(np.max(arr)),
                    mean=float(np.mean(arr)),
                    std_dev=float(np.std(arr)),
                )
            )

            time_series[key] = values

        return AnalysisStatisticsResponse(
            sim_time=sim_time,
            sample_count=len(history),
            metrics=metric_summaries,
            time_series=time_series,
        )
    except ImportError as exc:
        if logger:
            logger.error("Statistics computation error: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Statistics computation failed: {str(exc)}"
        ) from exc


# ──────────────────────────────────────────────────────────────
#  Data Export (See issue #1203)
# ──────────────────────────────────────────────────────────────


@router.post("/analysis/export")
@precondition(
    lambda request, engine_manager=None, logger=None: request.format in ("csv", "json"),
    "Export format must be 'csv' or 'json'",
)
async def export_analysis_data(  # noqa: C901
    request: DataExportRequest,
    engine_manager: EngineManager = Depends(get_engine_manager),
    logger: Any = Depends(get_request_logger),
) -> StreamingResponse:
    """Export analysis data as CSV or JSON download.

    Exports the stored metric history and current statistics
    as a downloadable file.

    Args:
        request: Export parameters (format, filters).
        engine_manager: Injected engine manager.
        logger: Injected logger.

    Returns:
        Streaming file response.
    """
    history = _get_metric_history(engine_manager)

    if not history:
        raise HTTPException(
            status_code=400,
            detail="No data to export. Run a simulation first.",
        )

    # Apply time range filter
    filtered = history
    if request.time_range and len(request.time_range) == 2:
        t_start, t_end = request.time_range
        filtered = [s for s in history if t_start <= s.get("sim_time", 0) <= t_end]

    try:
        if request.format == "csv":
            # Build CSV
            output = io.StringIO()
            if filtered:
                # Get all keys from all snapshots
                all_keys: set[str] = set()
                for snapshot in filtered:
                    for key, value in snapshot.items():
                        if isinstance(value, int | float):
                            all_keys.add(key)

                fieldnames = sorted(all_keys)
                writer = csv.DictWriter(output, fieldnames=fieldnames)
                writer.writeheader()
                for snapshot in filtered:
                    row = {}
                    for key in fieldnames:
                        val = snapshot.get(key)
                        if isinstance(val, int | float):
                            row[key] = val
                    writer.writerow(row)

            content = output.getvalue()
            return StreamingResponse(
                io.BytesIO(content.encode("utf-8")),
                media_type="text/csv",
                headers={
                    "Content-Disposition": "attachment; filename=analysis_export.csv"
                },
            )
        # JSON export
        export_data = {
            "format": "json",
            "record_count": len(filtered),
            "data": filtered,
        }
        content = json.dumps(export_data, indent=2, default=str)
        return StreamingResponse(
            io.BytesIO(content.encode("utf-8")),
            media_type="application/json",
            headers={
                "Content-Disposition": "attachment; filename=analysis_export.json"
            },
        )
    except ImportError as exc:
        if logger:
            logger.error("Export error: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Export failed: {str(exc)}"
        ) from exc


# ──────────────────────────────────────────────────────────────
#  Body Positioning (See issue #1179)
# ──────────────────────────────────────────────────────────────


@router.post("/simulation/position", response_model=BodyPositionResponse)
@precondition(
    lambda request, engine_manager=None, logger=None: (
        request.body_name is not None and len(request.body_name.strip()) > 0
    ),
    "Body name must be a non-empty string",
)
async def set_body_position(
    request: BodyPositionUpdateRequest,
    engine_manager: EngineManager = Depends(get_engine_manager),
    logger: Any = Depends(get_request_logger),
) -> BodyPositionResponse:
    """Set the position and/or rotation of a body in the simulation.

    Allows interactive repositioning of bodies for setup and analysis.

    Args:
        request: Body name, position, and rotation.
        engine_manager: Injected engine manager.
        logger: Injected logger.

    Returns:
        Applied position and rotation.

    Raises:
        HTTPException: If no engine is loaded or body not found.
    """
    engine = engine_manager.get_active_physics_engine()
    if engine is None:
        raise HTTPException(
            status_code=400,
            detail="No physics engine loaded. Load an engine first.",
        )

    position = request.position or [0.0, 0.0, 0.0]
    rotation = request.rotation or [0.0, 0.0, 0.0]

    try:
        # Raise 501 early if the engine supports neither setter
        _check_position_support(engine)

        if hasattr(engine, "set_body_position"):
            engine.set_body_position(request.body_name, position)
        if hasattr(engine, "set_body_rotation"):
            engine.set_body_rotation(request.body_name, rotation)

        return BodyPositionResponse(
            body_name=request.body_name,
            position=position,
            rotation=rotation,
            status=f"Position set for {request.body_name}",
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=404, detail=f"Body not found: {str(exc)}"
        ) from exc
    except ImportError as exc:
        if logger:
            logger.error("Body positioning error: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Positioning failed: {str(exc)}"
        ) from exc


# ──────────────────────────────────────────────────────────────
#  Measurement Tools (See issue #1179)
# ──────────────────────────────────────────────────────────────


@router.post("/simulation/measure", response_model=MeasurementResult)
@precondition(
    lambda request, engine_manager=None, logger=None: (
        request.body_a is not None
        and len(request.body_a.strip()) > 0
        and request.body_b is not None
        and len(request.body_b.strip()) > 0
    ),
    "Both body_a and body_b must be non-empty strings",
)
async def measure_distance(
    request: MeasurementRequest,
    engine_manager: EngineManager = Depends(get_engine_manager),
    logger: Any = Depends(get_request_logger),
) -> MeasurementResult:
    """Measure the distance between two bodies.

    Args:
        request: Body names to measure between.
        engine_manager: Injected engine manager.
        logger: Injected logger.

    Returns:
        Distance measurement and body positions.

    Raises:
        HTTPException: If no engine loaded or bodies not found.
    """
    engine = engine_manager.get_active_physics_engine()
    if engine is None:
        raise HTTPException(
            status_code=400,
            detail="No physics engine loaded. Load an engine first.",
        )

    try:
        # Raise 501 if engine doesn't support body position queries
        pos_a, pos_b = _get_body_position_vectors(
            engine, request.body_a, request.body_b
        )

        # Compute distance
        # ⚡ Bolt: math.dist is ~8x faster than list comprehension and math.sqrt(sum(...))
        distance = math.dist(pos_a, pos_b)
        delta = [b - a for a, b in zip(pos_a, pos_b, strict=True)]

        return MeasurementResult(
            body_a=request.body_a,
            body_b=request.body_b,
            distance=distance,
            position_a=pos_a,
            position_b=pos_b,
            delta=delta,
        )
    except HTTPException:
        raise
    except (ValueError, RuntimeError, AttributeError) as exc:
        if logger:
            logger.error("Measurement error: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Measurement failed: {str(exc)}"
        ) from exc


@router.get("/simulation/measurements", response_model=MeasurementToolsResponse)
async def get_measurement_tools(
    engine_manager: EngineManager = Depends(get_engine_manager),
    logger: Any = Depends(get_request_logger),
) -> MeasurementToolsResponse:
    """Get all joint angles and active measurements.

    Provides a unified view of joint angle displays and any
    active distance measurements for the simulation toolbar.

    Returns:
        Joint angles and measurements data.

    Raises:
        HTTPException: If no engine is loaded.
    """
    engine = engine_manager.get_active_physics_engine()
    if engine is None:
        raise HTTPException(
            status_code=400,
            detail="No physics engine loaded. Load an engine first.",
        )

    try:
        joint_angles: list[JointAngleDisplay] = []

        # Get state
        q, v = engine.get_state()
        q_list = q.tolist() if hasattr(q, "tolist") else list(q)
        v_list = v.tolist() if hasattr(v, "tolist") else list(v)

        # Get joint names if available
        joint_names: list[str] = []
        if hasattr(engine, "get_joint_names"):
            joint_names = engine.get_joint_names()

        # Get torques if available
        torques: list[float] = []
        if hasattr(engine, "get_applied_torques"):
            t = engine.get_applied_torques()
            torques = t.tolist() if hasattr(t, "tolist") else list(t)

        for i in range(len(q_list)):
            name = joint_names[i] if i < len(joint_names) else f"joint_{i}"
            angle_rad = q_list[i]
            vel = v_list[i] if i < len(v_list) else 0.0
            torque = torques[i] if i < len(torques) else 0.0

            joint_angles.append(
                JointAngleDisplay(
                    joint_name=name,
                    angle_rad=angle_rad,
                    angle_deg=math.degrees(angle_rad),
                    velocity=vel,
                    torque=torque,
                )
            )

        return MeasurementToolsResponse(
            joint_angles=joint_angles,
            measurements=[],
        )
    except (ValueError, RuntimeError, AttributeError) as exc:
        if logger:
            logger.error("Measurement tools error: %s", exc)
        raise HTTPException(
            status_code=500, detail=f"Measurement tools failed: {str(exc)}"
        ) from exc
