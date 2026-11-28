"""Slice 0 playback responsibility map and 60 FPS quiver profiling fixture (Epic #15660, #15661)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import hashlib
import math
import time

import numpy as np
from numpy.typing import NDArray

TARGET_FPS: float = 60.0
FRAME_DT_S: float = 1.0 / TARGET_FPS
FRAME_BUDGET_MS: float = 1000.0 / TARGET_FPS
REQUIRED_STAGE_NAMES: tuple[str, ...] = (
    "source",
    "interpolator",
    "plane_quiver_builder",
    "renderer",
    "ui_transport",
)
_SHA256_HEX_LEN = 64
_MIN_DT_TOL_S = 1e-12


class FrameBudgetExceededError(RuntimeError):
    """Raised when synthetic quiver frame profiling exceeds the 60 FPS frame budget."""


def _validate_nonempty_str(value: str, field_name: str) -> str:
    """Return stripped non-empty string or raise ValueError."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _validate_pos_finite(value: float, field_name: str) -> float:
    """Validate that a scalar float is finite and strictly positive."""
    if not (math.isfinite(value) and value > 0.0):
        raise ValueError(f"{field_name} must be positive and finite")
    return float(value)


def _validate_matrix_n3(
    arr: NDArray[np.float64], field_name: str
) -> NDArray[np.float64]:
    """Coerce and validate a finite (N, 3) float64 array."""
    matrix = np.asarray(arr, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 3:
        raise ValueError(f"{field_name} must have shape (N, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError(f"{field_name} must contain only finite values")
    return matrix


@dataclass(frozen=True)
class ResponsibilityStage:
    """Single-responsibility stage node in the unified playback pipeline DAG."""

    name: str
    owner: str
    responsibility: str
    depends_on: tuple[str, ...] = field(default_factory=tuple)
    inputs: tuple[str, ...] = field(default_factory=tuple)
    outputs: tuple[str, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        """Validate stage identity, ownership, and dependency hygiene."""
        name = _validate_nonempty_str(self.name, "name")
        owner = _validate_nonempty_str(self.owner, "owner")
        resp = _validate_nonempty_str(self.responsibility, "responsibility")
        deps = tuple(_validate_nonempty_str(d, "depends_on") for d in self.depends_on)
        if name in deps or len(set(deps)) != len(deps):
            raise ValueError(f"Invalid dependencies for stage '{name}'")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "owner", owner)
        object.__setattr__(self, "responsibility", resp)
        object.__setattr__(self, "depends_on", deps)


@dataclass(frozen=True)
class PlaybackResponsibilityMap:
    """Validated acyclic DAG mapping ownership across playback pipeline stages."""

    stages: tuple[ResponsibilityStage, ...]

    def __post_init__(self) -> None:
        """Reject empty maps, duplicate names/owners, unknown deps, or cycles."""
        stages = tuple(self.stages)
        if not stages:
            raise ValueError("PlaybackResponsibilityMap must define at least one stage")
        object.__setattr__(self, "stages", stages)
        names = [s.name for s in stages]
        owners = [s.owner for s in stages]
        if len(set(names)) != len(names):
            raise ValueError("Duplicate stage name in PlaybackResponsibilityMap")
        if len(set(owners)) != len(owners):
            raise ValueError("Duplicate stage owner in PlaybackResponsibilityMap")
        name_set = set(names)
        for stage in stages:
            unknown = [d for d in stage.depends_on if d not in name_set]
            if unknown:
                raise ValueError(f"Unknown dependency '{unknown[0]}' in '{stage.name}'")
        _ = self.topological_order()

    def stage_names(self) -> tuple[str, ...]:
        """Return stage names in declaration order."""
        return tuple(stage.name for stage in self.stages)

    def has_required_stages(
        self, required: tuple[str, ...] = REQUIRED_STAGE_NAMES
    ) -> bool:
        """Return True when all required canonical stages are present."""
        return set(required).issubset(set(self.stage_names()))

    def topological_order(self) -> tuple[str, ...]:
        """Return topological execution order, raising ValueError on any cycle."""
        stage_by_name = {stage.name: stage for stage in self.stages}
        visit_state: dict[str, int] = {}
        order: list[str] = []

        def _visit(name: str) -> None:
            """Depth-first visit verifying acyclicity for a single stage."""
            state = visit_state.get(name, 0)
            if state == 1:
                raise ValueError(f"Circular dependency detected involving '{name}'")
            if state == 2:
                return
            visit_state[name] = 1
            for dep in stage_by_name[name].depends_on:
                _visit(dep)
            visit_state[name] = 2
            order.append(name)

        for stage in self.stages:
            _visit(stage.name)
        return tuple(order)


def default_playback_responsibility_map() -> PlaybackResponsibilityMap:
    """Construct the canonical five-stage Slice 0 playback responsibility DAG."""
    specs = (
        (
            "source",
            "src.shared.python.force_overlay.series",
            "Load swing samples and timestamps.",
            (),
        ),
        (
            "interpolator",
            "src.shared.python.video_timing.frame_schedule",
            "Resample states onto 60 FPS grid.",
            ("source",),
        ),
        (
            "plane_quiver_builder",
            "src.shared.python.playback.quiver_profiling_fixture",
            "Build plane & WSCG quiver geometry.",
            ("interpolator",),
        ),
        (
            "renderer",
            "src.shared.python.playback.bidirectional_overlay",
            "Composite model, planes, and quivers.",
            ("plane_quiver_builder",),
        ),
        (
            "ui_transport",
            "src.shared.python.playback.camera_perspective",
            "Sync transport controls and camera.",
            ("source", "renderer"),
        ),
    )
    return PlaybackResponsibilityMap(tuple(ResponsibilityStage(*s) for s in specs))


@dataclass(frozen=True)
class QuiverFrameSample:
    """Single 60 FPS frame of world-space quiver origins, vectors, and labels."""

    frame_index: int
    time_s: float
    origins_m: NDArray[np.float64]
    vectors: NDArray[np.float64]
    labels: tuple[str, ...]

    def __post_init__(self) -> None:
        """Validate finite coordinates, matching shapes, and non-negative index."""
        if self.frame_index < 0 or not math.isfinite(self.time_s):
            raise ValueError("frame_index must be >= 0 and time_s must be finite")
        origins = _validate_matrix_n3(self.origins_m, "origins_m")
        vectors = _validate_matrix_n3(self.vectors, "vectors")
        labels = tuple(self.labels)
        if origins.shape != vectors.shape or len(labels) != origins.shape[0]:
            raise ValueError("origins_m, vectors, and labels counts must match")
        object.__setattr__(self, "origins_m", origins)
        object.__setattr__(self, "vectors", vectors)
        object.__setattr__(self, "labels", labels)


@dataclass(frozen=True)
class QuiverProfilingFixture:
    """Deterministic 60 FPS whole-swing quiver fixture with content SHA-256."""

    name: str
    fps: float
    dt_s: float
    duration_s: float
    frames: tuple[QuiverFrameSample, ...]
    fixture_sha256: str
    responsibility_map: PlaybackResponsibilityMap = field(
        default_factory=default_playback_responsibility_map
    )

    def __post_init__(self) -> None:
        """Validate 60 FPS cadence, monotonic timestamps, and fixture hash."""
        name = _validate_nonempty_str(self.name, "name")
        fps = _validate_pos_finite(self.fps, "fps")
        _validate_pos_finite(self.duration_s, "duration_s")
        if not math.isfinite(self.dt_s) or not math.isclose(
            self.dt_s, 1.0 / fps, rel_tol=0.0, abs_tol=_MIN_DT_TOL_S
        ):
            raise ValueError("dt_s must match 1.0 / fps")
        frames = tuple(self.frames)
        if len(frames) < 2:
            raise ValueError("Fixture must contain at least 2 frames")
        prev_time = -math.inf
        for expected_idx, frame in enumerate(frames):
            if frame.frame_index != expected_idx or frame.time_s <= prev_time:
                raise ValueError(
                    "Frames must have contiguous indices and monotonic time"
                )
            prev_time = frame.time_s
        if len(self.fixture_sha256) != _SHA256_HEX_LEN:
            raise ValueError("fixture_sha256 must be a 64-character hex digest")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "frames", frames)


def _compute_fixture_sha256(
    name: str, fps: float, duration_s: float, frames: tuple[QuiverFrameSample, ...]
) -> str:
    """Compute a deterministic SHA-256 digest over fixture metadata and arrays."""
    digest = hashlib.sha256()
    digest.update(f"{name}|{fps:.12g}|{duration_s:.12g}|{len(frames)}".encode())
    for frame in frames:
        digest.update(np.array([frame.time_s], dtype="<f8").tobytes())
        digest.update(np.asarray(frame.origins_m, dtype="<f8").tobytes())
        digest.update(np.asarray(frame.vectors, dtype="<f8").tobytes())
        digest.update("|".join(frame.labels).encode())
    return digest.hexdigest()


def build_60fps_quiver_fixture(
    duration_s: float = 1.5,
    n_quivers: int = 8,
    seed: int = 15661,
    name: str = "whole_swing_60fps",
) -> QuiverProfilingFixture:
    """Build a deterministic 60 FPS whole-swing quiver profiling fixture."""
    duration = _validate_pos_finite(duration_s, "duration_s")
    if n_quivers < 1:
        raise ValueError("n_quivers must be at least 1")
    n_frames = max(2, int(round(duration * TARGET_FPS)) + 1)
    base_offsets = np.random.default_rng(seed).uniform(-0.25, 0.25, size=(n_quivers, 3))
    labels = tuple(f"wscg:anchor_{idx:02d}" for idx in range(n_quivers))
    ap = np.linspace(0.0, math.pi, num=n_quivers, dtype=np.float64)

    samples: list[QuiverFrameSample] = []
    for idx in range(n_frames):
        t = idx * FRAME_DT_S
        ph = 2.0 * math.pi * (t / duration)
        origins = base_offsets + np.column_stack(
            [
                0.85 * np.sin(ph + ap * 0.25),
                0.35 * np.cos(ph * 0.5 + ap * 0.15),
                1.1 + 0.45 * np.cos(ph + ap * 0.2),
            ]
        )
        vecs = np.column_stack(
            [
                6.0 * np.cos(ph + ap * 0.3),
                -2.5 * np.sin(ph * 1.5 + ap * 0.2),
                4.0 * np.sin(ph + ap * 0.1),
            ]
        )
        samples.append(QuiverFrameSample(idx, t, origins, vecs, labels))

    frames = tuple(samples)
    digest = _compute_fixture_sha256(name, TARGET_FPS, duration, frames)
    return QuiverProfilingFixture(
        name, TARGET_FPS, FRAME_DT_S, duration, frames, digest
    )


def _default_quiver_builder(sample: QuiverFrameSample) -> NDArray[np.float64]:
    """Compute synthetic normalized quiver tip endpoints for one frame."""
    norms = np.linalg.norm(sample.vectors, axis=1, keepdims=True)
    return sample.origins_m + 0.15 * (
        sample.vectors / np.where(norms > 1e-12, norms, 1.0)
    )


@dataclass(frozen=True)
class QuiverProfileReport:
    """Summary timing report comparing per-frame quiver build times to 16.67 ms."""

    fixture_sha256: str
    frame_count: int
    budget_ms: float
    frame_times_ms: tuple[float, ...]
    mean_ms: float
    p95_ms: float
    max_ms: float
    over_budget_frames: int
    within_budget: bool


def profile_quiver_frame_budget(
    fixture: QuiverProfilingFixture,
    builder: Callable[[QuiverFrameSample], object] | None = None,
    *,
    budget_ms: float = FRAME_BUDGET_MS,
    fail_on_exceeded: bool = True,
    clock_ns: Callable[[], int] | None = None,
) -> QuiverProfileReport:
    """Measure quiver build times per frame against budget and fail closed if exceeded."""
    budget = _validate_pos_finite(budget_ms, "budget_ms")
    build_fn = builder if builder is not None else _default_quiver_builder
    now_ns = clock_ns if clock_ns is not None else time.perf_counter_ns
    durations_ms: list[float] = []
    for frame in fixture.frames:
        t0 = now_ns()
        _ = build_fn(frame)
        durations_ms.append(max(0.0, (now_ns() - t0) / 1_000_000.0))

    times_arr = np.asarray(durations_ms, dtype=np.float64)
    over_budget = int(np.sum(times_arr > budget))
    max_ms = float(np.max(times_arr))
    report = QuiverProfileReport(
        fixture.fixture_sha256,
        len(durations_ms),
        budget,
        tuple(durations_ms),
        float(np.mean(times_arr)),
        float(np.percentile(times_arr, 95)),
        max_ms,
        over_budget,
        over_budget == 0 and max_ms <= budget,
    )
    if fail_on_exceeded and not report.within_budget:
        raise FrameBudgetExceededError(
            f"Quiver frame build exceeded {budget:.2f} ms budget "
            f"({over_budget}/{len(durations_ms)} frames over, max={max_ms:.3f} ms)"
        )
    return report
