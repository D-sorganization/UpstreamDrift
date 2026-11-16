"""Offline grip-loss prediction from recorded swing loads (#15128)."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

import numpy as np

from src.shared.python.grip_contact.grip_loss_config import GripLossPhysicsConfig
from src.shared.python.grip_contact.grip_loss_loads import (
    DEFAULT_GRIP_LOAD_PROVIDER,
    GripLoadSeries,
    resolve_grip_load_provider,
)
from src.shared.python.grip_contact.grip_loss_state import (
    _PatchRuntime,
    advance_patch_state,
)
from src.shared.python.grip_contact.grip_margin import UNLOADED, pad_margins
from src.shared.python.grip_contact.pad_contact import PadWrench

PREDICTION_DISCLAIMER: str = (
    "Offline grip-loss prediction from recorded loads and Coulomb margin "
    "hysteresis only; not a simulated fly-off or dynamic clamp release."
)

__all__ = [
    "PREDICTION_DISCLAIMER",
    "GripLoadSeries",
    "GripLossEvaluator",
    "GripLossReport",
    "GripLossEvaluatorBackend",
    "register_grip_loss_evaluator",
    "resolve_grip_loss_evaluator",
    "serialise_grip_loss_report",
]


@dataclass(frozen=True)
class GripLossReport:
    """Prediction report for one recorded swing."""

    club: str
    source: str
    disclaimer: str
    physics: GripLossPhysicsConfig
    time_s: np.ndarray
    margin_trace: Mapping[str, tuple[float | str, ...]]
    state_trace: Mapping[str, tuple[str, ...]]
    first_slip_s: float | None
    first_slip_patch: str | None
    first_release_s: float | None
    first_release_patch: str | None
    sensitivity: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if self.disclaimer != PREDICTION_DISCLAIMER:
            raise ValueError("report disclaimer must be the canonical prediction text")


@runtime_checkable
class GripLossEvaluatorBackend(Protocol):
    def evaluate(
        self, loads: GripLoadSeries, physics: GripLossPhysicsConfig
    ) -> GripLossReport: ...


_EVALUATOR_REGISTRY: dict[str, GripLossEvaluatorBackend] = {}


def register_grip_loss_evaluator(name: str, backend: GripLossEvaluatorBackend) -> None:
    if not isinstance(name, str) or not name.strip():
        raise TypeError("evaluator name must be a non-empty string")
    if not isinstance(backend, GripLossEvaluatorBackend):
        raise TypeError("backend must implement GripLossEvaluatorBackend")
    _EVALUATOR_REGISTRY[name] = backend


def resolve_grip_loss_evaluator(name: str) -> GripLossEvaluatorBackend:
    backend = _EVALUATOR_REGISTRY.get(name)
    if backend is None:
        raise ValueError(f"unknown grip loss evaluator {name!r}")
    return backend


def _scale_loads(loads: GripLoadSeries, squeeze_scale: float) -> GripLoadSeries:
    if not (math.isfinite(squeeze_scale) and squeeze_scale > 0.0):
        raise ValueError("squeeze_scale must be positive and finite")
    return GripLoadSeries(
        club=loads.club,
        source=loads.source,
        patch_labels=loads.patch_labels,
        time_s=loads.time_s,
        normal_force_n=np.asarray(loads.normal_force_n, float) * squeeze_scale,
        tangential_force_n=np.asarray(loads.tangential_force_n, float) * squeeze_scale,
    )


class MarginHysteresisGripLossEvaluator:
    """Coulomb pad margin with per-patch HELD → SLIPPING → RELEASED hysteresis."""

    def evaluate(
        self, loads: GripLoadSeries, physics: GripLossPhysicsConfig
    ) -> GripLossReport:
        if not isinstance(loads, GripLoadSeries):
            raise TypeError("loads must be GripLoadSeries")
        if not isinstance(physics, GripLossPhysicsConfig):
            raise TypeError("physics must be GripLossPhysicsConfig")

        time_s = np.asarray(loads.time_s, float)
        dt = float(np.median(np.diff(time_s)))
        n_patches = len(loads.patch_labels)
        runtimes = [_PatchRuntime() for _ in range(n_patches)]
        margin_trace: dict[str, list[float | str]] = {
            label: [] for label in loads.patch_labels
        }
        state_trace: dict[str, list[str]] = {label: [] for label in loads.patch_labels}
        first_slip_s: float | None = None
        first_slip_patch: str | None = None
        first_release_s: float | None = None
        first_release_patch: str | None = None

        for frame in range(time_s.size):
            for patch_idx, label in enumerate(loads.patch_labels):
                wrench = PadWrench(
                    np.zeros(3),
                    np.zeros(3),
                    np.asarray([loads.normal_force_n[frame, patch_idx]], float),
                    np.asarray([loads.tangential_force_n[frame, patch_idx]], float),
                )
                margins = pad_margins(
                    wrench,
                    physics.friction,
                    physics.normal_floor_n,
                )
                margin = margins[0]
                margin_trace[label].append(margin)
                if margin == UNLOADED:
                    state_trace[label].append(runtimes[patch_idx].state.value)
                    continue
                step = advance_patch_state(
                    runtimes[patch_idx],
                    float(margin),
                    dt_s=dt,
                    thresholds=physics.thresholds,
                )
                state_trace[label].append(step.state.value)
                if step.entered_slip and first_slip_s is None:
                    first_slip_s = float(time_s[frame])
                    first_slip_patch = label
                if step.entered_release and first_release_s is None:
                    first_release_s = float(time_s[frame])
                    first_release_patch = label

        return GripLossReport(
            club=loads.club,
            source=loads.source,
            disclaimer=PREDICTION_DISCLAIMER,
            physics=physics,
            time_s=time_s,
            margin_trace={k: tuple(v) for k, v in margin_trace.items()},
            state_trace={k: tuple(v) for k, v in state_trace.items()},
            first_slip_s=first_slip_s,
            first_slip_patch=first_slip_patch,
            first_release_s=first_release_s,
            first_release_patch=first_release_patch,
        )


class GripLossEvaluator:
    """Engine-free grip-loss predictor with swappable load and evaluator backends."""

    def __init__(
        self,
        physics: GripLossPhysicsConfig,
        *,
        load_provider_name: str = DEFAULT_GRIP_LOAD_PROVIDER,
        evaluator_name: str = "margin_hysteresis",
    ) -> None:
        if not isinstance(physics, GripLossPhysicsConfig):
            raise TypeError("physics must be GripLossPhysicsConfig")
        self._physics = physics
        self._load_provider_name = load_provider_name
        self._evaluator_name = evaluator_name

    def evaluate(self, loads: GripLoadSeries) -> GripLossReport:
        backend = resolve_grip_loss_evaluator(self._evaluator_name)
        scaled = _scale_loads(loads, self._physics.squeeze_scale)
        return backend.evaluate(scaled, self._physics)

    def evaluate_club(self, club: str, *, evidence_root: Path) -> GripLossReport:
        provider = resolve_grip_load_provider(self._load_provider_name)
        loads = provider.load(club, evidence_root=evidence_root)
        return self.evaluate(loads)

    def evaluate_with_sensitivity(
        self,
        loads: GripLoadSeries,
        friction_scales: Sequence[float],
        squeeze_scales: Sequence[float],
    ) -> GripLossReport:
        base = self.evaluate(loads)
        rows: list[dict[str, Any]] = []
        for mu_scale in friction_scales:
            for sq_scale in squeeze_scales:
                child_physics = GripLossPhysicsConfig(
                    friction=base.physics.friction * mu_scale,
                    normal_floor_n=base.physics.normal_floor_n,
                    thresholds=base.physics.thresholds,
                    squeeze_scale=sq_scale,
                )
                child = GripLossEvaluator(
                    child_physics,
                    load_provider_name=self._load_provider_name,
                    evaluator_name=self._evaluator_name,
                )
                rep = child.evaluate(loads)
                rows.append(
                    {
                        "friction_scale": mu_scale,
                        "squeeze_scale": sq_scale,
                        "first_slip_s": rep.first_slip_s,
                        "first_release_s": rep.first_release_s,
                    }
                )
        return replace(base, sensitivity=tuple(rows))


def serialise_grip_loss_report(report: GripLossReport) -> dict[str, Any]:
    """JSON-safe dict for evidence receipts (margin trace only on summary keys)."""
    thresholds = report.physics.thresholds
    return {
        "club": report.club,
        "source": report.source,
        "disclaimer": report.disclaimer,
        "friction": report.physics.friction,
        "normal_floor_n": report.physics.normal_floor_n,
        "squeeze_scale": report.physics.squeeze_scale,
        "thresholds": {
            "slip_enter_margin": thresholds.slip_enter_margin,
            "slip_exit_margin": thresholds.slip_exit_margin,
            "release_enter_margin": thresholds.release_enter_margin,
            "min_slip_dwell_s": thresholds.min_slip_dwell_s,
            "min_release_dwell_s": thresholds.min_release_dwell_s,
        },
        "first_slip_s": report.first_slip_s,
        "first_slip_patch": report.first_slip_patch,
        "first_release_s": report.first_release_s,
        "first_release_patch": report.first_release_patch,
        "sample_count": int(report.time_s.size),
        "sensitivity": list(report.sensitivity),
    }


register_grip_loss_evaluator("margin_hysteresis", MarginHysteresisGripLossEvaluator())
