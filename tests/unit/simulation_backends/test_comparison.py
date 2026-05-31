"""Tests for CC-27 cross-engine comparison reports."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.simulation_backends import (
    BackendCapabilities,
    ComparisonInput,
    DivergenceRegistry,
    SimState,
    Trace,
    compare,
    compare_traces,
    render_markdown_report,
    write_report,
)
from src.shared.python.simulation_backends.compare_cli import main as compare_cli_main

pytestmark = pytest.mark.unit


class _FakeBackend:
    def __init__(self, name: str, offset: float = 0.0) -> None:
        self._name = name
        self._offset = offset
        self._state = SimState(q=np.array([0.0, 0.0]), v=np.array([0.0, 0.0]))

    @property
    def capabilities(self) -> BackendCapabilities:
        return BackendCapabilities(name=self._name, provides_dynamics=True)

    def reset(self, state: SimState | None = None) -> None:
        self._state = (
            state.copy()
            if state is not None
            else SimState(
                q=np.zeros(2),
                v=np.zeros(2),
            )
        )

    def step(self, dt: float | None = None) -> None:
        del dt

    def get_state(self) -> SimState:
        return self._state.copy()

    def set_control(self, u: np.ndarray) -> None:
        del u

    def get_time(self) -> float:
        return self._state.time

    def forward_dynamics(
        self, q: np.ndarray, v: np.ndarray, u: np.ndarray | None = None
    ) -> np.ndarray:
        tau = np.zeros_like(q) if u is None else np.asarray(u, dtype=float)
        return tau - self.bias_forces(q, v)

    def mass_matrix(self, q: np.ndarray) -> np.ndarray:
        del q
        return np.eye(2)

    def bias_forces(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        return np.asarray(q, dtype=float) + 0.1 * np.asarray(v, dtype=float)

    def rollout(
        self,
        controls: np.ndarray | None,
        horizon: int,
        dt: float,
    ) -> Trace:
        t = np.arange(horizon + 1, dtype=float) * dt
        q = np.column_stack(
            [
                self._state.q[0] + t + self._offset,
                self._state.q[1] - t + self._offset,
            ]
        )
        v = np.column_stack(
            [
                np.full(horizon + 1, 1.0 + self._offset),
                np.full(horizon + 1, -1.0 + self._offset),
            ]
        )
        if controls is None:
            u = np.zeros((horizon + 1, 2))
        else:
            u = np.zeros((horizon + 1, 2))
            u[:horizon] = controls
        wrench = np.column_stack(
            [
                np.full(horizon + 1, self._offset),
                np.zeros((horizon + 1, 5)),
            ]
        )
        return Trace(
            t=t,
            q=q,
            v=v,
            u=u,
            dt=dt,
            backend=self._name,
            meta={
                "provenance_engine": self._name,
                "provenance_created_at": "2026-05-31T00:00:00Z",
            },
            wrench=wrench,
        )


def test_compare_builds_side_by_side_panels_with_provenance() -> None:
    report = compare(
        [_FakeBackend("alpha"), _FakeBackend("beta", offset=0.02)],
        ComparisonInput(
            horizon=3,
            dt=0.01,
            controls=np.ones((3, 2)),
            initial_state=SimState(q=np.array([1.2, -0.6]), v=np.zeros(2)),
        ),
    )

    panel_names = {panel.name for panel in report.panels}
    assert panel_names == {"kinematics", "kinetics", "ztcf", "zvcf", "wrench"}
    kinematics = next(panel for panel in report.panels if panel.name == "kinematics")
    assert set(kinematics.metrics) == {"q", "v"}
    assert kinematics.provenance_by_engine["alpha"]["stamp"]["engine"] == "alpha"
    assert report.divergences
    first = report.divergences[0]
    assert first.registry.key in DivergenceRegistry.default().entries
    assert first.severity in {"minor", "major"}


def test_compare_omits_within_tolerance_from_divergence_summary() -> None:
    report = compare(
        [_FakeBackend("alpha"), _FakeBackend("alpha")],
        ComparisonInput(horizon=2, dt=0.01),
        labels=("alpha", "alpha"),
    )

    assert report.engines == ("alpha", "alpha#2")
    assert report.divergences == ()
    assert any(
        annotation.severity == "within_tolerance"
        for panel in report.panels
        for annotation in panel.annotations
    )


def test_compare_traces_skips_counterfactual_panels_without_engine_objects() -> None:
    trace_a = Trace(
        t=np.array([0.0, 0.1]),
        q=np.zeros((2, 2)),
        v=np.zeros((2, 2)),
        dt=0.1,
        backend="a",
    )
    trace_b = Trace(
        t=np.array([0.0, 0.1]),
        q=np.ones((2, 2)),
        v=np.zeros((2, 2)),
        dt=0.1,
        backend="b",
    )

    report = compare_traces({"a": trace_a, "b": trace_b})

    ztcf = next(panel for panel in report.panels if panel.name == "ztcf")
    assert ztcf.metrics == {}
    assert report.divergences[0].registry.key == "kinematics.q"


def test_markdown_and_json_writers_include_annotations(tmp_path: Path) -> None:
    report = compare(
        [_FakeBackend("alpha"), _FakeBackend("beta", offset=0.1)],
        ComparisonInput(horizon=2, dt=0.01),
    )

    markdown = render_markdown_report(report)
    assert "# Cross-Engine Comparison Report" in markdown
    assert "Provenance:" in markdown
    assert "kinematics.q" in markdown

    json_path = tmp_path / "report.json"
    md_path = tmp_path / "report.md"
    write_report(report, json_path, format="json")
    write_report(report, md_path)

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["baseline"] == "alpha"
    assert payload["divergences"]
    assert "Cross-Engine" in md_path.read_text(encoding="utf-8")


def test_cli_single_command_writes_markdown_report(tmp_path: Path) -> None:
    output = tmp_path / "cc27.md"

    exit_code = compare_cli_main(
        [
            "--engines",
            "ode,ode",
            "--horizon",
            "2",
            "--dt",
            "0.01",
            "--output",
            str(output),
        ]
    )

    assert exit_code == 0
    content = output.read_text(encoding="utf-8")
    assert "Cross-Engine Comparison Report" in content
    assert "`ode#2`" in content
