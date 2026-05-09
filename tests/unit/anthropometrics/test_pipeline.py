"""Unit tests for :mod:`anthropometrics.pipeline`.

Issue #4822 — Child #10 of EPIC #4797. The integration suite at
``tests/integration/anthropometrics/test_pipeline_e2e.py`` exercises
the real C3D fixture; here we lean on lightweight stubs so the
DbC contract, fallback logic, validation report, and engine
dispatch can be tested without touching ``ezc3d``.
"""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import anthropometrics
from anthropometrics import (
    SubjectAnthropometrics,
    run_pipeline,
)
from anthropometrics import pipeline as pipeline_module
from anthropometrics._types import Sex
from anthropometrics.readers.c3d_subject_info import C3DSubjectMetadata


# --------------------------------------------------------------------------- #
# Fixtures.                                                                   #
# --------------------------------------------------------------------------- #
@pytest.fixture
def fake_c3d(tmp_path: Path) -> Path:
    """Return a path to a placeholder ``.c3d`` file (contents irrelevant)."""
    path = tmp_path / "fake.c3d"
    path.write_bytes(b"not a real c3d")
    return path


@pytest.fixture
def stub_c3d_metadata(monkeypatch: pytest.MonkeyPatch) -> C3DSubjectMetadata:
    """Stub :func:`read_c3d_subject_metadata` to return a known record."""
    meta = C3DSubjectMetadata(
        subject_id="stub_subject",
        height_m=1.78,
        mass_kg=72.5,
        age_years=29.0,
        sex=Sex.MALE,
        leg_length_m=None,
        arm_length_m=None,
    )

    def _fake_reader(_path: Path | str) -> C3DSubjectMetadata:
        return meta

    monkeypatch.setattr(pipeline_module, "read_c3d_subject_metadata", _fake_reader)
    return meta


@pytest.fixture
def stub_marker_loader(monkeypatch: pytest.MonkeyPatch) -> dict[str, np.ndarray]:
    """Stub the ezc3d-backed marker loader with a synthetic trajectory.

    The synthetic trajectory satisfies a few of the default mocap
    segments (LShoulderTop / LElbowOut / LWristTop) so the pipeline
    exercises the mocap-length branch.
    """
    np.random.seed(0)
    n_frames = 20
    base = np.zeros((n_frames, 3))
    elbow_offset = np.tile(np.array([0.0, 0.30, 0.0]), (n_frames, 1))
    wrist_offset = np.tile(np.array([0.0, 0.30 + 0.27, 0.0]), (n_frames, 1))
    markers: dict[str, np.ndarray] = {
        "LShoulderTop": base.copy(),
        "LElbowOut": elbow_offset.copy(),
        "LWristTop": wrist_offset.copy(),
    }

    def _fake_loader(_path: Path) -> dict[str, np.ndarray]:
        return markers

    monkeypatch.setattr(pipeline_module, "_load_marker_trajectories", _fake_loader)
    return markers


# --------------------------------------------------------------------------- #
# Public-API smoke / happy path.                                              #
# --------------------------------------------------------------------------- #
def test_run_pipeline_is_in_top_level_all() -> None:
    """``run_pipeline`` is part of the public ``anthropometrics`` surface."""
    assert "run_pipeline" in anthropometrics.__all__
    assert anthropometrics.run_pipeline is run_pipeline


def test_run_pipeline_happy_path_with_explicit_scalars(
    tmp_path: Path,
    fake_c3d: Path,
    stub_c3d_metadata: C3DSubjectMetadata,
    stub_marker_loader: dict[str, np.ndarray],
) -> None:
    """Full pipeline with explicit height/mass produces every artefact."""
    out = tmp_path / "out"
    record = run_pipeline(
        fake_c3d,
        subject_height_m=1.80,
        subject_mass_kg=75.0,
        estimator="de_leva",
        target_engines=("drake", "pinocchio", "opensim"),
        output_dir=out,
    )

    assert isinstance(record, SubjectAnthropometrics)
    assert record.height_m == pytest.approx(1.80)
    assert record.mass_kg == pytest.approx(75.0)

    assert (out / "subject.json").exists()
    assert (out / "report.html").exists()
    assert (out / "drake.urdf").exists()
    assert (out / "pinocchio.urdf").exists()
    assert (out / "opensim.osim").exists()


def test_run_pipeline_falls_back_to_c3d_metadata(
    tmp_path: Path,
    fake_c3d: Path,
    stub_c3d_metadata: C3DSubjectMetadata,
    stub_marker_loader: dict[str, np.ndarray],
) -> None:
    """When height/mass are omitted they come from the C3D metadata stub."""
    out = tmp_path / "out"
    record = run_pipeline(
        fake_c3d,
        target_engines=(),
        output_dir=out,
    )
    assert record.height_m == pytest.approx(stub_c3d_metadata.height_m)
    assert record.mass_kg == pytest.approx(stub_c3d_metadata.mass_kg)
    assert record.sex == "M"
    assert record.age_years == pytest.approx(stub_c3d_metadata.age_years)


def test_run_pipeline_dempster_estimator(
    tmp_path: Path,
    fake_c3d: Path,
    stub_c3d_metadata: C3DSubjectMetadata,
    stub_marker_loader: dict[str, np.ndarray],
) -> None:
    """Dempster estimator is dispatchable through the orchestrator."""
    out = tmp_path / "out"
    record = run_pipeline(
        fake_c3d,
        subject_height_m=1.80,
        subject_mass_kg=75.0,
        estimator="dempster",
        target_engines=(),
        output_dir=out,
    )
    assert "dempster" in record.source_method.lower()


def test_run_pipeline_zatsiorsky_estimator(
    tmp_path: Path,
    fake_c3d: Path,
    stub_c3d_metadata: C3DSubjectMetadata,
    stub_marker_loader: dict[str, np.ndarray],
) -> None:
    """Zatsiorsky estimator is dispatchable through the orchestrator."""
    out = tmp_path / "out"
    record = run_pipeline(
        fake_c3d,
        subject_height_m=1.80,
        subject_mass_kg=75.0,
        estimator="zatsiorsky",
        target_engines=(),
        output_dir=out,
    )
    assert "zatsiorsky" in record.source_method.lower()


def test_run_pipeline_creates_missing_output_dir(
    tmp_path: Path,
    fake_c3d: Path,
    stub_c3d_metadata: C3DSubjectMetadata,
    stub_marker_loader: dict[str, np.ndarray],
) -> None:
    """Nested output_dir is created automatically (DbC postcondition)."""
    out = tmp_path / "deeply" / "nested" / "out"
    assert not out.exists()
    run_pipeline(
        fake_c3d,
        subject_height_m=1.80,
        subject_mass_kg=75.0,
        target_engines=(),
        output_dir=out,
    )
    assert out.is_dir()
    assert (out / "subject.json").exists()


def test_run_pipeline_skips_unknown_engine_with_warning(
    tmp_path: Path,
    fake_c3d: Path,
    stub_c3d_metadata: C3DSubjectMetadata,
    stub_marker_loader: dict[str, np.ndarray],
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Unknown engines are skipped with a warning, never raise."""
    out = tmp_path / "out"
    with caplog.at_level("WARNING", logger="anthropometrics.pipeline"):
        record = run_pipeline(
            fake_c3d,
            subject_height_m=1.80,
            subject_mass_kg=75.0,
            target_engines=("drake", "no_such_engine"),
            output_dir=out,
        )
    assert isinstance(record, SubjectAnthropometrics)
    assert (out / "drake.urdf").exists()
    assert not (out / "no_such_engine").exists()
    assert any("no_such_engine" in rec.message for rec in caplog.records)


def test_run_pipeline_mujoco_alias_routes_to_myosuite(
    tmp_path: Path,
    fake_c3d: Path,
    stub_c3d_metadata: C3DSubjectMetadata,
    stub_marker_loader: dict[str, np.ndarray],
) -> None:
    """``mujoco`` alias produces the MJCF + meta.json sidecar via MyoSuite."""
    out = tmp_path / "out"
    run_pipeline(
        fake_c3d,
        subject_height_m=1.80,
        subject_mass_kg=75.0,
        target_engines=("mujoco",),
        output_dir=out,
    )
    assert (out / "mujoco.xml").exists()


# --------------------------------------------------------------------------- #
# Validation / DbC.                                                           #
# --------------------------------------------------------------------------- #
def test_run_pipeline_raises_for_missing_file(tmp_path: Path) -> None:
    """Missing mocap file raises FileNotFoundError."""
    with pytest.raises(FileNotFoundError):
        run_pipeline(
            tmp_path / "missing.c3d",
            subject_height_m=1.8,
            subject_mass_kg=75.0,
            output_dir=tmp_path / "out",
        )


def test_run_pipeline_rejects_unknown_estimator(
    tmp_path: Path,
    fake_c3d: Path,
) -> None:
    """Unknown estimator name raises ValueError before doing any work."""
    with pytest.raises(ValueError, match="estimator must be one of"):
        run_pipeline(
            fake_c3d,
            subject_height_m=1.8,
            subject_mass_kg=75.0,
            estimator="unknown",  # type: ignore[arg-type]
            output_dir=tmp_path / "out",
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("subject_height_m", 0.0),
        ("subject_height_m", -1.0),
        ("subject_height_m", float("nan")),
        ("subject_mass_kg", 0.0),
        ("subject_mass_kg", -10.0),
        ("subject_mass_kg", float("inf")),
    ],
)
def test_run_pipeline_rejects_non_positive_scalars(
    tmp_path: Path, fake_c3d: Path, field: str, value: float
) -> None:
    """Subject height/mass must be positive finite floats."""
    kwargs: dict[str, Any] = {
        "subject_height_m": 1.8,
        "subject_mass_kg": 75.0,
        "output_dir": tmp_path / "out",
    }
    kwargs[field] = value
    with pytest.raises(ValueError, match=field):
        run_pipeline(fake_c3d, **kwargs)


def test_run_pipeline_raises_when_metadata_lacks_scalars(
    tmp_path: Path,
    fake_c3d: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If neither caller nor C3D supply height/mass, raise ValueError."""
    blank = C3DSubjectMetadata(
        subject_id="blank",
        height_m=None,
        mass_kg=None,
        age_years=None,
        sex=Sex.UNSPECIFIED,
        leg_length_m=None,
        arm_length_m=None,
    )

    def _fake_reader(_path: Path | str) -> C3DSubjectMetadata:
        return blank

    monkeypatch.setattr(pipeline_module, "read_c3d_subject_metadata", _fake_reader)
    monkeypatch.setattr(pipeline_module, "_load_marker_trajectories", lambda _p: {})
    with pytest.raises(ValueError, match="height/mass"):
        run_pipeline(fake_c3d, target_engines=(), output_dir=tmp_path / "out")


# --------------------------------------------------------------------------- #
# Validation report.                                                          #
# --------------------------------------------------------------------------- #
def test_validation_report_contents(
    tmp_path: Path,
    fake_c3d: Path,
    stub_c3d_metadata: C3DSubjectMetadata,
    stub_marker_loader: dict[str, np.ndarray],
) -> None:
    """The HTML report exposes the three closure / spectral checks."""
    out = tmp_path / "out"
    run_pipeline(
        fake_c3d,
        subject_height_m=1.80,
        subject_mass_kg=75.0,
        target_engines=(),
        output_dir=out,
    )
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "Mass closure" in html
    assert "Inertia spectral check" in html
    assert "Length closure" in html
    assert "Mocap-derived segment lengths" in html
    # Mass closure should be exactly 1.0 (estimators normalise mass).
    assert "1.000000" in html


def test_validation_report_handles_no_mocap_lengths(
    tmp_path: Path,
    fake_c3d: Path,
    stub_c3d_metadata: C3DSubjectMetadata,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When marker loading fails the report still renders cleanly."""
    monkeypatch.setattr(
        pipeline_module,
        "_load_marker_trajectories",
        lambda _p: (_ for _ in ()).throw(OSError("no mocap")),
    )
    out = tmp_path / "out"
    run_pipeline(
        fake_c3d,
        subject_height_m=1.80,
        subject_mass_kg=75.0,
        target_engines=(),
        output_dir=out,
    )
    html = (out / "report.html").read_text(encoding="utf-8")
    assert "No mocap-derived segment lengths" in html


# --------------------------------------------------------------------------- #
# Internal helpers.                                                           #
# --------------------------------------------------------------------------- #
def test_load_marker_trajectories_imports_ezc3d_lazily(
    tmp_path: Path,
    fake_c3d: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """_load_marker_trajectories opens via ezc3d when available."""
    captured: dict[str, str] = {}

    class _StubC3D:
        def __init__(self, path: str) -> None:
            captured["path"] = path

        def __getitem__(self, key: str) -> Any:  # pragma: no cover - exhaustive
            if key == "parameters":
                return {
                    "POINT": {
                        "LABELS": {"value": ["a", "b"]},
                        "UNITS": {"value": ["m"]},
                    }
                }
            if key == "data":
                return {
                    "points": np.zeros((4, 2, 5), dtype=float),
                }
            raise KeyError(key)

    fake_module = types.ModuleType("ezc3d")
    fake_module.c3d = _StubC3D  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "ezc3d", fake_module)

    out = pipeline_module._load_marker_trajectories(fake_c3d)
    assert set(out.keys()) == {"a", "b"}
    assert all(arr.shape == (5, 3) for arr in out.values())
    assert captured["path"] == str(fake_c3d)


def test_estimate_mocap_lengths_returns_empty_when_no_markers(
    fake_c3d: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """No marker overlap with the default segment list yields ``{}``."""
    monkeypatch.setattr(
        pipeline_module,
        "_load_marker_trajectories",
        lambda _p: {"NoneOfTheUsualMarkers": np.zeros((3, 3))},
    )
    assert pipeline_module._estimate_mocap_lengths_safely(fake_c3d) == {}


def test_slugify_replaces_unsafe_chars() -> None:
    """_slugify produces a non-empty filesystem-safe id."""
    assert pipeline_module._slugify("a b c!") == "a_b_c"
    assert pipeline_module._slugify("***") == "subject"
