"""End-to-end test for :func:`anthropometrics.run_pipeline`.

Drives the orchestrator against the real ``data/C3D_TA_Driver.c3d``
fixture, exports to all four engine formats, round-trips each, and
snapshots the validation report. Closes #4822.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from anthropometrics import (
    SubjectAnthropometrics,
    run_pipeline,
)
from anthropometrics.engine_adapters import ADAPTER_REGISTRY


_REPO_ROOT = Path(__file__).resolve().parents[3]
_C3D_FIXTURE = _REPO_ROOT / "data" / "C3D_TA_Driver.c3d"
_EXPECTED_REPORT = (
    _REPO_ROOT / "tests" / "fixtures" / "anthropometrics" / "expected_report.html"
)


pytestmark = [pytest.mark.integration]


@pytest.fixture(scope="module")
def pipeline_outputs(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[Path, SubjectAnthropometrics]:
    """Run the full pipeline once per module."""
    if not _C3D_FIXTURE.exists():
        pytest.skip(f"C3D fixture not present: {_C3D_FIXTURE}")
    pytest.importorskip("ezc3d")
    out_dir = tmp_path_factory.mktemp("anth_e2e")
    record = run_pipeline(
        _C3D_FIXTURE,
        subject_height_m=1.80,
        subject_mass_kg=75.0,
        estimator="de_leva",
        target_engines=("drake", "mujoco", "pinocchio", "opensim"),
        output_dir=out_dir,
    )
    return out_dir, record


def test_all_four_engine_outputs_exist(
    pipeline_outputs: tuple[Path, SubjectAnthropometrics],
) -> None:
    """Each requested engine writes its native model file."""
    out_dir, _ = pipeline_outputs
    expected = {
        "drake.urdf",
        "pinocchio.urdf",
        "opensim.osim",
        "mujoco.xml",
    }
    actual = {p.name for p in out_dir.iterdir()}
    assert expected.issubset(actual), f"missing engine outputs: {expected - actual}"


def test_subject_json_is_loadable(
    pipeline_outputs: tuple[Path, SubjectAnthropometrics],
) -> None:
    """``output_dir/subject.json`` round-trips via ``load_subject``."""
    from anthropometrics import load_subject

    out_dir, record = pipeline_outputs
    reloaded = load_subject(out_dir / "subject.json")
    assert reloaded.subject_id == record.subject_id
    assert len(reloaded.segments) == len(record.segments)


def test_each_engine_round_trip_recovers_record(
    pipeline_outputs: tuple[Path, SubjectAnthropometrics],
) -> None:
    """``adapter.import_back`` recovers the canonical record per engine.

    Single-precision serialisation in some formats (Simscape's MAT,
    OpenSim's textual XML) means we use ``rtol=1e-6`` rather than
    the inertia adapter's stricter native ``1e-9``.
    """
    out_dir, record = pipeline_outputs

    cases: list[tuple[str, Path]] = [
        ("drake", out_dir / "drake.urdf"),
        ("pinocchio", out_dir / "pinocchio.urdf"),
        ("opensim", out_dir / "opensim.osim"),
        ("myosuite", out_dir / "mujoco.xml"),
    ]
    for engine_name, path in cases:
        adapter = ADAPTER_REGISTRY[engine_name]
        recovered = adapter.import_back(path)
        assert recovered.subject_id == record.subject_id
        assert recovered.height_m == pytest.approx(record.height_m, rel=1e-6)
        assert recovered.mass_kg == pytest.approx(record.mass_kg, rel=1e-6)
        assert len(recovered.segments) == len(record.segments)
        rec_by_name = dict(recovered.segments)
        for name, props in record.segments:
            assert name in rec_by_name, f"{engine_name}: missing segment {name}"
            other = rec_by_name[name]
            assert other.mass_kg == pytest.approx(props.mass_kg, rel=1e-6)
            assert other.length_m == pytest.approx(props.length_m, rel=1e-6)
            np.testing.assert_allclose(
                other.com_xyz_m, props.com_xyz_m, rtol=1e-6, atol=1e-9
            )
            np.testing.assert_allclose(
                other.inertia_tensor, props.inertia_tensor, rtol=1e-6, atol=1e-9
            )


def _normalise_html(text: str) -> str:
    """Collapse runs of whitespace so report-text comparison is robust."""
    return re.sub(r"\s+", " ", text).strip()


def test_validation_report_snapshot(
    pipeline_outputs: tuple[Path, SubjectAnthropometrics],
) -> None:
    """``report.html`` matches the committed snapshot (text equality, normalised).

    On first run the snapshot is written automatically. Subsequent
    runs assert text-equality after collapsing whitespace.
    """
    out_dir, _ = pipeline_outputs
    actual_html = (out_dir / "report.html").read_text(encoding="utf-8")
    if not _EXPECTED_REPORT.exists():
        _EXPECTED_REPORT.parent.mkdir(parents=True, exist_ok=True)
        _EXPECTED_REPORT.write_text(actual_html, encoding="utf-8")
        pytest.skip(
            f"Wrote initial report snapshot to {_EXPECTED_REPORT}; "
            "rerun to verify text-equality."
        )
    expected = _EXPECTED_REPORT.read_text(encoding="utf-8")
    assert _normalise_html(actual_html) == _normalise_html(expected)
