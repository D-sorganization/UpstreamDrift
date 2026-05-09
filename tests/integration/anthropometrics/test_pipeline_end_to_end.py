"""Issue #4819 — full :func:`run_pipeline` against a real C3D fixture.

Exercises the high-level orchestrator end-to-end against
``data/C3D_TA_Driver.c3d`` and asserts the issue's acceptance
criteria:

* mass closure: ``sum(seg.mass) / subject_mass ∈ [0.99, 1.01]``;
* every segment inertia tensor has strictly positive eigenvalues
  (`np.linalg.eigvalsh`);
* every requested engine produces a file that round-trips back to the
  canonical record at ``rtol=1e-6``;
* ``report.html`` exists and contains both the ``Mass closure`` and
  ``Inertia spectral`` headings.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anthropometrics import ADAPTER_REGISTRY, SubjectAnthropometrics
from anthropometrics.pipeline import run_pipeline

_REPO_ROOT = Path(__file__).resolve().parents[3]
_C3D_FIXTURE = _REPO_ROOT / "data" / "C3D_TA_Driver.c3d"

# Expected on-disk file names per ``pipeline._ENGINE_EXTENSIONS``.
_EXPECTED_OUTPUT_FILES: dict[str, str] = {
    "drake": "drake.urdf",
    "mujoco": "mujoco.xml",
    "pinocchio": "pinocchio.urdf",
    "opensim": "opensim.osim",
}

# Looser tolerance than the cross-engine round-trip: the pipeline goes
# subject → estimator → engine file → reload, and JSON / XML number
# formatting introduces floats-printable rounding at the ~1e-7 scale.
_ROUNDTRIP_RTOL: float = 1.0e-6


@pytest.fixture(scope="module")
def pipeline_run(
    tmp_path_factory: pytest.TempPathFactory,
) -> tuple[SubjectAnthropometrics, Path]:
    """Drive ``run_pipeline`` once and reuse the outputs across tests.

    The fixture is module-scoped so the (relatively slow) C3D read
    only runs once. Each test below operates on the cached
    ``(record, output_dir)`` pair.
    """
    if not _C3D_FIXTURE.exists():
        pytest.skip(f"C3D fixture not bundled: {_C3D_FIXTURE}")
    pytest.importorskip("ezc3d", reason="run_pipeline needs ezc3d to read C3D")

    output_dir = tmp_path_factory.mktemp("pipeline_e2e")
    record = run_pipeline(
        _C3D_FIXTURE,
        subject_height_m=1.78,
        subject_mass_kg=75.0,
        estimator="de_leva",
        target_engines=("drake", "mujoco", "pinocchio", "opensim"),
        output_dir=output_dir,
    )
    return record, output_dir


def test_mass_closure_within_one_percent(
    pipeline_run: tuple[SubjectAnthropometrics, Path],
) -> None:
    """Sum of segment masses divided by subject mass lies in [0.99, 1.01]."""
    record, _ = pipeline_run
    total = float(sum(float(p.mass_kg) for _, p in record.segments))
    ratio = total / float(record.mass_kg)
    assert 0.99 <= ratio <= 1.01, f"mass closure out of band: ratio={ratio:.6f}"


def test_every_segment_inertia_has_positive_eigenvalues(
    pipeline_run: tuple[SubjectAnthropometrics, Path],
) -> None:
    """``np.linalg.eigvalsh`` of each inertia tensor returns positive values."""
    record, _ = pipeline_run
    for name, props in record.segments:
        eigs = np.linalg.eigvalsh(np.asarray(props.inertia_tensor))
        assert np.all(eigs > 0), (
            f"non-positive eigenvalues on segment {name}: {eigs.tolist()}"
        )


@pytest.mark.parametrize("engine", sorted(_EXPECTED_OUTPUT_FILES))
def test_engine_export_file_exists(
    engine: str,
    pipeline_run: tuple[SubjectAnthropometrics, Path],
) -> None:
    """Every requested engine produces its canonical on-disk file."""
    _, output_dir = pipeline_run
    expected = output_dir / _EXPECTED_OUTPUT_FILES[engine]
    assert expected.exists(), f"missing {engine} output: {expected}"
    assert expected.stat().st_size > 0, f"{engine} output is empty: {expected}"


def _adapter_for(engine: str):
    """Return the canonical :class:`EngineAdapter` for *engine*.

    ``mujoco`` is an alias for the MyoSuite (MuJoCo-based) adapter.
    """
    canonical = "myosuite" if engine == "mujoco" else engine
    return ADAPTER_REGISTRY[canonical]


@pytest.mark.parametrize("engine", sorted(_EXPECTED_OUTPUT_FILES))
def test_engine_output_round_trips_to_same_record(
    engine: str,
    pipeline_run: tuple[SubjectAnthropometrics, Path],
) -> None:
    """Reading any engine output back yields the same record (rtol=1e-6)."""
    record, output_dir = pipeline_run
    target = output_dir / _EXPECTED_OUTPUT_FILES[engine]
    adapter = _adapter_for(engine)
    recovered = adapter.import_back(target)

    assert recovered.subject_id == record.subject_id
    assert recovered.source_method == record.source_method
    assert len(recovered.segments) == len(record.segments)
    assert recovered.height_m == pytest.approx(record.height_m, rel=_ROUNDTRIP_RTOL)
    assert recovered.mass_kg == pytest.approx(record.mass_kg, rel=_ROUNDTRIP_RTOL)
    for (rec_n, rec_p), (org_n, org_p) in zip(
        recovered.segments, record.segments, strict=True
    ):
        assert rec_n == org_n
        assert rec_p.mass_kg == pytest.approx(org_p.mass_kg, rel=_ROUNDTRIP_RTOL)
        assert rec_p.length_m == pytest.approx(org_p.length_m, rel=_ROUNDTRIP_RTOL)
        np.testing.assert_allclose(
            rec_p.com_xyz_m, org_p.com_xyz_m, rtol=_ROUNDTRIP_RTOL, atol=1e-9
        )
        np.testing.assert_allclose(
            rec_p.inertia_tensor,
            org_p.inertia_tensor,
            rtol=_ROUNDTRIP_RTOL,
            atol=1e-12,
        )


def test_report_html_exists_with_required_headings(
    pipeline_run: tuple[SubjectAnthropometrics, Path],
) -> None:
    """``report.html`` exists and contains the two required headings."""
    _, output_dir = pipeline_run
    report = output_dir / "report.html"
    assert report.exists(), f"report.html missing in {output_dir}"
    text = report.read_text(encoding="utf-8")
    assert "Mass closure" in text, "report.html missing 'Mass closure' heading"
    assert "Inertia spectral" in text, "report.html missing 'Inertia spectral' heading"


def test_canonical_subject_json_persists(
    pipeline_run: tuple[SubjectAnthropometrics, Path],
) -> None:
    """The pipeline persists the canonical record to ``subject.json``."""
    _, output_dir = pipeline_run
    canonical = output_dir / "subject.json"
    assert canonical.exists(), f"subject.json missing in {output_dir}"
    assert canonical.stat().st_size > 0
