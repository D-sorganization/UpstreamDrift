"""Operator jobs select models and publish existing library assets."""

from pathlib import Path

import pytest

from src.motion_capture.reference.fit_job import FitJob, run_fit_job
from src.motion_capture.reference.fit_catalog import available_models

pytestmark = pytest.mark.unit


def test_catalog_imports_bundled_native_urdf_trees() -> None:
    root = Path(__file__).resolve().parents[2]
    catalog = available_models(root)
    assert {
        "golfer",
        "double_pendulum",
        "triple_pendulum",
        "pinocchio_golfer",
        "drake_golfer",
        "simple_humanoid",
    } <= set(catalog)
    assert len(catalog["pinocchio_golfer"].spec.joints) > 30
    assert len(catalog["simple_humanoid"].spec.joints) < 20


def test_job_rejects_invalid_stride() -> None:
    with pytest.raises(ValueError):
        FitJob(source="file.c3d", output="out", stride=0)


def test_job_rejects_unsafe_model_output_name() -> None:
    with pytest.raises(ValueError):
        FitJob(source="file.c3d", output="out", models=("../escape",))


def test_job_runs_real_c3d_to_library(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    job = FitJob(
        source=str(root / "data/C3D_TA_Driver.c3d"),
        output=str(tmp_path / "run"),
        models=("double_pendulum",),
        stride=36,
        max_iterations=3,
        library=str(tmp_path / "library"),
    )
    report = run_fit_job(job, root)
    assert report["models"]["double_pendulum"]["status"] == "fitted"
    assert len(list((tmp_path / "library").glob("*.json"))) == 1
    assert (tmp_path / "run/job.json").is_file()


def test_cli_lists_presets(capsys: pytest.CaptureFixture[str]) -> None:
    from src.motion_capture.reference.fit_job import main

    assert main(["--list-models"]) == 0
    assert "pinocchio_golfer" in capsys.readouterr().out


def test_inventory_exposes_unavailable_models() -> None:
    from src.motion_capture.reference.fit_catalog import model_inventory

    root = Path(__file__).resolve().parents[2]
    inventory = model_inventory(root)
    assert inventory["myosuite_body"]["status"] == "unavailable"
    assert "placeholder" in inventory["myosuite_body"]["reason"]
    assert inventory["opensim_golfer"]["status"] == "unavailable"


def test_custom_model_job_uses_explicit_native_mapping(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[2]
    model_path = tmp_path / "native.urdf"
    model_path.write_text('<robot name="single"><link name="root"/></robot>')
    job = FitJob.model_validate(
        {
            "source": str(root / "data/C3D_TA_Driver.c3d"),
            "output": str(tmp_path / "custom"),
            "models": ["custom"],
            "stride": 36,
            "max_iterations": 3,
            "custom_models": {
                "custom": {
                    "path": str(model_path),
                    "format": "urdf",
                    "landmarks": {"root": "mid_hip"},
                }
            },
        }
    )
    report = run_fit_job(job, root)
    assert report["models"]["custom"]["status"] == "fitted"
