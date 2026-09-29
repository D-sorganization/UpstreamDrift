from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPT_PATH = (
    Path(__file__).resolve().parents[2]
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "MachineLearning"
    / "run_matching_pareto_sweep.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("golf_ml_pareto_sweep", SCRIPT_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {SCRIPT_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _write_minimal_csv(path: Path) -> None:
    path.write_text("time,value\n0,0\n1,1\n", encoding="utf-8")


def test_positive_float_grid_validation() -> None:
    module = _load_module()

    assert module.parse_positive_float_grid("1e-6, 0.25,2", "weights") == [
        1.0e-6,
        0.25,
        2.0,
    ]
    with pytest.raises(ValueError, match="at least one"):
        module.parse_positive_float_grid(" , ", "weights")
    with pytest.raises(ValueError, match="positive finite"):
        module.parse_positive_float_grid("0,1", "weights")
    with pytest.raises(ValueError, match="non-float"):
        module.parse_positive_float_grid("abc", "weights")


def test_pareto_sweep_writes_run_outputs_and_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    checkpoint = tmp_path / "checkpoint.pt"
    desired = tmp_path / "desired.csv"
    reference = tmp_path / "reference.csv"
    output_dir = tmp_path / "pareto"
    checkpoint.write_text("synthetic checkpoint", encoding="utf-8")
    _write_minimal_csv(desired)
    _write_minimal_csv(reference)
    calls: list[dict[str, object]] = []

    def fake_optimize_sequence(**kwargs):
        output_csv = kwargs["output_csv"]
        effort_weight = float(kwargs["effort_weight"])
        smoothness_weight = float(kwargs["smoothness_weight"])
        calls.append(kwargs)
        rows = [
            {"time": "0.0", "hip_torque": "0.0", "shoulder_torque": "0.0"},
            {
                "time": "0.5",
                "hip_torque": str(effort_weight * 10.0),
                "shoulder_torque": str(smoothness_weight * 100.0),
            },
            {
                "time": "1.0",
                "hip_torque": str(effort_weight * 20.0),
                "shoulder_torque": str(smoothness_weight * 50.0),
            },
        ]
        with output_csv.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        tracking_loss = 1.0 / effort_weight
        summary = {
            "best_loss": tracking_loss + effort_weight,
            "history": [
                {
                    "step": 1,
                    "loss": tracking_loss + smoothness_weight,
                    "tracking_loss": tracking_loss,
                    "effort_loss": effort_weight,
                    "smoothness_loss": smoothness_weight,
                }
            ],
        }
        output_csv.with_suffix(".summary.json").write_text(
            json.dumps(summary), encoding="utf-8"
        )

    monkeypatch.setattr(module, "optimize_sequence", fake_optimize_sequence)

    runs = module.run_sweep(
        checkpoint=checkpoint,
        desired_club_csv=desired,
        reference_body_csv=reference,
        output_dir=output_dir,
        effort_weights=[0.1, 1.0],
        smoothness_weights=[0.01, 0.1],
        steps=3,
        learning_rate=0.05,
        scenario="synthetic",
        device="cpu",
    )

    assert len(calls) == 4
    assert len(runs) == 4
    assert (output_dir / "pareto_summary.csv").exists()
    summary_md = (output_dir / "pareto_summary.md").read_text(encoding="utf-8")
    assert "best_low_error" in summary_md
    assert "best_low_effort" in summary_md
    assert "knee_point" in summary_md

    run_summary = json.loads(runs[0].summary_json.read_text(encoding="utf-8"))
    assert run_summary["scenario"] == "synthetic"
    assert run_summary["torque_effort"]["available"] is True
    assert runs[0].torque_csv.exists()

    with (output_dir / "pareto_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 4
    assert any("best_low_error" in row["candidate_roles"] for row in rows)
    assert any("best_low_effort" in row["candidate_roles"] for row in rows)


def test_run_sweep_rejects_invalid_runtime_settings(tmp_path: Path) -> None:
    module = _load_module()

    with pytest.raises(ValueError, match="steps"):
        module.run_sweep(
            checkpoint=tmp_path / "checkpoint.pt",
            desired_club_csv=tmp_path / "desired.csv",
            reference_body_csv=tmp_path / "reference.csv",
            output_dir=tmp_path / "out",
            effort_weights=[1.0],
            smoothness_weights=[1.0],
            steps=0,
            learning_rate=0.1,
            scenario="synthetic",
            device="cpu",
        )
