"""Saved operator jobs for reproducible capture-to-reference fitting."""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from src.motion_capture.provenance import sha256_of
from src.motion_capture.reconstruct.model import FitOptions
from src.motion_capture.rig.documents import write_document
from src.shared.python.logging_pkg.logging_config import get_logger

from .fit_catalog import URDF_PRESETS, MJCF_PRESETS, available_models, model_inventory
from .fit_pipeline import fit_reference, save_reference_fit
from .fitting import MarkerProfile, TOUR_AVERAGE_PROFILE
from .importers import load_motion_draft
from .storage import ReferenceLibrary
from .urdf_models import load_urdf_model
from .mjcf_models import load_mjcf_model

logger = get_logger(__name__)
ModelName = Annotated[str, Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")]


class ModelFile(BaseModel):
    """An external native file and explicit body/link-to-reconstruction mapping."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    path: str = Field(min_length=1)
    format: Literal["urdf", "mjcf"]
    landmarks: dict[str, str | tuple[str, ...]]


class FitJob(BaseModel):
    """Portable input configuration; strides are explicit sample selection."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    schema_version: Literal["reference-fit-job/1.0"] = "reference-fit-job/1.0"
    source: str = Field(min_length=1)
    output: str = Field(min_length=1)
    models: tuple[ModelName, ...] = ("golfer",)
    custom_models: dict[ModelName, ModelFile] = Field(default_factory=dict)
    profile: MarkerProfile = TOUR_AVERAGE_PROFILE
    stride: int = Field(default=1, ge=1, strict=True)
    max_iterations: int = Field(default=100, ge=1, strict=True)
    fit_lengths: bool = False
    outlier_gate_sigma: float = Field(default=1000.0, gt=0, allow_inf_nan=False)
    huber_delta: float = Field(default=1000.0, gt=0, allow_inf_nan=False)
    library: str | None = None


def run_fit_job(job: FitJob, repo_root: Path) -> dict[str, Any]:
    """Run selected models, persist every outcome, optionally publish to a library.

    Output must be new. Failed models are recorded and never published; other
    models continue. A completed solver is reported as fitted, not scientifically
    validated. Source C3D and model files are hashed for reproducibility.
    """
    catalog = available_models(repo_root)
    for name, model_file in job.custom_models.items():
        if name in catalog:
            raise ValueError(f"Custom model name collides with catalog: {name}")
        loader = load_urdf_model if model_file.format == "urdf" else load_mjcf_model
        catalog[name] = loader(
            Path(model_file.path), name=name, landmark_map=model_file.landmarks
        )
    selected = tuple(catalog) if job.models == ("all",) else job.models
    if (
        not selected
        or len(set(selected)) != len(selected)
        or set(selected) - set(catalog)
    ):
        raise ValueError("Select unique available model names, or 'all'")
    source = load_motion_draft(Path(job.source))
    source = replace(
        source, points=source.points[:: job.stride], time_s=source.time_s[:: job.stride]
    )
    output = Path(job.output)
    output.mkdir(parents=True, exist_ok=False)
    write_document(output / "job.json", job.model_dump(mode="json"))
    report: dict[str, Any] = {
        "schema_version": "reference-fit-run/1.0",
        "source": source.source.model_dump(),
        "models": {},
        "inventory": model_inventory(repo_root),
    }
    for name in selected:
        model = catalog[name]
        try:
            options = FitOptions(
                max_iterations=job.max_iterations,
                gate=job.outlier_gate_sigma,
                huber_delta=job.huber_delta,
                fit_lengths=model.learnable_lengths if job.fit_lengths else (),
                sigma_length_m=0.05 if job.fit_lengths else 0.003,
            )
            result = fit_reference(source, job.profile, model, options=options)
            native_presets = URDF_PRESETS | MJCF_PRESETS
            if name in native_presets:
                model_path = repo_root / native_presets[name][0]
                result.manifest["model_source"] = {
                    "path": native_presets[name][0],
                    "sha256": sha256_of(model_path),
                }
            elif name in job.custom_models:
                model_path = Path(job.custom_models[name].path)
                result.manifest["model_source"] = {
                    "path": str(model_path),
                    "sha256": sha256_of(model_path),
                }
            save_reference_fit(result, output / name)
            if job.library:
                ReferenceLibrary(Path(job.library)).save(result.asset)
            report["models"][name] = {
                "status": "fitted",
                "reference_id": result.asset.id,
                **result.report,
            }
            logger.info(
                "%s all-observed RMS %.4f m", name, result.report["all_observed_rms_m"]
            )
        except (ValueError, OSError) as exc:
            logger.exception("Reference fitting failed for %s", name)
            report["models"][name] = {"status": "failed", "reason": str(exc)}
        write_document(output / "report.json", report)
    return report


def main(argv: list[str] | None = None) -> int:
    """Run a saved JSON job or list model names; failures return a nonzero code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repo-root", type=Path, default=Path(__file__).resolve().parents[3]
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--config", type=Path)
    mode.add_argument("--list-models", action="store_true")
    args = parser.parse_args(argv)
    if args.list_models:
        sys.stdout.write("\n".join(available_models(args.repo_root)) + "\n")
        return 0
    job = FitJob.model_validate_json(args.config.read_text(encoding="utf-8"))
    report = run_fit_job(job, args.repo_root)
    sys.stdout.write(json.dumps(report, allow_nan=False, indent=2) + "\n")
    return int(any(item["status"] != "fitted" for item in report["models"].values()))


if __name__ == "__main__":
    raise SystemExit(main())
