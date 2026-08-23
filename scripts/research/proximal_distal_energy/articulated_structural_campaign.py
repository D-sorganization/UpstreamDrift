"""Run and publish the registered structural headline propagation campaign."""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_atlas_authority import (
    ArticulatedAtlasAuthority,
)
from scripts.research.proximal_distal_energy.articulated_ground_atlas import (
    ArticulatedGroundAtlasConfig,
)
from scripts.research.proximal_distal_energy.articulated_scaled_authority import (
    load_scaled_authority,
)
from scripts.research.proximal_distal_energy.articulated_shaft_atlas import (
    ArticulatedShaftAtlasConfig,
)
from scripts.research.proximal_distal_energy.articulated_structural_atlas_execution import (
    StructuralAtlasExecution,
    execute_structural_ground_atlas,
    execute_structural_shaft_atlas,
)
from scripts.research.proximal_distal_energy.articulated_structural_axis_evidence import (
    assemble_structural_axis_pathway_evidence,
)
from scripts.research.proximal_distal_energy.articulated_structural_cell_evidence import (
    write_structural_cell_evidence,
)
from scripts.research.proximal_distal_energy.articulated_structural_corner_evidence import (
    StructuralCornerEvidenceRequest,
    StructuralCornerPathwayEvidence,
    assemble_structural_corner_pathway_evidence,
)
from scripts.research.proximal_distal_energy.articulated_structural_propagation_plan import (
    DEFAULT_OUTPUT as DEFAULT_PLAN,
    validate_structural_propagation_plan,
)
from scripts.research.proximal_distal_energy.articulated_structural_publication import (
    publish_structural_figure_bundle,
)
from scripts.research.proximal_distal_energy.articulated_structural_result import (
    assemble_structural_propagation_result,
    validate_structural_propagation_bundle_against_plan,
    write_structural_propagation_result,
)

ROOT = Path(__file__).resolve().parents[3]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
FIGURES = ROOT / "docs/research/proximal_distal_energy_transfer/figures"
DEFAULT_RESULT = DATA / "articulated_structural_propagation_result.json"
DEFAULT_FIGURE_DATA = DATA / "articulated_structural_figure_data.json"
DEFAULT_FIGURE = FIGURES / "articulated_structural_sensitivity.svg"
PathwayExecutor = Callable[..., StructuralAtlasExecution]
ReleaseBuilder = Callable[..., dict[str, Any]]
ArrayMap = dict[str, NDArray[Any]]
AXIS_SCALE_KEYS = {
    "height_scale": "height",
    "body_mass_scale": "body_mass",
    "joint_limit_scale": "joint_limit",
}


@dataclass(frozen=True, slots=True)
class StructuralCampaignDependencies:
    """Injectable pathway and release implementations for one campaign."""

    shaft_executor: PathwayExecutor = execute_structural_shaft_atlas
    ground_executor: PathwayExecutor = execute_structural_ground_atlas
    release_builder: ReleaseBuilder | None = None


def _json_bytes(record: Mapping[str, Any]) -> bytes:
    return (json.dumps(record, indent=2) + "\n").encode("utf-8")


def _write_json(record: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(_json_bytes(record))
    temporary.replace(path)


def _write_arrays(arrays: Mapping[str, Any], path: Path) -> None:
    safe = {name: np.asarray(value) for name, value in arrays.items()}
    if any(array.dtype.hasobject for array in safe.values()):
        raise ValueError("structural atlas arrays may not require pickle")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **safe)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_arrays(path: Path) -> ArrayMap:
    with np.load(path, allow_pickle=False) as source:
        return {name: np.asarray(source[name]).copy() for name in source.files}


def _artifact_stem(corner_id: str, pathway: str) -> str:
    return f"articulated_structural_{corner_id.replace('-', '_')}_{pathway}"


def _authority(
    corner: Mapping[str, Any], *, data_directory: Path
) -> ArticulatedAtlasAuthority:
    scaled = load_scaled_authority(
        data_directory / str(corner["record_artifact"]),
        data_directory / str(corner["array_artifact"]),
    )
    authority = ArticulatedAtlasAuthority.from_scaled(scaled)
    if authority.provenance_record() != corner["authority"]:
        raise RuntimeError("structural campaign authority does not reproduce the plan")
    return authority


def _write_execution(
    execution: StructuralAtlasExecution,
    *,
    output_directory: Path,
    corner_id: str,
    pathway: str,
) -> dict[str, Any]:
    stem = _artifact_stem(corner_id, pathway)
    record_path = output_directory / f"{stem}.json"
    arrays_path = output_directory / f"{stem}.npz"
    _write_json(execution.record, record_path)
    _write_arrays(execution.arrays, arrays_path)
    return {
        "corner_id": corner_id,
        "pathway": pathway,
        "record_artifact": record_path.name,
        "array_artifact": arrays_path.name,
        "checkpoint_audit": execution.checkpoint_audit,
    }


def _assemble_release(
    *,
    completed: Sequence[Mapping[str, Any]],
    plan: dict[str, Any],
    plan_path: Path,
    output_directory: Path,
    figure_directory: Path,
) -> dict[str, Any]:
    rows = {(row["corner_id"], row["pathway"]): row for row in completed}
    plan_corners = {row["corner_id"]: row for row in plan["corners"]}
    nominal = {
        pathway: _load_arrays(
            output_directory / rows[("nominal", pathway)]["array_artifact"]
        )
        for pathway in ("shaft", "ground")
    }
    evidence: dict[tuple[str, str], StructuralCornerPathwayEvidence] = {}
    for corner in plan["corners"]:
        corner_id = corner["corner_id"]
        for pathway in ("shaft", "ground"):
            row = rows[(corner_id, pathway)]
            record = json.loads(
                (output_directory / row["record_artifact"]).read_text(encoding="utf-8")
            )
            arrays = _load_arrays(output_directory / row["array_artifact"])
            pack_name = f"{_artifact_stem(corner_id, pathway)}_cells.npz"
            item = assemble_structural_corner_pathway_evidence(
                pathway,  # type: ignore[arg-type]
                nominal[pathway],
                arrays,
                request=StructuralCornerEvidenceRequest(
                    corner_id=corner_id,
                    cell_evidence_artifact=pack_name,
                    requested_state_count=int(corner["requested_state_count"]),
                    feasible_state_count=int(corner["feasible_state_count"]),
                    retained_failures=tuple(corner["retained_failures"]),
                    planned_headline_cell_count=(
                        int(corner["requested_state_count"]) * 32
                    ),
                    all_registered_gates_passed=bool(
                        record["results"]["all_registered_gates_passed"]
                    ),
                    authority=corner["authority"],
                ),
            )
            write_structural_cell_evidence(
                item.cell_evidence,
                output_directory / pack_name,
            )
            evidence[(corner_id, pathway)] = item
    axes = []
    for axis, scale_key in AXIS_SCALE_KEYS.items():
        low_corner = plan_corners[f"{axis}-low"]
        high_corner = plan_corners[f"{axis}-high"]
        nominal_corner = plan_corners["nominal"]
        for pathway in ("shaft", "ground"):
            axes.append(
                assemble_structural_axis_pathway_evidence(
                    axis,
                    evidence[(f"{axis}-low", pathway)],
                    evidence[(f"{axis}-high", pathway)],
                    low_scale=low_corner["authority"]["scales"][scale_key],
                    nominal_scale=nominal_corner["authority"]["scales"][scale_key],
                    high_scale=high_corner["authority"]["scales"][scale_key],
                ).axis_record
            )
    result = assemble_structural_propagation_result(
        plan_contract_sha256=plan["contract_sha256"],
        corner_records=tuple(item.corner_record for item in evidence.values()),
        axis_records=tuple(axes),
    )
    result_path = output_directory / DEFAULT_RESULT.name
    write_structural_propagation_result(result, result_path)
    validate_structural_propagation_bundle_against_plan(result_path, plan_path)
    figure_directory.mkdir(parents=True, exist_ok=True)
    publish_structural_figure_bundle(
        result_path=result_path,
        plan_path=plan_path,
        figure_data_output=output_directory / DEFAULT_FIGURE_DATA.name,
        figure_output=figure_directory / DEFAULT_FIGURE.name,
    )
    return result


def run_structural_propagation_campaign(
    *,
    checkpoint_directory: Path,
    output_directory: Path,
    figure_directory: Path,
    worker_count: int,
    plan_path: Path = DEFAULT_PLAN,
    data_directory: Path = DATA,
    dependencies: StructuralCampaignDependencies | None = None,
) -> dict[str, Any]:
    """Run all registered corners sequentially and promote only complete evidence."""

    if type(worker_count) is not int or worker_count <= 0:
        raise ValueError("worker_count must be a positive integer")
    dependencies = dependencies or StructuralCampaignDependencies()
    plan = validate_structural_propagation_plan(plan_path)
    shaft_config = replace(ArticulatedShaftAtlasConfig(), worker_count=worker_count)
    ground_config = replace(ArticulatedGroundAtlasConfig(), worker_count=worker_count)
    status_path = output_directory / "articulated_structural_campaign_status.json"
    status: dict[str, Any] = {
        "schema_version": "articulated-structural-campaign-status/v1",
        "state": "running",
        "plan_contract_sha256": plan["contract_sha256"],
        "worker_count": worker_count,
        "worker_count_role": "operational_only",
        "completed": [],
        "retained_execution_failures": [],
        "release_evidence": False,
    }
    _write_json(status, status_path)
    executors = {
        "shaft": dependencies.shaft_executor,
        "ground": dependencies.ground_executor,
    }
    configurations = {"shaft": shaft_config, "ground": ground_config}
    for corner in plan["corners"]:
        corner_id = str(corner["corner_id"])
        authority = _authority(corner, data_directory=data_directory)
        for pathway in ("shaft", "ground"):
            try:
                execution = executors[pathway](
                    authority,
                    corner_id=corner_id,
                    checkpoint_directory=(checkpoint_directory / corner_id / pathway),
                    config=configurations[pathway],
                    plan_path=plan_path,
                )
                status["completed"].append(
                    _write_execution(
                        execution,
                        output_directory=output_directory,
                        corner_id=corner_id,
                        pathway=pathway,
                    )
                )
                _write_json(status, status_path)
            except Exception as error:
                status["state"] = "failed_retained"
                status["retained_execution_failures"].append(
                    {
                        "corner_id": corner_id,
                        "pathway": pathway,
                        "exception_type": type(error).__name__,
                        "message": str(error),
                    }
                )
                _write_json(status, status_path)
                raise RuntimeError(
                    f"structural campaign failed: corner={corner_id}, pathway={pathway}"
                ) from error
    release_builder = dependencies.release_builder or _assemble_release
    result = release_builder(
        completed=tuple(status["completed"]),
        plan=plan,
        plan_path=plan_path,
        output_directory=output_directory,
        figure_directory=figure_directory,
    )
    status.update(
        {
            "state": "complete",
            "result_sha256": result["result_sha256"],
            "release_evidence": True,
        }
    )
    _write_json(status, status_path)
    return status


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoints", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DATA)
    parser.add_argument("--figures", type=Path, default=FIGURES)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--plan", type=Path, default=DEFAULT_PLAN)
    parser.add_argument("--authority-data", type=Path, default=DATA)
    args = parser.parse_args(argv)
    run_structural_propagation_campaign(
        checkpoint_directory=args.checkpoints,
        output_directory=args.output,
        figure_directory=args.figures,
        worker_count=args.workers,
        plan_path=args.plan,
        data_directory=args.authority_data,
    )


if __name__ == "__main__":
    main()


__all__ = [
    "StructuralCampaignDependencies",
    "main",
    "run_structural_propagation_campaign",
]
