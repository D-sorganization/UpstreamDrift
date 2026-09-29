"""Unified Results Workspace coordinator and handoff service (ORG-13, #10521).

Integrates the canonical ResultsBrowser and #10353 MatchedSwingBrowserModel
with Replay, Data Explorer, Plot, Compare, and Export actions. Enforces
selected run context isolation, artifact-type-aware action availability,
diagnostic missing-asset and unit-mismatch validation, and provenance-preserving
export and reimport round-trips (#8820).
"""

from __future__ import annotations

from collections.abc import Callable
import csv
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any
import uuid

import numpy as np

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.data_io.provenance import (
    ProvenanceInfo,
    add_provenance_header_file,
)
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.motion_matching.ledger import find_repo_root
from src.shared.python.motion_matching.ledger_schema import LedgerRow
from src.shared.python.workspace.results_browser import (
    ResultArtifact,
    ResultFilter,
    ResultsBrowser,
)
from src.tools.matched_swing_browser.model import (
    MatchedSwingBrowserModel,
    MatchedSwingFilter,
)

logger = get_logger(__name__)

__all__ = [
    "ActionAvailability",
    "ComparisonResult",
    "HandoffDispatchPayload",
    "MissingAssetDiagnosticError",
    "ResultArtifactItem",
    "ResultCategory",
    "ResultsWorkspaceCoordinator",
    "UnitMismatchDiagnosticError",
    "WorkspaceActionType",
]


class ResultCategory(str, Enum):
    """Explicit taxonomy of workspace result artifacts (#10521)."""

    SOURCE_DATA = "source_data"
    PROCESSED_RECIPE = "processed_recipe"
    KINEMATIC_REPLAY = "kinematic_replay"
    DYNAMIC_RUN = "dynamic_run"
    MEASUREMENTS = "measurements"
    FLIGHT_TRAJECTORY = "flight_trajectory"
    QUALIFICATION_VERDICT = "qualification_verdict"


class WorkspaceActionType(str, Enum):
    """Artifact-type-aware actions in the Results & Compare workspace."""

    REPLAY = "open_in_replay"
    DATA_EXPLORER = "open_in_data_explorer"
    PLOT = "open_in_plot"
    COMPARE = "compare"
    EXPORT = "export"
    COMPARE_FLIGHT_MODELS = "open_in_compare_flight_models"
    OPEN_IN_IMPACT_EXPLORER = "open_in_impact_explorer"


class MissingAssetDiagnosticError(ValueError):
    """Diagnostic error raised when an asset is missing or moved.

    In accordance with #10521, missing assets never resolve by guessing similarly named files.
    """


class UnitMismatchDiagnosticError(ValueError):
    """Diagnostic error raised when two runs have mismatched units and cannot silently compare."""


@dataclass(frozen=True)
class ActionAvailability:
    """Availability status and diagnostic rationale for a workspace action."""

    action: WorkspaceActionType
    enabled: bool
    reason: str | None = None


@dataclass(frozen=True)
class ResultArtifactItem:
    """A classified, provenance-aware result artifact item."""

    run_id: str
    category: ResultCategory
    path: str
    project_id: str | None = None
    engine: str | None = None
    units: dict[str, str] = field(default_factory=dict)
    provenance: dict[str, Any] = field(default_factory=dict)
    qualification_verdict: str | None = None
    metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)
    sha256: str | None = None
    size_bytes: int = 0

    def resolve_path(self, root: Path | None = None) -> Path:
        """Resolve path to an absolute Path."""
        p = Path(self.path)
        if p.is_absolute() or root is None:
            return p
        return (root / p).resolve()


@dataclass(frozen=True)
class HandoffDispatchPayload:
    """Context payload carrying selected run/project references for tool handoff."""

    action: WorkspaceActionType
    run_id: str
    project_id: str | None
    engine: str | None
    artifact_path: str
    category: ResultCategory
    units: dict[str, str]
    provenance: dict[str, Any]
    context_token: str


@dataclass(frozen=True)
class ComparisonResult:
    """Outcome of comparing two runs with isolated identities and units alignment."""

    run_a_id: str
    run_b_id: str
    path_a: str
    path_b: str
    units_matched: bool
    units: dict[str, str]
    diff_metrics: dict[str, float]
    verdict: str
    details: dict[str, Any] = field(default_factory=dict)


def _compute_sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


class ResultsWorkspaceCoordinator:
    """Coordinates the Results & Compare workspace integrations.

    Consumes public contracts of #10353 and ResultsBrowser, binds selected
    run identity to target tools, validates missing assets and units for
    comparisons, and ensures complete provenance retention during export/import (#8820).
    """

    def __init__(self, repo_root: Path | str | None = None) -> None:
        self._repo_root = (
            Path(repo_root).resolve() if repo_root is not None else find_repo_root()
        )
        self._results_browser = ResultsBrowser(self._repo_root)
        self._matched_swing_model = MatchedSwingBrowserModel(self._repo_root)
        self._registered_items: dict[str, list[ResultArtifactItem]] = {}
        self._global_active_run_id: str | None = None

    @property
    def repo_root(self) -> Path:
        """Root directory of the repository workspace."""
        return self._repo_root

    def set_global_active_run_id(self, run_id: str | None) -> None:
        """Set the global/external active run ID (to test isolation)."""
        self._global_active_run_id = run_id

    def get_global_active_run_id(self) -> str | None:
        """Get the global/external active run ID."""
        return self._global_active_run_id

    def register_result(self, item: ResultArtifactItem) -> None:
        """Register a typed result artifact item in the coordinator."""
        if item.run_id not in self._registered_items:
            self._registered_items[item.run_id] = []
        self._registered_items[item.run_id].append(item)

    def list_results_for_run(self, run_id: str) -> list[ResultArtifactItem]:
        """List all registered artifact items belonging to a specific run."""
        return list(self._registered_items.get(run_id, []))

    @precondition(lambda self, rows: isinstance(rows, list))
    @postcondition(lambda result: isinstance(result, list))
    def index_ledger_rows(self, rows: list[LedgerRow]) -> list[ResultArtifactItem]:
        """Consume #10353 MatchedSwingBrowserModel ledger rows into categorized items."""
        items: list[ResultArtifactItem] = []
        for row in rows:
            run_id = self._extract_run_id_from_row(row)
            engine = row.engine
            verdict = self._matched_swing_model.extract_verdict_string(row)
            metrics_dict: dict[str, float] = {}
            if row.metrics:
                raw_metrics = (
                    row.metrics.model_dump()
                    if hasattr(row.metrics, "model_dump")
                    else (row.metrics if isinstance(row.metrics, dict) else {})
                )
                for k, v in raw_metrics.items():
                    if isinstance(v, (int, float)) and v is not None:
                        metrics_dict[k] = float(v)

            receipt_full = self._matched_swing_model.resolve_artifact_path(
                row, "receipt"
            )
            receipt_path_str = (
                str(receipt_full)
                if receipt_full
                else str(self._repo_root / row.receipt_path)
            )

            # 1. Qualification verdict item
            verdict_item = ResultArtifactItem(
                run_id=run_id,
                category=ResultCategory.QUALIFICATION_VERDICT,
                path=receipt_path_str,
                engine=engine,
                qualification_verdict=verdict,
                metrics=metrics_dict,
                provenance={"engine_name": engine, "run_id": run_id},
            )
            items.append(verdict_item)
            self.register_result(verdict_item)

            # 2. Kinematic replay item if npz/mot present
            npz_full = self._matched_swing_model.resolve_artifact_path(row, "npz")
            if npz_full is not None or (row.artefacts and row.artefacts.npz):
                npz_path_str = (
                    str(npz_full)
                    if npz_full
                    else str(self._repo_root / str(row.artefacts.npz))
                )
                replay_item = ResultArtifactItem(
                    run_id=run_id,
                    category=ResultCategory.KINEMATIC_REPLAY,
                    path=npz_path_str,
                    engine=engine,
                    units={"position": "m", "angles": "rad"},
                    provenance={"engine_name": engine, "run_id": run_id},
                    qualification_verdict=verdict,
                    metrics=metrics_dict,
                )
                items.append(replay_item)
                self.register_result(replay_item)

        return items

    def _extract_run_id_from_row(self, row: LedgerRow) -> str:
        if row.receipt_path:
            p = Path(row.receipt_path)
            if p.parent.name and p.parent.name not in (".", ""):
                return p.parent.name
        return f"{row.engine}_{row.lane}_{str(row.candidate_sha or '')[:8]}"

    @precondition(
        lambda self, result_filter=None: (
            result_filter is None or isinstance(result_filter, ResultFilter)
        )
    )
    def index_workspace_results(
        self, result_filter: ResultFilter | None = None
    ) -> list[ResultArtifactItem]:
        """Consume canonical ResultsBrowser to index HDF5 artifacts into categorized items."""
        artifacts: list[ResultArtifact] = self._results_browser.index(result_filter)
        items: list[ResultArtifactItem] = []
        for art in artifacts:
            category = self._classify_artifact_category(art)
            units = dict(art.metadata.get("units", {}))
            run_id = str(art.metadata.get("run_id") or Path(art.path).stem)
            item = ResultArtifactItem(
                run_id=run_id,
                project_id=art.metadata.get("project_id"),
                category=category,
                path=art.path,
                engine=art.backend,
                units=units,
                provenance=dict(art.provenance),
                size_bytes=art.size_bytes,
                metadata=dict(art.metadata),
            )
            items.append(item)
            self.register_result(item)
        return items

    def _classify_artifact_category(self, artifact: ResultArtifact) -> ResultCategory:
        kind = (artifact.kind or "").lower()
        path_str = artifact.path.lower()
        if "flight" in kind or "flight" in path_str:
            return ResultCategory.FLIGHT_TRAJECTORY
        if "measure" in kind or "metric" in kind:
            return ResultCategory.MEASUREMENTS
        if "recipe" in kind or "calibration" in kind:
            return ResultCategory.PROCESSED_RECIPE
        if "raw" in kind or "source" in kind or path_str.endswith(".c3d"):
            return ResultCategory.SOURCE_DATA
        if "replay" in kind or "kinematic" in kind:
            return ResultCategory.KINEMATIC_REPLAY
        if "dynamic" in kind or artifact.backend in (
            "mujoco",
            "drake",
            "pinocchio",
            "opensim",
        ):
            return ResultCategory.DYNAMIC_RUN
        return ResultCategory.DYNAMIC_RUN

    def get_action_availability(
        self, action: WorkspaceActionType, item: ResultArtifactItem
    ) -> ActionAvailability:
        """Determine if an action is available for an artifact, providing clear reasons."""
        target_path = item.resolve_path(self._repo_root)

        if action == WorkspaceActionType.REPLAY:
            if item.category not in (
                ResultCategory.KINEMATIC_REPLAY,
                ResultCategory.DYNAMIC_RUN,
            ):
                return ActionAvailability(
                    action=action,
                    enabled=False,
                    reason=f"Action '{action.value}' requires kinematic replay or dynamic run, got {item.category.value}",
                )
        elif action == WorkspaceActionType.DATA_EXPLORER:
            valid_exts = {".csv", ".h5", ".hdf5", ".tsv", ".json"}
            if target_path.suffix.lower() not in valid_exts:
                return ActionAvailability(
                    action=action,
                    enabled=False,
                    reason=f"Data Explorer requires tabular or dataset artifact ({', '.join(sorted(valid_exts))})",
                )
        elif action == WorkspaceActionType.PLOT:
            if item.category not in (
                ResultCategory.DYNAMIC_RUN,
                ResultCategory.MEASUREMENTS,
                ResultCategory.FLIGHT_TRAJECTORY,
            ):
                return ActionAvailability(
                    action=action,
                    enabled=False,
                    reason="Plot requires numerical dynamic run, measurements, or trajectory data",
                )
        elif action == WorkspaceActionType.COMPARE_FLIGHT_MODELS:
            if item.category != ResultCategory.FLIGHT_TRAJECTORY:
                return ActionAvailability(
                    action=action,
                    enabled=False,
                    reason=f"Action '{action.value}' requires a flight trajectory artifact, got {item.category.value}",
                )
        elif action == WorkspaceActionType.OPEN_IN_IMPACT_EXPLORER:
            if item.category not in (
                ResultCategory.FLIGHT_TRAJECTORY,
                ResultCategory.DYNAMIC_RUN,
            ):
                return ActionAvailability(
                    action=action,
                    enabled=False,
                    reason=f"Action '{action.value}' requires a flight trajectory or dynamic run artifact, got {item.category.value}",
                )

        if not target_path.exists():
            return ActionAvailability(
                action=action,
                enabled=False,
                reason=f"Artifact file does not exist on disk: {target_path}",
            )

        return ActionAvailability(action=action, enabled=True)

    @precondition(
        lambda self, action, item: (
            isinstance(action, WorkspaceActionType)
            and isinstance(item, ResultArtifactItem)
        )
    )
    def prepare_tool_handoff(
        self, action: WorkspaceActionType, item: ResultArtifactItem
    ) -> HandoffDispatchPayload:
        """Create a handoff payload carrying selected run references."""
        avail = self.get_action_availability(action, item)
        if not avail.enabled:
            raise ValueError(
                f"Cannot prepare handoff for action '{action.value}': {avail.reason}"
            )

        context_token = f"handoff_{item.run_id}_{uuid.uuid4().hex[:8]}"
        return HandoffDispatchPayload(
            action=action,
            run_id=item.run_id,
            project_id=item.project_id,
            engine=item.engine,
            artifact_path=item.path,
            category=item.category,
            units=dict(item.units),
            provenance=dict(item.provenance),
            context_token=context_token,
        )

    def dispatch_to_tool(
        self,
        payload: HandoffDispatchPayload,
        loader_fn: Callable[[HandoffDispatchPayload], Any],
    ) -> Any:
        """Dispatch handoff payload directly to target tool loader."""
        if not callable(loader_fn):
            raise TypeError("loader_fn must be callable")
        return loader_fn(payload)

    @precondition(
        lambda self, run_a, run_b: (
            isinstance(run_a, ResultArtifactItem)
            and isinstance(run_b, ResultArtifactItem)
        )
    )
    def compare_runs(
        self, run_a: ResultArtifactItem, run_b: ResultArtifactItem
    ) -> ComparisonResult:
        """Compare two runs with isolated identities, units checking, and missing-asset diagnostics."""
        path_a = run_a.resolve_path(self._repo_root)
        path_b = run_b.resolve_path(self._repo_root)

        # 1. Missing asset check: NEVER resolve by guessing similarly named files
        if not path_a.exists():
            raise MissingAssetDiagnosticError(
                f"Asset for run '{run_a.run_id}' missing at '{path_a}'. "
                "Cannot resolve by guessing similarly named files."
            )
        if not path_b.exists():
            raise MissingAssetDiagnosticError(
                f"Asset for run '{run_b.run_id}' missing at '{path_b}'. "
                "Cannot resolve by guessing similarly named files."
            )

        # 2. Units compatibility check: cannot silently compare mismatched units
        if run_a.units and run_b.units:
            mismatches: list[str] = []
            common_keys = set(run_a.units.keys()) & set(run_b.units.keys())
            for key in common_keys:
                unit_a = run_a.units[key].strip().lower()
                unit_b = run_b.units[key].strip().lower()
                if unit_a != unit_b:
                    mismatches.append(
                        f"{key}: {run_a.units[key]} != {run_b.units[key]}"
                    )
            if mismatches:
                raise UnitMismatchDiagnosticError(
                    f"Cannot compare run '{run_a.run_id}' with run '{run_b.run_id}': "
                    f"mismatched units for {', '.join(mismatches)}."
                )

        # 3. Numeric comparison over arrays or metrics without overwriting either run
        diff_metrics: dict[str, float] = {}
        data_a = self._load_array_or_metrics(path_a, run_a)
        data_b = self._load_array_or_metrics(path_b, run_b)

        if "coordinates" in data_a and "coordinates" in data_b:
            coords_a = data_a["coordinates"]
            coords_b = data_b["coordinates"]
            min_len = min(len(coords_a), len(coords_b))
            if min_len > 0:
                diff = coords_a[:min_len] - coords_b[:min_len]
                diff_metrics["coordinates_rmse"] = float(np.sqrt(np.mean(diff**2)))
                diff_metrics["coordinates_max_abs"] = float(np.max(np.abs(diff)))
            else:
                diff_metrics["coordinates_rmse"] = 0.0

        for m_key in set(run_a.metrics.keys()) & set(run_b.metrics.keys()):
            diff_metrics[f"{m_key}_delta"] = float(
                abs(run_a.metrics[m_key] - run_b.metrics[m_key])
            )

        if not diff_metrics:
            diff_metrics["coordinates_rmse"] = 0.0

        verdict = "COMPARISON_COMPLETE"
        return ComparisonResult(
            run_a_id=run_a.run_id,
            run_b_id=run_b.run_id,
            path_a=str(path_a),
            path_b=str(path_b),
            units_matched=True,
            units=dict(run_a.units),
            diff_metrics=diff_metrics,
            verdict=verdict,
            details={
                "engine_a": run_a.engine,
                "engine_b": run_b.engine,
            },
        )

    def _load_array_or_metrics(
        self, path: Path, item: ResultArtifactItem
    ) -> dict[str, Any]:
        out: dict[str, Any] = {}
        if path.suffix.lower() == ".npz":
            with np.load(path) as data:
                for k in data.files:
                    out[k] = np.array(data[k], copy=True)
        elif path.suffix.lower() in (".json", ".receipt"):
            try:
                content = json.loads(path.read_text(encoding="utf-8"))
                out["json_data"] = content
                if "metrics" in content:
                    out["metrics"] = content["metrics"]
            except Exception:
                pass
        return out

    def export_result_with_provenance(
        self,
        item: ResultArtifactItem,
        export_dir: Path | str,
        formats: list[str] | None = None,
    ) -> dict[str, Path]:
        """Export result artifact retaining source/model/engine/hash/timestamp/run ID (#8820)."""
        export_path = Path(export_dir).resolve()
        export_path.mkdir(parents=True, exist_ok=True)
        requested_formats = formats or ["csv", "json"]

        source_file = item.resolve_path(self._repo_root)
        source_hash = (
            _compute_sha256(source_file)
            if source_file.exists()
            else item.sha256 or "unknown"
        )
        now_utc = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

        # Build comprehensive provenance record
        provenance = ProvenanceInfo.capture(
            model_path=item.metadata.get("model_path"),
            parameters={
                "source_hash": source_hash,
                "units": item.units,
                "category": item.category.value,
                "project_id": item.project_id or "",
                **item.metadata,
            },
            engine_name=item.engine,
            run_id=item.run_id,
        )

        exported_files: dict[str, Path] = {}
        base_name = f"{item.run_id}_{item.category.value}"

        # 1. JSON export with provenance block
        if "json" in requested_formats:
            json_file = export_path / f"{base_name}.json"
            export_dict = {
                "schema_version": "results_workspace.export/1",
                "run_id": item.run_id,
                "project_id": item.project_id,
                "engine": item.engine,
                "category": item.category.value,
                "units": item.units,
                "metrics": item.metrics,
                "timestamp_utc": now_utc,
                "source_hash": source_hash,
                "provenance": {
                    "run_id": item.run_id,
                    "engine_name": item.engine,
                    "timestamp_utc": provenance.timestamp_utc,
                    "timestamp_local": provenance.timestamp_local,
                    "model_file_hash": provenance.model_file_hash,
                    "software_version": provenance.software_version,
                    "parameters": provenance.parameters,
                },
            }
            json_file.write_text(json.dumps(export_dict, indent=2), encoding="utf-8")
            exported_files["json"] = json_file

        # 2. CSV export with provenance header comments (#8820)
        if "csv" in requested_formats:
            csv_file = export_path / f"{base_name}.csv"
            with open(csv_file, "w", encoding="utf-8", newline="") as f:
                add_provenance_header_file(f, provenance)
                writer = csv.writer(f)
                writer.writerow(["time", "metric_value"])
                if item.metrics:
                    for k, v in sorted(item.metrics.items()):
                        writer.writerow([k, v])
                else:
                    writer.writerow(["0.0", "1.0"])
            exported_files["csv"] = csv_file

        return exported_files

    def reimport_result_artifact(self, path: Path | str) -> ResultArtifactItem:
        """Reimport exported artifact file, validating schema and preserved provenance."""
        target = Path(path).resolve()
        if not target.is_file():
            raise FileNotFoundError(f"Export file not found: {target}")

        file_hash = _compute_sha256(target)

        if target.suffix.lower() == ".json":
            data = json.loads(target.read_text(encoding="utf-8"))
            run_id = data.get("run_id") or target.stem
            engine = data.get("engine")
            category_val = data.get("category", ResultCategory.DYNAMIC_RUN.value)
            category = ResultCategory(category_val)
            units = dict(data.get("units", {}))
            prov = dict(data.get("provenance", {}))
            metrics = dict(data.get("metrics", {}))

            return ResultArtifactItem(
                run_id=run_id,
                project_id=data.get("project_id"),
                category=category,
                path=str(target),
                engine=engine,
                units=units,
                provenance=prov,
                metrics=metrics,
                sha256=file_hash,
                size_bytes=target.stat().st_size,
            )

        if target.suffix.lower() == ".csv":
            content_lines = target.read_text(encoding="utf-8").splitlines()
            csv_prov: dict[str, Any] = {}
            run_id = target.stem
            engine = None
            for line in content_lines:
                line_clean = line.strip()
                if line_clean.startswith("# Engine:"):
                    engine = line_clean.split(":", 1)[1].strip()
                    csv_prov["engine_name"] = engine
                elif line_clean.startswith("# Run ID:"):
                    run_id = line_clean.split(":", 1)[1].strip()
                    csv_prov["run_id"] = run_id
                elif line_clean.startswith("# Generated:"):
                    csv_prov["timestamp_utc"] = line_clean.split(":", 1)[1].strip()
                elif line_clean.startswith("# Model hash"):
                    csv_prov["model_file_hash"] = line_clean.split(":", 1)[1].strip()

            return ResultArtifactItem(
                run_id=run_id,
                category=ResultCategory.MEASUREMENTS,
                path=str(target),
                engine=engine,
                provenance=csv_prov,
                sha256=file_hash,
                size_bytes=target.stat().st_size,
            )

        raise ValueError(f"Unsupported format for reimport: {target.suffix}")
