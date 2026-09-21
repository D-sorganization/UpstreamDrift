"""Research capability lifecycle management and contract enforcement (ORG-22, #10530).

Governs intentionally excluded packages, manifest-only services, and incomplete
research capabilities per ADR-0047 and Epic #10508:
- Untiled packages under src/tools/ must be explicitly justified and registered.
- Legitimate headless tools (contraction, drift_control, offline_validation,
  model_converter, hmr2_sidecar, etc.) remain discoverable as CLI/library tools,
  never disguised with fake GUIs.
- SG optimizer Phase 1-3 lifecycle is accurately classified (active CLI with
  concrete Phase 3 PyQt6 UI follow-up per #6272, not abandoned).
- Incomplete/planned capabilities record owner, tracking issue, useful access,
  supported inputs/outputs, missing acceptance, and next actionable step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import re
from typing import Any, Final

import yaml

from src.config.capability_migration import CapabilityMigrationInventory, MigrationEntry
from src.config.launcher_manifest_loader import LauncherManifest
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class CLINotInteractiveGUIError(ValueError):
    """Raised when attempting to configure or adapt a CLI-only capability as a GUI tile."""


@dataclass(frozen=True)
class IncompleteCapabilityRecord:
    """Disclose owner, issue, current access, and next action for an incomplete capability."""

    capability_id: str
    name: str
    entity_kind: str
    primary_workspace: str
    owner: str
    issue: str
    useful_access: str
    supported_inputs: tuple[str, ...] = field(default_factory=tuple)
    supported_outputs: tuple[str, ...] = field(default_factory=tuple)
    missing_acceptance: str = ""
    next_action: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Serialize record to dictionary."""
        return {
            "capability_id": self.capability_id,
            "name": self.name,
            "entity_kind": self.entity_kind,
            "primary_workspace": self.primary_workspace,
            "owner": self.owner,
            "issue": self.issue,
            "useful_access": self.useful_access,
            "supported_inputs": list(self.supported_inputs),
            "supported_outputs": list(self.supported_outputs),
            "missing_acceptance": self.missing_acceptance,
            "next_action": self.next_action,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> IncompleteCapabilityRecord:
        """Deserialize record from dictionary."""
        return cls(
            capability_id=str(data["capability_id"]),
            name=str(data.get("name", "")),
            entity_kind=str(data.get("entity_kind", "cli_tool")),
            primary_workspace=str(data.get("primary_workspace", "simulation")),
            owner=str(data.get("owner", "@core")),
            issue=str(data.get("issue", "")),
            useful_access=str(data.get("useful_access", "")),
            supported_inputs=tuple(data.get("supported_inputs", [])),
            supported_outputs=tuple(data.get("supported_outputs", [])),
            missing_acceptance=str(data.get("missing_acceptance", "")),
            next_action=str(data.get("next_action", "")),
        )


# Authoritative baseline of research/incomplete capabilities
KNOWN_RESEARCH_RECORDS: Final[dict[str, dict[str, Any]]] = {
    "sg_optimizer": {
        "capability_id": "sg_optimizer",
        "name": "Strokes Gained Optimizer",
        "entity_kind": "cli_tool",
        "primary_workspace": "training",
        "owner": "@core",
        "issue": "#6272",
        "useful_access": "python -m src.tools.sg_optimizer --profile P --baseline B --hole-spec H [--conditions tournament]",
        "supported_inputs": [
            "player_profile_yaml",
            "baseline_bag_yaml",
            "hole_spec_python",
            "conditions_preset",
        ],
        "supported_outputs": [
            "optimal_tee_action_json",
            "expected_strokes_metric",
        ],
        "missing_acceptance": "PyQt6 profile editor (profile_editor.py), conditions panel (conditions_panel.py), live expected strokes preview.",
        "next_action": "Implement PyQt6 profile editor and conditions panel per #6272 tied to existing sg_optimizer CLI/MDP engine.",
    },
    "contraction": {
        "capability_id": "contraction",
        "name": "Contraction & Floquet Analysis",
        "entity_kind": "cli_tool",
        "primary_workspace": "simulation",
        "owner": "@physics-team",
        "issue": "#8863",
        "useful_access": "python -m src.tools.contraction",
        "supported_inputs": ["trajectory_npz", "state_channels"],
        "supported_outputs": ["contraction_metrics_json", "floquet_multipliers"],
        "missing_acceptance": "Interactive viewer integration deferred; terminal workflow complete.",
        "next_action": "Retain as headless research tool; link docs in simulation workspace contextual help.",
    },
    "drift_control": {
        "capability_id": "drift_control",
        "name": "Drift-to-Control Ratio Calculator",
        "entity_kind": "cli_tool",
        "primary_workspace": "simulation",
        "owner": "@physics-team",
        "issue": "#8863",
        "useful_access": "python -m src.tools.drift_control",
        "supported_inputs": ["trajectory_npz"],
        "supported_outputs": ["drift_control_ratio_metrics"],
        "missing_acceptance": "No GUI planned; terminal-only analysis workflow.",
        "next_action": "Retain as headless research tool; maintain CLI entry point with NPZ inputs.",
    },
    "model_converter": {
        "capability_id": "model_converter",
        "name": "URDF and MJCF Model Converter",
        "entity_kind": "cli_tool",
        "primary_workspace": "simulation",
        "owner": "@physics-team",
        "issue": "#9965",
        "useful_access": "python -m src.tools.model_converter.build_models",
        "supported_inputs": ["canonical_model_spec", "inertia_parameters"],
        "supported_outputs": ["urdf_model", "mjcf_model"],
        "missing_acceptance": "Build-time orchestrator; no runtime UI required.",
        "next_action": "Retain as build-time code generator; ensure schema consistency with ADR-0007.",
    },
    "offline_validation": {
        "capability_id": "offline_validation",
        "name": "Offline Gradient-Oracle Validator",
        "entity_kind": "library",
        "primary_workspace": "simulation",
        "owner": "@physics-team",
        "issue": "#8863",
        "useful_access": "python -m src.tools.offline_validation",
        "supported_inputs": ["nimblephysics_kinematics", "analytical_gradient_oracle"],
        "supported_outputs": ["validation_report_json"],
        "missing_acceptance": "Requires nimblephysics isolated environment; not bundled in base runtime.",
        "next_action": "Maintain offline CI gate and dedicated requirements-nimble test harness.",
    },
    "hmr2_sidecar": {
        "capability_id": "hmr2_sidecar",
        "name": "4D-Humans/HMR 2.0 Subprocess Sidecar",
        "entity_kind": "library",
        "primary_workspace": "capture",
        "owner": "@core",
        "issue": "#8863",
        "useful_access": "Invoked via HMR2_COMMAND environment variable by motion capture pipeline",
        "supported_inputs": ["video_frames", "bounding_boxes"],
        "supported_outputs": ["smpl_mesh_parameters_npz"],
        "missing_acceptance": "Direct user launch disallowed to preserve CC-BY-NC license isolation.",
        "next_action": "Retain subprocess license boundary and verification assertions in capture rig.",
    },
}


class ResearchCapabilityLifecycleManager:
    """Coordinates lifecycle verification, tile eligibility, and incomplete records."""

    def __init__(
        self,
        repo_root: Path,
        exclusions: list[dict[str, Any]],
        inventory: CapabilityMigrationInventory,
        manifest: LauncherManifest,
        records: dict[str, IncompleteCapabilityRecord],
    ) -> None:
        self.repo_root = repo_root
        self.exclusions = exclusions
        self.inventory = inventory
        self.manifest = manifest
        self.records = records

    @classmethod
    def load(cls, repo_root: Path | None = None) -> ResearchCapabilityLifecycleManager:
        """Load canonical manager from repository root."""
        root = repo_root or Path(__file__).resolve().parents[2]
        exclusions_path = root / "src" / "config" / "registry_exclusions.yaml"
        migration_path = root / "src" / "config" / "capability_migration.json"

        exclusions_data = yaml.safe_load(exclusions_path.read_text(encoding="utf-8"))
        exclusions = list(exclusions_data.get("exclusions", []))
        inventory = CapabilityMigrationInventory.load(migration_path)
        manifest = LauncherManifest.load()

        records = {
            cid: IncompleteCapabilityRecord.from_dict(data)
            for cid, data in KNOWN_RESEARCH_RECORDS.items()
        }

        return cls(
            repo_root=root,
            exclusions=exclusions,
            inventory=inventory,
            manifest=manifest,
            records=records,
        )

    def find_unaccounted_tools(self, tools_dir: Path) -> list[str]:
        """Find src/tools directories that are neither tiles nor excluded."""
        tool_packages = {
            d.name
            for d in tools_dir.iterdir()
            if d.is_dir() and (d / "__init__.py").exists()
        }

        tile_packages: set[str] = set()
        for tile in self.manifest.tiles:
            path = (tile.path or "").replace("\\", "/")
            match = re.match(r"src/tools/([A-Za-z0-9_]+)", path)
            if match:
                tile_packages.add(match.group(1))

        excluded = {entry["package"] for entry in self.exclusions}
        unaccounted = sorted(tool_packages - tile_packages - excluded)
        return unaccounted

    def validate_exclusions_in_migration(self) -> list[str]:
        """Verify that every excluded tool package is cataloged in capability_migration.json."""
        missing: list[str] = []
        for entry in self.exclusions:
            pkg = entry["package"]
            if (
                pkg not in self.inventory.entries
                and f"cli_{pkg}" not in self.inventory.entries
                and f"tools_{pkg}" not in self.inventory.entries
            ):
                missing.append(pkg)
        return missing

    def verify_tile_eligibility(
        self, capability_id: str, requested_surface: str = "desktop"
    ) -> bool:
        """Enforce that CLI-only / headless capabilities cannot be claimed as GUI tiles."""
        entries = self.inventory.entries
        entry = entries.get(capability_id)
        if entry is None:
            # Check exclusions directly
            for excl in self.exclusions:
                if excl.get("package") == capability_id:
                    if requested_surface in ("desktop", "gui"):
                        raise CLINotInteractiveGUIError(
                            f"Capability '{capability_id}' is an intentionally headless or "
                            f"library tool and cannot be represented as a GUI-ready tile "
                            f"on surface '{requested_surface}'. Reason: {excl.get('reason')}"
                        )
                    return True
            raise KeyError(f"Unknown capability: {capability_id!r}")

        # Check entity_kind and lifecycle
        if requested_surface in ("desktop", "gui"):
            if entry.entity_kind in ("cli_tool", "library", "service"):
                # Exception: unless it has an explicit GUI entrypoint in manifest
                has_manifest_tile = any(
                    t.id == capability_id for t in self.manifest.tiles
                )
                if not has_manifest_tile:
                    raise CLINotInteractiveGUIError(
                        f"Capability '{capability_id}' is a {entry.entity_kind} and "
                        f"cannot be represented as a GUI-ready tile on surface '{requested_surface}'."
                    )
            if entry.lifecycle == "intentionally_headless":
                raise CLINotInteractiveGUIError(
                    f"Capability '{capability_id}' has lifecycle 'intentionally_headless' and "
                    f"cannot be represented as a GUI-ready tile on surface '{requested_surface}'."
                )

        return True

    def get_record(self, capability_id: str) -> IncompleteCapabilityRecord | None:
        """Get research / incomplete capability record."""
        return self.records.get(capability_id)

    def get_incomplete_records(self) -> list[IncompleteCapabilityRecord]:
        """Get all registered incomplete capability records."""
        return list(self.records.values())


def audit_research_and_excluded_capabilities(
    repo_root: Path | None = None,
) -> dict[str, Any]:
    """Audit all research, headless, and excluded capabilities across the repo."""
    manager = ResearchCapabilityLifecycleManager.load(repo_root)
    unaccounted = manager.find_unaccounted_tools(manager.repo_root / "src" / "tools")
    missing_migration = manager.validate_exclusions_in_migration()
    incomplete = [r.to_dict() for r in manager.get_incomplete_records()]

    return {
        "unaccounted_tools": unaccounted,
        "missing_from_migration": missing_migration,
        "total_exclusions": len(manager.exclusions),
        "incomplete_records": incomplete,
        "status": "clean"
        if not unaccounted and not missing_migration
        else "drift_detected",
    }
