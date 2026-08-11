"""Deterministic release manifest and qualification for the open resource."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any


ARTICLE_REL = Path("docs/research/proximal_distal_energy_transfer")
_EXCLUDED = frozenset({"release_manifest.json", "CHECKSUMS.sha256"})


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _artifact_paths(root: Path) -> tuple[Path, ...]:
    article = root / ARTICLE_REL
    selected: set[Path] = set()
    for pattern in (
        "*.md",
        "*.qmd",
        "CITATION.cff",
        "references.bib",
        "proximal_distal_energy_transfer.pdf",
        "chapters/*.qmd",
        "data/**/*.json",
        "data/**/*.npz",
        "data/**/*.csv",
        "figures/*.pdf",
        "figures/*.svg",
    ):
        selected.update(path for path in article.glob(pattern) if path.is_file())
    scripts = root / "scripts/research/proximal_distal_energy"
    selected.update(path for path in scripts.glob("*.py") if path.is_file())
    selected.add(root / "src/shared/python/biomechanics/interaction_evidence.py")
    return tuple(
        sorted(
            (path for path in selected if path.name not in _EXCLUDED),
            key=lambda path: path.relative_to(root).as_posix(),
        )
    )


def build_release_manifest(root: str | Path) -> dict[str, Any]:
    """Build the current deterministic release qualification record."""
    root_path = Path(root).resolve()
    artifacts = {
        path.relative_to(root_path).as_posix(): {
            "sha256": _sha256(path),
            "bytes": path.stat().st_size,
        }
        for path in _artifact_paths(root_path)
    }
    return {
        "schema_version": "proximal-distal-open-release-v1",
        "release_id": "proximal-distal-model-ladder-2026-08",
        "resource_framing": "neutral_open_research_resource",
        "presets": {
            "double_pendulum": {
                "command": "python -m scripts.research.proximal_distal_energy.run_experiments",
                "tier": "planar_open_chain",
            },
            "forward_two_hand": {
                "command": "python -m scripts.research.proximal_distal_energy.run_forward_two_arm_study",
                "tier": "planar_constrained_forward",
            },
            "moving_base_flexible_club": {
                "command": "python -m scripts.research.proximal_distal_energy.run_moving_base_flexible_study",
                "tier": "planar_coupled_base_flex",
            },
            "shaft_beam_reference": {
                "command": "python -m scripts.research.proximal_distal_energy.run_shaft_beam_reference",
                "tier": "synthetic_distributed_shaft_comparison",
            },
            "spatial_common_state": {
                "command": "python -m scripts.research.proximal_distal_energy.run_spatial_full_body_study",
                "tier": "reduced_full_body_common_state",
            },
            "uncertainty_control": {
                "command": "python -m scripts.research.proximal_distal_energy.run_uncertainty_control_study",
                "tier": "coupled_uncertainty_control",
            },
            "experimental_readiness": {
                "command": "python -m scripts.research.proximal_distal_energy.run_experimental_protocol_dry_run",
                "tier": "synthetic_protocol_qualification_only",
            },
        },
        "claims": {
            "interaction_dynamics_planar": "supported_at_declared_model_tier",
            "geometry_transfer_spatial_common_state": "supported_at_declared_model_tier",
            "distributed_shaft_modal_reduction": "supported_on_synthetic_structural_case",
            "passive_negative_couple_spatial_forward": "untested",
            "universal_control_strategy": "unsupported",
            "human_experimental": "untested",
        },
        "known_open_gates": [
            "forward spatial contact in two independent engines",
            "equipment-calibrated distributed beam coupled into the forward two-hand solve",
            "governed held-out human experimental evaluation",
            "external archive deposit and persistent identifier",
        ],
        "archive": {
            "persistent_identifier_status": "pending_external_archive",
            "reason": "Archive deposition is an external publication action and has not been executed.",
        },
        "artifacts": artifacts,
    }


def validate_release_manifest(
    root: str | Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    """Fail closed on missing, changed, unsafe, or unsupported artifacts."""
    root_path = Path(root).resolve()
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("release manifest validation failed: artifacts are missing")
    mismatches: list[str] = []
    for relative, expected in artifacts.items():
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            mismatches.append(f"unsafe path: {relative}")
            continue
        path = root_path / relative_path
        if not path.is_file():
            mismatches.append(f"missing: {relative}")
            continue
        if not isinstance(expected, dict):
            mismatches.append(f"invalid record: {relative}")
            continue
        if _sha256(path) != expected.get("sha256"):
            mismatches.append(f"hash mismatch: {relative}")
        if path.stat().st_size != expected.get("bytes"):
            mismatches.append(f"size mismatch: {relative}")
    if mismatches:
        raise ValueError(
            "release manifest validation failed: " + "; ".join(mismatches[:8])
        )
    return {"valid": True, "artifact_count": len(artifacts), "mismatches": []}


def checksum_lines(manifest: dict[str, Any]) -> tuple[str, ...]:
    """Return sorted sha256sum-compatible records."""
    artifacts = manifest["artifacts"]
    return tuple(f"{artifacts[path]['sha256']}  {path}" for path in sorted(artifacts))
