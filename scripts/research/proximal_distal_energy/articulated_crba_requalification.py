"""Preregister the fail-closed Pinocchio CRBA evidence requalification.

The registration freezes source identity, execution order, replay rules, and
promotion gates before corrected native-engine outcomes are inspected.  It is
protocol evidence only and cannot qualify or promote a scientific result.
"""

from __future__ import annotations

from argparse import ArgumentParser
import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
REGISTRATION_PATH = ARTICLE / "data/articulated_crba_requalification.json"
SCHEMA_VERSION = "proximal-distal-articulated-crba-requalification/v1"

CORRECTED_SOURCE_PATHS = (
    Path("scripts/research/proximal_distal_energy/articulated_inertia_cross_engine.py"),
    Path("scripts/research/proximal_distal_energy/articulated_forward_integration.py"),
    Path("scripts/research/proximal_distal_energy/articulated_contact_projection.py"),
    Path(
        "scripts/research/proximal_distal_energy/"
        "articulated_drift_contact_attribution.py"
    ),
)

PRIMARY_ARTIFACTS = (
    "articulated_inertia_cross_engine",
    "articulated_native_constraint_discrepancy",
    "articulated_contact_projection",
    "articulated_drift_contact_attribution",
    "articulated_forward_contact",
    "articulated_distributed_grip_atlas",
)

_PIPELINES = (
    (
        "articulated_inertia_cross_engine",
        "run_articulated_inertia_cross_engine",
        "make_articulated_inertia_cross_engine_figure",
        "register_articulated_inertia_claims.py",
        ("pdf", "svg"),
    ),
    (
        "articulated_native_constraint_discrepancy",
        "run_articulated_native_constraint_discrepancy",
        "make_articulated_native_constraint_discrepancy_figure",
        "register_articulated_native_constraint_discrepancy_claims.py",
        ("pdf",),
    ),
    (
        "articulated_contact_projection",
        "run_articulated_contact_projection",
        "make_articulated_contact_projection_figure",
        "register_articulated_contact_projection_claims.py",
        ("pdf", "svg"),
    ),
    (
        "articulated_drift_contact_attribution",
        "run_articulated_drift_contact_attribution",
        "make_articulated_drift_contact_attribution_figure",
        "register_articulated_drift_contact_attribution_claims.py",
        ("pdf", "svg"),
    ),
    (
        "articulated_forward_contact",
        "run_articulated_forward_contact",
        "make_articulated_forward_contact_figure",
        "register_articulated_forward_contact_claims.py",
        ("pdf", "svg"),
    ),
    (
        "articulated_distributed_grip_atlas",
        "run_distributed_grip_atlas",
        "make_distributed_grip_figure",
        "register_distributed_grip_claims.py",
        ("pdf", "svg"),
    ),
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _authority(root: Path, relative: Path) -> dict[str, object]:
    path = root / relative
    if not path.is_file():
        raise ValueError(f"missing requalification authority: {relative.as_posix()}")
    return {
        "path": relative.as_posix(),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
    }


def _pipeline_phase(
    artifact: str,
    runner: str,
    figure: str,
    registration: str,
    figure_suffixes: tuple[str, ...],
) -> dict[str, object]:
    module_root = "scripts.research.proximal_distal_energy"
    script_root = "scripts/research/proximal_distal_energy"
    outputs = [
        f"docs/research/proximal_distal_energy_transfer/data/{artifact}.json",
        f"docs/research/proximal_distal_energy_transfer/data/{artifact}.npz",
    ]
    outputs.extend(
        f"docs/research/proximal_distal_energy_transfer/figures/fig_{artifact}.{suffix}"
        for suffix in figure_suffixes
    )
    return {
        "phase_id": f"regenerate_{artifact}",
        "commands": [
            f"python3 -m {module_root}.{runner}",
            f"python3 -m {module_root}.{figure}",
            f"python3 {script_root}/{registration}",
        ],
        "expected_outputs": outputs,
        "outcome_rule": "retain_all_declared_cases_and_typed_failures",
    }


def _execution_phases() -> list[dict[str, object]]:
    environment = {
        "phase_id": "qualify_native_environment",
        "commands": [
            'python3 -c "import mujoco, pinocchio as pin; '
            "assert mujoco.__version__ == '3.8.0'; "
            "assert pin.__version__ == '3.8.0'; "
            "assert all(hasattr(pin, name) for name in ('Model','crba','rnea'))\""
        ],
        "expected_outputs": [],
        "outcome_rule": "stop_before_execution_on_any_identity_mismatch",
    }
    phases = [environment]
    phases.extend(_pipeline_phase(*pipeline) for pipeline in _PIPELINES)
    phases.append(
        {
            "phase_id": "validate_and_promote",
            "commands": [
                "python3 -m scripts.research.proximal_distal_energy."
                "articulated_crba_requalification validate",
                "python3 -m scripts.research.proximal_distal_energy."
                "build_claim_numeric_comparison_evidence check",
                "python3 -m scripts.research.proximal_distal_energy."
                "register_numeric_claim_evidence check",
                "python3 -m scripts.research.proximal_distal_energy.claim_audit validate",
                "python3 -m scripts.research.proximal_distal_energy.claim_audit numeric",
                "python3 -m scripts.research.proximal_distal_energy."
                "claim_evidence_integrity write",
                "quarto render docs/research/proximal_distal_energy_transfer/"
                "proximal_distal_energy_transfer.qmd --to pdf",
                "python3 -m scripts.research.proximal_distal_energy.optimize_article_pdf",
                "python3 -m scripts.research.proximal_distal_energy."
                "qualify_open_release write --source-revision $(git rev-parse HEAD) "
                "--publication-profile computational",
                "python3 -m scripts.research.proximal_distal_energy."
                "qualify_open_release validate --source-revision $(git rev-parse HEAD) "
                "--publication-profile computational",
            ],
            "expected_outputs": [
                "docs/research/proximal_distal_energy_transfer/data/"
                "claim_audit_registry.json",
                "docs/research/proximal_distal_energy_transfer/data/"
                "claim_evidence_manifest.json",
                "docs/research/proximal_distal_energy_transfer/"
                "proximal_distal_energy_transfer.pdf",
                "docs/research/proximal_distal_energy_transfer/release_manifest.json",
                "docs/research/proximal_distal_energy_transfer/CHECKSUMS.sha256",
            ],
            "outcome_rule": "promotion_requires_every_registered_gate",
        }
    )
    return phases


def build_registration(root: Path = ROOT) -> dict[str, object]:
    """Build the deterministic outcome-blind requalification registration."""
    root = root.resolve()
    return {
        "schema_version": SCHEMA_VERSION,
        "classification": "prospective_native_evidence_requalification",
        "evidence_status": "prospective_no_requalified_outcome",
        "trigger": {
            "defect": (
                "Pinocchio CRBA populates the upper triangle; four research paths "
                "previously passed the returned storage without explicit mirroring"
            ),
            "scientific_effect_status": "unknown_until_requalified",
            "pre_correction_artifact_status": "retained_but_stale_not_promotable",
        },
        "corrected_source_authorities": [
            _authority(root, path) for path in CORRECTED_SOURCE_PATHS
        ],
        "primary_artifacts": list(PRIMARY_ARTIFACTS),
        "qualified_environment": {
            "supported_python": "3.11.x_or_3.12.x",
            "operating_system": "linux_x86_64",
            "pinocchio_distribution": "pin",
            "pinocchio_version": "3.8.0",
            "mujoco_distribution": "mujoco",
            "mujoco_version": "3.8.0",
            "pinocchio_operator_probe": ["Model", "crba", "rnea"],
            "maximum_workers": 1,
            "thread_limits": {
                "OMP_NUM_THREADS": "1",
                "OPENBLAS_NUM_THREADS": "1",
                "MKL_NUM_THREADS": "1",
            },
            "environment_capture": [
                "python_version",
                "platform",
                "machine",
                "numpy_version",
                "mujoco_version",
                "pinocchio_version",
                "installed_distribution_hashes",
            ],
        },
        "execution_phases": _execution_phases(),
        "replay_contract": {
            "clean_execution_count": 2,
            "outcome_inspection_before_second_replay": False,
            "json_comparison": "canonical_exact",
            "npz_comparison": "memberwise_exact_equal_nan",
            "figure_comparison": "svg_canonical_exact_and_pdf_page_render_equivalent",
            "case_order_comparison": "exact",
            "typed_failure_comparison": "exact",
        },
        "required_gates": [
            "all corrected source hashes match this registration",
            "all six primary JSON source maps match current source content",
            "all six primary JSON and NPZ outputs pass their focused tests",
            "both clean executions satisfy the replay contract",
            "the poisoned-lower-triangle CRBA regression remains green",
            "all declared cases and typed failures are retained",
            "claim numeric and evidence-integrity authorities are current",
            "the paper is regenerated and every PDF page passes inspection",
            "the release manifest and checksums match the final tree",
            "protected CI succeeds on the exact final head",
        ],
        "stop_conditions": [
            "native package identity or version mismatch",
            "source hash drift after registration",
            "any primary artifact omits a declared case or typed failure",
            "any unexplained difference between the two clean executions",
            "any focused falsification or corruption killswitch failure",
            "any claim, figure, paper, checksum, or release freshness failure",
        ],
        "promotion_authority": "none_until_all_gates_pass",
        "promotion_eligible": False,
        "inference_boundary": (
            "Requalification can establish reproducibility and internal numerical "
            "consistency only for the declared synthetic articulated models. It "
            "cannot establish human motor intent, anatomical force allocation, "
            "population effects, injury risk, or coaching guidance."
        ),
    }


def validate_registration(
    report: dict[str, Any], root: Path = ROOT
) -> dict[str, object]:
    """Fail closed on source, scope, replay, or promotion drift."""
    expected = build_registration(root)
    if report != expected:
        raise ValueError("registration differs from deterministic authority")
    sources = report.get("corrected_source_authorities")
    artifacts = report.get("primary_artifacts")
    phases = report.get("execution_phases")
    if not isinstance(sources, list) or len(sources) != len(CORRECTED_SOURCE_PATHS):
        raise ValueError("all corrected source authorities are required")
    if not isinstance(artifacts, list) or artifacts != list(PRIMARY_ARTIFACTS):
        raise ValueError("complete primary artifact closure is required")
    if not isinstance(phases, list) or len(phases) != 8:
        raise ValueError("the eight dependency-ordered phases are required")
    if report.get("promotion_eligible") is not False:
        raise ValueError("prospective registration cannot promote evidence")
    return {
        "corrected_source_count": len(sources),
        "primary_artifact_count": len(artifacts),
        "execution_phase_count": len(phases),
        "promotion_eligible": False,
    }


def _parser() -> ArgumentParser:
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "validate"))
    return parser


def main() -> None:
    """Write or validate the prospective registration."""
    action = _parser().parse_args().action
    if action == "write":
        REGISTRATION_PATH.parent.mkdir(parents=True, exist_ok=True)
        REGISTRATION_PATH.write_text(
            json.dumps(build_registration(), indent=2) + "\n", encoding="utf-8"
        )
        print(REGISTRATION_PATH)
        return
    report = json.loads(REGISTRATION_PATH.read_text(encoding="utf-8"))
    print(json.dumps(validate_registration(report), indent=2))


if __name__ == "__main__":
    main()
