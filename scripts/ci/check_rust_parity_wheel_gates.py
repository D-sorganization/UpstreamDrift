"""Ratchet Rust facade parity coverage and built-wheel import gates."""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DOC_PATH = REPO_ROOT / "docs" / "development" / "rust_parity_wheel_gates.md"
WORKFLOW_PATH = REPO_ROOT / ".github" / "workflows" / "ci-standard.yml"
IMPORT_SCRIPT = REPO_ROOT / "scripts" / "ci" / "import_built_rust_wheels.py"


@dataclass(frozen=True)
class Gate:
    facade: str
    crate: str
    parity_test: str
    wheel_module: str


REQUIRED_GATES = (
    Gate(
        "src/shared/python/physics/rust_kernel.py",
        "rust_core/upstream-physics",
        "rust_core/upstream-physics/tests/parity_physics.rs",
        "upstream_physics",
    ),
    Gate(
        "src/shared/python/physics/ball_flight_physics.py",
        "rust_core/upstream-physics",
        "rust_core/upstream-physics/tests/parity_physics.rs",
        "upstream_physics",
    ),
    Gate(
        "src/shared/python/motion_pipeline/preprocessing",
        "rust_core/upstream-mocap-preproc",
        "tests/unit/motion_pipeline/preprocessing/test_rust_parity.py",
        "upstream_mocap_preproc",
    ),
    Gate(
        "src/shared/python/motion_pipeline/sources",
        "rust_core/upstream-mocap-io",
        "tests/unit/motion_pipeline/sources/test_mocap_io_rust_parity.py",
        "upstream_mocap_io",
    ),
    Gate(
        "src/shared/python/biomechanics/rust_muscle.py",
        "rust_core/upstream-muscle",
        "rust_core/upstream-muscle/tests/parity_full.rs",
        "upstream_muscle",
    ),
    Gate(
        "src/shared/python/motion_pipeline/matching",
        "rust_core/upstream-motion-matching",
        "rust_core/upstream-motion-matching/tests/parity_finite_diff.rs",
        "upstream_motion_matching",
    ),
)


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def _has_markdown_row(text: str, gate: Gate) -> bool:
    pattern = re.compile(
        r"^\|\s*"
        + re.escape(gate.facade)
        + r"\s*\|\s*"
        + re.escape(gate.crate)
        + r"\s*\|\s*"
        + re.escape(gate.parity_test)
        + r"\s*\|\s*"
        + re.escape(gate.wheel_module)
        + r"\s*\|",
        re.MULTILINE,
    )
    return bool(pattern.search(text))


def audit(root: Path = REPO_ROOT) -> list[str]:
    failures: list[str] = []
    doc_path = root / DOC_PATH.relative_to(REPO_ROOT)
    workflow_path = root / WORKFLOW_PATH.relative_to(REPO_ROOT)
    import_script = root / IMPORT_SCRIPT.relative_to(REPO_ROOT)

    if not doc_path.exists():
        failures.append(
            f"missing Rust parity/wheel mapping doc: {doc_path.relative_to(root)}"
        )
        doc_text = ""
    else:
        doc_text = _read(doc_path)

    for gate in REQUIRED_GATES:
        if not (root / gate.parity_test).exists():
            failures.append(f"missing Rust parity test path: {gate.parity_test}")
        if doc_text and not _has_markdown_row(doc_text, gate):
            failures.append(
                "missing documented facade -> crate -> parity test -> wheel row for "
                f"{gate.facade}"
            )

    if not import_script.exists():
        failures.append(
            f"missing built-wheel import script: {import_script.relative_to(root)}"
        )

    workflow_text = _read(workflow_path) if workflow_path.exists() else ""
    if (
        "python -m pip install --force-reinstall target/wheels/*.whl"
        not in workflow_text
    ):
        failures.append(
            "ci-standard.yml must install every built Rust wheel before import smoke"
        )
    if "python scripts/ci/import_built_rust_wheels.py" not in workflow_text:
        failures.append("ci-standard.yml must run import_built_rust_wheels.py")

    for gate in REQUIRED_GATES:
        if gate.wheel_module not in workflow_text:
            failures.append(
                f"ci-standard.yml import smoke missing module {gate.wheel_module}"
            )

    return failures


def main() -> int:
    failures = audit()
    if failures:
        for failure in failures:
            print(f"ERROR: {failure}", file=sys.stderr)
        return 1
    print("Rust parity/wheel gates are documented and enforced.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
