"""unit-test-gate records the numeric-kernel host behind exact-hash results (#14397).

The grip fixture hashes and saved-comparison receipts drift by the last bit on
some hosted runners. The drift follows the runner CPU (it appears on AVX-512
hosts such as AMD EPYC 9V45 and not on AVX2-only EPYC 7763 hosts), so every
gate run logs the CPU model, whether it has AVX-512, and numpy's SIMD dispatch.

Pinning OPENBLAS_CORETYPE=Haswell and disabling numpy's AVX-512 targets was
tried and did not hold: on an EPYC 9V45 with both pins active the same four
hashes still drifted. The gate must not carry those variables, because a pin
that does not hold would make the logs claim a kernel the run did not use.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/ci-standard.yml"


def _unit_test_gate() -> dict:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return workflow["jobs"]["unit-test-gate"]


def test_unit_test_gate_reports_the_numeric_kernel_host() -> None:
    steps = [
        step
        for step in _unit_test_gate()["steps"]
        if step.get("name") == "Report numeric kernel host (#14397)"
    ]

    assert len(steps) == 1
    script = steps[0]["run"]
    assert 'grep -m1 "model name" /proc/cpuinfo' in script
    assert "grep -m1 -o -w avx512f /proc/cpuinfo" in script
    assert "numpy.show_runtime()" in script


def test_unit_test_gate_does_not_claim_an_unproven_kernel_pin() -> None:
    env = _unit_test_gate().get("env", {})

    assert "OPENBLAS_CORETYPE" not in env
    assert "NPY_DISABLE_CPU_FEATURES" not in env
