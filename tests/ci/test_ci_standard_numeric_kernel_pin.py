"""unit-test-gate pins its numeric kernels so exact-hash fixtures match on any host (#14397).

The bundled OpenBLAS picks a GEMM kernel per CPU, and numpy picks AVX-512 loops
where the host has them. Both change the last bit of the small matrix products
behind the grip fixtures and the saved-comparison receipts, so the hashes moved
with the runner's CPU rather than with the code. The gate therefore pins the
AVX2 OpenBLAS kernel and turns off numpy's AVX-512 dispatch; every hosted and
fleet runner supports AVX2.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.unit

WORKFLOW = Path(__file__).resolve().parents[2] / ".github/workflows/ci-standard.yml"

# The AVX-512 dispatch targets of the pinned numpy 2.2.6. Each one is checked on
# its own, so disabling AVX512F alone would leave the newer targets enabled.
# AVX512_SPR is not a 2.2.6 target and naming it raises an ImportWarning.
AVX512_TARGETS = {
    "AVX512F",
    "AVX512CD",
    "AVX512_KNL",
    "AVX512_KNM",
    "AVX512_SKX",
    "AVX512_CLX",
    "AVX512_CNL",
    "AVX512_ICL",
}


def _unit_test_gate_env() -> dict[str, str]:
    workflow = yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))
    return {
        key: str(value)
        for key, value in workflow["jobs"]["unit-test-gate"]["env"].items()
    }


def test_unit_test_gate_pins_the_avx2_openblas_kernel() -> None:
    assert _unit_test_gate_env().get("OPENBLAS_CORETYPE") == "Haswell"


def test_unit_test_gate_disables_every_numpy_avx512_target() -> None:
    disabled = set(_unit_test_gate_env().get("NPY_DISABLE_CPU_FEATURES", "").split())

    assert disabled >= AVX512_TARGETS, sorted(AVX512_TARGETS - disabled)
