"""Regression tests for the ZTCF / drift sign convention (issue #11553).

With the equation of motion ``M(q) a + h(q, v) = tau + J^T lambda`` and
``h = C(q, v) v + g(q)``, setting ``tau = 0`` gives
``a_ZTCF = M(q)^-1 (J^T lambda_0 - h(q, v))`` (methods reference
``model_aware_matching.tex``, eq. ``eq:ztcf``).  The bias enters with a minus
sign.  The engine contract documented the opposite sign, so an engine written
against that contract would report gravity pushing a link *up*.

The analytic answer pinned here: a single uniform link pinned at the origin,
horizontal along +x, at rest, under gravity ``(0, -g)``.  The torque of
gravity about the pin is ``-m g l_c``, so the link's angular acceleration is
``-m g l_c / I_o`` (clockwise: it falls).
"""

from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pytest

from src.shared.python.estimation.mosaic.planar_chain import PlanarChain

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[3]

# Every place that documents the drift / ZTCF formula of the engine contract.
_CONTRACT_DOCS = (
    "src/shared/python/engine_core/_dynamics_interface.py",
    "src/shared/python/engine_core/sub_protocols.py",
    "src/shared/python/control_features_registry.py",
    "src/engines/physics_engines/mujoco/python/mujoco_humanoid_golf/physics_engine.py",
)

# ``M(q)^-1`` (Unicode or ASCII) applied to a bias that starts with a positive
# ``C(q,v)`` term: the wrong-sign form ``M^-1 (C v + g ...)``.
_POSITIVE_BIAS = re.compile(r"M\(q\)\s*(?:⁻¹|\^-1)\s*[·*]\s*\(\s*C\(q,\s*v\)")


def _single_link() -> tuple[PlanarChain, np.ndarray, float]:
    mass, length, lc, i_c = 2.0, 1.0, 0.5, 0.1
    chain = PlanarChain(link_lengths=np.array([length]), actuated=np.array([True]))
    i_o = i_c + mass * lc**2
    pi = np.array([mass, mass * lc, 0.0, i_o])
    expected = -mass * 9.81 * lc / i_o
    return chain, pi, expected


def test_ztcf_of_horizontal_link_at_rest_points_down() -> None:
    """ZTCF on the planar oracle equals the analytic ``-m g l_c / I_o``."""
    chain, pi, expected = _single_link()
    q = np.zeros((1, 1))
    v = np.zeros((1, 1))

    drift = chain.drift_decomposition(q, v, pi)

    assert expected < 0.0
    np.testing.assert_allclose(drift.ztcf[0, 0], expected, rtol=1e-12)
    np.testing.assert_allclose(drift.zvcf[0, 0], expected, rtol=1e-12)


def test_ztcf_equals_minus_inverse_mass_times_bias() -> None:
    """With velocity the ZTCF is ``-M^-1 h`` exactly, never ``+M^-1 h``."""
    chain, pi, _ = _single_link()
    q = np.array([[0.4]])
    v = np.array([[1.7]])
    mass = chain.mass_matrix(q, pi)
    bias = chain.bias(q, v, pi)

    ztcf = chain.drift_decomposition(q, v, pi).ztcf

    np.testing.assert_allclose(ztcf, -np.linalg.solve(mass, bias[..., None])[..., 0])
    assert not np.allclose(ztcf, np.linalg.solve(mass, bias[..., None])[..., 0])


@pytest.mark.parametrize("relative_path", _CONTRACT_DOCS)
def test_contract_documents_negative_bias(relative_path: str) -> None:
    """No contract docstring may document ``M^-1 (C(q,v) v + g)`` as the drift."""
    text = (_REPO_ROOT / relative_path).read_text(encoding="utf-8")

    offending = [
        line.strip() for line in text.splitlines() if _POSITIVE_BIAS.search(line)
    ]

    assert offending == [], f"{relative_path} documents the wrong drift sign"
