"""Parity verification and acceptance tests for issue #13560.

Validates that:
1. Sibling-implementation sweep: no `_net_moment_xfail` remains in tests/.
2. All six MuJoCo and OpenSim determinate net_moment_com cases are active without xfails.
3. The balanced t = 0 preload zero-friction at-rest invariant holds in the shared law.
4. OpenSim Hollars static friction matches the shared Stribeck law (0.664 for 0.9/0.7).
5. OpenSim integrator accuracy selects 5e-10 for swing and 5e-8 for hold.
6. Gate tolerances remain strictly unchanged at 5.0 % peak / 2.0 % RMS.
"""

from __future__ import annotations

from pathlib import Path
import re

import pytest

from src.engines.physics_engines.opensim.python import grip_contact_osim_sim as osim_sim
from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.grip_contact.friction_curve_match import (
    matched_static_friction,
    shared_law_curve,
)
from src.shared.python.grip_contact.parity import (
    DETERMINATE_QUANTITIES,
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
)
from tests.unit.grip_contact.test_grip_style_parity_report import (
    _DETERMINATE_ENGINE_CASES,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

ROOT = Path(__file__).resolve().parents[3]
TESTS_DIR = ROOT / "tests"


def test_sibling_implementation_sweep_no_net_moment_xfails_remain() -> None:
    """Sibling sweep: no test file across tests/ may declare or use _net_moment_xfail."""
    pattern = re.compile(r"\b_net_moment_xfail\b")
    violations: list[str] = []

    this_file = Path(__file__).resolve()
    for test_file in TESTS_DIR.rglob("*.py"):
        if test_file.resolve() == this_file:
            continue
        text = test_file.read_text(encoding="utf-8")
        if pattern.search(text):
            violations.append(str(test_file.relative_to(ROOT)))

    assert not violations, (
        f"Found remaining _net_moment_xfail references in: {violations}"
    )


def test_determinate_parity_cases_registered_without_xfail() -> None:
    """All 6 MuJoCo and 6 OpenSim cases must be registered without xfail marks."""
    registered: dict[tuple[str, str, str], list[pytest.Mark]] = {}
    for param in _DETERMINATE_ENGINE_CASES:
        engine, style, motion = param.values
        style_val = style.value if isinstance(style, GripStyle) else str(style)
        registered[(engine, style_val, motion)] = list(param.marks)

    styles = ["overlap", "interlock", "ten_finger"]
    motions = ["hold", "swing"]

    for engine in ("mujoco", "opensim"):
        for style in styles:
            for motion in motions:
                key = (engine, style, motion)
                assert key in registered, f"Missing case: {key}"
                marks = registered[key]
                mark_names = [m.name for m in marks]
                assert "xfail" not in mark_names, f"Case {key} has xfail mark: {marks}"


def test_balanced_preload_at_rest_zero_friction_invariant() -> None:
    """At t = 0 rest (slip ratio r = 0), shared friction law evaluates to exactly 0."""
    import numpy as np

    zero_slip = np.array([0.0])
    mu = shared_law_curve(zero_slip, mu_s=0.9, mu_d=0.7)
    assert float(mu[0]) == pytest.approx(0.0, abs=1e-12)


def test_opensim_hollars_static_friction_match() -> None:
    """Simbody Hollars static friction fits shared Stribeck law to ~0.664."""
    mu_s_fitted = matched_static_friction(mu_s=0.9, mu_d=0.7)
    assert mu_s_fitted == pytest.approx(0.664, abs=0.01)
    assert 0.65 < mu_s_fitted < 0.70


def test_opensim_integrator_accuracy_by_motion() -> None:
    """Swings integrate at 5e-10 to resolve net-moment gap; holds keep 5e-8."""
    assert osim_sim.topology_contact_accuracy(hold=False) == 5e-10
    assert osim_sim.topology_contact_accuracy(hold=True) == 5e-8


def test_determinate_gate_tolerances_remain_strictly_unchanged() -> None:
    """Parity tolerances must stay 5.0 % peak / 2.0 % RMS (#13560 non-regression)."""
    assert PEAK_TOLERANCE == 0.05
    assert RMS_TOLERANCE == 0.02
    assert "net_moment_com" in DETERMINATE_QUANTITIES
    assert "net_force" in DETERMINATE_QUANTITIES
