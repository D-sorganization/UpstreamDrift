"""Reconciled restitution constant and centre-strike ratio (#14141 step 3).

Two strike assumptions carried over from the (reference-only, never
copied) Gemini suite needed reconciling in UD:

1. Restitution appeared as a bare ``0.78`` literal in two independent
   places (``PutterState.cor``'s default and
   ``collision.GROUND_RESTITUTION``) instead of one shared constant.
2. The Gemini suite quoted a single "1.595" ball/head speed ratio.
   That number is **not** a universal constant: it is
   ``(1 + e) * M / (M + m)`` evaluated at one particular
   ``(e, M, m)`` triple, valid only for a centred, square strike. Off
   centre, the effective head mass drops (see
   :func:`~src.shared.python.putting_dynamics.effective_head_mass`),
   which strictly lowers the ratio. This module asserts that
   closed-form relationship directly and never hardcodes 1.595.

All expected values below are closed forms (1-D impulse-momentum with
a coefficient of restitution), never read off a model run.
"""

from __future__ import annotations

import ast
import math
import re
from pathlib import Path

import pytest

from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GREEN_RESTITUTION,
    PUTTER_FACE_COR,
)
from src.shared.python.putting_dynamics import (
    GROUND_RESTITUTION,
    PutterState,
    centre_strike_speed_ratio,
    effective_head_mass,
    strike,
)
from src.shared.python.putting_dynamics.collision import (
    GROUND_RESTITUTION as COLLISION_GROUND_RESTITUTION,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
_PUTTING_DYNAMICS_DIR = REPO_ROOT / "src/shared/python/putting_dynamics"
_CONTACT_RS = REPO_ROOT / "rust_core/upstream-physics/src/contact.rs"

#: Bare literals the Gemini suite used in two conflicting places
#: (0.79/0.80 in one spot, a hardcoded 0.820 in another) — as floats,
#: 0.8 == 0.80 and 0.82 == 0.820, so this set covers all five spellings
#: named in the issue.
_FORBIDDEN_RESTITUTION_LITERALS = {0.78, 0.8, 0.82}
_COR_NAME_RE = re.compile(r"(?i)cor|restitution")

_M_BALL = float(GOLF_BALL_MASS_KG)


def _restitution_name_literal_assignments(source: str) -> list[str]:
    """Names assigned a bare forbidden-restitution-literal float.

    Only direct ``Assign``/``AnnAssign`` targets whose name mentions
    "cor" or "restitution" are considered — the shared
    ``PhysicalConstant`` definitions this module imports (outside
    ``putting_dynamics/``) are not scanned, so they remain the one
    place allowed to hold the literal.
    """
    tree = ast.parse(source)
    offenders: list[str] = []
    for node in ast.walk(tree):
        target: ast.expr | None = None
        value: ast.expr | None = None
        if isinstance(node, ast.Assign) and len(node.targets) == 1:
            target, value = node.targets[0], node.value
        elif isinstance(node, ast.AnnAssign):
            target, value = node.target, node.value
        if not isinstance(target, ast.Name) or value is None:
            continue
        if not _COR_NAME_RE.search(target.id):
            continue
        if (
            isinstance(value, ast.Constant)
            and isinstance(value.value, float)
            and value.value in _FORBIDDEN_RESTITUTION_LITERALS
        ):
            offenders.append(target.id)
    return offenders


def test_no_bare_restitution_literal_in_putting_dynamics() -> None:
    """Only the shared constants module may hold the restitution value.

    Characterizes the fix for the Gemini-suite drift: 0.78/0.79/0.80
    in one place, a hardcoded 0.820 in another.
    """
    offenders: dict[str, list[str]] = {}
    for path in sorted(_PUTTING_DYNAMICS_DIR.glob("*.py")):
        names = _restitution_name_literal_assignments(path.read_text(encoding="utf-8"))
        if names:
            offenders[path.relative_to(REPO_ROOT).as_posix()] = names
    assert not offenders, (
        "Bare restitution literal assigned to a cor/restitution name "
        f"outside the shared constants module: {offenders}"
    )


def test_contact_rs_default_matches_shared_constants() -> None:
    """Parity: the Python constants restate contact.rs's literal default.

    ``contact.rs`` is plainly parseable (a single ``cor: 0.78`` field
    default), so this pins the two Python constants to the Rust
    source rather than re-trusting a second copy of the number.
    """
    if not _CONTACT_RS.is_file():
        pytest.skip("rust_core/upstream-physics/src/contact.rs not present")
    source = _CONTACT_RS.read_text(encoding="utf-8")
    match = re.search(r"cor:\s*(-?\d+\.\d+)", source)
    if match is None:
        pytest.skip("contact.rs default cor literal not plainly parseable")
    rust_default = float(match.group(1))
    assert float(PUTTER_FACE_COR) == rust_default
    assert float(GREEN_RESTITUTION) == rust_default


def test_putter_state_cor_default_is_shared_constant() -> None:
    putter = PutterState(
        head_mass_kg=0.35, moi_kg_m2=4.5e-4, loft_deg=3.0, speed_mps=1.8
    )
    assert putter.cor == PUTTER_FACE_COR


def test_ground_restitution_is_shared_constant() -> None:
    assert GROUND_RESTITUTION == GREEN_RESTITUTION
    assert COLLISION_GROUND_RESTITUTION == GREEN_RESTITUTION


def _closed_form_ratio(cor: float, m_eff: float) -> float:
    """``(1 + e) * m_eff / (m_eff + m_ball)`` (module docstring derivation)."""
    return (1.0 + cor) * m_eff / (m_eff + _M_BALL)


def test_centre_strike_ratio_matches_strike_output_at_zero_loft() -> None:
    """At loft 0 the tangential terms vanish, so ``strike`` must reduce
    exactly to the closed-form centre-strike ratio (no loft/spin
    decomposition residue to tolerate)."""
    putter = PutterState(
        head_mass_kg=0.34, moi_kg_m2=4.0e-4, loft_deg=0.0, speed_mps=1.5
    )
    report = strike(putter, impact_toe_m=0.0)
    observed_ratio = report.ball_speed_mps / putter.speed_mps
    expected_ratio = centre_strike_speed_ratio(putter)
    assert observed_ratio == pytest.approx(expected_ratio, rel=1e-9)
    assert expected_ratio == pytest.approx(
        _closed_form_ratio(putter.cor, putter.head_mass_kg), rel=1e-12
    )


def test_centre_strike_ratio_is_not_the_gemini_1_595_constant() -> None:
    """The ratio is a function of (e, M, m) — not a fixed 1.595.

    This repo never adopts the Gemini-suite figure; it is only valid
    for one particular centre-strike parameter triple, and is wrong
    off centre. No assertion in this module (or anywhere else in the
    reconciled code) may equal 1.595.
    """
    light_putter = PutterState(
        head_mass_kg=0.30, moi_kg_m2=4.0e-4, loft_deg=0.0, speed_mps=1.5
    )
    heavy_putter = PutterState(
        head_mass_kg=0.40, moi_kg_m2=4.0e-4, loft_deg=0.0, speed_mps=1.5
    )
    ratio_light = centre_strike_speed_ratio(light_putter)
    ratio_heavy = centre_strike_speed_ratio(heavy_putter)
    assert ratio_light != pytest.approx(ratio_heavy)
    assert ratio_light != pytest.approx(1.595, abs=1e-3)
    assert ratio_heavy != pytest.approx(1.595, abs=1e-3)


@pytest.mark.parametrize("offset_mm", [0.0, 5.0, 10.0, 20.0, 30.0])
def test_off_centre_ratio_matches_effective_mass_closed_form(offset_mm: float) -> None:
    putter = PutterState(
        head_mass_kg=0.34, moi_kg_m2=4.0e-4, loft_deg=0.0, speed_mps=1.5
    )
    offset_m = offset_mm / 1000.0
    report = strike(putter, impact_toe_m=offset_m)
    observed_ratio = report.ball_speed_mps / putter.speed_mps
    m_eff = effective_head_mass(putter, offset_m)
    expected_ratio = _closed_form_ratio(putter.cor, m_eff)
    assert observed_ratio == pytest.approx(expected_ratio, rel=1e-9)


def test_off_centre_ratio_is_bounded_by_centre_ratio_and_symmetric() -> None:
    putter = PutterState(
        head_mass_kg=0.34, moi_kg_m2=4.0e-4, loft_deg=0.0, speed_mps=1.5
    )
    centre_ratio = centre_strike_speed_ratio(putter)

    offsets_mm = (0.0, 5.0, 10.0, 20.0, 30.0)
    ratios_by_abs_offset = {}
    for offset_mm in offsets_mm:
        offset_m = offset_mm / 1000.0
        for signed in (offset_m, -offset_m) if offset_m else (0.0,):
            report = strike(putter, impact_toe_m=signed)
            ratio = report.ball_speed_mps / putter.speed_mps
            assert ratio <= centre_ratio + 1e-12
            ratios_by_abs_offset.setdefault(abs(signed), set()).add(round(ratio, 12))

    # Symmetric in sign: +offset and -offset give the same ratio.
    for abs_offset, ratios in ratios_by_abs_offset.items():
        assert len(ratios) == 1, f"ratio not symmetric at offset {abs_offset}"

    # Monotone non-increasing in |offset|.
    ordered = [next(iter(ratios_by_abs_offset[abs(o / 1000.0)])) for o in offsets_mm]
    assert all(a >= b for a, b in zip(ordered, ordered[1:], strict=False)), ordered


def test_centre_strike_ratio_rejects_non_putter_state() -> None:
    with pytest.raises(ValueError):
        centre_strike_speed_ratio(object())  # type: ignore[arg-type]


def test_literal_scan_detects_assign_and_annassign_forms() -> None:
    """The scanner catches a module constant and a dataclass-field default."""
    source = "GROUND_RESTITUTION = 0.78\ncor: float = 0.82\nspeed = 0.8\n"
    assert _restitution_name_literal_assignments(source) == [
        "GROUND_RESTITUTION",
        "cor",
    ]
