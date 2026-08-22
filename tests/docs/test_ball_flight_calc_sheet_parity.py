"""Doc/code parity gate for the ball-flight calculation sheet (issue #8845).

The calculation sheet ``docs/physics/BALL_FLIGHT_MODEL_DOCUMENTATION.md`` and
the provenance record ``docs/physics/GOLF_BALL_FLIGHT_IMPACT_SOURCE_MAP.md``
carry machine-readable markers of the form::

    <!-- calc:NAME -->VALUE<!-- /calc -->

Each marker is a *validated copy* of a constant whose single source of truth
is the physics code (DRY: the code owns the value; the doc quotes it and this
gate enforces the quote). If a constant changes in code, this test fails until
the doc is updated -- preventing the drift documented in issue #8845 (wrong
lift law, dead cl0/cl1/cl2 presented as live, 5x cd1 discrepancy, stale
assumptions).
"""

from __future__ import annotations

import math
import re
from pathlib import Path

import pytest

from src.shared.python.core.physics_constants import (
    AIR_DENSITY_SEA_LEVEL_KG_M3,
    GOLF_BALL_DIAMETER_M,
    GOLF_BALL_MASS_KG,
    SPIN_DECAY_RATE_S,
)
from src.shared.python.physics.ball_properties import (
    MAX_LIFT_COEFFICIENT,
    MIN_SPEED_THRESHOLD,
    PENNER_LIFT_EXPONENT,
    PENNER_LIFT_SCALE,
    BallProperties,
    calculate_spin_lift_coefficient,
)
from src.shared.python.physics.flight_models import WATERLOO_PENNER_COEFFICIENTS

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[2]
CALC_SHEET = REPO_ROOT / "docs" / "physics" / "BALL_FLIGHT_MODEL_DOCUMENTATION.md"
SOURCE_MAP = REPO_ROOT / "docs" / "physics" / "GOLF_BALL_FLIGHT_IMPACT_SOURCE_MAP.md"

MARKER_RE = re.compile(
    r"<!--\s*calc:([A-Za-z0-9_.]+)\s*-->\s*([^<]+?)\s*<!--\s*/calc\s*-->"
)

_BALL = BallProperties()

# Single source of truth: values imported from the physics code.
# Keys are marker names; values are the authoritative code values.
CALC_SHEET_EXPECTED: dict[str, float] = {
    "ball_mass_kg": float(GOLF_BALL_MASS_KG),
    "ball_diameter_m": float(GOLF_BALL_DIAMETER_M),
    "air_density_sea_level_kg_m3": float(AIR_DENSITY_SEA_LEVEL_KG_M3),
    "cd0": _BALL.cd0,
    "cd1": _BALL.cd1,
    "cd2": _BALL.cd2,
    "penner_lift_scale": PENNER_LIFT_SCALE,
    "penner_lift_exponent": PENNER_LIFT_EXPONENT,
    "max_lift_coefficient": MAX_LIFT_COEFFICIENT,
    "spin_decay_rate_s": float(SPIN_DECAY_RATE_S),
    "min_speed_threshold_m_s": MIN_SPEED_THRESHOLD,
    # Dead legacy fields: still on the dataclass, never read by the shipped
    # lift law. The sheet must present them as unused-legacy, but the quoted
    # values must still match the code so the sheet cannot silently rot.
    "legacy_cl0": _BALL.cl0,
    "legacy_cl1": _BALL.cl1,
    "legacy_cl2": _BALL.cl2,
    # Second live model family (issue #8978): the constant-spin multi-model
    # framework's Waterloo/Penner set is deliberately different from the
    # core set above; the sheet documents both and this gate pins both.
    "wp_cd1": WATERLOO_PENNER_COEFFICIENTS.cd1,
    "wp_max_lift_coefficient": WATERLOO_PENNER_COEFFICIENTS.cl_max,
}

SOURCE_MAP_EXPECTED: dict[str, float] = {
    "cd0": _BALL.cd0,
    "legacy_cl1": _BALL.cl1,
    "max_lift_coefficient": MAX_LIFT_COEFFICIENT,
    "spin_decay_rate_s": float(SPIN_DECAY_RATE_S),
}


def _parse_markers(path: Path) -> dict[str, list[float]]:
    """Extract all calc markers from a doc as name -> list of parsed values."""
    text = path.read_text(encoding="utf-8")
    found: dict[str, list[float]] = {}
    for name, raw in MARKER_RE.findall(text):
        found.setdefault(name, []).append(float(raw))
    return found


@pytest.fixture(scope="module")
def calc_sheet_markers() -> dict[str, list[float]]:
    return _parse_markers(CALC_SHEET)


@pytest.fixture(scope="module")
def source_map_markers() -> dict[str, list[float]]:
    return _parse_markers(SOURCE_MAP)


class TestCalcSheetParity:
    """BALL_FLIGHT_MODEL_DOCUMENTATION.md must match the physics code."""

    def test_doc_exists(self) -> None:
        assert CALC_SHEET.is_file(), f"missing calculation sheet: {CALC_SHEET}"

    @pytest.mark.parametrize("name", sorted(CALC_SHEET_EXPECTED))
    def test_marker_matches_code(
        self, name: str, calc_sheet_markers: dict[str, list[float]]
    ) -> None:
        expected = CALC_SHEET_EXPECTED[name]
        assert name in calc_sheet_markers, (
            f"calc sheet is missing marker '<!-- calc:{name} -->' "
            f"(code value: {expected})"
        )
        for value in calc_sheet_markers[name]:
            assert math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12), (
                f"calc sheet quotes {name} = {value}, but code has {expected}"
            )

    def test_no_unknown_markers(
        self, calc_sheet_markers: dict[str, list[float]]
    ) -> None:
        unknown = set(calc_sheet_markers) - set(CALC_SHEET_EXPECTED)
        assert not unknown, (
            f"calc sheet has markers with no code-backed expectation: {sorted(unknown)}. "
            "Add them to CALC_SHEET_EXPECTED with an authoritative import."
        )


class TestSourceMapParity:
    """GOLF_BALL_FLIGHT_IMPACT_SOURCE_MAP.md must match the physics code."""

    def test_doc_exists(self) -> None:
        assert SOURCE_MAP.is_file(), f"missing source map: {SOURCE_MAP}"

    @pytest.mark.parametrize("name", sorted(SOURCE_MAP_EXPECTED))
    def test_marker_matches_code(
        self, name: str, source_map_markers: dict[str, list[float]]
    ) -> None:
        expected = SOURCE_MAP_EXPECTED[name]
        assert name in source_map_markers, (
            f"source map is missing marker '<!-- calc:{name} -->' "
            f"(code value: {expected})"
        )
        for value in source_map_markers[name]:
            assert math.isclose(value, expected, rel_tol=1e-9, abs_tol=1e-12), (
                f"source map quotes {name} = {value}, but code has {expected}"
            )

    def test_no_unknown_markers(
        self, source_map_markers: dict[str, list[float]]
    ) -> None:
        unknown = set(source_map_markers) - set(SOURCE_MAP_EXPECTED)
        assert not unknown, (
            f"source map has markers with no code-backed expectation: {sorted(unknown)}"
        )


class TestDocumentedLawsMatchCode:
    """The laws stated in the sheet must be the laws the code executes."""

    def test_lift_law_is_penner_power_law_not_quadratic(self) -> None:
        """C_L = min(cap, scale * S**exponent); the quadratic cl-fields are dead."""
        for s in (0.05, 0.1, 0.2, 0.3):
            expected = min(
                MAX_LIFT_COEFFICIENT, PENNER_LIFT_SCALE * s**PENNER_LIFT_EXPONENT
            )
            assert math.isclose(_BALL.calculate_cl(s), expected, rel_tol=1e-12)
        # Changing the legacy quadratic fields must NOT change the lift output.
        modified = BallProperties(cl0=9.0, cl1=9.0, cl2=9.0)
        assert modified.calculate_cl(0.2) == _BALL.calculate_cl(0.2), (
            "cl0/cl1/cl2 now affect calculate_cl -- the calc sheet documents "
            "them as unused-legacy; update the sheet and this gate"
        )

    def test_lift_cap_engages(self) -> None:
        assert calculate_spin_lift_coefficient(10.0) == MAX_LIFT_COEFFICIENT

    def test_drag_law_is_quadratic_in_spin_parameter(self) -> None:
        """C_D = cd0 + cd1*S + cd2*S**2, as the calc sheet states."""
        for s in (0.0, 0.1, 0.25):
            expected = _BALL.cd0 + _BALL.cd1 * s + _BALL.cd2 * s**2
            assert math.isclose(_BALL.calculate_cd(s), expected, rel_tol=1e-12)
