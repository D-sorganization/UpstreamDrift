"""Keep standard gravity literals out of owned source (issue #13791)."""

from __future__ import annotations

import ast
import re
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
GRAVITY_LITERAL = re.compile(r"(?<![\w.])(?:-?9\.81|9\.8)(?![\w.])")
MJCF_GRAVITY = re.compile(r'\bgravity\s*=\s*["\'][^"\']*(?:-?9\.81|9\.8)(?![\w.])')

# These files intentionally retain values that cannot be converted in this
# slice. Every entry names the reason, and the test rejects stale entries.
ALLOWLIST: dict[str, str] = {
    "src/engines/physics_engines/mujoco/myo_sim/arm/myoarm_simple.xml": (
        "Vendored MyoSuite model asset; preserve upstream source unchanged."
    ),
    "src/engines/physics_engines/mujoco/myo_sim/body/myobody.xml": (
        "Vendored MyoSuite model asset; preserve upstream source unchanged."
    ),
    "src/engines/physics_engines/mujoco/myo_sim/body/myoupperbody.xml": (
        "Vendored MyoSuite model asset; preserve upstream source unchanged."
    ),
    "src/tools/shot_pattern_analysis/academic_plots.py": (
        "9.8 is the plot height in inches, not a gravity value."
    ),
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/Scripts/Dataset Generator/golf_swing_dataset_20251030/trial_002_20251030_192719.csv": (
        "Recorded capture data; values are observations, not source constants."
    ),
}


def _gravity_literal_files() -> set[str]:
    matches: set[str] = set()
    for path in (REPO_ROOT / "src").rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".py",
            ".xml",
            ".m",
            ".csv",
        }:
            continue
        try:
            source = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        if path.suffix.lower() == ".py":
            if not GRAVITY_LITERAL.search(source):
                continue
            tree = ast.parse(source, filename=str(path))
            has_literal = any(
                isinstance(node, ast.Constant)
                and isinstance(node.value, float)
                and node.value in {9.8, 9.81}
                for node in ast.walk(tree)
            )
        elif path.suffix.lower() == ".xml":
            has_literal = bool(MJCF_GRAVITY.search(source))
        else:
            has_literal = bool(GRAVITY_LITERAL.search(source))
        if has_literal:
            matches.add(path.relative_to(REPO_ROOT).as_posix())
    return matches


def test_owned_source_uses_canonical_gravity_constant() -> None:
    matches = _gravity_literal_files()
    stale_allowlist = set(ALLOWLIST) - matches
    unallowlisted = matches - set(ALLOWLIST)

    assert not unallowlisted, (
        "Replace standard-gravity literals with GRAVITY_M_S2 or add a reasoned, "
        f"scoped follow-up allowlist entry: {sorted(unallowlisted)}"
    )
    assert not stale_allowlist, (
        f"Stale gravity literal allowlist entries: {sorted(stale_allowlist)}"
    )


def test_gravity_allowlist_entries_have_reasons() -> None:
    assert all(reason.strip() for reason in ALLOWLIST.values())


def test_ud_31_tuple_literal_detected() -> None:
    """UD-31: AST lint guard catches 9.81 in tuple literals and negative forms."""
    source = "gravity = (0.0, 0.0, -9.81)"
    tree = ast.parse(source)
    has_literal = any(
        isinstance(node, ast.Constant)
        and isinstance(node.value, float)
        and node.value in {9.8, 9.81}
        for node in ast.walk(tree)
    )
    assert has_literal
