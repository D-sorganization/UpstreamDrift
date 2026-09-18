"""CI gate for the impact acceptance matrix (issue #9550, epic #9546).

``src/config/impact_acceptance.json`` is the frozen acceptance record for the
Impact Explorer product: the model-capability matrix, the acceptance work
items with their evidence or explicit open state, and the bounded release
claim. These tests ARE the gate. They fail when:

1. A model row uses a value outside the closed vocabulary, or claims spin /
   gear-effect / COR / inertia behaviour the library does not exhibit — a model
   lacking spin support must say ``unavailable`` (normal-only), never imply a
   predicted zero-spin result.
2. A ``met`` work item lacks evidence or a test; an ``open``/``partial`` item
   lacks blockers, an owner or a plan.
3. A referenced path does not exist, or the pinned Tools revision drifts from
   ``requirements-tools.txt`` without the matrix being re-reconciled.
4. The release claim reads as predictive while any work item is outstanding.

The library probes double as the shared golden cases of acceptance item 2:
centered / off-center, no-hit, low / high speed, spin sign, momentum,
passivity and integrator convergence, all in SI.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.core.physics_constants import GOLF_BALL_MASS_KG
from src.shared.python.physics.impact_model import (
    ImpactModelType,
    ImpactParameters,
    ImpactSolverAPI,
    PreImpactState,
    SpringDamperImpactModel,
    create_impact_model,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
MATRIX_PATH = REPO_ROOT / "src" / "config" / "impact_acceptance.json"
PACKET_PATH = REPO_ROOT / "docs" / "development" / "impact_acceptance_matrix.md"
TOOLS_PIN_PATH = REPO_ROOT / "requirements-tools.txt"

_COMMIT_RE = re.compile(r"^[0-9a-f]{40}$")
WORK_STATUSES = {"met", "partial", "open"}
VOCABULARY: dict[str, set[str]] = {
    "contact": {
        "instantaneous",
        "finite_duration_reported_only",
        "finite_duration_integrated",
    },
    "tangential_spin": {"supported", "unavailable"},
    "gear_effect": {"empirical_overlay_via_solver_api", "unavailable"},
    "inertia_tensor": {"scalar_moi_effective_mass", "ignored"},
    "velocity_dependent_cor": {"unavailable_constant_cor"},
    "shaft_grip": {"not_modelled_free_clubhead"},
}


@pytest.fixture(scope="module")
def matrix() -> dict[str, Any]:
    return json.loads(MATRIX_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def models(matrix: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in matrix["model_capabilities"]["models"]}


def _lofted_strike(
    speed: float = 45.0,
    loft_deg: float = 10.5,
    offset: list[float] | None = None,
) -> PreImpactState:
    """Driver-like strike: face normal tilted by loft, ball at rest, SI units."""
    loft = np.radians(loft_deg)
    return PreImpactState(
        clubhead_velocity=np.array([speed, 0.0, 0.0]),
        clubhead_angular_velocity=np.zeros(3),
        clubhead_orientation=np.array([np.cos(loft), 0.0, np.sin(loft)]),
        ball_position=np.zeros(3),
        ball_velocity=np.zeros(3),
        ball_angular_velocity=np.zeros(3),
        clubhead_mass=0.200,
        impact_offset=None if offset is None else np.array(offset),
    )


# ---------------------------------------------------------------------------
# The committed matrix
# ---------------------------------------------------------------------------


def test_matrix_identifies_issue_epic_and_exact_revisions(
    matrix: dict[str, Any],
) -> None:
    assert matrix["issue"] == 9550
    assert matrix["epic"] == 9546
    for block in ("audit_snapshot", "reconciled_against"):
        for repo in ("upstreamdrift", "tools"):
            sha = matrix[block][repo]
            assert _COMMIT_RE.fullmatch(sha), f"{block}.{repo} is not a full commit"
    assert matrix["audit_snapshot"] != matrix["reconciled_against"]


def test_reconciled_tools_revision_matches_the_pinned_provider(
    matrix: dict[str, Any],
) -> None:
    """Exact-pin evidence: the matrix speaks about the pin actually consumed."""
    pin = re.search(r"Tools\.git@([0-9a-f]{40})", TOOLS_PIN_PATH.read_text())
    assert pin is not None, "requirements-tools.txt carries no Tools commit pin"
    assert matrix["reconciled_against"]["tools"] == pin.group(1), (
        "impact_acceptance.json was reconciled against a different Tools pin; "
        "re-run the capability probes and served-bundle verifier, then update "
        "reconciled_against"
    )


def test_every_referenced_path_exists(matrix: dict[str, Any]) -> None:
    cited: list[str] = []
    for row in matrix["model_capabilities"]["models"]:
        cited.append(row["module"])
    for item in matrix["acceptance_work"]:
        cited.extend(item.get("tests", []))
        cited.extend(item.get("evidence_paths", []))
    missing = [p for p in cited if not (REPO_ROOT / p).exists()]
    assert not missing, f"impact_acceptance.json cites missing paths: {missing}"


def test_model_rows_use_the_closed_vocabulary(
    models: dict[str, dict[str, Any]],
) -> None:
    assert set(models) == {"rigid_body", "spring_damper", "finite_time"}
    for model_id, row in models.items():
        for field, allowed in VOCABULARY.items():
            assert row[field] in allowed, f"{model_id}.{field}={row[field]!r}"
        assert row["numerical_limits"], f"{model_id} must state numerical limits"
        assert row["user_facing_statement"], f"{model_id} needs a user statement"


def test_spinless_models_say_unavailable_not_zero(
    models: dict[str, dict[str, Any]],
) -> None:
    """Rule from #9550 item 1: no implied zero-spin prediction."""
    for model_id, row in models.items():
        if row["tangential_spin"] == "unavailable":
            statement = row["user_facing_statement"].lower()
            assert "unavailable" in statement or "normal-only" in statement, (
                f"{model_id} lacks spin support but its statement does not say so"
            )
            assert "zero spin" not in statement and "no spin" not in statement, (
                f"{model_id} must not present missing spin support as a result"
            )


def test_work_items_carry_evidence_or_explicit_open_state(
    matrix: dict[str, Any],
) -> None:
    items = matrix["acceptance_work"]
    assert [i["item"] for i in items] == [1, 2, 3, 4, 5, 6]
    for item in items:
        assert item["status"] in WORK_STATUSES, item["item"]
        if item["status"] == "met":
            assert item["evidence"] and item["tests"], f"item {item['item']}"
            assert not item.get("blockers"), f"met item {item['item']} has blockers"
        else:
            assert item["blockers"], f"item {item['item']} is open with no blocker"
            assert item["owner"], f"item {item['item']} has no owner"
            assert item["plan"], f"item {item['item']} has no plan"


def test_release_claim_stays_bounded_while_work_is_outstanding(
    matrix: dict[str, Any],
) -> None:
    outstanding = [i for i in matrix["acceptance_work"] if i["status"] != "met"]
    assert matrix["predictive_accuracy_claim"] is False
    if outstanding:
        assert matrix["release_claim"] == "code_verified_only"
    for entry in matrix["closure"]:
        if entry["status"] != "met":
            assert entry["blockers"], f"closure {entry['id']} unmet without blockers"


def test_review_packet_names_every_model(models: dict[str, dict[str, Any]]) -> None:
    packet = PACKET_PATH.read_text(encoding="utf-8")
    for model_id in models:
        assert f"`{model_id}`" in packet, f"{PACKET_PATH.name} omits {model_id}"


# ---------------------------------------------------------------------------
# Library probes: the matrix must describe the code that ships (item 1 & 2)
# ---------------------------------------------------------------------------


def _solve(model_id: str, pre: PreImpactState, params: ImpactParameters | None = None):
    model = create_impact_model(ImpactModelType[model_id.upper()])
    return model.solve(pre, params or ImpactParameters())


@pytest.mark.parametrize("model_id", ["rigid_body", "spring_damper", "finite_time"])
def test_contact_row_matches_library(
    model_id: str, models: dict[str, dict[str, Any]]
) -> None:
    params = ImpactParameters()
    post = _solve(model_id, _lofted_strike(), params)
    rigid = _solve("rigid_body", _lofted_strike(), params)
    row = models[model_id]["contact"]
    if row == "instantaneous":
        assert post.contact_duration == 0.0
    elif row == "finite_duration_reported_only":
        assert post.contact_duration == params.contact_duration
        np.testing.assert_allclose(post.ball_velocity, rigid.ball_velocity)
    else:
        assert 0.0 < post.contact_duration < 0.005


@pytest.mark.parametrize("model_id", ["rigid_body", "spring_damper", "finite_time"])
def test_spin_row_matches_library(
    model_id: str, models: dict[str, dict[str, Any]]
) -> None:
    """Lofted strike has a tangential approach; supported models spin the ball."""
    post = _solve(model_id, _lofted_strike())
    spins = bool(np.linalg.norm(post.ball_angular_velocity) > 1.0)
    assert spins == (models[model_id]["tangential_spin"] == "supported")
    if spins:
        # Backspin about -y for a +x launch in a z-up frame (t x n axis).
        assert post.ball_angular_velocity[1] < 0.0
        assert abs(post.ball_angular_velocity[0]) < 1e-9
        assert abs(post.ball_angular_velocity[2]) < 1e-9


@pytest.mark.parametrize("model_id", ["rigid_body", "spring_damper", "finite_time"])
def test_inertia_row_matches_library(
    model_id: str, models: dict[str, dict[str, Any]]
) -> None:
    """Off-center strike reduces ball speed only where scalar MOI is used."""
    centered = _solve(model_id, _lofted_strike())
    toe = _solve(model_id, _lofted_strike(offset=[0.02, 0.0]))
    slower = np.linalg.norm(toe.ball_velocity) < np.linalg.norm(centered.ball_velocity)
    expected = models[model_id]["inertia_tensor"] == "scalar_moi_effective_mass"
    assert slower == expected
    # No model applies gear-effect spin on its own; the solver API overlays it.
    np.testing.assert_allclose(
        toe.ball_angular_velocity, centered.ball_angular_velocity
    )


@pytest.mark.parametrize("model_id", ["rigid_body", "spring_damper", "finite_time"])
def test_gear_effect_row_matches_solver_api(
    model_id: str, models: dict[str, dict[str, Any]]
) -> None:
    api = ImpactSolverAPI(ImpactModelType[model_id.upper()])
    post = api.solve_pre_impact_state(0.0, _lofted_strike(offset=[0.02, 0.0]), False)
    overlay = models[model_id]["gear_effect"] == "empirical_overlay_via_solver_api"
    assert overlay
    assert abs(post.ball_angular_velocity[2]) > 1.0  # toe offset -> vertical-axis spin


@pytest.mark.parametrize("model_id", ["rigid_body", "spring_damper", "finite_time"])
def test_cor_is_speed_independent(model_id: str) -> None:
    """Low- and high-speed golden: constant COR gives one smash factor."""
    low = np.linalg.norm(_solve(model_id, _lofted_strike(20.0)).ball_velocity) / 20.0
    high = np.linalg.norm(_solve(model_id, _lofted_strike(60.0)).ball_velocity) / 60.0
    assert low == pytest.approx(high, rel=1e-3)


@pytest.mark.parametrize("model_id", ["rigid_body", "spring_damper", "finite_time"])
def test_no_hit_momentum_and_passivity_goldens(model_id: str) -> None:
    still = _solve(model_id, _lofted_strike(0.0))
    np.testing.assert_allclose(still.ball_velocity, 0.0, atol=1e-12)
    np.testing.assert_allclose(still.ball_angular_velocity, 0.0, atol=1e-12)

    pre = _lofted_strike()
    post = _solve(model_id, pre)
    p_pre = pre.clubhead_mass * pre.clubhead_velocity
    p_post = (
        pre.clubhead_mass * post.clubhead_velocity
        + GOLF_BALL_MASS_KG * post.ball_velocity
    )
    np.testing.assert_allclose(p_post, p_pre, atol=1e-6)
    ke_pre = (
        0.5 * pre.clubhead_mass * np.dot(pre.clubhead_velocity, pre.clubhead_velocity)
    )
    ke_post = 0.5 * pre.clubhead_mass * np.dot(
        post.clubhead_velocity, post.clubhead_velocity
    ) + 0.5 * GOLF_BALL_MASS_KG * np.dot(post.ball_velocity, post.ball_velocity)
    assert ke_post < ke_pre


def test_spring_damper_converges_in_time_step() -> None:
    coarse = SpringDamperImpactModel(dt=1e-7).solve(
        _lofted_strike(), ImpactParameters()
    )
    fine = SpringDamperImpactModel(dt=5e-8).solve(_lofted_strike(), ImpactParameters())
    np.testing.assert_allclose(fine.ball_velocity, coarse.ball_velocity, rtol=1e-4)
    assert fine.contact_duration == pytest.approx(coarse.contact_duration, rel=1e-3)
