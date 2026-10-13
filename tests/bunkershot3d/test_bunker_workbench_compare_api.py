"""Contract tests for the v1 A/B comparison and record verification (#9545).

The comparison route is a thin client of
:meth:`~src.tools.bunker_shot_gui.model.WorkbenchModel.compare`, so it must
carry the **same** objective standing the model attaches (#9239) rather than
a vocabulary of its own:

* the response carries the standing's disposition and reasons verbatim;
* ``winner`` is ``null`` unless the standing is ``supported``, and any lead
  the bands do show is reported as ``exploratory_leader`` beside the reasons;
* malformed input -- a missing or duplicated design, an unknown field, a
  left-handed or predictive request -- is a precise 4xx before any solve.

A record also survives being saved, reloaded and handed back: the verify
route accepts it unchanged and refuses it after any edit (corruption is
rejected, not re-trusted).

The Rust flight kernel is optional, so most tests replace only the carry
grid of an otherwise real evaluation -- the bootstrap, the bands and the
standing still run for real.
"""

from __future__ import annotations

import dataclasses
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("fastapi")

from fastapi import FastAPI  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from bunkershot3d.ball import ObjectiveDisposition  # noqa: E402
from src.api.routes.bunker_workbench import (  # noqa: E402
    COMPARISON_SCHEMA_VERSION,
    SCHEMA_VERSION,
    record_digest,
    router,
)
from src.tools.bunker_shot_gui import model as model_module  # noqa: E402
from src.tools.bunker_shot_gui.model import WorkbenchModel  # noqa: E402
from src.tools.bunker_shot_gui.objective import (  # noqa: E402
    SHIPPED_CARRY_DISPOSITION,
    UNCALIBRATED_REASON,
)

pytestmark = [pytest.mark.contract, pytest.mark.integration]

COMPARE = "/tools/bunker-workbench/v1/compare"
EVALUATE = "/tools/bunker-workbench/v1/evaluate"
VERIFY = "/tools/bunker-workbench/v1/verify"

TARGET_CARRY_M = 12.0
"""The workbench default target, which the comparison is judged against."""

PAIR: dict[str, Any] = {
    "handedness": "right",
    "left": {"name": "A", "grind_preset": "sm9_58_m"},
    "right": {"name": "B", "grind_preset": "sm9_58_m", "marketed_bounce_deg": 13.0},
    "sand": {"preset": "firm"},
    "swing": {"clubhead_speed_mps": 25.0, "attack_angle_deg": -6.0},
}

EXPLORATORY_OBJECTIVE: dict[str, Any] = {
    "use": "exploratory",
    "target_carry_m": TARGET_CARRY_M,
    "tolerance_fraction": 0.1,
}


def _separated_grids() -> dict[str, np.ndarray]:
    """Finite carry grids: ``A`` inside the 1.2 m window, ``B`` far outside.

    One cell of each is nudged so neither grid is constant.
    """
    grids = {
        "A": np.full((2, 2), TARGET_CARRY_M - 0.5),
        "B": np.full((2, 2), TARGET_CARRY_M - 10.0),
    }
    for grid in grids.values():
        grid[0, 0] -= 0.1
    return grids


@pytest.fixture(scope="module")
def client() -> TestClient:
    """A bare app carrying only the workbench router."""
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


@pytest.fixture
def ranked_grids(monkeypatch: pytest.MonkeyPatch) -> None:
    """Kernel-free ranked path: a real nominal shot, a substituted carry grid."""
    real_evaluate = WorkbenchModel.evaluate
    grids = _separated_grids()

    def evaluate(self, design, sand, swing, *, include_playability=True):  # type: ignore[no-untyped-def]
        evaluation = real_evaluate(self, design, sand, swing, include_playability=False)
        playability = dataclasses.replace(
            evaluation.playability,
            carry_m=grids[design.name],
            carry_verdict=evaluation.verdict,
        )
        return dataclasses.replace(evaluation, playability=playability)

    monkeypatch.setattr(WorkbenchModel, "evaluate", evaluate)


@pytest.fixture(scope="module")
def unranked(client: TestClient) -> dict[str, Any]:
    """The real comparison, run once: no kernel locally, so often unranked."""
    response = client.post(COMPARE, json=PAIR)
    assert response.status_code == 200, response.text
    return response.json()


def _detail_code(response: Any) -> str:
    """The machine-readable error code of a route-raised 4xx."""
    return str(response.json()["detail"]["code"])


# --------------------------------------------------------------- versioning


def test_compare_and_verify_are_versioned_under_the_tools_prefix() -> None:
    paths = {route.path for route in router.routes}
    assert {COMPARE, VERIFY} <= paths


# ------------------------------------------------- the real (shipped) pair


def test_the_shipped_comparison_names_no_winner(unranked: dict[str, Any]) -> None:
    """Whatever the kernel did, the uncalibrated objective names nobody."""
    assert unranked["schema_version"] == COMPARISON_SCHEMA_VERSION
    assert unranked["winner"] is None
    assert unranked["predictive"] is False
    standing = unranked["objective_standing"]
    assert standing["disposition"] == SHIPPED_CARRY_DISPOSITION.value
    assert UNCALIBRATED_REASON in standing["reasons"]
    assert standing["predictive"] is False


def test_each_side_is_a_full_stamped_evaluation_record(
    unranked: dict[str, Any],
) -> None:
    for side, name in (("left", "A"), ("right", "B")):
        record = unranked[side]
        assert record["schema_version"] == SCHEMA_VERSION
        assert record["inputs"]["design"]["name"] == name
        assert record["stamp"]
        body = {key: value for key, value in record.items() if key != "digest"}
        assert record_digest(body) == record["digest"]


def test_the_comparison_digest_covers_the_whole_record(
    unranked: dict[str, Any],
) -> None:
    assert re.fullmatch(r"[0-9a-f]{64}", unranked["digest"])
    body = {key: value for key, value in unranked.items() if key != "digest"}
    assert record_digest(body) == unranked["digest"]


def test_an_unranked_comparison_says_why(unranked: dict[str, Any]) -> None:
    ranking = unranked["ranking"]
    if ranking["available"]:
        pytest.skip("the flight kernel is installed, so the pair was ranked")
    assert ranking["unavailable_reason"]
    assert unranked["exploratory_leader"] is None
    assert unranked["verdict_statement"] == ranking["unavailable_reason"]


# ------------------------------------------------------- the ranked path


@pytest.mark.usefixtures("ranked_grids")
def test_a_decided_uncalibrated_ranking_is_exploratory_only(
    client: TestClient,
) -> None:
    body = client.post(COMPARE, json=PAIR).json()
    ranking = body["ranking"]
    assert ranking["available"] is True
    assert ranking["shared_points"] == 4
    assert ranking["banded_verdict"] == "a-better"
    assert body["winner"] is None
    assert body["exploratory_leader"] == "A"
    assert body["predictive"] is False
    assert body["verdict_statement"].startswith("EXPLORATORY")
    standing = body["objective_standing"]
    assert standing["disposition"] == SHIPPED_CARRY_DISPOSITION.value
    assert standing["reasons"] == [UNCALIBRATED_REASON]


@pytest.mark.usefixtures("ranked_grids")
def test_the_route_reports_the_model_standing_not_its_own(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Were a qualification shipped, the same route would name the winner.

    The route must not hard-code the refusal: it carries whatever standing
    the model attached, so a supported objective passes straight through.
    """
    monkeypatch.setattr(
        model_module, "SHIPPED_CARRY_DISPOSITION", ObjectiveDisposition.SUPPORTED
    )
    body = client.post(COMPARE, json=PAIR).json()
    assert body["objective_standing"] == {
        "disposition": ObjectiveDisposition.SUPPORTED.value,
        "reasons": [],
        "predictive": True,
    }
    assert body["predictive"] is True
    assert body["winner"] == "A"
    assert not body["verdict_statement"].startswith("EXPLORATORY")


@pytest.mark.usefixtures("ranked_grids")
def test_an_exploratory_objective_sets_the_window_the_standing_judges(
    client: TestClient,
) -> None:
    """A 2 % window (0.24 m) is empty for both grids: the reason says so."""
    objective = {**EXPLORATORY_OBJECTIVE, "tolerance_fraction": 0.02}
    body = client.post(COMPARE, json={**PAIR, "objective": objective}).json()
    assert body["objective"] == objective
    reasons = body["objective_standing"]["reasons"]
    assert any("playability window is empty" in reason for reason in reasons)
    assert body["winner"] is None


# ------------------------------------------------------ fail-closed input


def test_two_designs_with_one_name_are_refused(client: TestClient) -> None:
    right = {**PAIR["right"], "name": "A"}
    response = client.post(COMPARE, json={**PAIR, "right": right})
    assert response.status_code == 422
    assert _detail_code(response) == "design_names_not_distinct"


@pytest.mark.parametrize("missing", ["left", "right", "handedness"])
def test_a_missing_section_is_rejected(client: TestClient, missing: str) -> None:
    request = {key: value for key, value in PAIR.items() if key != missing}
    response = client.post(COMPARE, json=request)
    assert response.status_code == 422
    assert any(error["loc"][-1] == missing for error in response.json()["detail"])


def test_an_unknown_field_is_rejected(client: TestClient) -> None:
    response = client.post(COMPARE, json={**PAIR, "winner": "A"})
    assert response.status_code == 422
    assert any(error["loc"][-1] == "winner" for error in response.json()["detail"])


def test_an_out_of_range_rival_is_rejected(client: TestClient) -> None:
    right = {**PAIR["right"], "loft_deg": 0.5}
    response = client.post(COMPARE, json={**PAIR, "right": right})
    assert response.status_code == 422
    assert any(error["loc"][-1] == "loft_deg" for error in response.json()["detail"])


def test_a_left_handed_comparison_is_refused(client: TestClient) -> None:
    response = client.post(COMPARE, json={**PAIR, "handedness": "left"})
    assert response.status_code == 422
    assert _detail_code(response) == "handedness_unsupported"


def test_a_predictive_comparison_is_refused_with_the_standing(
    client: TestClient,
) -> None:
    objective = {**EXPLORATORY_OBJECTIVE, "use": "predictive"}
    response = client.post(COMPARE, json={**PAIR, "objective": objective})
    assert response.status_code == 422
    detail = response.json()["detail"]
    assert detail["code"] == "objective_not_predictive"
    assert detail["disposition"] == SHIPPED_CARRY_DISPOSITION.value


def test_an_unconstructible_rival_is_a_precise_422(client: TestClient) -> None:
    right = {**PAIR["right"], "grind_preset": "no_such_grind"}
    response = client.post(COMPARE, json={**PAIR, "right": right})
    assert response.status_code == 422
    assert _detail_code(response) == "invalid_workbench_input"


# ------------------------------------------ save -> reload -> verify/export


def _save_and_reload(record: dict[str, Any], path: Path) -> dict[str, Any]:
    """Write a record as a client would, then read it back."""
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.usefixtures("ranked_grids")
def test_a_saved_comparison_reloads_and_verifies(
    client: TestClient, tmp_path: Path
) -> None:
    record = client.post(COMPARE, json=PAIR).json()
    reloaded = _save_and_reload(record, tmp_path / "comparison.json")
    assert reloaded == record
    response = client.post(VERIFY, json=reloaded)
    assert response.status_code == 200, response.text
    assert response.json() == {
        "valid": True,
        "schema_version": COMPARISON_SCHEMA_VERSION,
        "digest": record["digest"],
    }
    # The standing, units and frame survive the round trip unchanged.
    assert reloaded["winner"] is None
    assert reloaded["objective_standing"] == record["objective_standing"]
    assert reloaded["conventions"]["handedness"] == "right"
    assert reloaded["conventions"]["units"]["target_carry_m"] == "m"


def test_a_saved_evaluation_reloads_and_verifies(
    client: TestClient, tmp_path: Path
) -> None:
    request = {
        "handedness": "right",
        "design": PAIR["left"],
        "sand": PAIR["sand"],
        "swing": PAIR["swing"],
    }
    record = client.post(EVALUATE, json=request).json()
    reloaded = _save_and_reload(record, tmp_path / "evaluation.json")
    response = client.post(VERIFY, json=reloaded)
    assert response.status_code == 200, response.text
    assert response.json()["schema_version"] == SCHEMA_VERSION


@pytest.mark.usefixtures("ranked_grids")
def test_a_promoted_leader_is_rejected_as_corrupt(
    client: TestClient, tmp_path: Path
) -> None:
    """Editing ``winner`` into a saved exploratory record breaks its digest."""
    record = _save_and_reload(
        client.post(COMPARE, json=PAIR).json(), tmp_path / "comparison.json"
    )
    record["winner"] = record["exploratory_leader"]
    response = client.post(VERIFY, json=record)
    assert response.status_code == 422
    assert _detail_code(response) == "record_digest_mismatch"


@pytest.mark.parametrize(
    ("edit", "code"),
    [
        ({"digest": None}, "record_digest_invalid"),
        ({"digest": "0" * 63}, "record_digest_invalid"),
        ({"schema_version": "bunker-workbench-evaluation/0"}, "record_schema_unknown"),
        ({"schema_version": None}, "record_schema_unknown"),
    ],
)
def test_a_malformed_record_is_refused(
    client: TestClient, edit: dict[str, Any], code: str
) -> None:
    record = {"schema_version": SCHEMA_VERSION, "digest": "0" * 64, **edit}
    record = {key: value for key, value in record.items() if value is not None}
    response = client.post(VERIFY, json=record)
    assert response.status_code == 422
    assert _detail_code(response) == code


def test_a_non_object_record_is_rejected(client: TestClient) -> None:
    response = client.post(VERIFY, json=["not", "a", "record"])
    assert response.status_code == 422
