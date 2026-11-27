"""The current golfer model and its deprecated predecessors (#15648).

The skinned human hands and skinned body are the main golfer model. The
registry names them, marks the capsule hands and loft body deprecated, and a
spec that still resolves to a deprecated part says so, so neither people nor
agents pick the old model by accident.
"""

from __future__ import annotations

import copy
import json
import logging
from pathlib import Path

import pytest

from src.shared.python.model_appearance import golfer_model_versions as gmv
from src.shared.python.model_appearance.body_skin_export import (
    LOFT_BACKEND,
    SKINNED_BACKEND,
    body_skin_enabled,
)
from src.shared.python.model_appearance.hand_spec import (
    HAND_VISUAL_BACKENDS,
    resolved_hand_spec,
)

pytestmark = pytest.mark.unit
REPO = Path(__file__).resolve().parents[3]
SPEC = json.loads(
    (REPO / "tests/fixtures/club_face/spec_driver.json").read_text(encoding="utf-8")
)


def _current_spec() -> dict:
    spec = copy.deepcopy(SPEC)
    spec["subject"]["hand_visual_backend"] = "skinned_human"
    spec["subject"]["body_visual_backend"] = "skinned_cc0"
    return spec


def test_current_model_is_the_skinned_hands_and_body() -> None:
    current = gmv.current_golfer_model()
    assert current.hand_visual_backend == "skinned_human" in HAND_VISUAL_BACKENDS
    assert current.body_visual_backend == SKINNED_BACKEND
    assert current.spec_revision_configuration == "skinned_hand_wrists"


def test_the_tools_that_make_the_current_model_exist() -> None:
    current = gmv.current_golfer_model()
    for reference in (current.make_with, current.refit_poses_with, current.docs):
        assert (REPO / reference.split("::")[0]).is_file(), reference


def test_capsule_hands_and_loft_body_are_deprecated_with_replacements() -> None:
    parts = {(p.field, p.value): p for p in gmv.deprecated_values()}
    capsule = parts[("hand_visual_backend", "capsule")]
    loft = parts[("body_visual_backend", LOFT_BACKEND)]
    assert capsule.replacement == "skinned_human"
    assert loft.replacement == SKINNED_BACKEND
    assert capsule.reason and capsule.still_allowed_for
    assert loft.reason and loft.still_allowed_for


def test_a_spec_without_backends_resolves_to_both_deprecated_parts() -> None:
    found = {(p.field, p.value) for p in gmv.deprecated_parts(SPEC)}
    assert found == {
        ("hand_visual_backend", "capsule"),
        ("body_visual_backend", "loft"),
    }


def test_a_current_model_spec_has_no_deprecated_parts() -> None:
    assert gmv.deprecated_parts(_current_spec()) == ()


def test_resolving_old_hands_and_body_logs_a_deprecation(caplog) -> None:
    gmv.reset_deprecation_warnings()
    with caplog.at_level(logging.WARNING):
        resolved_hand_spec(SPEC)
        body_skin_enabled(SPEC)
    text = caplog.text
    assert "deprecated" in text and "capsule" in text and "loft" in text
    assert "golfer_model_versions.json" in text


def test_the_deprecation_is_logged_once_per_part(caplog) -> None:
    gmv.reset_deprecation_warnings()
    with caplog.at_level(logging.WARNING):
        for _ in range(3):
            resolved_hand_spec(SPEC)
    assert caplog.text.count("'capsule'") == 1


def test_resolving_the_current_model_logs_nothing(caplog) -> None:
    gmv.reset_deprecation_warnings()
    with caplog.at_level(logging.WARNING):
        body_skin_enabled(_current_spec())
    assert "deprecated" not in caplog.text


def test_registry_rejects_an_unknown_current_backend(tmp_path) -> None:
    raw = json.loads(gmv.SETTINGS_PATH.read_text(encoding="utf-8"))
    raw["current"]["hand_visual_backend"] = "mittens"
    bad = tmp_path / "versions.json"
    bad.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises(ValueError, match="mittens"):
        gmv.load_golfer_model_versions(bad)


def test_agent_docs_point_at_the_registry() -> None:
    agents = (REPO / "AGENTS.md").read_text(encoding="utf-8")
    assert "src/config/golfer_model_versions.json" in agents
