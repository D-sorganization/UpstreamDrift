"""Deterministic refinement orchestration uses a fake endpoint engine only."""

from types import SimpleNamespace
import pytest

pytestmark = pytest.mark.unit


def _api():
    from docs.research.shot_pattern_analysis.phase2.validation import (
        supplemental_refinement as api,
    )

    return api


def test_tail_grid_is_predeclared_and_contains_axis_and_joint_extremes():
    cases = _api().tail_cases()
    assert len(cases) == 37
    offsets = [tuple(row["offsets"]) for row in cases]
    assert len(set(offsets)) == 37
    assert (6.0, 0, 0, 0, 0) in offsets
    assert (-6.0, 0, 0, 0, 0) in offsets
    assert any(all(v != 0 for v in row) for row in offsets)


def test_refinement_uses_rematched_nominals_and_exact_primary_setting():
    api = _api()
    seen = []

    class Engine:
        def simulate_delivery(self, face, path, config, **kwargs):
            seen.append((face, path, config, kwargs))
            return SimpleNamespace(
                outcome=SimpleNamespace(carry_x_m=100 + config.dt_s, carry_y_m=0.0)
            )

    config = api.MatchedConfig()
    nominal = {"face_deg": 4.8, "path_deg": 7.8, "speed_mps": 56.345}
    row = api.evaluate_case(
        Engine(), config, nominal, [6, 1.5, 2.68224, 0.01905, -0.01905]
    )
    assert row["comparisons"]["primary_to_fine"]["pass"] is True
    assert len(seen) == 6
    assert seen[-1][2].steps_per_compression == 128 and seen[-1][2].dt_s == 0.02
    assert seen[0][2].club_speed_mps == pytest.approx(59.02724)
    assert seen[0][1] == pytest.approx(9.3)
    assert seen[0][3]["shaft_rotation_deg"] == 6


@pytest.mark.parametrize("inconsistent", [False, True])
def test_unsupported_statuses_are_retained_not_resampled(inconsistent):
    api = _api()

    class Engine:
        def simulate_delivery(self, face, path, config, **kwargs):
            if not inconsistent or config.steps_per_compression == 128:
                raise api.UnsupportedShot("nonairborne_launch")
            return SimpleNamespace(outcome=SimpleNamespace(carry_x_m=100, carry_y_m=0))

    row = api.evaluate_case(
        Engine(),
        api.MatchedConfig(),
        {"face_deg": 0, "path_deg": 0, "speed_mps": 45},
        [0] * 5,
    )
    assert row["comparisons"]["primary_to_fine"]["pass"] is (
        False if inconsistent else None
    )


def test_unqualified_source_review_rejects_before_any_flight(tmp_path):
    import hashlib
    import json

    review = tmp_path / "review.json"
    review.write_text(json.dumps({"status": "pending", "new_execution": {}}))
    manifest = tmp_path / "inputs.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "supplement-refinement-inputs/1",
                "execution": {},
                "source_review": {
                    "path": "review.json",
                    "sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
                },
            }
        )
    )
    with pytest.raises(ValueError, match="source-transition review"):
        _api().load_qualified_nominals(manifest)


def test_changed_bound_evidence_rejects_before_any_flight(tmp_path):
    import json

    (tmp_path / "review.json").write_text("{}")
    manifest = tmp_path / "inputs.json"
    manifest.write_text(
        json.dumps(
            {
                "schema": "supplement-refinement-inputs/1",
                "source_review": {"path": "review.json", "sha256": "0" * 64},
            }
        )
    )
    with pytest.raises(ValueError, match="evidence hash mismatch"):
        _api().load_qualified_nominals(manifest)
