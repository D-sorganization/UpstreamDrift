"""Public authenticated extraction with independent screw velocity expectations."""

import json
from dataclasses import replace

import numpy as np
import pytest
from impact_fixture import impact_case
from src.shared.python.workspace import necromatcher_impact as owner
from test_necromatcher_impact_contracts import geometry, selection

pytestmark = pytest.mark.unit


def test_extracts_exact_screw_velocity_and_authored_clock(monkeypatch) -> None:
    library = impact_case(monkeypatch)
    g = replace(geometry(), local_head_point_m=(0.1, 0, 0))
    state = owner.extract_replay_impact_state(library, "replay", g, selection())
    # At pi/2, point [0,.1,0], translating +2m/s and rotating +3rad/s.
    np.testing.assert_allclose(state.clubhead_velocity, [1.7, 0, 0], atol=1e-14)
    np.testing.assert_allclose(state.clubhead_angular_velocity, [0, 0, 3], atol=1e-14)
    np.testing.assert_allclose(state.clubhead_orientation, [0, 1, 0], atol=1e-14)
    assert state.metadata["recorded_time_s"] == 2.1
    assert state.metadata["physical_source_time_qualified"] is False
    assert state.metadata["scientific_qualified"] is False
    assert state.metadata["frame_id"] == "flight_xfwd_yleft_zup"
    assert library.active is False


@pytest.mark.parametrize("fault", ["labels", "pose", "nonrigid"])
def test_rejects_inconsistent_native_marker_contract(monkeypatch, fault: str) -> None:
    library = impact_case(monkeypatch)
    plant = library.binding.plant
    original_factory = plant.create_marker_linearizer

    def factory(attachments):
        original = original_factory(attachments)

        class FaultyLinearizer:
            def marker_linearization(self, q):
                row = original.marker_linearization(q)
                if fault == "labels":
                    return replace(
                        row, marker_labels=tuple(reversed(row.marker_labels))
                    )
                if fault == "nonrigid":
                    jacobian = row.jacobian.copy()
                    jacobian[1, :, 0] += row.positions[1] - row.positions[0]
                    return replace(row, jacobian=jacobian)
                return row

        return FaultyLinearizer()

    monkeypatch.setattr(plant, "create_marker_linearizer", factory)
    if fault == "pose":
        original_pose = plant.frame_poses

        def wrong_pose(mapping, q):
            rotation, translation = original_pose(mapping, q)["club"]
            return {"club": (rotation, translation + 1)}

        monkeypatch.setattr(plant, "frame_poses", wrong_pose)
    with pytest.raises(ValueError):
        owner.extract_replay_impact_state(library, "replay", geometry(), selection())
    assert library.active is False


def test_fresh_parent_check_rejects_mid_extraction_mutation(monkeypatch) -> None:
    library = impact_case(monkeypatch)
    original = library.binding.plant.frame_poses

    def mutated_pose(mapping, q):
        library.assets["profile"].metadata["hash"] = "sha256:" + "f" * 64
        return original(mapping, q)

    monkeypatch.setattr(library.binding.plant, "frame_poses", mutated_pose)
    with pytest.raises(ValueError, match="hash differs"):
        owner.extract_replay_impact_state(library, "replay", geometry(), selection())
    assert library.active is False


def test_recorded_zero_rates_and_source_identity_are_not_pts_derivatives(
    monkeypatch,
) -> None:
    library = impact_case(monkeypatch)
    library.trace.v[:] = 0
    library.trace.meta["source_frame_json"] = json.dumps({"pts_numerator": 999999})
    state = owner.extract_replay_impact_state(
        library, "replay", geometry(), selection()
    )
    np.testing.assert_array_equal(state.clubhead_velocity, np.zeros(3))
    np.testing.assert_array_equal(state.clubhead_angular_velocity, np.zeros(3))
    assert state.metadata["capture_initial_frame"] == {"pts_numerator": 999999}
    assert state.metadata["capture_pts_used_for_velocity"] is False
    json.dumps(state.metadata, allow_nan=False)


def test_rotates_all_vectors_without_translating_them(monkeypatch) -> None:
    library = impact_case(monkeypatch)
    rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]])
    s = replace(
        selection(),
        world_to_flight_rotation=rotation,
        world_to_flight_translation_m=(99, 98, 97),
    )
    g = replace(geometry(), local_head_point_m=(0.1, 0, 0))
    state = owner.extract_replay_impact_state(library, "replay", g, s)
    np.testing.assert_allclose(state.clubhead_velocity, [0, 0, -1.7], atol=1e-14)
    np.testing.assert_allclose(state.clubhead_angular_velocity, [3, 0, 0], atol=1e-14)
    np.testing.assert_allclose(state.clubhead_orientation, [0, 1, 0], atol=1e-14)


@pytest.mark.parametrize(
    "fault", ["hash", "units", "order", "sample", "body", "capability"]
)
def test_wrong_bound_inputs_rejected(monkeypatch, fault: str) -> None:
    library = impact_case(monkeypatch)
    g, s = geometry(), selection()
    if fault == "hash":
        library.trace.meta["fit_hash"] = "sha256:" + "f" * 64
    elif fault == "units":
        library.trace.meta["coordinate_units_json"] = json.dumps(["rad", "rad"])
    elif fault == "order":
        library.trace.meta["coordinate_order_json"] = json.dumps(
            ["rotation", "translation"]
        )
    elif fault == "sample":
        s = replace(s, recorded_sample_index=2)
    elif fault == "body":
        g = replace(g, body="not-club")
    else:
        library.binding = replace(library.binding, plant=object())
    with pytest.raises((ValueError, IndexError)):
        owner.extract_replay_impact_state(library, "replay", g, s)
    assert library.active is False


@pytest.mark.parametrize(
    "key", ["fit_id", "coordinate_order_json", "source_frame_json"]
)
def test_rejects_nonstring_parent_and_json_metadata(monkeypatch, key: str) -> None:
    library = impact_case(monkeypatch)
    library.trace.meta[key] = []
    with pytest.raises(ValueError, match="string"):
        owner.extract_replay_impact_state(library, "replay", geometry(), selection())
    assert library.active is False
