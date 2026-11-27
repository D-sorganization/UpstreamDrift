"""A schema-versioned model card in every clip manifest (#15150, E2)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.same_input import InputBundle
from src.sim_run.native_export_jobs import manifest_of
from src.tools.native_viewer_export import model_card as mc
from src.tools.native_viewer_export.core import ExportSettings, SwingInput
from src.tools.native_viewer_export.core import export_swing

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

COORDS = ("A", "B")
FIELDS = (
    "replay_kind",
    "fit_error",
    "golfer_model",
    "club",
    "shot_setup",
    "impact_model",
)


def _swing(
    spec_extra: dict | None = None,
    provenance: dict | None = None,
    efforts: float = 0.0,
) -> SwingInput:
    spec = {"coordinate_order": list(COORDS), **(spec_extra or {})}
    q = np.zeros((101, 2))
    bundle = InputBundle(
        spec_bytes=json.dumps(spec).encode(),
        coordinate_order=COORDS,
        dt_s=0.001,
        q0=q[0],
        v0=q[0],
        efforts=np.full((100, 2), efforts),
        reference_q=q,
        reference_v=q.copy(),
        reference_engine="mujoco",
        provenance=provenance or {},
    )
    return SwingInput(bundle, bundle.reference_q, "driver", "Driver", "mujoco")


def _chain_spec() -> dict:
    shaft = {
        "model": "chain",
        "flex": "stiff",
        "flex_cpm": 258.0,
        "n_links": 6,
        "profile_status": "default_not_measured",
    }
    return {"club": {"name": "Driver", "shaft": shaft}}


def test_card_has_every_schema_key_and_a_provenance_tag() -> None:
    card = mc.model_card_of(_swing(), ExportSettings(), None).to_dict()
    assert card["schema"] == mc.load_card_settings()["schema"]
    assert set(card) == {"schema", *FIELDS}
    for name in FIELDS:
        entry = card[name]
        assert set(entry) == {"value", "tag", "source"}
        assert entry["tag"] in mc.TAGS
        assert entry["source"].strip()
    assert set(mc.TAGS) == {"measured", "fitted", "assumed", "unknown"}


def test_missing_data_is_an_explicit_unknown() -> None:
    """A bare bundle has no fit error, club record or impact: all ``unknown``."""
    card = mc.model_card_of(_swing(efforts=1.0), ExportSettings(), None).to_dict()
    for name in ("replay_kind", "fit_error", "club", "impact_model"):
        assert card[name]["value"] == mc.UNKNOWN, name
        assert card[name]["tag"] == "unknown", name
    assert card["shot_setup"]["value"]["ball"] == mc.UNKNOWN


def test_kinematic_replay_motion_is_measured() -> None:
    card = mc.model_card_of(_swing(), ExportSettings(), None)
    assert card.replay_kind.value == "kinematic_replay"
    assert card.replay_kind.tag == "measured"


def test_tracker_fit_error_is_fitted_with_its_key_as_source() -> None:
    prov = {"controller": "computed-torque", "max_pose_drift_m": 0.004}
    card = mc.model_card_of(
        _swing(provenance=prov, efforts=1.0), ExportSettings(), None
    )
    assert card.replay_kind.value == "feedback_tracked_dynamics"
    assert card.fit_error.value == {"max_pose_drift_m": 0.004}
    assert card.fit_error.tag == "fitted"
    assert "max_pose_drift_m" in card.fit_error.source


def test_nested_kinetic_match_error_is_read() -> None:
    prov = {
        "replay_mode": "inverse_dynamics",
        "kinetic_match": {"closure_gap_max_m": 1e-5},
    }
    card = mc.model_card_of(
        _swing(provenance=prov, efforts=1.0), ExportSettings(), None
    )
    assert card.fit_error.value == {"kinetic_match.closure_gap_max_m": 1e-5}


def test_chain_shaft_flex_is_assumed_from_the_spec() -> None:
    card = mc.model_card_of(_swing(_chain_spec()), ExportSettings(), None)
    value = card.club.value
    assert value["club"] == "Driver" and value["shaft_model"] == "chain"
    assert value["flex"] == "stiff" and value["flex_cpm"] == 258.0
    assert card.club.tag == "assumed"
    assert "default_not_measured" in card.club.source


def test_rigid_club_without_shaft_record_is_rigid() -> None:
    card = mc.model_card_of(
        _swing({"club": {"name": "Driver"}}), ExportSettings(), None
    )
    assert card.club.value["shaft_model"] == "rigid"
    assert card.club.value["flex"] == mc.UNKNOWN


def test_shot_setup_records_settings_and_ball() -> None:
    settings = ExportSettings(ball_position_shift_m=0.02, club_roll_in_hands_deg=1.5)
    ball = {"position_m": [0.1, 0.0, 0.0213], "source": "strike_geometry"}
    card = mc.model_card_of(_swing(), settings, ball)
    value = card.shot_setup.value
    assert value["ball_position_shift_m"] == 0.02
    assert value["club_roll_in_hands_deg"] == 1.5
    assert value["shot_shape"] == mc.UNKNOWN
    assert value["ball"]["source"] == "strike_geometry"
    assert value["ball"]["ground_height_m"] == 0.0
    assert card.shot_setup.tag == "assumed"


def test_recorded_ball_impact_names_the_impact_model() -> None:
    impact = {"cor": 0.78, "ball_mass_kg": 0.0459, "frame": "Clubhead"}
    card = mc.model_card_of(
        _swing(provenance={"ball_impact": impact}), ExportSettings(), None
    )
    assert card.impact_model.value["bundle_impact"]["cor"] == 0.78
    assert card.impact_model.value["flight_launch"] == mc.UNKNOWN
    assert card.impact_model.tag == "assumed"


def test_json_round_trip_and_schema_check() -> None:
    card = mc.model_card_of(_swing(_chain_spec()), ExportSettings(), None)
    text = json.dumps(card.to_dict())
    assert mc.ModelCard.from_dict(json.loads(text)) == card
    bad = {**card.to_dict(), "schema": "native-export-model-card/v0"}
    with pytest.raises(ValueError, match="schema"):
        mc.ModelCard.from_dict(bad)
    missing = {k: v for k, v in card.to_dict().items() if k != "club"}
    with pytest.raises(ValueError, match="club"):
        mc.ModelCard.from_dict(missing)


def test_card_field_rejects_a_bad_tag_or_source() -> None:
    with pytest.raises(ValueError, match="tag"):
        mc.CardField(1.0, "guessed", "x")
    with pytest.raises(ValueError, match="source"):
        mc.CardField(1.0, "measured", " ")


class _Writer:
    def append_data(self, frame: np.ndarray) -> None:
        pass

    def close(self) -> None:
        pass


class _FakeBackend:
    engine = "fake"

    def unavailable_reason(self) -> str | None:
        return None

    def render(self, swing, settings, indices, overlay, ball=None):
        for _k in indices:
            yield {
                v: np.full((settings.height, settings.width, 3), 40, np.uint8)
                for v in settings.views
            }


def test_card_is_in_the_manifest_the_exporter_writes(tmp_path: Path) -> None:
    settings = ExportSettings(
        views=("overhead",), multiview=False, width=64, height=48, speeds=(1.0,)
    )
    result = export_swing(
        _FakeBackend(),
        _swing(),
        settings,
        tmp_path,
        writer_factory=lambda p, fps: _Writer(),
    )
    row = json.loads(json.dumps(manifest_of([result]), default=str))["engines"][0]
    card = mc.ModelCard.from_dict(row["model_card"])
    assert card.replay_kind.value == row["replay_mode"] == "kinematic_replay"
