"""Roll-model provenance gates for the putting-green engine (ADR-0045 F1, #9343).

ADR-0045 preserves two putting roll models and makes the choice explicit
instead of silent: UpstreamDrift's agronomic law is ``ud-legacy-roll/1`` and
the Tools stimpmeter law is ``usga-stimp-roll/1``.  Every result document the
``putting_green`` engine emits must name the model that produced it, readers
of those documents must refuse an unnamed payload (fail-closed), and the
physics itself must be untouched by the plumbing (bit-identical regression
pins below).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.engines.physics_engines.putting_green.python._checkpoint import (
    PUTTING_CHECKPOINT_SCHEMA_VERSION,
    CheckpointProvenance,
    read_checkpoint_provenance,
)
from src.engines.physics_engines.putting_green.python._practice_mode import (
    load_result,
)
from src.engines.physics_engines.putting_green.python.ball_roll_physics import (
    ROLL_MODEL_FIELD,
    UD_LEGACY_ROLL_MODEL,
    USGA_STIMP_ROLL_MODEL,
    BallRollPhysics,
    BallState,
    RollModelProvenanceError,
    require_roll_model,
)
from src.engines.physics_engines.putting_green.python.green_surface import GreenSurface
from src.engines.physics_engines.putting_green.python.putter_stroke import (
    StrokeParameters,
)
from src.engines.physics_engines.putting_green.python.simulator import (
    PuttingGreenSimulator,
    SimulationConfig,
    SimulationResult,
)
from src.engines.physics_engines.putting_green.python.turf_properties import (
    TurfProperties,
)
from src.shared.python.engine_core.checkpoint import StateCheckpoint

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------


def _pinned_simulator() -> PuttingGreenSimulator:
    """Build the pinned regression simulator (never change these numbers)."""
    green = GreenSurface(
        width=20.0, height=20.0, turf=TurfProperties(stimp_rating=10.0)
    )
    green.set_hole_position(np.array([10.0, 15.0]))
    return PuttingGreenSimulator(
        green=green,
        config=SimulationConfig(record_trajectory=True),
        random_seed=8345,
    )


def _feed(digest: Any, *arrays: Any) -> None:
    """Feed float64 arrays into a hash in a shape-sensitive, stable order."""
    for array in arrays:
        contiguous = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
        digest.update(repr(contiguous.shape).encode("utf-8"))
        digest.update(contiguous.tobytes())


@pytest.fixture
def simulator() -> PuttingGreenSimulator:
    green = GreenSurface(
        width=20.0, height=20.0, turf=TurfProperties(stimp_rating=10.0)
    )
    green.set_hole_position(np.array([10.0, 12.0]))
    return PuttingGreenSimulator(green=green, config=SimulationConfig())


@pytest.fixture
def putt_result(simulator: PuttingGreenSimulator) -> SimulationResult:
    stroke = StrokeParameters(speed=1.4, direction=np.array([0.0, 1.0]))
    return simulator.simulate_putt(stroke, ball_position=np.array([10.0, 8.0]))


# ---------------------------------------------------------------------------
# The names themselves
# ---------------------------------------------------------------------------


class TestRollModelNames:
    """The two ADR-0045 model identifiers are stable, distinct strings."""

    def test_ud_legacy_identifier_is_pinned(self) -> None:
        assert UD_LEGACY_ROLL_MODEL == "ud-legacy-roll/1"

    def test_usga_counterpart_identifier_is_pinned(self) -> None:
        assert USGA_STIMP_ROLL_MODEL == "usga-stimp-roll/1"

    def test_models_are_distinct(self) -> None:
        assert UD_LEGACY_ROLL_MODEL != USGA_STIMP_ROLL_MODEL

    def test_field_name_is_pinned(self) -> None:
        assert ROLL_MODEL_FIELD == "roll_model"

    def test_contract_is_documented_beside_the_physics(self) -> None:
        """The constant carries the ADR-0045 contract in its own module."""
        from src.engines.physics_engines.putting_green.python import ball_roll_physics

        text = ball_roll_physics.__doc__ or ""
        assert "ADR-0045" in text
        assert USGA_STIMP_ROLL_MODEL in text
        assert "2.854" in text


# ---------------------------------------------------------------------------
# Fail-closed reader gate
# ---------------------------------------------------------------------------


class TestRequireRollModel:
    """``require_roll_model`` is the single fail-closed reader gate."""

    def test_accepts_named_document(self) -> None:
        document = {"holed": True, ROLL_MODEL_FIELD: UD_LEGACY_ROLL_MODEL}
        assert require_roll_model(document, source="test") == UD_LEGACY_ROLL_MODEL

    def test_accepts_the_counterpart_model(self) -> None:
        document = {ROLL_MODEL_FIELD: USGA_STIMP_ROLL_MODEL}
        assert require_roll_model(document, source="test") == USGA_STIMP_ROLL_MODEL

    def test_refuses_missing_field(self) -> None:
        with pytest.raises(RollModelProvenanceError, match="roll_model"):
            require_roll_model({"holed": True}, source="test")

    def test_refuses_none(self) -> None:
        with pytest.raises(RollModelProvenanceError):
            require_roll_model({ROLL_MODEL_FIELD: None}, source="test")

    def test_refuses_blank(self) -> None:
        with pytest.raises(RollModelProvenanceError):
            require_roll_model({ROLL_MODEL_FIELD: "   "}, source="test")

    def test_refuses_unknown_model(self) -> None:
        with pytest.raises(RollModelProvenanceError, match="unknown"):
            require_roll_model({ROLL_MODEL_FIELD: "made-up-roll/9"}, source="test")

    def test_refuses_non_mapping(self) -> None:
        with pytest.raises(RollModelProvenanceError):
            require_roll_model([UD_LEGACY_ROLL_MODEL], source="test")  # type: ignore[arg-type]

    def test_message_names_the_source(self) -> None:
        with pytest.raises(RollModelProvenanceError, match="putt-archive.json"):
            require_roll_model({}, source="putt-archive.json")


# ---------------------------------------------------------------------------
# Result documents carry the name
# ---------------------------------------------------------------------------


class TestResultDocumentsNameTheirModel:
    """Every result document the engine emits names its roll model."""

    def test_simulation_result_defaults_to_ud_legacy(
        self, putt_result: SimulationResult
    ) -> None:
        assert putt_result.roll_model == UD_LEGACY_ROLL_MODEL

    def test_simulation_result_dict_carries_the_name(
        self, putt_result: SimulationResult
    ) -> None:
        assert putt_result.to_dict()[ROLL_MODEL_FIELD] == UD_LEGACY_ROLL_MODEL

    def test_simulation_result_refuses_blank_model(self) -> None:
        with pytest.raises(RollModelProvenanceError):
            SimulationResult(
                positions=np.zeros((1, 2)),
                velocities=np.zeros((1, 2)),
                times=np.zeros(1),
                holed=False,
                final_position=np.zeros(2),
                roll_model="",
            )

    def test_simulation_result_refuses_unknown_model(self) -> None:
        with pytest.raises(RollModelProvenanceError):
            SimulationResult(
                positions=np.zeros((1, 2)),
                velocities=np.zeros((1, 2)),
                times=np.zeros(1),
                holed=False,
                final_position=np.zeros(2),
                roll_model="ud-legacy-roll/99",
            )

    def test_result_round_trips_through_from_dict(
        self, putt_result: SimulationResult
    ) -> None:
        restored = SimulationResult.from_dict(putt_result.to_dict())
        assert restored.roll_model == UD_LEGACY_ROLL_MODEL
        assert restored.holed == putt_result.holed
        np.testing.assert_allclose(restored.final_position, putt_result.final_position)

    def test_from_dict_refuses_unnamed_payload(
        self, putt_result: SimulationResult
    ) -> None:
        payload = putt_result.to_dict()
        del payload[ROLL_MODEL_FIELD]
        with pytest.raises(RollModelProvenanceError):
            SimulationResult.from_dict(payload)

    def test_physics_trajectory_document_names_the_model(self) -> None:
        physics = BallRollPhysics(turf=TurfProperties(stimp_rating=11.0))
        state = BallState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.2, 0.0]),
            spin=np.zeros(3),
        )
        document = physics.simulate_putt(state, max_time=4.0, dt=0.005)
        assert document[ROLL_MODEL_FIELD] == UD_LEGACY_ROLL_MODEL

    def test_practice_feedback_names_the_model(
        self, simulator: PuttingGreenSimulator
    ) -> None:
        stroke = StrokeParameters(speed=1.4, direction=np.array([0.0, 1.0]))
        feedback = simulator.simulate_with_feedback(stroke)
        assert feedback[ROLL_MODEL_FIELD] == UD_LEGACY_ROLL_MODEL

    def test_scatter_results_name_the_model(
        self, simulator: PuttingGreenSimulator
    ) -> None:
        results = simulator.simulate_scatter(
            start_position=np.array([10.0, 8.0]),
            stroke_params=StrokeParameters(speed=1.4, direction=np.array([0.0, 1.0])),
            n_simulations=3,
        )
        assert [r.roll_model for r in results] == [UD_LEGACY_ROLL_MODEL] * 3

    def test_aim_line_names_the_model(self, simulator: PuttingGreenSimulator) -> None:
        aim = simulator.compute_aim_line(np.array([10.0, 8.0]))
        assert aim[ROLL_MODEL_FIELD] == UD_LEGACY_ROLL_MODEL

    def test_green_reading_names_the_model(
        self, simulator: PuttingGreenSimulator
    ) -> None:
        reading = simulator.read_green(np.array([10.0, 8.0]), np.array([10.0, 12.0]))
        assert reading[ROLL_MODEL_FIELD] == UD_LEGACY_ROLL_MODEL

    def test_current_trajectory_names_the_model(
        self, simulator: PuttingGreenSimulator
    ) -> None:
        simulator.set_ball_velocity(np.array([0.6, 0.0]))
        simulator.step()
        trajectory = simulator.get_current_trajectory()
        assert trajectory[ROLL_MODEL_FIELD] == UD_LEGACY_ROLL_MODEL


# ---------------------------------------------------------------------------
# Exported result files: fail-closed on both ends
# ---------------------------------------------------------------------------


class TestExportedResultFiles:
    """``export_result`` writes the name; ``load_result`` refuses without it."""

    def test_exported_file_carries_the_name(
        self,
        simulator: PuttingGreenSimulator,
        putt_result: SimulationResult,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "putt.json"
        simulator.export_result(putt_result, str(path))
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload[ROLL_MODEL_FIELD] == UD_LEGACY_ROLL_MODEL

    def test_exported_file_round_trips(
        self,
        simulator: PuttingGreenSimulator,
        putt_result: SimulationResult,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "putt.json"
        simulator.export_result(putt_result, str(path))
        assert load_result(str(path)).roll_model == UD_LEGACY_ROLL_MODEL

    def test_loader_refuses_unnamed_file(
        self,
        simulator: PuttingGreenSimulator,
        putt_result: SimulationResult,
        tmp_path: Path,
    ) -> None:
        path = tmp_path / "legacy_putt.json"
        simulator.export_result(putt_result, str(path))
        payload = json.loads(path.read_text(encoding="utf-8"))
        del payload[ROLL_MODEL_FIELD]
        path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(RollModelProvenanceError):
            load_result(str(path))


# ---------------------------------------------------------------------------
# Checkpoints: versioned payload, archive-tolerant reads
# ---------------------------------------------------------------------------


class TestCheckpointProvenance:
    """Checkpoints are versioned, so old archives stay readable but untagged."""

    def test_current_checkpoints_are_version_two(
        self, simulator: PuttingGreenSimulator
    ) -> None:
        assert PUTTING_CHECKPOINT_SCHEMA_VERSION == 2
        checkpoint = simulator.get_checkpoint()
        assert checkpoint.engine_state["schema_version"] == 2

    def test_current_checkpoints_name_the_model(
        self, simulator: PuttingGreenSimulator
    ) -> None:
        checkpoint = simulator.get_checkpoint()
        assert checkpoint.engine_state[ROLL_MODEL_FIELD] == UD_LEGACY_ROLL_MODEL

    def test_provenance_of_current_checkpoint(
        self, simulator: PuttingGreenSimulator
    ) -> None:
        provenance = read_checkpoint_provenance(simulator.get_checkpoint())
        assert provenance == CheckpointProvenance(
            roll_model=UD_LEGACY_ROLL_MODEL, schema_version=2, is_archive=False
        )

    def test_version_one_archive_is_tagged_not_defaulted(self) -> None:
        """An unversioned archive reads back with ``roll_model=None``."""
        archive = StateCheckpoint.create(
            engine_type="putting_green",
            engine_state={"spin": [0.0, 0.0, 0.0]},
            q=np.array([1.0, 2.0]),
            v=np.array([0.0, 0.0]),
            timestamp=0.25,
        )
        provenance = read_checkpoint_provenance(archive)
        assert provenance.is_archive is True
        assert provenance.schema_version == 1
        assert provenance.roll_model is None

    def test_version_one_archive_still_restores(
        self, simulator: PuttingGreenSimulator
    ) -> None:
        archive = StateCheckpoint.create(
            engine_type="putting_green",
            engine_state={"spin": [0.0, 0.0, 0.0]},
            q=np.array([3.0, 4.0]),
            v=np.array([0.1, 0.2]),
            timestamp=0.5,
        )
        simulator.restore_checkpoint(archive)
        position, velocity = simulator.get_state()
        np.testing.assert_allclose(position, np.array([3.0, 4.0]))
        np.testing.assert_allclose(velocity, np.array([0.1, 0.2]))

    def test_current_version_checkpoint_without_model_is_refused(self) -> None:
        """Fail-closed inside the current version: v2 must name its model."""
        broken = StateCheckpoint.create(
            engine_type="putting_green",
            engine_state={"spin": [0.0, 0.0, 0.0], "schema_version": 2},
            q=np.array([1.0, 1.0]),
            v=np.array([0.0, 0.0]),
            timestamp=0.0,
        )
        with pytest.raises(RollModelProvenanceError):
            read_checkpoint_provenance(broken)

    def test_foreign_engine_checkpoint_is_refused(self) -> None:
        foreign = StateCheckpoint.create(
            engine_type="mujoco",
            engine_state={"schema_version": 2, ROLL_MODEL_FIELD: UD_LEGACY_ROLL_MODEL},
            q=np.array([0.0]),
            v=np.array([0.0]),
            timestamp=0.0,
        )
        with pytest.raises(ValueError, match="putting_green"):
            read_checkpoint_provenance(foreign)

    def test_simulator_exposes_provenance_reader(
        self, simulator: PuttingGreenSimulator
    ) -> None:
        checkpoint = simulator.get_checkpoint()
        assert simulator.checkpoint_provenance(checkpoint).roll_model == (
            UD_LEGACY_ROLL_MODEL
        )


# ---------------------------------------------------------------------------
# Physics regression: provenance plumbing changes no number
# ---------------------------------------------------------------------------


class TestPhysicsIsBitIdentical:
    """Digests pinned from ``origin/main`` before the provenance change.

    A mismatch means the F1 plumbing perturbed the physics, which ADR-0045
    forbids: naming a model must never move it.
    """

    SIMULATOR_DIGEST = (
        "ac36342f4c12c9c296db6465d54f32af66bda77a50b98c7b88578f005e63d1b9"
    )
    PHYSICS_DIGEST = "bbe25afb71eac5b28126a91874c709c270f66fd3e118f12730b224e2212c5268"

    def test_pinned_simulator_putt_is_bit_identical(self) -> None:
        result = _pinned_simulator().simulate_putt(
            StrokeParameters(speed=1.0, direction=np.array([0.0, 1.0])),
            ball_position=np.array([10.0, 5.0]),
        )
        digest = hashlib.sha256()
        _feed(
            digest,
            result.positions,
            result.velocities,
            result.times,
            result.final_position,
        )
        digest.update(repr(bool(result.holed)).encode("utf-8"))
        assert digest.hexdigest() == self.SIMULATOR_DIGEST

    def test_pinned_roll_physics_is_bit_identical(self) -> None:
        physics = BallRollPhysics(
            turf=TurfProperties(stimp_rating=11.0), integrator="rk4"
        )
        state = BallState(
            position=np.array([0.0, 0.0]),
            velocity=np.array([1.6, 0.4]),
            spin=np.array([0.0, -30.0, 0.0]),
        )
        roll = physics.simulate_putt(state, max_time=12.0, dt=0.002)
        digest = hashlib.sha256()
        _feed(
            digest,
            roll["positions"],
            roll["velocities"],
            roll["spins"],
            roll["times"],
            roll["final_position"],
            roll["final_velocity"],
        )
        digest.update(repr(bool(roll["holed"])).encode("utf-8"))
        assert digest.hexdigest() == self.PHYSICS_DIGEST
