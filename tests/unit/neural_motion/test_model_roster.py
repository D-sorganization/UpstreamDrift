"""NM-01 (#10616): model roster freeze keyed to TB-00 identities."""

from __future__ import annotations

import pytest

from src.shared.python.neural_motion.roster import (
    NeuralModelRoster,
    RosterStage,
    build_neural_model_roster,
    resolve_roster_entry,
)
from src.shared.python.tour_baselines.models import BackendType
from src.shared.python.tour_baselines.registry import list_golf_models

pytestmark = pytest.mark.unit


def test_roster_covers_every_registered_golf_model() -> None:
    roster = build_neural_model_roster()
    registered = {m.model_id for m in list_golf_models()}
    assert isinstance(roster, NeuralModelRoster)
    assert set(roster.model_ids()) == registered
    assert len(roster.entries) == len(registered)


def test_roster_entry_binds_model_id_to_backend_identity() -> None:
    roster = build_neural_model_roster()
    entry = resolve_roster_entry(roster, "driven_double_pendulum")
    assert entry.model_id == "driven_double_pendulum"
    assert entry.backend is BackendType.SCIPY_ODE
    assert entry.q_dim == 2
    assert entry.independent_dof == 2
    assert entry.checkpoint_layout != "hardcoded_27x7"
    assert entry.pilot_stage is RosterStage.PILOT_ELIGIBLE


def test_full_body_models_are_deferred_until_benefit_review() -> None:
    roster = build_neural_model_roster()
    simscape = resolve_roster_entry(roster, "full_body_simscape")
    assert simscape.pilot_stage is RosterStage.DEFERRED_PENDING_BENEFIT
    assert "benefit" in " ".join(simscape.blockers).lower()


def test_roster_rejects_unknown_model_id() -> None:
    roster = build_neural_model_roster()
    with pytest.raises(KeyError, match="unknown"):
        resolve_roster_entry(roster, "not_a_real_model")


def test_roster_digest_is_stable_and_content_addressed() -> None:
    a = build_neural_model_roster()
    b = build_neural_model_roster()
    assert a.content_digest() == b.content_digest()
    assert len(a.content_digest()) == 64
