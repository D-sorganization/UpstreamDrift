from __future__ import annotations

from dataclasses import replace
from typing import Any

import pytest

from src.engines.physics_engines.opensim.python.tour_matching import (
    native_muscle_effort_adapter as adapter,
)
from src.shared.python.biomechanics.muscle_allocation import MuscleAllocationResult
from src.shared.python.motion_matching.distributed_feedback import FeedbackRequest

pytestmark = pytest.mark.unit


def snapshot() -> adapter.NativeMuscleStateSnapshot:
    coordinate = adapter.NativeCoordinateMapping(
        "/jointset/hinge/hinge_coord",
        "/jointset/hinge/hinge_coord/value",
        "/jointset/hinge/hinge_coord/speed",
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        0,
        False,
        False,
        "rotational",
        "N*m",
        "actuator",
    )
    config = adapter.AnalyticRecruitmentConfig(
        10.0, 0.1, 0.2, 0.0, 0.01, 0.04, 0.001, (0.0, 1.0), 100.0
    )
    paths = (
        adapter.NativeMusclePathState(
            "flexor",
            "/forceset/flexor",
            0.3,
            0.05,
            0.1,
            0.2,
            "/forceset/flexor/activation",
            "/forceset/flexor/fiber_length",
            0.1,
            0.0,
            config,
        ),
        adapter.NativeMusclePathState(
            "extensor",
            "/forceset/extensor",
            0.3,
            -0.05,
            0.1,
            0.2,
            "/forceset/extensor/activation",
            "/forceset/extensor/fiber_length",
            0.1,
            0.0,
            config,
        ),
    )
    state = (
        (coordinate.coordinate_value_state_path, 0.0),
        (coordinate.coordinate_speed_state_path, 0.0),
        (paths[0].activation_state_path, 0.1),
        (paths[1].activation_state_path, 0.1),
        (paths[0].fiber_length_state_path, 0.1),
        (paths[1].fiber_length_state_path, 0.1),
    )
    base = adapter.NativeMuscleStateSnapshot(
        "a" * 64,
        "OpenSim test runtime",
        0.25,
        coordinate,
        paths,
        tuple(name for name, _ in state),
        state,
        None,
    )
    passive = adapter.NativePassiveEffortObservation(
        "a" * 64,
        "OpenSim test runtime",
        0.25,
        adapter.native_state_fingerprint(
            source_sha256=base.source_sha256,
            runtime_identity=base.runtime_identity,
            time_s=base.time_s,
            coordinate=base.coordinate,
            paths=base.paths,
            required_state_paths=base.required_state_paths,
            complete_state=base.complete_state,
        ),
        coordinate.coordinate_path,
        "N*m",
        ((paths[0].absolute_path, 0.5), (paths[1].absolute_path, 0.25)),
        (
            (
                paths[0].absolute_path,
                (adapter.NativePassiveComponent.PASSIVE_FIBER_FORCE_ALONG_TENDON,),
            ),
            (
                paths[1].absolute_path,
                (adapter.NativePassiveComponent.PASSIVE_FIBER_FORCE_ALONG_TENDON,),
            ),
        ),
    )
    return replace(base, passive_effort=passive)


def refresh_passive_binding(
    state: adapter.NativeMuscleStateSnapshot,
) -> adapter.NativeMuscleStateSnapshot:
    prior = state.passive_effort
    if prior is None:
        return state
    fingerprint = adapter.native_state_fingerprint(
        source_sha256=state.source_sha256,
        runtime_identity=state.runtime_identity,
        time_s=state.time_s,
        coordinate=state.coordinate,
        paths=state.paths,
        required_state_paths=state.required_state_paths,
        complete_state=state.complete_state,
    )
    return replace(
        state,
        passive_effort=replace(
            prior,
            time_s=state.time_s,
            native_state_sha256=fingerprint,
        ),
    )


def result(target: float) -> MuscleAllocationResult:
    raw = {"flexor": 0.3, "extensor": 0.25}
    return MuscleAllocationResult(
        "feasible",
        True,
        raw,
        raw.copy(),
        {"flexor": 0.2, "extensor": 0.2},
        {"flexor": (0.0, 1.0), "extensor": (0.0, 1.0)},
        {"flexor": {"tendon_n": 10.0}, "extensor": {"tendon_n": 5.0}},
        {"hinge_coord": target},
        {"hinge_coord": 0.0},
        3,
        "fixture",
    )


def request(
    basis: adapter.EffortBasis = adapter.EffortBasis.TOTAL_MUSCULOTENDINOUS,
) -> adapter.NativeMuscleEffortRequest:
    return adapter.NativeMuscleEffortRequest(basis)


def feedback(
    target: float,
    time_s: float = 0.25,
    schedule_index: int = 0,
    channel_id: str = "actuator",
) -> FeedbackRequest:
    return FeedbackRequest(
        time_s=time_s,
        dt_s=0.01,
        schedule_index=schedule_index,
        phase="fixture",
        active_tasks=(),
        nominal_feedforward=(target,),
        feedback_correction=(0.0,),
        total_requested=(target,),
        generalized_effort=(target,),
        channel_ids=(channel_id,),
        generalized_indices=(0,),
        selected_coordinate_mask=(True,),
    )


def propose(
    session: adapter.NativeMuscleEffortSession,
    effort: float = 1.0,
    basis: adapter.EffortBasis = adapter.EffortBasis.TOTAL_MUSCULOTENDINOUS,
    *,
    time_s: float = 0.25,
    schedule_index: int = 0,
) -> adapter.NativeMuscleEffortProposal:
    return session.propose(
        request(basis),
        feedback(effort, time_s=time_s, schedule_index=schedule_index),
        effort_tolerance_n_m=0.1,
    )


def test_total_and_active_increment_effort_accounting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    targets: list[float] = []

    def allocate(*args: Any, **kwargs: Any) -> MuscleAllocationResult:
        targets.append(float(args[2][0]))
        return result(targets[-1])

    monkeypatch.setattr(adapter, "allocate_muscle_excitations", allocate)
    session = adapter.NativeMuscleEffortSession(
        snapshot(), dt_s=0.01, first_schedule_index=0
    )
    total = propose(session, 2.0)
    active = propose(session, 2.0, adapter.EffortBasis.ACTIVE_INCREMENT)
    assert targets == [2.0, 2.75]
    assert total.requested_total_effort_n_m == 2.0
    assert active.requested_total_effort_n_m == 2.75


def test_ambiguous_active_increment_never_reaches_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> MuscleAllocationResult:
        raise AssertionError("ambiguous request reached allocator")

    monkeypatch.setattr(adapter, "allocate_muscle_excitations", unexpected)
    initial = replace(snapshot(), passive_effort=None)
    session = adapter.NativeMuscleEffortSession(
        initial, dt_s=0.01, first_schedule_index=0
    )
    with pytest.raises(ValueError, match="native passive effort observation"):
        propose(session, 2.0, adapter.EffortBasis.ACTIVE_INCREMENT)


def test_proposal_rejection_is_retryable_and_snapshot_is_immutable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter, "allocate_muscle_excitations", lambda *a, **k: result(1.0)
    )
    initial = snapshot()
    session = adapter.NativeMuscleEffortSession(
        initial, dt_s=0.01, first_schedule_index=0
    )
    old = propose(session)
    assert tuple(name for name, _ in old.raw_excitations) == ("flexor", "extensor")
    assert tuple(name for name, _ in old.effective_bounds) == ("flexor", "extensor")
    assert isinstance(old.muscle_forces_n[0][1], tuple)
    assert isinstance(old.raw_excitations, tuple)
    with pytest.raises((AttributeError, TypeError)):
        old.raw_excitations[0] = ("flexor", 0.9)  # type: ignore[index]
    session.reject(old)
    current = propose(session)
    assert current.start_time_s == initial.time_s
    assert current.proposal_id != old.proposal_id


def test_accept_requires_fresh_complete_state_and_is_single_use(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter, "allocate_muscle_excitations", lambda *a, **k: result(1.0)
    )
    session = adapter.NativeMuscleEffortSession(
        snapshot(), dt_s=0.01, first_schedule_index=0
    )
    stale = propose(session)
    current = propose(session)
    with pytest.raises(ValueError, match="stale proposal"):
        session.begin_advance(stale)
    token = session.begin_advance(current)
    paths = tuple(
        replace(
            p,
            activation=0.21,
            previous_raw_excitation=dict(current.raw_excitations)[p.name],
        )
        for p in snapshot().paths
    )
    accepted = refresh_passive_binding(
        replace(
            snapshot(),
            time_s=current.expected_end_time_s,
            paths=paths,
            passive_effort=None,
            complete_state=(
                ("/jointset/hinge/hinge_coord/value", 0.01),
                ("/jointset/hinge/hinge_coord/speed", 0.1),
                ("/forceset/flexor/activation", 0.21),
                ("/forceset/extensor/activation", 0.21),
                ("/forceset/flexor/fiber_length", 0.1),
                ("/forceset/extensor/fiber_length", 0.1),
            ),
        )
    )
    session.accept_advanced(token, accepted)
    assert session.snapshot.time_s == pytest.approx(0.26)
    with pytest.raises(ValueError, match="no native advance is pending"):
        session.accept_advanced(token, accepted)


def test_post_advance_exception_poison_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter, "allocate_muscle_excitations", lambda *a, **k: result(1.0)
    )
    session = adapter.NativeMuscleEffortSession(
        snapshot(), dt_s=0.01, first_schedule_index=0
    )
    token = session.begin_advance(propose(session))
    with pytest.raises(RuntimeError, match="session is poisoned"):
        session.fail_advance(token, RuntimeError("Manager may have partially advanced"))
    assert session.poisoned
    with pytest.raises(RuntimeError, match="poisoned"):
        propose(session)


def test_invalid_allocator_controls_fail_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = replace(result(1.0), excitations={"flexor": 0.2})
    monkeypatch.setattr(adapter, "allocate_muscle_excitations", lambda *a, **k: invalid)
    session = adapter.NativeMuscleEffortSession(
        snapshot(), dt_s=0.01, first_schedule_index=0
    )
    with pytest.raises(ValueError, match="omitted or added"):
        propose(session)


def test_out_of_bounds_raw_control_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invalid = replace(
        result(1.0),
        excitations={"flexor": 1.1, "extensor": 0.25},
    )
    monkeypatch.setattr(adapter, "allocate_muscle_excitations", lambda *a, **k: invalid)
    session = adapter.NativeMuscleEffortSession(
        snapshot(), dt_s=0.01, first_schedule_index=0
    )
    with pytest.raises(ValueError, match="invalid raw/effective"):
        propose(session)


def test_nonfeasible_or_exhausted_candidate_cannot_advance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for status in ("infeasible", "budget_exhausted", "solver_failed"):
        candidate = replace(result(1.0), status=status, feasible=False)
        monkeypatch.setattr(
            adapter,
            "allocate_muscle_excitations",
            lambda *a, _candidate=candidate, **k: _candidate,
        )
        session = adapter.NativeMuscleEffortSession(
            snapshot(), dt_s=0.01, first_schedule_index=0
        )
        with pytest.raises(ValueError, match="only feasible"):
            session.begin_advance(propose(session))


def test_feedback_mask_and_policy_effort_must_match_mapping() -> None:
    session = adapter.NativeMuscleEffortSession(
        snapshot(), dt_s=0.01, first_schedule_index=0
    )
    wrong_mask = replace(feedback(1.0), selected_coordinate_mask=(False,))
    with pytest.raises(ValueError, match="coordinate mask"):
        session.propose(request(), wrong_mask, effort_tolerance_n_m=0.1)
    wrong_order = replace(feedback(1.0), generalized_indices=(1,))
    with pytest.raises(ValueError, match="generalized indices"):
        session.propose(request(), wrong_order, effort_tolerance_n_m=0.1)
    duplicate_mapping = replace(feedback(1.0), generalized_indices=(0, 0))
    with pytest.raises(ValueError, match="generalized indices"):
        session.propose(request(), duplicate_mapping, effort_tolerance_n_m=0.1)
    multiple_channels = replace(
        feedback(1.0),
        channel_ids=("actuator-a", "actuator-b"),
        total_requested=(0.5, 0.5),
    )
    with pytest.raises(ValueError, match="feedback channel values"):
        session.propose(request(), multiple_channels, effort_tolerance_n_m=0.1)
    forged_generalized = replace(feedback(1.0), generalized_effort=(2.0,))
    with pytest.raises(ValueError, match="differs from channel request"):
        session.propose(request(), forged_generalized, effort_tolerance_n_m=0.1)


def test_request_configuration_rejects_implicit_or_mutable_basis_data() -> None:
    with pytest.raises(ValueError, match="explicit EffortBasis"):
        adapter.NativeMuscleEffortRequest("active_increment")  # type: ignore[arg-type]
    bounds = [0.1, 0.9]
    config = adapter.AnalyticRecruitmentConfig(
        10.0,
        0.1,
        0.2,
        0.0,
        0.01,
        0.04,
        0.001,
        bounds,
        100.0,  # type: ignore[arg-type]
    )
    bounds[0] = 0.7
    assert config.excitation_bounds == (0.1, 0.9)
    with pytest.raises((AttributeError, TypeError)):
        config.excitation_bounds[0] = 0.7  # type: ignore[index]


def test_passive_observation_must_match_snapshot_source_and_complete_state() -> None:
    initial = snapshot()
    stale_time = replace(initial.passive_effort, time_s=0.0)
    with pytest.raises(ValueError, match="stale or mismatched"):
        replace(initial, passive_effort=stale_time)

    changed_paths = tuple(replace(path, activation=0.2) for path in initial.paths)
    changed_state = tuple(
        (name, 0.2 if name.endswith("/activation") else value)
        for name, value in initial.complete_state
    )
    changed = replace(
        initial,
        paths=changed_paths,
        complete_state=changed_state,
        passive_effort=None,
    )
    with pytest.raises(ValueError, match="stale or mismatched"):
        replace(changed, passive_effort=initial.passive_effort)


def test_cross_session_advance_token_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter, "allocate_muscle_excitations", lambda *a, **k: result(1.0)
    )
    first = adapter.NativeMuscleEffortSession(
        snapshot(), dt_s=0.01, first_schedule_index=0
    )
    second = adapter.NativeMuscleEffortSession(
        snapshot(), dt_s=0.01, first_schedule_index=0
    )
    first_token = first.begin_advance(propose(first))
    second_token = second.begin_advance(propose(second))
    with pytest.raises(ValueError, match="no native advance is pending"):
        first.accept_advanced(second_token, snapshot())
    with pytest.raises(RuntimeError, match="session is poisoned"):
        first.fail_advance(first_token, RuntimeError("fixture cleanup"))
    with pytest.raises(RuntimeError, match="session is poisoned"):
        second.fail_advance(second_token, RuntimeError("fixture cleanup"))


def test_feedback_channel_and_schedule_index_are_bound_to_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter, "allocate_muscle_excitations", lambda *a, **k: result(1.0)
    )
    session = adapter.NativeMuscleEffortSession(
        snapshot(), dt_s=0.01, first_schedule_index=4
    )
    with pytest.raises(ValueError, match="schedule index"):
        session.propose(
            request(), feedback(1.0, schedule_index=3), effort_tolerance_n_m=0.1
        )
    with pytest.raises(ValueError, match="feedback channel"):
        session.propose(
            request(),
            feedback(1.0, schedule_index=4, channel_id="different actuator"),
            effort_tolerance_n_m=0.1,
        )
    proposal = propose(session, schedule_index=4)
    assert proposal.schedule_index == 4
    assert proposal.feedback_channel_id == "actuator"
    session.reject(proposal)
    retry = propose(session, schedule_index=4)
    token = session.begin_advance(retry)
    accepted_paths = tuple(
        replace(path, previous_raw_excitation=dict(retry.raw_excitations)[path.name])
        for path in snapshot().paths
    )
    advanced = refresh_passive_binding(
        replace(
            snapshot(),
            time_s=retry.expected_end_time_s,
            paths=accepted_paths,
            passive_effort=None,
        )
    )
    session.accept_advanced(token, advanced)
    with pytest.raises(ValueError, match="schedule index"):
        propose(session, time_s=advanced.time_s, schedule_index=4)
    next_proposal = propose(session, time_s=advanced.time_s, schedule_index=5)
    assert next_proposal.schedule_index == 5


def test_passive_observation_rejects_nonpassive_component_labels() -> None:
    initial = snapshot()
    assert initial.passive_effort is not None
    invalid_components = tuple(
        (path, ("getActuation",))
        for path, _ in initial.passive_effort.component_names_by_path
    )
    with pytest.raises(
        ValueError, match="passive effort must be a complete native observation"
    ):
        replace(initial.passive_effort, component_names_by_path=invalid_components)


def test_nonadvancing_time_is_rejected_before_allocator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected(*args: Any, **kwargs: Any) -> MuscleAllocationResult:
        raise AssertionError("invalid clock reached allocator")

    monkeypatch.setattr(adapter, "allocate_muscle_excitations", unexpected)
    initial = replace(snapshot(), time_s=1.0e16, passive_effort=None)
    session = adapter.NativeMuscleEffortSession(
        initial, dt_s=0.01, first_schedule_index=0
    )
    req = replace(feedback(1.0), time_s=initial.time_s)
    with pytest.raises(ValueError, match="non-finite or non-advancing"):
        session.propose(request(), req, effort_tolerance_n_m=0.1)


def test_raw_floor_separation_and_next_history_use_raw_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = {"flexor": 0.0005, "extensor": 0.25}
    effective = {"flexor": 0.001, "extensor": 0.25}
    floored = replace(result(1.0), excitations=raw, effective_excitations=effective)
    monkeypatch.setattr(adapter, "allocate_muscle_excitations", lambda *a, **k: floored)
    session = adapter.NativeMuscleEffortSession(
        snapshot(), dt_s=0.01, first_schedule_index=0
    )
    proposal = propose(session)
    assert dict(proposal.raw_excitations)["flexor"] == 0.0005
    assert dict(proposal.effective_excitations)["flexor"] == 0.001
    token = session.begin_advance(proposal)
    accepted_paths = tuple(
        replace(path, previous_raw_excitation=dict(proposal.raw_excitations)[path.name])
        for path in snapshot().paths
    )
    advanced = refresh_passive_binding(
        replace(
            snapshot(),
            time_s=proposal.expected_end_time_s,
            paths=accepted_paths,
            passive_effort=None,
        )
    )
    session.accept_advanced(token, advanced)
    assert session.snapshot.paths[0].previous_raw_excitation == 0.0005


def test_accept_rejects_changed_static_path_binding_and_poisons_session(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter, "allocate_muscle_excitations", lambda *a, **k: result(1.0)
    )
    session = adapter.NativeMuscleEffortSession(
        snapshot(), dt_s=0.01, first_schedule_index=0
    )
    proposal = propose(session)
    token = session.begin_advance(proposal)
    changed = replace(
        snapshot().paths[0],
        analytic=replace(snapshot().paths[0].analytic, max_isometric_force_n=11.0),
    )
    advanced = refresh_passive_binding(
        replace(
            snapshot(),
            time_s=proposal.expected_end_time_s,
            paths=(changed, snapshot().paths[1]),
            passive_effort=None,
        )
    )
    with pytest.raises(ValueError, match="identity differs"):
        session.accept_advanced(token, advanced)
    assert session.poisoned


def test_snapshot_requires_complete_exactly_ordered_state_schema() -> None:
    initial = snapshot()
    with pytest.raises(ValueError, match="paths/order"):
        replace(initial, complete_state=initial.complete_state[:-1])
    with pytest.raises(ValueError, match="paths/order"):
        replace(
            initial,
            complete_state=initial.complete_state + (("/unexpected/state", 0.0),),
        )
    changed = list(initial.complete_state)
    changed[2] = (changed[2][0], 0.3)
    with pytest.raises(ValueError, match="activation differs"):
        replace(initial, complete_state=tuple(changed))


def test_mapping_rejects_multidof_or_constrained_model() -> None:
    with pytest.raises(ValueError, match="one unlocked, unconstrained rotational"):
        adapter.NativeCoordinateMapping(
            "/jointset/hinge/hinge_coord",
            "/jointset/hinge/hinge_coord/value",
            "/jointset/hinge/hinge_coord/speed",
            0,
            0,
            0,
            0,
            1,
            2,
            1,
            0,
            False,
            False,
            "rotational",
            "N*m",
            "actuator",
        )
    with pytest.raises(ValueError, match="one unlocked, unconstrained rotational"):
        adapter.NativeCoordinateMapping(
            "/jointset/hinge/hinge_coord",
            "/jointset/hinge/hinge_coord/value",
            "/jointset/hinge/hinge_coord/speed",
            0,
            0,
            0,
            0,
            1,
            1,
            1,
            1,
            False,
            False,
            "rotational",
            "N*m",
            "actuator",
        )


from src.engines.physics_engines.opensim.python.tour_matching.native_muscle_effort_runtime import (
    NativeMuscleEffortExecutor,
    _OpenSimNativeEffortBackend,
)


class _FakeNativeBackend:
    def __init__(self, state: adapter.NativeMuscleStateSnapshot) -> None:
        self.state = state
        self.events: list[str] = []
        self.raw: dict[str, float] = {}
        self.fail_advance = False
        self.alter_readback = False

    def observe(self) -> adapter.NativeMuscleStateSnapshot:
        self.events.append("observe")
        return self.state

    def apply_raw_excitations(
        self, proposal: adapter.NativeMuscleEffortProposal
    ) -> None:
        self.events.append("apply")
        self.raw = dict(proposal.raw_excitations)

    def advance_to(self, end_time_s: float) -> adapter.NativeMuscleStateSnapshot:
        self.events.append("advance")
        if self.fail_advance:
            raise RuntimeError("fixture Manager failure")
        paths = tuple(
            replace(path, previous_raw_excitation=self.raw[path.name])
            for path in self.state.paths
        )
        self.state = refresh_passive_binding(
            replace(
                self.state,
                time_s=end_time_s,
                paths=paths,
                passive_effort=None,
            )
        )
        return self.state

    def applied_raw_excitations(self) -> dict[str, float]:
        self.events.append("readback")
        result_values = dict(self.raw)
        if self.alter_readback:
            result_values["flexor"] += 0.01
        return result_values


def test_executor_applies_verifies_then_accepts_one_feasible_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        adapter, "allocate_muscle_excitations", lambda *a, **k: result(1.0)
    )
    initial = snapshot()
    session = adapter.NativeMuscleEffortSession(
        initial, dt_s=0.01, first_schedule_index=0
    )
    backend = _FakeNativeBackend(initial)
    executor = NativeMuscleEffortExecutor(session, backend)

    stepped = executor.execute_one_interval(
        request(), feedback(1.0), effort_tolerance_n_m=0.1
    )

    assert backend.events == ["observe", "apply", "advance", "readback"]
    assert stepped.accepted_snapshot is not None
    assert session.snapshot.time_s == pytest.approx(0.26)
    assert (
        session.snapshot.paths[0].previous_raw_excitation
        == dict(stepped.proposal.raw_excitations)["flexor"]
    )


def test_executor_never_advances_infeasible_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = replace(result(1.0), status="budget_exhausted", feasible=False)
    monkeypatch.setattr(
        adapter,
        "allocate_muscle_excitations",
        lambda *a, **k: candidate,
    )
    initial = snapshot()
    session = adapter.NativeMuscleEffortSession(
        initial, dt_s=0.01, first_schedule_index=0
    )
    backend = _FakeNativeBackend(initial)

    stepped = NativeMuscleEffortExecutor(session, backend).execute_one_interval(
        request(), feedback(1.0), effort_tolerance_n_m=0.1
    )

    assert not stepped.proposal.feasible
    assert stepped.accepted_snapshot is None
    assert backend.events == ["observe"]
    assert not session.poisoned


@pytest.mark.parametrize("failure", ["manager", "readback"])
def test_executor_poison_session_after_partial_or_unverified_advance(
    monkeypatch: pytest.MonkeyPatch, failure: str
) -> None:
    monkeypatch.setattr(
        adapter, "allocate_muscle_excitations", lambda *a, **k: result(1.0)
    )
    initial = snapshot()
    session = adapter.NativeMuscleEffortSession(
        initial, dt_s=0.01, first_schedule_index=0
    )
    backend = _FakeNativeBackend(initial)
    backend.fail_advance = failure == "manager"
    backend.alter_readback = failure == "readback"

    with pytest.raises(RuntimeError):
        NativeMuscleEffortExecutor(session, backend).execute_one_interval(
            request(), feedback(1.0), effort_tolerance_n_m=0.1
        )
    assert session.poisoned


def test_pinned_backend_rejects_source_mismatch_before_model_load(tmp_path) -> None:
    from src.engines.physics_engines.opensim.python.tour_matching.native_muscle_effort_runtime import (
        load_pinned_opensim_backend,
    )

    source = tmp_path / "fixture.osim"
    source.write_text("source bytes", encoding="utf-8")

    class Runtime:
        @staticmethod
        def GetVersionAndDate() -> str:
            raise AssertionError("runtime should not be queried after source mismatch")

        @staticmethod
        def Model(path: str) -> object:
            raise AssertionError("native model construction must not be reached")

    with pytest.raises(ValueError, match="source hash"):
        load_pinned_opensim_backend(
            osim=Runtime(),
            source_path=source,
            expected_source_sha256="0" * 64,
            expected_runtime_identity="pinned runtime",
            coordinate_mapping=snapshot().coordinate,
            analytic_by_muscle={},
            initial_state={},
            initial_time_s=0.0,
            integrator_accuracy=1e-8,
        )


def test_fresh_replay_gate_uses_constant_raw_knots_and_compares_full_state(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import numpy as np
    from types import SimpleNamespace

    from src.engines.physics_engines.opensim.python.tour_matching import muscle_replay
    from src.engines.physics_engines.opensim.python.tour_matching.native_muscle_effort_runtime import (
        verify_fresh_constant_replay,
    )

    initial_base = snapshot()
    initial_paths = tuple(
        replace(path, native_actuation_force_n=force)
        for path, force in zip(initial_base.paths, (5.0, 6.0), strict=True)
    )
    initial_without_passive = replace(
        initial_base, paths=initial_paths, passive_effort=None
    )
    passive = replace(
        initial_base.passive_effort,
        native_state_sha256=adapter.native_state_fingerprint(
            source_sha256=initial_without_passive.source_sha256,
            runtime_identity=initial_without_passive.runtime_identity,
            time_s=initial_without_passive.time_s,
            coordinate=initial_without_passive.coordinate,
            paths=initial_without_passive.paths,
            required_state_paths=initial_without_passive.required_state_paths,
            complete_state=initial_without_passive.complete_state,
        ),
    )
    initial = replace(initial_without_passive, passive_effort=passive)
    candidate = replace(result(1.0), excitations={"flexor": 0.3, "extensor": 0.25})
    monkeypatch.setattr(
        adapter, "allocate_muscle_excitations", lambda *a, **k: candidate
    )
    session = adapter.NativeMuscleEffortSession(
        initial, dt_s=0.01, first_schedule_index=0
    )
    proposal = propose(session)
    raw = dict(proposal.raw_excitations)
    end_state = tuple(
        (
            name,
            0.01
            if name.endswith("/value")
            else (0.1 if name.endswith("/speed") else value),
        )
        for name, value in initial.complete_state
    )
    advanced_paths = tuple(
        replace(path, previous_raw_excitation=raw[path.name]) for path in initial.paths
    )
    advanced = replace(
        initial,
        time_s=proposal.expected_end_time_s,
        paths=advanced_paths,
        complete_state=end_state,
        passive_effort=None,
    )
    captured: dict[str, object] = {}

    def replay(
        source_path: str,
        initial_state: dict[str, float],
        times: np.ndarray,
        excitations: dict[str, np.ndarray],
        *,
        accuracy: float,
    ) -> object:
        captured["times"] = times
        captured["excitations"] = excitations
        names = initial.required_state_paths
        rows = np.asarray(
            [
                [dict(initial.complete_state)[name] for name in names],
                [dict(advanced.complete_state)[name] for name in names],
            ],
            dtype=float,
        )
        applied = np.asarray(
            [[raw[name] for name, _ in proposal.raw_excitations]] * 2, dtype=float
        )
        return SimpleNamespace(
            model_sha256=initial.source_sha256,
            state_names=names,
            muscle_names=tuple(name for name, _ in proposal.raw_excitations),
            times=times.copy(),
            applied_excitations=applied,
            states=rows,
            muscle_forces_n=np.asarray([[5.0, 6.0], [5.0, 6.0]]),
            policy={"provider": initial.runtime_identity},
        )

    monkeypatch.setattr(muscle_replay, "replay_muscle_excitations", replay)
    tolerances = dict.fromkeys(initial.required_state_paths, 1e-12)
    result_value = verify_fresh_constant_replay(
        source_path="fixture.osim",
        initial=initial,
        proposal=proposal,
        advanced=advanced,
        state_absolute_tolerances=tolerances,
        muscle_force_absolute_tolerances_n={"flexor": 1e-12, "extensor": 1e-12},
    )

    assert result_value.states.shape == (2, len(initial.required_state_paths))
    assert captured["times"] == pytest.approx([0.25, 0.26])
    assert all(
        values[0] == values[1] == raw[name]
        for name, values in captured["excitations"].items()
    )


@pytest.mark.parametrize(
    ("getter_name", "wrong_value"),
    [
        ("getMaxIsometricForce", 10.1),
        ("getOptimalFiberLength", 0.11),
        ("getTendonSlackLength", 0.21),
        ("getPennationAngleAtOptimalFiberLength", 0.01),
        ("getActivationTimeConstant", 0.02),
        ("getDeactivationTimeConstant", 0.05),
        ("getMinimumActivation", 0.002),
    ],
)
def test_analytic_physical_parameters_must_match_loaded_native_getters(
    getter_name: str, wrong_value: float
) -> None:
    from types import SimpleNamespace

    config = snapshot().paths[0].analytic
    base_values = {
        "getMaxIsometricForce": config.max_isometric_force_n,
        "getOptimalFiberLength": config.optimal_fiber_length_m,
        "getTendonSlackLength": config.tendon_slack_length_m,
        "getPennationAngleAtOptimalFiberLength": config.pennation_angle_rad,
    }
    derived_values = {
        "getActivationTimeConstant": config.activation_time_constant_s,
        "getDeactivationTimeConstant": config.deactivation_time_constant_s,
        "getMinimumActivation": config.minimum_activation,
    }
    values = derived_values if getter_name in derived_values else base_values
    values[getter_name] = wrong_value
    derived = SimpleNamespace(
        **{name: (lambda value=value: value) for name, value in derived_values.items()}
    )
    base = SimpleNamespace(
        **{name: (lambda value=value: value) for name, value in base_values.items()}
    )
    osim = SimpleNamespace(
        Millard2012EquilibriumMuscle=SimpleNamespace(
            safeDownCast=lambda muscle: derived if muscle is base else None
        )
    )
    with pytest.raises(ValueError, match="differs from native"):
        _OpenSimNativeEffortBackend._verify_analytic_parameters(osim, base, config)


def test_native_parameter_admission_rejects_failed_millard_downcast() -> None:
    from types import SimpleNamespace

    config = snapshot().paths[0].analytic
    base = SimpleNamespace(
        getMaxIsometricForce=lambda: config.max_isometric_force_n,
        getOptimalFiberLength=lambda: config.optimal_fiber_length_m,
        getTendonSlackLength=lambda: config.tendon_slack_length_m,
        getPennationAngleAtOptimalFiberLength=lambda: config.pennation_angle_rad,
    )
    osim = SimpleNamespace(
        Millard2012EquilibriumMuscle=SimpleNamespace(safeDownCast=lambda _: None)
    )
    with pytest.raises(ValueError, match="cannot be admitted as Millard2012"):
        _OpenSimNativeEffortBackend._verify_analytic_parameters(osim, base, config)


def test_analytic_control_bounds_and_rate_are_explicit_caller_policy() -> None:
    from types import SimpleNamespace

    config = snapshot().paths[0].analytic
    base_values = {
        "getMaxIsometricForce": config.max_isometric_force_n,
        "getOptimalFiberLength": config.optimal_fiber_length_m,
        "getTendonSlackLength": config.tendon_slack_length_m,
        "getPennationAngleAtOptimalFiberLength": config.pennation_angle_rad,
    }
    derived_values = {
        "getActivationTimeConstant": config.activation_time_constant_s,
        "getDeactivationTimeConstant": config.deactivation_time_constant_s,
        "getMinimumActivation": config.minimum_activation,
    }
    derived = SimpleNamespace(
        **{name: (lambda value=value: value) for name, value in derived_values.items()}
    )
    native = SimpleNamespace(
        **{name: (lambda value=value: value) for name, value in base_values.items()}
    )
    osim = SimpleNamespace(
        Millard2012EquilibriumMuscle=SimpleNamespace(
            safeDownCast=lambda muscle: derived if muscle is native else None
        )
    )
    _OpenSimNativeEffortBackend._verify_analytic_parameters(osim, native, config)


@pytest.mark.parametrize("changed_flag", ["ignore", "locked", "prescribed"])
def test_native_mode_admission_is_rechecked_on_each_observation(
    changed_flag: str,
) -> None:
    from types import SimpleNamespace

    from src.engines.physics_engines.opensim.python.tour_matching.native_muscle_effort_runtime import (
        _OpenSimNativeEffortBackend,
    )

    accepted = snapshot()
    config_by_name = {path.name: path.analytic for path in accepted.paths}
    state_flags = {"ignore": False, "locked": False, "prescribed": False}

    class ComponentSet:
        def __init__(self, components: tuple[object, ...]) -> None:
            self._components = components

        def getSize(self) -> int:
            return len(self._components)

        def get(self, index: int) -> object:
            return self._components[index]

    coordinate = SimpleNamespace(
        getAbsolutePathString=lambda: accepted.coordinate.coordinate_path,
        getLocked=lambda state: state["locked"],
        isPrescribed=lambda state: state["prescribed"],
    )
    muscles = []
    for path in accepted.paths:
        config = config_by_name[path.name]
        values = {
            "getMaxIsometricForce": config.max_isometric_force_n,
            "getOptimalFiberLength": config.optimal_fiber_length_m,
            "getTendonSlackLength": config.tendon_slack_length_m,
            "getPennationAngleAtOptimalFiberLength": config.pennation_angle_rad,
            "getActivationTimeConstant": config.activation_time_constant_s,
            "getDeactivationTimeConstant": config.deactivation_time_constant_s,
            "getMinimumActivation": config.minimum_activation,
        }
        muscle = SimpleNamespace(
            getName=lambda name=path.name: name,
            getAbsolutePathString=lambda name=path.absolute_path: name,
            getConcreteClassName=lambda: "Millard2012EquilibriumMuscle",
            get_ignore_activation_dynamics=lambda: False,
            get_ignore_tendon_compliance=lambda: False,
            getIgnoreActivationDynamics=lambda state: state["ignore"],
            getIgnoreTendonCompliance=lambda state: False,
            **{key: (lambda value=value: value) for key, value in values.items()},
        )
        muscles.append(muscle)
    muscle_set = ComponentSet(tuple(muscles))
    state_names = accepted.required_state_paths
    model = SimpleNamespace(
        getCoordinateSet=lambda: ComponentSet((coordinate,)),
        getConstraintSet=lambda: ComponentSet(()),
        getStateVariableNames=lambda: ComponentSet(state_names),
        getNumControls=lambda: 2,
    )
    backend = object.__new__(_OpenSimNativeEffortBackend)
    backend._osim = SimpleNamespace(
        Millard2012EquilibriumMuscle=SimpleNamespace(safeDownCast=lambda muscle: muscle)
    )
    backend._analytic_by_muscle = config_by_name
    backend._coordinate_mapping = accepted.coordinate
    backend._model = model
    backend._muscles = muscle_set
    backend._muscle_names = tuple(path.name for path in accepted.paths)
    backend._coordinate = coordinate

    backend._validate_model_boundary(state_flags)
    state_flags[changed_flag] = True
    with pytest.raises(ValueError):
        backend._validate_model_boundary(state_flags)
