"""Unit tests for the cross-model grip qualification harness (HRG-GS-M3, #13325).

Acceptance criteria:
- Tolerances pinned: PEAK_TOLERANCE == 0.05 and RMS_TOLERANCE == 0.02.
- A series with a NaN channel cannot produce passed=True (parametrised over
  the channel kinds: pair_normal, pair_tangential, pair_spin, hand_force,
  hand_moment).
- A missing channel gives status="unavailable", and a mapping-hash mismatch
  gives status="unavailable" with reason "mapping hash mismatch".
- Identical series pass, and a series scaled by 1.06 in normal force fails peak.
- Allocated channels never affect the physical verdict.
- Zero allocated force is applied in a physical run (counter test).
- In a slow, requires_pinocchio test: a G6 Pinocchio topology run (0.04 s hold,
  one style), converted with from_contact_run, passes against law_level_series
  at its recorded states.
- AST check: no attribute chain deeper than 2 dots, functions <= 100 lines and
  <= 8 parameters, files <= 1200 lines.
"""

from __future__ import annotations

import ast
import json
from collections.abc import Mapping
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from src.shared.python.grip_contact import (
    BushingState,
    GripInterface,
    load_coordinate_swing,
)
from src.shared.python.grip_contact.binding import (
    BodyStates,
    GripCapabilities,
    GripStepReport,
    HandClubBinding,
    SiteBinding,
    apply_report,
    bind_rigid_hand_sites,
    evaluate_contacts,
)
from src.shared.python.grip_contact.contact_topology import (
    ContactTopology,
    GripStyle,
    contact_topology,
)
from src.shared.python.grip_contact.parity import (
    NUMERICAL_ZERO_PEAK,
    PEAK_TOLERANCE,
    RMS_TOLERANCE,
    ChannelParityResult,
    TopologyParityReport,
    compare_topology_series,
    compute_quantity_error,
)
from src.shared.python.grip_contact.topology_contact import (
    TopologyContactModel,
    build_topology_model,
)
from src.shared.python.grip_contact.topology_series import (
    CHANNEL_KINDS,
    SAMPLE_DT_S,
    TopologyKineticsSeries,
    law_level_series,
    total_applied_allocated_force_n,
)
from src.shared.python.motion_matching.grip_contract import GripSpec

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
MODELS = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
SPEC_PATH = MODELS / "full_body_spec_anthro_driver.json"
TOPOLOGY_SERIES_PY = ROOT / "src/shared/python/grip_contact/topology_series.py"
PARITY_PY = ROOT / "src/shared/python/grip_contact/parity.py"
BINDING_PY = ROOT / "src/shared/python/grip_contact/binding.py"


@pytest.fixture(scope="module")
def spec_dict() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def interface(spec_dict: dict[str, Any]) -> GripInterface:
    return GripInterface.from_spec(spec_dict)


@pytest.fixture(scope="module")
def overlap_topology() -> ContactTopology:
    return contact_topology(GripSpec(style=GripStyle.OVERLAP))


@pytest.fixture(scope="module")
def overlap_model(
    interface: GripInterface, overlap_topology: ContactTopology
) -> TopologyContactModel:
    return build_topology_model(interface, overlap_topology, squeeze_per_hand_n=1100.0)


def _make_series(
    model: TopologyContactModel,
    *,
    n: int = 21,
    mapping_hash: str = "sha256:test_mapping_v1",
    engine: str = "reference_engine",
    model_id: str = "golfer_v1",
    normal_scale: float = 1.0,
    allocated_finite: bool = False,
    allocated_scale: float = 1.0,
) -> TopologyKineticsSeries:
    """Build a synthetic TopologyKineticsSeries on the 2 ms grid."""
    time_s = np.arange(n, dtype=float) * SAMPLE_DT_S
    wave = 1.0 + 0.15 * np.sin(2.0 * np.pi * 5.0 * time_s)

    pair_normal: dict[str, np.ndarray] = {}
    pair_tangential: dict[str, np.ndarray] = {}
    pair_spin: dict[str, np.ndarray] = {}
    labels: dict[str, str] = {}
    reasons: dict[str, str] = {}

    for idx, pair in enumerate(model.topology.pairs):
        factor = 1.0 + 0.05 * idx
        if pair.is_hand_grip:
            labels[pair.name] = "physical"
            pair_normal[pair.name] = normal_scale * 80.0 * factor * wave
            pair_tangential[pair.name] = 15.0 * factor * wave
            pair_spin[pair.name] = 0.08 * factor * wave
        else:
            labels[pair.name] = "allocated"
            if allocated_finite:
                pair_normal[pair.name] = allocated_scale * 25.0 * factor * wave
                pair_tangential[pair.name] = allocated_scale * 5.0 * factor * wave
                pair_spin[pair.name] = allocated_scale * 0.02 * factor * wave
            else:
                pair_normal[pair.name] = np.full(n, np.nan)
                pair_tangential[pair.name] = np.full(n, np.nan)
                pair_spin[pair.name] = np.full(n, np.nan)
                reasons[pair.name] = (
                    "hand-hand pair is allocated and no closure wrench was supplied"
                )

    force_l = np.column_stack([120.0 * wave, -35.0 * wave, 50.0 * wave])
    force_r = np.column_stack([-95.0 * wave, 40.0 * wave, 60.0 * wave])
    moment_l = np.column_stack([2.5 * wave, -1.8 * wave, 0.9 * wave])
    moment_r = np.column_stack([-2.1 * wave, 1.5 * wave, -0.7 * wave])

    return TopologyKineticsSeries(
        engine=engine,
        model_id=model_id,
        mapping_hash=mapping_hash,
        time_s=time_s,
        pair_normal_force_n=pair_normal,
        pair_tangential_force_n=pair_tangential,
        pair_spin_moment_nm=pair_spin,
        force_on_club_n={"L": force_l, "R": force_r},
        moment_on_club_nm={"L": moment_l, "R": moment_r},
        labels=labels,  # type: ignore[arg-type]
        channel_reasons=reasons,
    )


def test_parity_tolerances_pinned() -> None:
    """AC 1: PEAK_TOLERANCE == 0.05 (5 %) and RMS_TOLERANCE == 0.02 (2 %)."""
    assert PEAK_TOLERANCE == 0.05
    assert RMS_TOLERANCE == 0.02
    assert SAMPLE_DT_S == 0.002


def test_series_dbc_preconditions(overlap_model: TopologyContactModel) -> None:
    """DbC: monotonic time, equal lengths, NaN-reason bijection, and known labels."""
    base = _make_series(overlap_model)

    # 1. Non-monotonic time raises ValueError
    with pytest.raises(ValueError, match="increasing|monotonic"):
        replace(base, time_s=base.time_s[::-1])

    # 2. Mismatched channel length raises ValueError
    bad_normal = dict(base.pair_normal_force_n)
    phys_pair = overlap_model.topology.hand_grip_pairs()[0].name
    bad_normal[phys_pair] = bad_normal[phys_pair][:-2]
    with pytest.raises(ValueError, match="shape|length"):
        replace(base, pair_normal_force_n=bad_normal)

    # 3. NaN channel without a reason raises ValueError
    nan_normal = dict(base.pair_normal_force_n)
    arr = nan_normal[phys_pair].copy()
    arr[3] = np.nan
    nan_normal[phys_pair] = arr
    with pytest.raises(ValueError, match="reason"):
        replace(base, pair_normal_force_n=nan_normal)

    # 4. Reason attached to a finite channel raises ValueError
    bad_reasons = dict(base.channel_reasons)
    bad_reasons[f"normal:{phys_pair}"] = "spurious reason on finite channel"
    with pytest.raises(ValueError, match="finite|NaN"):
        replace(base, channel_reasons=bad_reasons)

    # 5. Unknown label raises ValueError
    bad_labels = dict(base.labels)
    bad_labels[phys_pair] = "magical"  # type: ignore[assignment]
    with pytest.raises(ValueError, match="label"):
        replace(base, labels=bad_labels)


@pytest.mark.parametrize("channel_kind", CHANNEL_KINDS)
def test_nan_channel_never_passes(
    overlap_model: TopologyContactModel, channel_kind: str
) -> None:
    """AC 2: A series with a NaN channel cannot produce passed=True across channel kinds."""
    ref = _make_series(overlap_model)
    phys_pair = overlap_model.topology.hand_grip_pairs()[0].name

    normal = {k: v.copy() for k, v in ref.pair_normal_force_n.items()}
    tangential = {k: v.copy() for k, v in ref.pair_tangential_force_n.items()}
    spin = {k: v.copy() for k, v in ref.pair_spin_moment_nm.items()}
    force = {k: v.copy() for k, v in ref.force_on_club_n.items()}
    moment = {k: v.copy() for k, v in ref.moment_on_club_nm.items()}
    reasons = dict(ref.channel_reasons)

    if channel_kind == "pair_normal":
        normal[phys_pair][5] = np.nan
        expected_key = f"normal:{phys_pair}"
        reasons[expected_key] = "normal sensor dropout"
    elif channel_kind == "pair_tangential":
        tangential[phys_pair][5] = np.nan
        expected_key = f"tangential:{phys_pair}"
        reasons[expected_key] = "tangential friction unavailable"
    elif channel_kind == "pair_spin":
        spin[phys_pair][5] = np.nan
        expected_key = f"spin:{phys_pair}"
        reasons[expected_key] = "point contact has no torsional friction"
    elif channel_kind == "hand_force":
        force["L"][5, 0] = np.nan
        expected_key = "force_L"
        reasons[expected_key] = "left hand resultant force unavailable"
    elif channel_kind == "hand_moment":
        moment["R"][5, 2] = np.nan
        expected_key = "moment_R"
        reasons[expected_key] = "right hand resultant moment unavailable"
    else:
        pytest.fail(f"unhandled channel_kind {channel_kind!r}")

    cand = replace(
        ref,
        pair_normal_force_n=normal,
        pair_tangential_force_n=tangential,
        pair_spin_moment_nm=spin,
        force_on_club_n=force,
        moment_on_club_nm=moment,
        channel_reasons=reasons,
    )

    for report in (
        compare_topology_series(ref, cand),
        compare_topology_series(cand, ref),
    ):
        assert report.passed is False
        assert report.status == "unavailable"
        ch_res = report.channels[expected_key]
        assert ch_res.status == "unavailable"
        assert ch_res.passed is False
        assert ch_res.reason == reasons[expected_key]
        assert expected_key in report.unavailable_reasons


def test_missing_channel_and_mapping_hash_mismatch_give_unavailable(
    overlap_model: TopologyContactModel,
) -> None:
    """AC 3: A missing channel gives unavailable; mapping-hash mismatch gives unavailable."""
    ref = _make_series(overlap_model)
    phys_pair = overlap_model.topology.hand_grip_pairs()[0].name

    # 1. Missing pair spin channel
    spin_missing = {k: v for k, v in ref.pair_spin_moment_nm.items() if k != phys_pair}
    cand_missing_ch = replace(ref, pair_spin_moment_nm=spin_missing)
    rep_missing = compare_topology_series(ref, cand_missing_ch)
    assert rep_missing.passed is False
    assert rep_missing.status == "unavailable"
    assert rep_missing.channels[f"spin:{phys_pair}"].status == "unavailable"
    assert rep_missing.channels[f"spin:{phys_pair}"].reason is not None

    # 2. Entire physical pair omitted from candidate
    cand_missing_pair = replace(
        ref,
        pair_normal_force_n={
            k: v for k, v in ref.pair_normal_force_n.items() if k != phys_pair
        },
        pair_tangential_force_n={
            k: v for k, v in ref.pair_tangential_force_n.items() if k != phys_pair
        },
        pair_spin_moment_nm=spin_missing,
        labels={k: v for k, v in ref.labels.items() if k != phys_pair},
    )
    rep_pair = compare_topology_series(ref, cand_missing_pair)
    assert rep_pair.passed is False
    assert rep_pair.status == "unavailable"
    assert rep_pair.channels[f"normal:{phys_pair}"].status == "unavailable"

    # 3. Mapping-hash mismatch
    cand_hash = replace(ref, mapping_hash="sha256:different_binding_hash")
    rep_hash = compare_topology_series(ref, cand_hash)
    assert rep_hash.passed is False
    assert rep_hash.status == "unavailable"
    assert rep_hash.reason == "mapping hash mismatch"


def test_identical_series_pass_and_scaled_normal_fails_peak(
    overlap_model: TopologyContactModel,
) -> None:
    """AC 4: Identical series pass; scaling normal force by 1.06 fails peak."""
    ref = _make_series(overlap_model)
    rep_ok = compare_topology_series(ref, ref)
    assert rep_ok.passed is True
    assert rep_ok.status == "passed"
    assert all(c.status == "passed" and c.passed for c in rep_ok.channels.values())

    cand_scaled = _make_series(overlap_model, normal_scale=1.06)
    rep_fail = compare_topology_series(ref, cand_scaled)
    assert rep_fail.passed is False
    assert rep_fail.status == "failed"
    phys_pair = overlap_model.topology.hand_grip_pairs()[0].name
    ch_norm = rep_fail.channels[f"normal:{phys_pair}"]
    assert ch_norm.status == "failed"
    assert ch_norm.passed is False
    assert ch_norm.error is not None
    assert ch_norm.error.peak_error == pytest.approx(0.06)
    assert ch_norm.error.peak_error > PEAK_TOLERANCE


def _with_spin(
    series: TopologyKineticsSeries, pair: str, values: np.ndarray
) -> TopologyKineticsSeries:
    """Return ``series`` with the spin channel of ``pair`` replaced by ``values``."""
    spin = dict(series.pair_spin_moment_nm)
    spin[pair] = values
    return replace(series, pair_spin_moment_nm=spin)


def test_numerical_zero_reference_spin_channel_is_compared_absolutely(
    overlap_model: TopologyContactModel,
) -> None:
    """A ~1e-14 N·m reference spin peak is numerical zero, not a 32 % miss.

    Reproduces the myoHand frozen-finger law-level failure on
    ``spin:trail_index_trigger_grip`` (reference 1.29e-14, candidate 8.75e-15).
    """
    assert NUMERICAL_ZERO_PEAK == 1e-8
    phys_pair = overlap_model.topology.hand_grip_pairs()[0].name
    ch = f"spin:{phys_pair}"
    base = _make_series(overlap_model)
    n = base.time_s.size
    rng = np.random.default_rng(13)
    ref_noise = 1.2e-14 * rng.uniform(-1.0, 1.0, n)
    ref_noise[n // 2] = 1.2910139841389068e-14
    cand_noise = 8.0e-15 * rng.uniform(-1.0, 1.0, n)
    cand_noise[n // 3] = 8.75405750832983e-15
    ref = _with_spin(base, phys_pair, ref_noise)
    cand = _with_spin(base, phys_pair, cand_noise)

    rep = compare_topology_series(ref, cand)
    assert rep.passed is True, rep.to_dict()["channels"][ch]
    err = rep.channels[ch].error
    assert err is not None
    assert err.numerical_zero is True
    assert err.peak_reference == pytest.approx(1.2910139841389068e-14)
    assert err.peak_error == 0.0 and err.rms_error == 0.0
    assert err.to_dict()["numerical_zero"] is True

    # A genuine candidate spin against a numerically-zero reference still fails.
    genuine = _with_spin(base, phys_pair, np.full(n, 0.08))
    rep_fail = compare_topology_series(ref, genuine)
    assert rep_fail.status == "failed"
    assert rep_fail.channels[ch].status == "failed"
    assert rep_fail.channels[ch].error is not None
    assert rep_fail.channels[ch].error.peak_error == float("inf")

    # Just above the floor the relative 5 % / 2 % metric still applies.
    small = _with_spin(base, phys_pair, np.full(n, 10.0 * NUMERICAL_ZERO_PEAK))
    small_cand = _with_spin(base, phys_pair, np.full(n, 7.0 * NUMERICAL_ZERO_PEAK))
    rep_small = compare_topology_series(small, small_cand)
    small_err = rep_small.channels[ch].error
    assert small_err is not None and small_err.numerical_zero is False
    assert small_err.peak_error == pytest.approx(0.3)
    assert rep_small.channels[ch].status == "failed"


def test_numerical_zero_floor_never_applies_to_strict_comparisons() -> None:
    """Without ``allow_zero_reference`` a tiny reference stays relative (no floor)."""
    ref = np.full((5, 3), 1e-14)
    err = compute_quantity_error(ref * 0.5, ref, name="force_L")
    assert err.numerical_zero is False
    assert err.peak_error == pytest.approx(0.5)


def test_allocated_channels_never_affect_physical_verdict(
    overlap_model: TopologyContactModel,
) -> None:
    """AC 5: Allocated channels are reported separately and never affect physical parity."""
    # 1. Both sides have finite allocated channels, but candidate's allocated forces
    #    are scaled by 5.0 (400 % error). Physical channels are identical -> PASSES.
    ref = _make_series(overlap_model, allocated_finite=True, allocated_scale=1.0)
    cand = _make_series(overlap_model, allocated_finite=True, allocated_scale=5.0)
    report = compare_topology_series(ref, cand)
    assert report.passed is True
    assert report.status == "passed"
    assert len(report.allocated_channels) > 0
    assert all(
        c.status == "failed" and c.label == "allocated"
        for c in report.allocated_channels.values()
    )

    # 2. Candidate has NaN allocated channels while reference has finite allocated
    #    channels. Physical channels are identical -> PASSES.
    cand_nan_alloc = _make_series(overlap_model, allocated_finite=False)
    report_nan_alloc = compare_topology_series(ref, cand_nan_alloc)
    assert report_nan_alloc.passed is True
    assert report_nan_alloc.status == "passed"
    assert all(
        c.status == "unavailable" for c in report_nan_alloc.allocated_channels.values()
    )


class _SpyAdapter:
    """Spy implementing ModelGripAdapter to verify zero allocated force application."""

    def __init__(
        self, binding: HandClubBinding, states: Mapping[str, np.ndarray]
    ) -> None:
        self._binding = binding
        self._states = dict(states)
        self.applied: list[dict[str, tuple[np.ndarray, np.ndarray]]] = []

    def capabilities(self) -> GripCapabilities:
        return GripCapabilities(
            attachment_modes=("contact", "contact_v1"),
            supports_physical_pairs=True,
            supports_wrench_application=True,
            native_contact=False,
            available_bodies=("club", "hand_l", "hand_r"),
        )

    def resolve_binding(self, topology: ContactTopology) -> HandClubBinding:
        return self._binding

    def sample_states(self, binding: HandClubBinding) -> BodyStates:
        return BodyStates(self._states)

    def apply_wrenches(
        self, wrenches: Mapping[str, tuple[np.ndarray, np.ndarray]]
    ) -> None:
        self.applied.append({k: (f.copy(), m.copy()) for k, (f, m) in wrenches.items()})


def test_zero_allocated_force_applied_in_physical_run(
    overlap_model: TopologyContactModel,
) -> None:
    """AC 6: Zero allocated force is applied in any physical run, and guard blocks tampering."""
    topo = overlap_model.topology
    sites = bind_rigid_hand_sites(
        topo, lead_hand_body="hand_l", trail_hand_body="hand_r"
    )
    binding = HandClubBinding(
        model_id="test_golfer_v1",
        model_hash="sha256:binding_v1",
        handedness="right",
        attachment_mode="contact",
        club_body="club",
        lead_hand_body="hand_l",
        trail_hand_body="hand_r",
        sites=sites,
        unavailable_sites={},
    )
    states = {"club": np.eye(4), "hand_l": np.eye(4), "hand_r": np.eye(4)}
    spy = _SpyAdapter(binding, states)

    closure_stub = SimpleNamespace(
        available=True,
        reason="",
        wrenches=[
            SimpleNamespace(
                force_n=np.array([400.0, -250.0, 150.0]),
                torque_nm=np.array([12.0, -8.0, 4.0]),
                point_m=np.zeros(3),
            )
        ],
    )

    times = [0.0, 0.001, 0.002, 0.003, 0.004]
    reports: list[GripStepReport] = []
    for _ in times:
        rep = evaluate_contacts(overlap_model, binding, states, closure=closure_stub)
        # Allocated pairs have finite positive normal force reported:
        for hh in topo.hand_hand_pairs():
            assert rep.pair_labels[hh.name] == "allocated"
            assert np.isfinite(rep.normal_force_n[hh.name])
            assert rep.normal_force_n[hh.name] > 0.0
        # Yet applied allocated force is strictly 0.0 N:
        assert rep.applied_allocated_force_n() == 0.0
        apply_report(spy, rep)
        reports.append(rep)

    assert len(spy.applied) == len(times)
    assert total_applied_allocated_force_n(reports) == 0.0

    # Build series from step reports (resampling from 1 ms onto 2 ms grid)
    series = TopologyKineticsSeries.from_step_reports(
        times, reports, engine="spy_engine", binding=binding
    )
    assert series.applied_allocated_force_n == 0.0
    np.testing.assert_allclose(series.time_s, [0.0, 0.002, 0.004])

    # Tamper with a report to slip an allocated pair into pair_wrenches -> caught!
    phys_pw = next(iter(reports[0].pair_wrenches.values()))
    alloc_name = topo.hand_hand_pairs()[0].name
    tampered_pw = dict(reports[0].pair_wrenches)
    tampered_pw[alloc_name] = phys_pw
    tampered_report = replace(reports[0], pair_wrenches=tampered_pw)
    assert tampered_report.applied_allocated_force_n() > 0.0
    with pytest.raises(ValueError, match="allocated"):
        apply_report(spy, tampered_report)


@pytest.mark.slow
@pytest.mark.requires_pinocchio
def test_pinocchio_topology_hold_passes_against_law_level_series(
    spec_dict: dict[str, Any],
    interface: GripInterface,
    overlap_model: TopologyContactModel,
) -> None:
    """AC 7: G6 Pinocchio 0.04 s hold converted with from_contact_run passes against law_level_series."""
    try:
        import pinocchio  # noqa: F401
    except ImportError as exc:
        pytest.skip(f"pinocchio not installed: {exc}")

    from src.engines.physics_engines.pinocchio.python import grip_contact_sim

    spec_bytes = SPEC_PATH.read_bytes()
    names = spec_dict["coordinate_order"]
    swing = load_coordinate_swing(
        FIXTURES / "swing_q_driver.npz",
        FIXTURES / "address_poses.json",
        "driver",
        names,
    )
    run = grip_contact_sim.simulate_grip_contact(
        spec_bytes,
        swing,
        interface=interface,
        t_end_s=0.04,
        hold=True,
        topology_model=overlap_model,
    )
    assert run.hand_states is not None
    assert run.club_states is not None

    mapping_hash = "sha256:g6_pinocchio_overlap_v1"
    cand = TopologyKineticsSeries.from_contact_run(
        run,
        overlap_model,
        engine="pinocchio",
        mapping_hash=mapping_hash,
    )
    ref = law_level_series(
        overlap_model,
        run.hand_states,
        run.club_states,
        run.series.time_s,
        mapping_hash=mapping_hash,
    )

    report = compare_topology_series(ref, cand)
    assert report.passed is True, f"Pinocchio hold failed parity: {report.to_dict()}"
    assert report.status == "passed"
    for ch_name, ch_res in report.channels.items():
        assert ch_res.passed is True, f"Channel {ch_name} failed: {ch_res}"
        assert ch_res.error is not None
        assert ch_res.error.peak_error <= PEAK_TOLERANCE
        assert ch_res.error.rms_error <= RMS_TOLERANCE


def _attribute_chain_depth(node: ast.AST) -> int:
    """Count chained attribute accesses (including through calls/subscripts)."""
    depth = 0
    cur = node
    while True:
        if isinstance(cur, ast.Attribute):
            depth += 1
            cur = cur.value
        elif isinstance(cur, ast.Call):
            cur = cur.func
        elif isinstance(cur, ast.Subscript):
            cur = cur.value
        else:
            break
    return depth


def test_ast_lod_and_function_budgets() -> None:
    """AST gate: <= 2 dot chain depth, <= 100 lines/function, <= 8 params, <= 1200 lines/file."""
    for path in (TOPOLOGY_SERIES_PY, PARITY_PY, BINDING_PY):
        source = path.read_text(encoding="utf-8")
        lines = source.splitlines()
        assert len(lines) <= 1200, f"{path.name} has {len(lines)} lines > 1200"
        tree = ast.parse(source, filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                depth = _attribute_chain_depth(node)
                assert depth <= 2, (
                    f"{path.name}:{node.lineno} has attribute chain depth {depth} > 2: "
                    f"{ast.unparse(node)}"
                )
            if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                assert node.end_lineno is not None
                length = node.end_lineno - node.lineno + 1
                assert length <= 100, f"{path.name}:{node.name} is {length} lines > 100"
                args = node.args
                n_params = (
                    len(args.posonlyargs)
                    + len(args.args)
                    + len(args.kwonlyargs)
                    + (1 if args.vararg else 0)
                    + (1 if args.kwarg else 0)
                )
                if args.args and args.args[0].arg in ("self", "cls"):
                    n_params -= 1
                assert n_params <= 8, (
                    f"{path.name}:{node.name} has {n_params} params > 8"
                )
