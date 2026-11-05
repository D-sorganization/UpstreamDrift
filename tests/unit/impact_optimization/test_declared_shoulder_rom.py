"""Shoulder ROM (D35/D39, #14557): per-side cited vs assumed declared ranges.

The v2 spec carries no shoulder range (``coordinate_ranges_deg`` is a hard IK
bound, so a design guess must not go there).  The ranges live in
``impact_envelope_limits.yaml`` under ``declared_rom_deg`` with one citation
and ``verification_status`` PER SIDE, because one coordinate is cited on one
side only (flexion, external rotation: Gill 2020) and assumed on the other
(extension, internal rotation, horizontal: nothing read in full text).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.kinematic_binding import (
    kinematic_binding,
)
from src.shared.python.impact_optimization.physiological_envelope import (
    DEFAULT_LIMITS_PATH,
    PhysiologicalEnvelope,
    is_assumed,
    load_declared_rom,
    load_limits,
)
from src.shared.python.loop_nullspace.capacity import CapacitySubject

pytestmark = pytest.mark.unit

SPEC_V2 = json.loads(
    Path(
        "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
    ).read_text(encoding="utf-8")
)
SUBJECT = CapacitySubject(mass_kg=78.0, stature_m=1.71)
SHOULDERS = tuple(f"{s}SInput{a}" for s in "LR" for a in "XYZ")


def _env(**kw) -> PhysiologicalEnvelope:
    return PhysiologicalEnvelope.from_spec(
        SPEC_V2, SUBJECT, load_limits(DEFAULT_LIMITS_PATH), **kw
    )


@pytest.fixture(scope="module")
def env() -> PhysiologicalEnvelope:
    return _env()


def _rom(env: PhysiologicalEnvelope, name: str) -> tuple[float, float]:
    lo, hi = env.rom_deg[env.names.index(name)]
    return float(lo), float(hi)


def test_v2_builds_with_no_unbounded_shoulder(env: PhysiologicalEnvelope) -> None:
    for name in SHOULDERS:
        lo, hi = _rom(env, name)
        assert np.isfinite(lo) and np.isfinite(hi) and lo < hi, name
        assert name not in env.unbounded_rom_slots


def test_fail_closed_without_declared_rom_names_the_shoulders() -> None:
    with pytest.raises(ValueError, match="LSInputX") as exc:
        _env(declared_rom={})
    assert "RSInputZ" in str(exc.value)
    assert "lumbar" not in str(exc.value)


@pytest.mark.parametrize(
    ("name", "side", "value"),
    [
        # flexion: Y lower = 90 - (mean + 1 SD of the male sample), Table 1
        ("LSInputY", 0, 90.0 - (159.9 + 18.1)),
        ("RSInputY", 0, 90.0 - (161.5 + 18.7)),
        # external rotation: left Z+ is ER, right Z+ is IR (sign map), Table 3
        ("LSInputZ", 1, 55.0 + 17.9),
        ("RSInputZ", 0, -(56.8 + 17.6)),
    ],
)
def test_cited_sides_reproduce_gill_2020_male_mean_plus_one_sd(
    env: PhysiologicalEnvelope, name: str, side: int, value: float
) -> None:
    assert _rom(env, name)[side] == pytest.approx(value, abs=0.051)


def test_exactly_the_uncited_sides_are_listed_as_assumed(
    env: PhysiologicalEnvelope,
) -> None:
    expected = {
        (f"{s}SInput{a}", side)
        for s in "LR"
        for a, side in (("X", "lower"), ("X", "upper"), ("Y", "upper"))
    } | {("LSInputZ", "lower"), ("RSInputZ", "upper")}
    assert set(env.assumed_rom_sides) == expected


def test_declared_rom_records_source_per_side() -> None:
    declared = load_declared_rom(DEFAULT_LIMITS_PATH)
    assert set(declared) == set(SHOULDERS)
    cited = [
        (n, s)
        for n, d in declared.items()
        for s, side in (("lower", d.lower), ("upper", d.upper))
        if side.verification_status == "VERIFIED"
    ]
    assert len(cited) == 4  # flexion and ER, both arms
    for n, d in declared.items():
        for side in (d.lower, d.upper):
            if side.verification_status == "VERIFIED":
                assert "Gill" in side.citation, n
                assert "10.1186/s12891-020-03665-9" in side.citation, n


def test_a_hard_spec_range_wins_over_a_declared_one() -> None:
    spec = json.loads(json.dumps(SPEC_V2))
    spec["coordinate_ranges_deg"]["LSInputY"] = [-30.0, 30.0]
    env = PhysiologicalEnvelope.from_spec(
        spec, SUBJECT, load_limits(DEFAULT_LIMITS_PATH)
    )
    assert _rom(env, "LSInputY") == (-30.0, 30.0)
    assert not any(j == "LSInputY" for j, _ in env.assumed_rom_sides)


def _traj(env: PhysiologicalEnvelope, name: str, peak_deg: float) -> np.ndarray:
    traj = np.zeros((4, len(env.names)))
    traj[2, env.names.index(name)] = peak_deg
    return np.radians(traj)


def test_hit_names_its_side_and_the_cited_side_is_not_assumed(
    env: PhysiologicalEnvelope,
) -> None:
    lo, _ = _rom(env, "LSInputY")
    hits = kinematic_binding(_traj(env, "LSInputY", lo - 5.0), np.arange(4) * 0.01, env)
    (h,) = [h for h in hits if h.joint == "LSInputY" and h.kind == "ROM"]
    assert h.side == "lower" and h.utilisation > 1.0
    assert not env.limit_is_assumed(h.joint, h.kind, h.side)  # cited flexion


def test_breach_on_an_assumed_side_reads_assumed(env: PhysiologicalEnvelope) -> None:
    _, hi = _rom(env, "LSInputY")  # extension side: no source read
    hits = kinematic_binding(_traj(env, "LSInputY", hi + 5.0), np.arange(4) * 0.01, env)
    (h,) = [h for h in hits if h.joint == "LSInputY" and h.kind == "ROM"]
    assert h.side == "upper"
    assert env.limit_is_assumed(h.joint, h.kind, h.side)
    assert is_assumed(h.joint, h.kind, h.side, env.assumed_limits())


def test_is_assumed_whole_joint_entry_covers_both_sides() -> None:
    assumed = frozenset({("SpineInputZ", "ROM")})
    assert is_assumed("SpineInputZ", "ROM", "lower", assumed)
    assert is_assumed("SpineInputZ", "ROM", "", assumed)
    assert not is_assumed("SpineInputY", "ROM", "upper", assumed)


def test_side_less_query_is_not_assumed_for_a_side_assumed_joint(
    env: PhysiologicalEnvelope,
) -> None:
    # LSInputY is assumed on ONE side only; without a side the answer is "no"
    # rather than guessing, so a cited breach is never relabelled.
    assert not env.limit_is_assumed("LSInputY", "ROM")


def _copy_yaml(tmp_path: Path, mutate) -> Path:
    raw = yaml.safe_load(DEFAULT_LIMITS_PATH.read_text(encoding="utf-8"))
    mutate(raw["declared_rom_deg"])
    out = tmp_path / "limits.yaml"
    out.write_text(yaml.safe_dump(raw), encoding="utf-8")
    return out


def test_loader_rejects_inverted_range(tmp_path: Path) -> None:
    def swap(entries) -> None:
        e = entries[0]
        e["lower"]["value"], e["upper"]["value"] = 10.0, -10.0

    with pytest.raises(ContractViolationError):
        load_declared_rom(_copy_yaml(tmp_path, swap))


def test_loader_rejects_a_verified_side_without_a_citation(tmp_path: Path) -> None:
    def strip(entries) -> None:
        for e in entries:
            for side in ("lower", "upper"):
                if e[side]["verification_status"] == "VERIFIED":
                    e[side]["citation"] = {"id": None, "note": "none"}
                    return

    with pytest.raises(ContractViolationError):
        load_declared_rom(_copy_yaml(tmp_path, strip))
