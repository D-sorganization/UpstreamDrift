"""Player matching: fit a ``SkillFamily`` from shot data with prior shrinkage (#14571)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.sg_optimizer.shot_model.skill_family import (
    IMPACT_FIELDS,
    ImpactDistribution,
    SkillFamily,
)
from src.shared.python.sg_optimizer.shot_model.skill_fit import (
    PRIOR_STRENGTH,
    club_to_skill,
    fit_skill_family,
    read_shot_csv,
    shrinkage_weight,
)
from src.shared.python.sg_optimizer.shot_model.skill_registry import get_skill_family

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
SAMPLE = ROOT / "data/sg_optimizer/skill_families/sample_shots.csv"
TRUTH = ImpactDistribution(
    mean=(100.0, 1.0, -2.0, -3.0, 3.0, 2.0),
    sd=(4.0, 2.0, 3.0, 2.0, 7.0, 4.0),
    corr=tuple(
        tuple(0.4 if {i, j} == {1, 2} else float(i == j) for j in range(6))
        for i in range(6)
    ),
)


def _draw(n: int, seed: int = 0) -> np.ndarray:
    return TRUTH.sample(np.random.default_rng(seed), n)


def test_sample_csv_fits_a_family_and_keeps_unobserved_skills_as_preset() -> None:
    prior = get_skill_family("10_hcp")
    fam = fit_skill_family(read_shot_csv(SAMPLE), prior, name="sample_player")
    assert isinstance(fam, SkillFamily) and fam.name == "sample_player"
    assert fam.skills["driver"].provenance.kind == "fit"
    assert "n=30" in fam.skills["driver"].provenance.source
    assert fam.skills["mid_iron"].provenance.kind == "fit"
    assert fam.skills["wedge"] == prior.skills["wedge"]  # no data -> preset
    assert fam.provenance.kind == "fit"


def test_large_sample_recovers_the_truth_and_the_fitted_ellipsoid_covers() -> None:
    prior = get_skill_family("tour")
    fam = fit_skill_family({"driver": _draw(4000)}, prior, name="p")
    fitted = fam.skills["driver"].distribution
    se = np.array(TRUTH.sd) / np.sqrt(4000)
    assert np.all(np.abs(np.array(fitted.mean) - TRUTH.mean) < 4 * se + 0.2)
    held_out = _draw(4000, seed=1) - np.array(fitted.mean)
    d2 = np.einsum(
        "ij,jk,ik->i", held_out, np.linalg.inv(fitted.covariance()), held_out
    )
    chi2_95 = 12.5916  # 6 degrees of freedom
    assert 0.93 < float(np.mean(d2 <= chi2_95)) < 0.97


def test_small_n_falls_back_toward_the_preset() -> None:
    prior = get_skill_family("tour")
    pm = prior.skills["driver"].distribution.mean[0]
    data = _draw(60)
    gaps = []
    for n in (2, 8, 60):
        fam = fit_skill_family({"driver": data[:n]}, prior, name="p")
        gaps.append(abs(fam.skills["driver"].distribution.mean[0] - pm))
    sample_gap = abs(data[:2, 0].mean() - pm)
    assert gaps[0] < sample_gap  # n=2 sits between the preset and the sample
    assert gaps[0] < gaps[1] < gaps[2] + 1e-9  # more data pulls away from the preset


def test_shrinkage_weight_is_monotone_and_bounded() -> None:
    ws = [shrinkage_weight(n) for n in (0, 1, 5, 50, 5000)]
    assert ws[0] == 0.0 and ws == sorted(ws) and ws[-1] < 1.0
    assert shrinkage_weight(PRIOR_STRENGTH) == pytest.approx(0.5)


def test_fewer_than_two_shots_keep_the_preset() -> None:
    prior = get_skill_family("tour")
    fam = fit_skill_family({"driver": _draw(1)}, prior, name="p")
    assert fam.skills["driver"] == prior.skills["driver"]


def test_absent_fields_fall_back_to_the_preset_distribution() -> None:
    prior = get_skill_family("tour")
    data = _draw(200)
    data[:, 4:] = np.nan  # launch monitor without strike location
    d = (
        fit_skill_family({"driver": data}, prior, name="p")
        .skills["driver"]
        .distribution
    )
    p = prior.skills["driver"].distribution
    assert d.mean[4:] == p.mean[4:] and d.sd[4:] == p.sd[4:]
    assert d.mean[0] != p.mean[0]


def test_fitted_covariance_is_psd_and_roundtrips(tmp_path: Path) -> None:
    fam = fit_skill_family({"driver": _draw(12)}, get_skill_family("tour"), name="p")
    cov = fam.skills["driver"].distribution.covariance()
    assert np.linalg.eigvalsh(cov).min() > -1e-9
    fam.to_yaml(tmp_path / "f.yaml")
    assert SkillFamily.from_yaml(tmp_path / "f.yaml").skills[
        "driver"
    ].distribution.mean == (pytest.approx(fam.skills["driver"].distribution.mean))


@pytest.mark.parametrize(
    ("club", "skill"),
    [("Driver", "driver"), ("3 Wood", "fairway_wood"), ("4 Iron", "long_iron"),
     ("7 Iron", "mid_iron"), ("9 Iron", "short_iron"), ("PW", "wedge"),
     ("Sand Wedge", "wedge"), ("putter", None)],
)  # fmt: skip
def test_club_names_map_to_skills(club: str, skill: str | None) -> None:
    assert club_to_skill(club) == skill


def test_bad_inputs_fail_closed(tmp_path: Path) -> None:
    prior = get_skill_family("tour")
    with pytest.raises(ContractViolationError):
        fit_skill_family({"putter": _draw(5)}, prior, name="p")
    bad = _draw(5)
    bad[0, 0] = np.inf
    with pytest.raises(ContractViolationError):
        fit_skill_family({"driver": bad}, prior, name="p")
    with pytest.raises(ContractViolationError):
        fit_skill_family({"driver": _draw(5)}, prior, name="p", prior_strength=0.0)
    csv = tmp_path / "s.csv"
    csv.write_text("skill,clubhead_speed_mph\nputter,50\n", encoding="utf-8")
    with pytest.raises(ContractViolationError):
        read_shot_csv(csv)
    assert len(IMPACT_FIELDS) == 6
