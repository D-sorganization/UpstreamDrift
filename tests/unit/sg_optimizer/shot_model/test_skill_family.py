"""``SkillFamily`` schema, presets, registry and ``PlayerProfile`` link (#14566)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_parameters.extract import ImpactParameters
from src.shared.python.sg_optimizer.shot_model.player_profile import PlayerProfile
from src.shared.python.sg_optimizer.shot_model.skill_family import (
    IMPACT_FIELDS,
    SKILL_NAMES,
    UNITS,
    ImpactDistribution,
    Provenance,
    SkillFamily,
)
from src.shared.python.sg_optimizer.shot_model.skill_registry import (
    derive_family,
    available_presets,
    get_skill_family,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
PGA_TOUR_YAML = REPO_ROOT / "data" / "sg_optimizer" / "baselines" / "pga_tour.yaml"
_SD = (2.0, 1.5, 2.0, 1.5, 6.0, 4.0)


def _dist(**kw: object) -> ImpactDistribution:
    args: dict[str, object] = {
        "mean": (100.0, 0.0, 0.0, -2.0, 0.0, 0.0),
        "sd": _SD,
        "corr": None,
    }
    args.update(kw)
    return ImpactDistribution(**args)  # type: ignore[arg-type]


def test_fields_reuse_impact_parameters_vocabulary() -> None:
    """Single vocabulary: every field is an ``ImpactParameters`` attribute."""
    known = set(ImpactParameters.__dataclass_fields__)
    assert set(IMPACT_FIELDS) <= known


def test_covariance_is_psd_and_matches_sd() -> None:
    d = _dist()
    cov = d.covariance()
    assert cov.shape == (6, 6)
    assert np.all(np.linalg.eigvalsh(cov) >= -1e-12)
    np.testing.assert_allclose(np.sqrt(np.diag(cov)), _SD)


def test_non_psd_correlation_rejected() -> None:
    bad = np.eye(6)
    bad[0, 1] = bad[1, 0] = 0.99
    bad[1, 2] = bad[2, 1] = 0.99
    bad[0, 2] = bad[2, 0] = -0.99
    with pytest.raises(ContractViolationError, match="PSD"):
        _dist(corr=tuple(map(tuple, bad)))


def test_negative_sd_and_nonfinite_mean_rejected() -> None:
    with pytest.raises(ContractViolationError):
        _dist(sd=(2.0, -1.0, 2.0, 1.5, 6.0, 4.0))
    with pytest.raises(ContractViolationError):
        _dist(mean=(float("nan"), 0, 0, 0, 0, 0))
    with pytest.raises(ContractViolationError):
        _dist(mean=(-5.0, 0, 0, 0, 0, 0))  # negative speed


def test_sample_reproduces_and_matches_moments() -> None:
    d = _dist()
    a = d.sample(np.random.default_rng(4), 20000)
    b = d.sample(np.random.default_rng(4), 20000)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (20000, 6)
    np.testing.assert_allclose(a.mean(axis=0), d.mean, atol=0.2)
    np.testing.assert_allclose(a.std(axis=0), _SD, rtol=0.05)


def test_provenance_requires_known_source_kind() -> None:
    Provenance(kind="preset", source="x", version="1")
    with pytest.raises(ContractViolationError):
        Provenance(kind="guess", source="x", version="1")
    with pytest.raises(ContractViolationError):
        Provenance(kind="fit", source="", version="1")


def test_presets_cover_every_level_and_skill() -> None:
    assert {"tour", "scratch", "10_hcp", "20_hcp"} <= set(available_presets())
    for name in ("tour", "scratch", "10_hcp", "20_hcp"):
        fam = get_skill_family(name)
        assert set(fam.skills) == set(SKILL_NAMES)
        assert fam.provenance.kind == "preset"
        assert fam.provenance.source  # cited, or marked assumption
        for spec in fam.skills.values():
            assert spec.provenance.source


def test_presets_order_by_skill_level() -> None:
    sd_tour = get_skill_family("tour").skills["mid_iron"].distribution.sd
    sd_20 = get_skill_family("20_hcp").skills["mid_iron"].distribution.sd
    v_tour = get_skill_family("tour").skills["driver"].distribution.mean[0]
    v_20 = get_skill_family("20_hcp").skills["driver"].distribution.mean[0]
    assert all(b > a for a, b in zip(sd_tour, sd_20, strict=True))
    assert v_20 < v_tour


def test_unknown_preset_fails_closed() -> None:
    with pytest.raises(ContractViolationError, match="unknown skill family"):
        get_skill_family("nope")


def test_yaml_round_trip_is_lossless(tmp_path: Path) -> None:
    fam = get_skill_family("10_hcp")
    p = tmp_path / "fam.yaml"
    fam.to_yaml(p)
    back = SkillFamily.from_yaml(p)
    assert back == fam


def test_units_are_serialized_and_wrong_or_missing_units_rejected() -> None:
    raw = get_skill_family("tour").to_dict()
    assert raw["units"] == UNITS and UNITS["clubhead_speed_mph"] == "mph"
    assert SkillFamily.from_dict(raw) == get_skill_family("tour")
    for bad in (
        {**UNITS, "clubhead_speed_mph": "m/s"},
        {**UNITS, "face_angle_deg": "rad"},
    ):
        with pytest.raises(ContractViolationError, match="units"):
            SkillFamily.from_dict({**raw, "units": bad})
    del raw["units"]
    with pytest.raises(ContractViolationError, match="units"):
        SkillFamily.from_dict(raw)


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_derived_scales_must_be_finite_and_positive(bad: float) -> None:
    spec = {"source": "x", "version": "1", "speed_scale": bad, "dispersion_scale": 1.0}
    with pytest.raises(ContractViolationError, match="finite"):
        derive_family(get_skill_family("tour"), "x", spec)


def test_family_rejects_unknown_skill_and_empty() -> None:
    fam = get_skill_family("tour")
    with pytest.raises(ContractViolationError):
        SkillFamily(name="x", skills={}, provenance=fam.provenance)
    with pytest.raises(ContractViolationError):
        SkillFamily(
            name="x",
            skills={"nine_wood": fam.skills["driver"]},
            provenance=fam.provenance,
        )


def test_player_profile_carries_family_through_yaml(tmp_path: Path) -> None:
    fam = get_skill_family("scratch")
    prof = PlayerProfile(name="p", baseline=str(PGA_TOUR_YAML), skill_family=fam)
    out = tmp_path / "p.yaml"
    prof.to_yaml(out)
    assert PlayerProfile.from_yaml(out).skill_family == fam


def test_player_profile_without_family_unchanged(tmp_path: Path) -> None:
    prof = PlayerProfile(name="p", baseline=str(PGA_TOUR_YAML))
    out = tmp_path / "p.yaml"
    prof.to_yaml(out)
    assert "skill_family" not in out.read_text(encoding="utf-8")
    assert PlayerProfile.from_yaml(out).skill_family is None
