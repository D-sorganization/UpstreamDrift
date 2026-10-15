"""Tests for golf literature context bands and their annotation (SPN-11b).

Every band value is cross-checked against the JSON rows of
``docs/research/segmental_spine/data_extraction.md``. Bands are reported beside
model results and are never acceptance criteria: ``is_acceptance`` is always
``False``.
"""

from __future__ import annotations

import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.core.constants import GRAVITY_FLOAT
from src.shared.python.spine.context_bands import (
    ContextBand,
    ContextComparison,
    annotate,
    context_bands,
    load_context_bands,
)
from src.shared.python.spine.levels import SpinalLevel
from src.shared.python.spine.loads import (
    REQUIRED_PROVENANCE_KEYS,
    LoadObservable,
    SpinalLoadSeries,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
EXTRACTION = REPO_ROOT / "docs" / "research" / "segmental_spine" / "data_extraction.md"
BANDS_JSON = REPO_ROOT / "src" / "config" / "spine_golf_context_bands.json"
EXPECTED_UNITS = {"G1_L3L4_loads": "N", "G2_L4L5_loads_abstract": "BW"}
MASS_KG = 80.0
WEIGHT_N = MASS_KG * GRAVITY_FLOAT
QUANTITIES = ("compression", "ap_shear", "ml_shear")


def _extraction_tables() -> dict[str, dict]:
    """Return every ```json block of the extraction keyed by table_id."""
    text = EXTRACTION.read_text(encoding="utf-8")
    tables: dict[str, dict] = {}
    for block in re.findall(r"```json\n(.*?)\n```", text, flags=re.DOTALL):
        payload = json.loads(block)
        if "table_id" in payload:
            tables[payload["table_id"]] = payload
    return tables


def _numbers(node: object) -> list[float]:
    """Collect every numeric leaf of a JSON-like structure."""
    if isinstance(node, bool):
        return []
    if isinstance(node, int | float):
        return [float(node)]
    if isinstance(node, dict):
        return [v for item in node.values() for v in _numbers(item)]
    if isinstance(node, list):
        return [v for item in node for v in _numbers(item)]
    return []


def _prov() -> dict[str, str]:
    return {key: f"value-{key}" for key in REQUIRED_PROVENANCE_KEYS}


def _series(levels, force, *, time_s=(0.0, 0.1)) -> SpinalLoadSeries:
    force = np.asarray(force, dtype=float)
    return SpinalLoadSeries(
        time_s=np.asarray(time_s, dtype=float),
        levels=levels,
        observable=LoadObservable.JOINT_REACTION,
        force_n=force,
        moment_nm=np.zeros_like(force),
        provenance=_prov(),
    )


def _synthetic_series() -> SpinalLoadSeries:
    """Two samples, levels (L4_L5, L3_L4); peak values are exact by design.

    L4_L5 compression peaks at 900 N (sample 1); L4_L5 AP shear at -100 N
    (sample 0); L3_L4 compression peaks at 3792 N (sample 1). Frames: index 1
    is the Y (compression) component; index 0 is X (AP shear).
    """
    force = np.zeros((2, 2, 3))
    force[0, 0, 1] = 200.0
    force[1, 0, 1] = 900.0
    force[0, 0, 0] = -100.0
    force[1, 0, 0] = 50.0
    force[0, 1, 1] = 500.0
    force[1, 1, 1] = 3792.0
    return _series((SpinalLevel.L4_L5, SpinalLevel.L3_L4), force)


# --- Data file ---------------------------------------------------------------


def test_bands_load_from_repository_file():
    bands = load_context_bands()
    assert isinstance(bands, tuple)
    assert len(bands) == 5
    assert all(isinstance(band, ContextBand) for band in bands)


def test_every_band_value_matches_data_extraction():
    tables = _extraction_tables()
    for band in load_context_bands():
        table = tables.get(band.source_table_id)
        assert table is not None, f"{band.band_id}: no table {band.source_table_id}"
        assert band.value in _numbers(table["rows"]), band.band_id
        assert band.verified_from_primary is table["verified_from_primary"], band.band_id
        assert band.units == EXPECTED_UNITS[band.source_table_id], band.band_id


def test_units_and_body_weight_flag_are_consistent():
    for band in load_context_bands():
        assert band.body_weight_normalised is (band.units == "BW"), band.band_id


def test_only_hosea_and_lim_golf_studies_are_present():
    studies = {band.study for band in load_context_bands()}
    assert studies == {"Hosea 1990", "Lim 2012"}


def test_hosea_bands_are_secondary_and_lim_bands_are_primary_abstract():
    bands = load_context_bands()
    hosea = [b for b in bands if b.study == "Hosea 1990"]
    lim = [b for b in bands if b.study == "Lim 2012"]
    assert hosea and lim
    assert all(b.verified_from_primary is False for b in hosea)
    assert all(b.verified_from_primary is True for b in lim)


def test_loader_rejects_unknown_units(tmp_path):
    bad = tmp_path / "bands.json"
    rows = json.loads(BANDS_JSON.read_text(encoding="utf-8"))
    rows[0]["units"] = "kN"
    bad.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError):
        load_context_bands(bad)


def test_loader_rejects_nonpositive_value(tmp_path):
    bad = tmp_path / "bands.json"
    rows = json.loads(BANDS_JSON.read_text(encoding="utf-8"))
    rows[0]["value"] = 0
    bad.write_text(json.dumps(rows), encoding="utf-8")
    with pytest.raises(ValueError):
        load_context_bands(bad)


def test_loader_rejects_non_list_document(tmp_path):
    bad = tmp_path / "bands.json"
    bad.write_text(json.dumps({"bands": []}), encoding="utf-8")
    with pytest.raises(TypeError):
        load_context_bands(bad)


# --- ContextBand and ContextComparison ---------------------------------------


def test_context_band_is_frozen():
    band = load_context_bands()[0]
    with pytest.raises(AttributeError):
        band.value = 1.0  # type: ignore[misc]


def test_comparison_is_acceptance_defaults_false():
    band = load_context_bands()[0]
    comparison = ContextComparison(band, 1.0, band.units, 1.0 / band.value)
    assert comparison.is_acceptance is False


def test_comparison_cannot_be_marked_acceptance():
    band = load_context_bands()[0]
    with pytest.raises(ValueError):
        ContextComparison(band, 1.0, band.units, 1.0, is_acceptance=True)  # type: ignore[arg-type]


# --- Filtering ---------------------------------------------------------------


def test_filter_by_level_and_quantity():
    hosea = context_bands(SpinalLevel.L3_L4, "compression")
    assert isinstance(hosea, tuple)
    assert {b.subjects for b in hosea} == {
        "4 professionals, mean age 37",
        "4 amateurs, mean age 34",
    }
    lim = context_bands(SpinalLevel.L4_L5, "ap_shear")
    assert len(lim) == 1 and lim[0].study == "Lim 2012"
    assert lim[0].value == pytest.approx(1.6)


def test_filter_returns_empty_when_no_band_matches():
    assert context_bands(SpinalLevel.T1_T2, "compression") == ()


def test_filter_rejects_unknown_quantity():
    with pytest.raises(ValueError):
        context_bands(SpinalLevel.L3_L4, "torque")


def test_filter_rejects_non_level():
    with pytest.raises(TypeError):
        context_bands("L3_L4", "compression")  # type: ignore[arg-type]


# --- Annotation --------------------------------------------------------------


def _by_band(comparisons):
    return {c.band.band_id: c for c in comparisons}


def test_annotate_compares_matching_quantity_in_band_units():
    comparisons = annotate(_synthetic_series(), MASS_KG)
    by_id = _by_band(comparisons)
    hosea_pro = next(b for b in load_context_bands() if "professional" in b.subjects)
    lim_comp = next(
        b for b in load_context_bands() if b.study == "Lim 2012" and b.quantity == "compression"
    )
    lim_ap = next(
        b for b in load_context_bands() if b.study == "Lim 2012" and b.quantity == "ap_shear"
    )
    assert by_id[hosea_pro.band_id].model_units == "N"
    assert by_id[hosea_pro.band_id].model_peak == pytest.approx(3792.0)
    assert by_id[hosea_pro.band_id].ratio == pytest.approx(3792.0 / 7584.0)
    assert by_id[lim_comp.band_id].model_units == "BW"
    assert by_id[lim_comp.band_id].model_peak == pytest.approx(900.0 / WEIGHT_N)
    assert by_id[lim_comp.band_id].ratio == pytest.approx(900.0 / WEIGHT_N / 6.0)
    assert by_id[lim_ap.band_id].model_peak == pytest.approx(100.0 / WEIGHT_N)


def test_annotate_only_covers_levels_present_in_series():
    comparisons = annotate(_synthetic_series(), MASS_KG)
    levels = {c.band.level for c in comparisons}
    assert levels == {SpinalLevel.L4_L5, SpinalLevel.L3_L4}


def test_annotate_result_is_independent_of_series_units():
    newton = annotate(_synthetic_series(), MASS_KG)
    bw = annotate(_synthetic_series().in_body_weights(MASS_KG), MASS_KG)
    newton_ratio = _by_band(newton)
    bw_ratio = _by_band(bw)
    assert newton_ratio.keys() == bw_ratio.keys()
    for band_id, comparison in newton_ratio.items():
        assert bw_ratio[band_id].ratio == pytest.approx(comparison.ratio)
        assert bw_ratio[band_id].model_units == comparison.model_units


def test_annotate_all_comparisons_are_not_acceptance():
    assert all(c.is_acceptance is False for c in annotate(_synthetic_series(), MASS_KG))


def test_annotate_rejects_non_series():
    with pytest.raises(TypeError):
        annotate(object(), MASS_KG)  # type: ignore[arg-type]


@pytest.mark.parametrize("mass", [0.0, -70.0])
def test_annotate_rejects_non_positive_mass(mass):
    with pytest.raises(ValueError):
        annotate(_synthetic_series(), mass)


def test_annotate_rejects_non_numeric_mass():
    with pytest.raises(TypeError):
        annotate(_synthetic_series(), "80")  # type: ignore[arg-type]


def test_annotate_rejects_bool_mass():
    with pytest.raises(TypeError):
        annotate(_synthetic_series(), True)  # type: ignore[arg-type]


def test_replace_cannot_flip_acceptance():
    band = load_context_bands()[0]
    comparison = ContextComparison(band, 1.0, band.units, 1.0)
    with pytest.raises(ValueError):
        replace(comparison, is_acceptance=True)  # type: ignore[arg-type]
