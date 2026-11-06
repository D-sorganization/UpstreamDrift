"""Law of Demeter guard for the table shot model (fast-track fix after #14726).

``TableShotModel.sample`` used ``self.table.provenance.get(...)``, a chain three
deep that the repo-wide ``check_lod`` gate rejects.  The table now answers the
question itself through ``DispersionTable.club_label``.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from src.shared.python.sg_optimizer.shot_model.dispersion_table import DispersionTable

ROOT = Path(__file__).resolve().parents[4]
TABLE_MODEL = ROOT / "src/shared/python/sg_optimizer/shot_model/table_model.py"


def _check_lod():
    spec = importlib.util.spec_from_file_location(
        "check_lod_under_test", ROOT / "scripts/ci/check_lod.py"
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.unit
def test_table_model_has_no_deep_attribute_chain() -> None:
    assert _check_lod().check_file(TABLE_MODEL) == []


@pytest.mark.unit
def test_club_label_maps_skill_to_the_recorded_club() -> None:
    table = DispersionTable(cells=(), provenance={"clubs": {"driver": "Driver"}})
    assert table.club_label("driver") == "Driver"


@pytest.mark.unit
def test_club_label_defaults_to_the_skill_name() -> None:
    assert DispersionTable(cells=(), provenance={}).club_label("wedge") == "wedge"
    assert DispersionTable(cells=(), provenance={}).club_label(None) is None
