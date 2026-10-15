"""Read an OpenSim motion into a SegmentalKinematicsSeries (SPN-9d, #12759).

The reader is the inverse of ``drive.export_mot``: a drive -> ``.mot`` -> read
round trip must return the driven level angles.  All numbers are invented.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.spine.drive import (
    FIXTURE_PELVIS,
    DriveOptions,
    DriveResult,
    drive_segmental_model,
    export_mot,
)
from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    FixtureSpec,
    fixture_level_names,
    write_fixture_osim,
)
from src.engines.physics_engines.opensim.python.spine.read_kinematics import (
    read_segmental_kinematics,
)
from src.shared.python.engine_core.opensim_runtime import import_opensim
from src.shared.python.spine.segmental import SegmentalKinematicsSeries
from tests.helpers.opensim_support import require_opensim
from tests.opensim.spine.test_drive import _golf_input, _law

pytestmark = [pytest.mark.requires_opensim, pytest.mark.integration]


@pytest.fixture(scope="module", autouse=True)
def _osim():
    return require_opensim()


@pytest.fixture(scope="module")
def driven(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("read")
    spec = FixtureSpec()
    law = _law()
    model = write_fixture_osim(spec, tmp / "fixture.osim", coupling=law)
    golf, _phi = _golf_input(spec, law)
    result = drive_segmental_model(
        golf,
        model_path=model,
        model_id="fixture",
        law=law,
        options=DriveOptions(pelvis=FIXTURE_PELVIS),
    )
    mot = export_mot(result, tmp / "drive.mot")
    return spec, model, result, mot, tmp


def _read(model: Path, mot: Path, **kw) -> SegmentalKinematicsSeries:
    kw.setdefault("distribution_source", "coupling:invented@1")
    return read_segmental_kinematics(mot, model, "fixture", **kw)


def test_round_trip_recovers_level_angles(driven) -> None:
    _spec, model, result, mot, _tmp = driven
    series = _read(model, mot)
    assert isinstance(series, SegmentalKinematicsSeries)
    np.testing.assert_allclose(series.time_s, result.time_s, atol=1e-12)
    assert series.poses[0].levels == result.kinematics.poses[0].levels
    for got, want in zip(series.poses, result.kinematics.poses, strict=True):
        for g, w in zip(got.transforms, want.transforms, strict=True):
            np.testing.assert_allclose(g.isb_angles(), w.isb_angles(), atol=1e-10)
            np.testing.assert_allclose(g.matrix(), w.matrix(), atol=1e-10)
        np.testing.assert_allclose(got.compose(), want.compose(), atol=1e-10)
    assert series.valid.all()


def test_distribution_source_is_carried(driven) -> None:
    _spec, model, _result, mot, _tmp = driven
    assert _read(model, mot).distribution_source == "coupling:invented@1"


def test_header_source_must_agree(driven) -> None:
    _spec, model, result, mot, tmp = driven
    table = import_opensim().TimeSeriesTable(str(mot))
    assert table.getTableMetaDataAsString("distribution_source") == (
        result.kinematics.distribution_source
    )
    with pytest.raises(ValueError, match="distribution_source"):
        _read(model, mot, distribution_source="ik:other.sto")
    agree = _read(model, mot, distribution_source=result.kinematics.distribution_source)
    assert agree.distribution_source == result.kinematics.distribution_source


def _rewrite(mot: Path, out: Path, *, degrees: bool, drop: str | None, header: bool):
    osim = import_opensim()
    table = osim.TimeSeriesTable(str(mot))
    labels = [str(x) for x in table.getColumnLabels()]
    new = osim.TimeSeriesTable()
    keep = [lab for lab in labels if lab != drop]
    vec = osim.StdVectorString()
    for lab in keep:
        vec.append(lab)
    new.setColumnLabels(vec)
    pelvis = set(FIXTURE_PELVIS.rotation)
    for r in range(int(table.getNumRows())):
        row = []
        for lab in keep:
            v = float(table.getDependentColumn(lab)[r])
            row.append(
                np.degrees(v)
                if degrees and lab not in {*pelvis} and not lab.startswith("pelvis_t")
                else v
            )
        new.appendRow(float(table.getIndependentColumn()[r]), osim.RowVector(row))
    new.addTableMetaDataString("inDegrees", "yes" if degrees else "no")
    if header:
        new.addTableMetaDataString("distribution_source", "x")
    osim.STOFileAdapter.write(new, str(out))
    return out


def test_degrees_motion_is_converted(driven) -> None:
    _spec, model, result, mot, tmp = driven
    deg = _rewrite(mot, tmp / "deg.mot", degrees=True, drop=None, header=False)
    series = _read(model, deg)
    for got, want in zip(series.poses, result.kinematics.poses, strict=True):
        np.testing.assert_allclose(got.compose(), want.compose(), atol=1e-8)


def test_headerless_motion_uses_argument(driven) -> None:
    _spec, model, _result, mot, tmp = driven
    plain = _rewrite(mot, tmp / "plain.mot", degrees=False, drop=None, header=False)
    assert _read(
        model, plain, distribution_source="ik:plain.mot"
    ).distribution_source == ("ik:plain.mot")


def test_missing_level_column_names_it(driven) -> None:
    spec, model, _result, mot, tmp = driven
    drop = fixture_level_names(spec)[1].coordinates[2]
    cut = _rewrite(mot, tmp / "cut.mot", degrees=False, drop=drop, header=False)
    with pytest.raises(ValueError, match=drop):
        _read(model, cut)


@pytest.mark.parametrize("bad", ["", "   "])
def test_empty_distribution_source_raises(driven, bad) -> None:
    _spec, model, _result, mot, _tmp = driven
    with pytest.raises(ValueError, match="distribution_source"):
        _read(model, mot, distribution_source=bad)
    with pytest.raises(TypeError, match="distribution_source"):
        _read(model, mot, distribution_source=None)


def test_missing_files_raise(driven) -> None:
    _spec, model, _result, mot, _tmp = driven
    with pytest.raises(ValueError, match="motion file not found"):
        _read(model, mot.with_name("nope.mot"))
    with pytest.raises(ValueError, match="model file not found"):
        _read(model.with_name("nope.osim"), mot)
    with pytest.raises(ValueError, match="unknown model_id"):
        read_segmental_kinematics(
            mot, model, "nope", distribution_source="coupling:invented@1"
        )


def test_native_verification_passes_on_fixture(driven) -> None:
    _spec, model, _result, mot, _tmp = driven
    series = _read(model, mot, verify_native=True)
    assert series.valid.all()


def test_native_verification_detects_tampering(driven) -> None:
    _spec, model, _result, mot, tmp = driven
    osim = import_opensim()
    table = osim.TimeSeriesTable(str(mot))
    labels = [str(x) for x in table.getColumnLabels()]
    bad = tmp / "bad.mot"
    new = osim.TimeSeriesTable()
    vec = osim.StdVectorString()
    for lab in labels:
        vec.append(lab)
    new.setColumnLabels(vec)
    for r in range(int(table.getNumRows())):
        row = [float(table.getDependentColumn(lab)[r]) for lab in labels]
        new.appendRow(float(table.getIndependentColumn()[r]), osim.RowVector(row))
    new.addTableMetaDataString("inDegrees", "no")
    osim.STOFileAdapter.write(new, str(bad))
    # Same file -> passes; a wrong level map axis sign breaks composition vs native.
    from dataclasses import replace

    from src.engines.physics_engines.opensim.python.spine.inventory import (
        load_level_maps,
    )

    entry = load_level_maps()["fixture"]
    lv = next(iter(entry.levels))
    flipped = replace(
        entry,
        levels={
            **entry.levels,
            lv: replace(entry.levels[lv], axis_sign=(-1, -1, -1)),
        },
    )
    assert _read(model, bad, level_map=entry, verify_native=True)
    with pytest.raises(ValueError, match="native"):
        _read(model, bad, level_map=flipped, verify_native=True)


def test_result_type_is_drive_result(driven) -> None:
    assert isinstance(driven[2], DriveResult)
