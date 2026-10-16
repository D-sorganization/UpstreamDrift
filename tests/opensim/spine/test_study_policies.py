"""Study request policies: muscle model and driven-coordinate locks (#12761).

Both are modelling decisions and therefore opt-in; the default fails with a
message naming the option.  All data are INVENTED (the five-level fixture).
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    FixtureSpec,
    fixture_level_names,
    write_fixture_osim,
)
from src.engines.physics_engines.opensim.python.spine.study import (
    SpineStudyRequest,
    request_fingerprint,
    run_spine_study,
)
from tests.helpers.opensim_support import require_opensim
from tests.opensim.spine.test_muscle_model import _retag_thelen
from tests.opensim.spine.test_study import (
    GOLF_MODEL,
    _law,
    _request,
    _retune,
    _write_motion,
)


@pytest.fixture(scope="module")
def models(tmp_path_factory):
    osim = require_opensim()
    root = tmp_path_factory.mktemp("policies")
    base = write_fixture_osim(
        FixtureSpec(), root / "fixture.osim", coupling=_law(), muscles=True
    )
    _retune(base)
    thelen = _retag_thelen(base, root / "thelen.osim")
    locked = root / "locked.osim"
    model = osim.Model(str(base))
    driven = fixture_level_names(FixtureSpec())[0].coordinates[0]
    model.getCoordinateSet().get(driven).set_locked(True)
    model.printToXML(str(locked))
    return {
        "motion": _write_motion(root / "swing.sto"),
        "model": base,
        "thelen": thelen,
        "locked": locked,
        "driven": driven,
    }


def _stage(receipt, name):
    return next(s for s in receipt.stages if s.stage == name)


@pytest.mark.unit
def test_request_fields_validate_and_enter_the_hash(tmp_path) -> None:
    (tmp_path / "m.sto").write_text("x")
    (tmp_path / "x.osim").write_text("x")
    base = {
        "study_id": "pol",
        "motion_path": tmp_path / "m.sto",
        "golf_model_path": GOLF_MODEL,
        "model_path": tmp_path / "x.osim",
        "model_id": "fixture",
        "output_dir": tmp_path,
    }
    default = SpineStudyRequest(**base)
    assert "muscle_model_policy" not in default.canonical_dict()
    assert "unlock_driven_coordinates" not in default.canonical_dict()
    millard = SpineStudyRequest(**base, muscle_model_policy="millard_equivalent")
    unlock = SpineStudyRequest(**base, unlock_driven_coordinates=True)
    assert millard.canonical_dict()["muscle_model_policy"] == "millard_equivalent"
    assert unlock.canonical_dict()["unlock_driven_coordinates"] is True
    assert len({request_fingerprint(r) for r in (default, millard, unlock)}) == 3
    with pytest.raises(ValueError, match="muscle_model_policy"):
        SpineStudyRequest(**base, muscle_model_policy="thelen")
    with pytest.raises(TypeError, match="unlock_driven_coordinates"):
        SpineStudyRequest(**base, unlock_driven_coordinates="yes")


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_thelen_model_fails_so_naming_the_policy_by_default(models, tmp_path) -> None:
    request = _request(models, tmp_path / "o", model_path=models["thelen"])
    receipt = run_spine_study(request, code_version="c")
    assert receipt.status == "failed"
    stage = _stage(receipt, "static_optimization")
    assert stage.status == "failed"
    assert "muscle_model_policy" in stage.error
    assert "millard_equivalent" in stage.error


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_millard_equivalent_runs_and_records_the_swap(models, tmp_path) -> None:
    request = _request(
        models,
        tmp_path / "o",
        model_path=models["thelen"],
        muscle_model_policy="millard_equivalent",
    )
    receipt = run_spine_study(request, code_version="c")
    assert receipt.status == "pass", (receipt.status_reasons, receipt.stages[-1])
    prov = _stage(receipt, "static_optimization").provenance
    assert prov["muscle_model_policy"] == "millard_equivalent"
    assert int(prov["muscles_swapped"]) > 0
    assert "Thelen2003Muscle" in prov["muscle_swap_note"]
    assert prov["stage2_solver"] in {"slsqp", "clarabel_qp", "clarabel_irls"}
    assert "path-length central difference" in prov["moment_arm_screen"]
    native = run_spine_study(
        _request(models, tmp_path / "n", model_path=models["model"]),
        code_version="c",
    )
    assert _stage(native, "static_optimization").provenance["muscles_swapped"] == "0"
    assert native.request_sha256 != receipt.request_sha256


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_driven_locked_coordinate_fails_by_default(models, tmp_path) -> None:
    request = _request(models, tmp_path / "o", model_path=models["locked"])
    receipt = run_spine_study(request, code_version="c")
    assert receipt.status == "failed"
    stage = _stage(receipt, "drive")
    assert stage.status == "failed"
    assert models["driven"] in stage.error
    assert "unlock_driven_coordinates" in stage.error


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_unlock_option_unlocks_only_driven_locked_coordinates(models, tmp_path) -> None:
    request = _request(
        models,
        tmp_path / "o",
        model_path=models["locked"],
        unlock_driven_coordinates=True,
    )
    receipt = run_spine_study(request, code_version="c")
    assert receipt.status == "pass", (receipt.status_reasons, receipt.stages[-1])
    note = _stage(receipt, "drive").provenance["coordinate_locks"]
    assert note.startswith("1 driven-and-locked coordinates unlocked")
    assert models["driven"] in note
    plain = run_spine_study(
        _request(models, tmp_path / "p", model_path=models["model"]),
        code_version="c",
    )
    assert (
        _stage(plain, "drive").provenance["coordinate_locks"]
        == "no driven-and-locked coordinates"
    )
    assert json.dumps(receipt.as_dict()).count(str(Path(tmp_path))) == 0
