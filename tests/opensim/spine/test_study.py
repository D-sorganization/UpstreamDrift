"""Spine study execution and receipts (SPN-13a, #12766).

Every number is INVENTED: a five-level fixture spine with an invented linear
coupling law and lumped muscles (tendon slack retuned test-locally so no
muscle is slack), driven by a five-sample synthetic golf motion on the in-repo
``golf_humanoid.osim``.  The tests check the request -> receipt contract
(hashes, determinism, fail-closed status, JSON round trip), not physiology.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    FixtureSpec,
    write_fixture_osim,
)
from src.engines.physics_engines.opensim.python.spine.study import (
    SpineStudyReceipt,
    SpineStudyRequest,
    load_receipt,
    request_fingerprint,
    run_spine_study,
    write_receipt,
)
from src.shared.python.spine.coupling import LinearCouplingLaw
from src.shared.python.spine.levels import SpinalLevel, levels_between
from tests.helpers.opensim_support import require_opensim

REPO = Path(__file__).resolve().parents[3]
GOLF_MODEL = REPO / "src/engines/physics_engines/opensim/models/golf_humanoid.osim"
LUMBAR = levels_between(SpinalLevel.L5_S1, SpinalLevel.L1_L2)
PHI = ("trunk_flex_ext", "trunk_lat_bend", "trunk_axial_rot")
SHA = "0123456789abcdef" * 4
_SHA_LEN = 64


def _law() -> LinearCouplingLaw:
    mats = {}
    for k, lv in enumerate(LUMBAR):
        w = 0.1 + 0.05 * k
        mats[lv] = np.array(
            [[w, 0.03 * (k + 1), 0.0], [0.0, w, 0.02 * (k + 1)], [0.0, 0.0, w / 2]]
        )
    return LinearCouplingLaw("invented", "1", mats, PHI, "invented, SPN-13a test")


def _write_motion(path: Path, n: int = 5) -> Path:
    t = np.linspace(0.0, 0.4, n)
    cols = {
        "lumbar_extension": 0.05 + 0.2 * np.sin(3 * t),
        "lumbar_bending": 0.1 - 0.3 * t,
        "lumbar_rotation": 0.1 + 0.8 * t,
    }
    lines = [
        "synthetic",
        "version=1",
        f"nRows={n}",
        f"nColumns={len(cols) + 1}",
        "inDegrees=no",
        "endheader",
        "time\t" + "\t".join(cols),
    ]
    for i, ti in enumerate(t):
        lines.append("\t".join([f"{ti:.12f}"] + [f"{cols[k][i]:.15g}" for k in cols]))
    path.write_text("\n".join(lines) + "\n")
    return path


def _retune(path: Path) -> None:
    osim = require_opensim()
    model = osim.Model(str(path))
    state = model.initSystem()
    muscles = model.getMuscles()
    for i in range(muscles.getSize()):
        muscle = muscles.get(i)
        muscle.set_tendon_slack_length(
            muscle.getLength(state) - muscle.get_optimal_fiber_length()
        )
    model.printToXML(str(path))


@pytest.fixture(scope="module")
def inputs(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("study_inputs")
    model = write_fixture_osim(
        FixtureSpec(), root / "fixture.osim", coupling=_law(), muscles=True
    )
    _retune(model)
    return {"motion": _write_motion(root / "swing.sto"), "model": model, "root": root}


def _request(inputs, out: Path, **kw) -> SpineStudyRequest:
    base = {
        "study_id": "unit_study",
        "motion_path": inputs["motion"],
        "golf_model_path": GOLF_MODEL,
        "model_path": inputs["model"],
        "model_id": "fixture",
        "pelvis": "fixture",
        "events": {"mid": 0.2, "late": 0.3},
        "stride": None,
        "max_frames": 5,
        "residual_force_fraction_bw": 1e3,
        "residual_moment_fraction": 1e3,
        "reserve_limit_nm": 1e6,
        "reserve_max_share": 1e3,
        "output_dir": out,
    }
    base.update(kw)
    return SpineStudyRequest(**base)


# ------------------------------------------------------------- pure contract


@pytest.mark.unit
def test_request_preconditions(tmp_path) -> None:
    ok = {
        "study_id": "s",
        "motion_path": tmp_path / "m.sto",
        "golf_model_path": tmp_path / "g.osim",
        "model_path": tmp_path / "x.osim",
        "model_id": "fixture",
        "output_dir": tmp_path,
    }
    SpineStudyRequest(**ok)
    with pytest.raises(ValueError, match="study_id"):
        SpineStudyRequest(**{**ok, "study_id": "../escape"})
    with pytest.raises(TypeError, match="motion_path"):
        SpineStudyRequest(**{**ok, "motion_path": "m.sto"})
    with pytest.raises(ValueError, match="cost_exponent"):
        SpineStudyRequest(**{**ok, "cost_exponent": 4})
    with pytest.raises(ValueError, match="pelvis"):
        SpineStudyRequest(**{**ok, "pelvis": "other"})
    with pytest.raises(ValueError, match="model_path or asset_id"):
        SpineStudyRequest(**{**ok, "model_path": None})
    with pytest.raises(ValueError, match="residual_force_fraction_bw"):
        SpineStudyRequest(**{**ok, "residual_force_fraction_bw": 0.0})
    with pytest.raises(ValueError, match="events"):
        SpineStudyRequest(**{**ok, "events": {"a": float("nan")}})
    with pytest.raises(TypeError, match="request"):
        run_spine_study("nope")  # type: ignore[arg-type]


@pytest.mark.unit
def test_missing_licensed_asset_is_unqualified(tmp_path) -> None:
    motion = _write_motion(tmp_path / "swing.sto")
    request = SpineStudyRequest(
        study_id="missing_asset",
        motion_path=motion,
        golf_model_path=GOLF_MODEL,
        model_id="bruno2015_fullbody_os4_v2",
        asset_id="bruno2015_fullbody_os4_v2",
        output_dir=tmp_path / "out",
    )
    receipt = run_spine_study(request, code_version="deadbeef", env={})
    assert receipt.status == "unqualified"
    assert receipt.status != "pass"
    assert receipt.asset is not None and receipt.asset["status"] == "missing"
    assert receipt.stages == ()
    assert any("asset" in r for r in receipt.status_reasons)
    roles = {f.role for f in receipt.inputs}
    assert {"motion", "golf_model"} <= roles
    assert all(len(f.sha256) == _SHA_LEN for f in receipt.inputs)
    assert receipt.scientific_status == "unqualified"
    assert str(tmp_path) not in json.dumps(receipt.as_dict())


@pytest.mark.unit
def test_receipt_json_round_trip(tmp_path) -> None:
    motion = _write_motion(tmp_path / "swing.sto")
    request = SpineStudyRequest(
        study_id="round_trip",
        motion_path=motion,
        golf_model_path=GOLF_MODEL,
        model_id="bruno2015_fullbody_os4_v2",
        asset_id="bruno2015_fullbody_os4_v2",
        output_dir=tmp_path / "out",
    )
    receipt = run_spine_study(request, code_version="deadbeef", env={})
    path = write_receipt(receipt, tmp_path / "out" / "receipt.json")
    again = load_receipt(path)
    assert again == receipt
    assert isinstance(again, SpineStudyReceipt)
    assert path.read_text(encoding="utf-8") == write_receipt(
        again, tmp_path / "second.json"
    ).read_text(encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        bad = tmp_path / "bad.json"
        bad.write_text(json.dumps({"schema_version": "nope"}), encoding="utf-8")
        load_receipt(bad)


@pytest.mark.unit
def test_request_fingerprint_uses_basenames_and_hashes(tmp_path) -> None:
    motion = _write_motion(tmp_path / "swing.sto")
    base = {
        "study_id": "fp",
        "golf_model_path": GOLF_MODEL,
        "model_id": "bruno2015_fullbody_os4_v2",
        "asset_id": "bruno2015_fullbody_os4_v2",
        "output_dir": tmp_path / "a",
    }
    one = SpineStudyRequest(motion_path=motion, **base)
    moved = SpineStudyRequest(motion_path=motion, **{**base, "output_dir": tmp_path})
    assert request_fingerprint(one) == request_fingerprint(moved)
    other = SpineStudyRequest(motion_path=motion, **{**base, "cost_exponent": 2})
    assert request_fingerprint(one) != request_fingerprint(other)
    assert str(tmp_path) not in json.dumps(one.canonical_dict())


def _loads_request(tmp_path, xml: Path) -> SpineStudyRequest:
    return SpineStudyRequest(
        study_id="loads",
        motion_path=_write_motion(tmp_path / "swing.sto"),
        golf_model_path=GOLF_MODEL,
        model_id="bruno2015_fullbody_os4_v2",
        asset_id="bruno2015_fullbody_os4_v2",
        external_loads=xml,
        output_dir=tmp_path / "out",
    )


def _loads_xml(path: Path, datafile: str) -> Path:
    path.write_text(
        f"<OpenSimDocument><ExternalLoads><datafile>{datafile}</datafile>"
        "</ExternalLoads></OpenSimDocument>",
        encoding="utf-8",
    )
    return path


@pytest.mark.unit
def test_missing_external_loads_datafile_raises(tmp_path) -> None:
    xml = _loads_xml(tmp_path / "loads.xml", "absent_grf.mot")
    request = _loads_request(tmp_path, xml)
    with pytest.raises(FileNotFoundError, match="external loads data"):
        request_fingerprint(request)
    with pytest.raises(FileNotFoundError, match="absent_grf.mot"):
        run_spine_study(request, code_version="c", env={})
    assert not (tmp_path / "out").exists()


@pytest.mark.unit
def test_relative_datafile_resolves_against_xml_directory(
    tmp_path, monkeypatch
) -> None:
    (tmp_path / "xml" / "data").mkdir(parents=True)
    (tmp_path / "elsewhere").mkdir()
    data = tmp_path / "xml" / "data" / "grf.mot"
    data.write_text("one", encoding="utf-8")
    xml = _loads_xml(tmp_path / "xml" / "loads.xml", "data/grf.mot")
    monkeypatch.chdir(tmp_path / "elsewhere")
    request = _loads_request(tmp_path, xml)
    before = request_fingerprint(request)
    data.write_text("two", encoding="utf-8")
    assert request_fingerprint(request) != before


@pytest.mark.unit
def test_absolute_datafile_is_hashed(tmp_path) -> None:
    (tmp_path / "xml").mkdir()
    data = tmp_path / "grf_abs.mot"
    data.write_text("one", encoding="utf-8")
    xml = _loads_xml(tmp_path / "xml" / "loads.xml", str(data))
    request = _loads_request(tmp_path, xml)
    before = request_fingerprint(request)
    data.write_text("two", encoding="utf-8")
    assert request_fingerprint(request) != before
    data.unlink()
    with pytest.raises(FileNotFoundError, match="external loads data"):
        request_fingerprint(request)


# ------------------------------------------------------------ native pipeline


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_end_to_end_receipt_has_all_hashes(inputs, tmp_path) -> None:
    require_opensim()
    receipt = run_spine_study(
        _request(inputs, tmp_path / "out"), code_version="cafe1234"
    )
    assert receipt.status == "pass", (receipt.status_reasons, receipt.reserve)
    assert receipt.code_version == "cafe1234"
    assert receipt.scientific_status == "unqualified"
    roles = {f.role for f in receipt.inputs}
    assert {"motion", "golf_model", "segmental_model"} <= roles
    assert {
        "config:spine_coupling_laws.json",
        "config:spine_model_level_maps.json",
        "config:spine_golf_context_bands.json",
    } <= roles
    assert all(len(f.sha256) == _SHA_LEN for f in receipt.inputs)
    assert [s.stage for s in receipt.stages] == [
        "golf_input",
        "drive",
        "inverse_dynamics",
        "static_optimization",
        "joint_reactions",
    ]
    assert all(s.status == "ok" for s in receipt.stages)
    assert all(s.provenance for s in receipt.stages[1:])
    assert receipt.distribution_source.startswith("coupling:")
    assert "@" in receipt.law
    names = {a.name for a in receipt.artifacts}
    assert names == {"unit_study_coordinates.mot", "unit_study_loads.npz"}
    out = tmp_path / "out"
    for art in receipt.artifacts:
        assert (out / art.name).is_file()
        assert len(art.sha256) == _SHA_LEN
    assert receipt.residual["drive_max_rad"] >= 0.0
    assert "n_flagged" in receipt.reserve
    assert receipt.context_comparisons
    assert all(c["is_acceptance"] is False for c in receipt.context_comparisons)
    assert str(tmp_path) not in json.dumps(receipt.as_dict())
    with np.load(out / "unit_study_loads.npz") as loads:
        assert {"time_s", "joint_reaction_force_n"} <= set(loads.files)


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_rerun_is_byte_identical(inputs, tmp_path) -> None:
    require_opensim()
    first = run_spine_study(_request(inputs, tmp_path / "a"), code_version="c")
    second = run_spine_study(_request(inputs, tmp_path / "b"), code_version="c")
    one = write_receipt(first, tmp_path / "a" / "receipt.json").read_bytes()
    two = write_receipt(second, tmp_path / "b" / "receipt.json").read_bytes()
    assert one == two
    for art in first.artifacts:
        assert (tmp_path / "a" / art.name).read_bytes() == (
            tmp_path / "b" / art.name
        ).read_bytes()


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_changed_input_changes_hash_and_receipt(inputs, tmp_path) -> None:
    require_opensim()
    base = run_spine_study(_request(inputs, tmp_path / "base"), code_version="c")
    edited = tmp_path / "swing_edited.sto"
    edited.write_text(
        inputs["motion"].read_text().replace("0.1000000", "0.1100000", 1),
        encoding="utf-8",
    )
    assert edited.read_text() != inputs["motion"].read_text()
    changed = run_spine_study(
        _request(inputs, tmp_path / "ch", motion_path=edited), code_version="c"
    )

    def hashes(receipt):
        return {f.role: f.sha256 for f in receipt.inputs}

    assert hashes(base)["motion"] != hashes(changed)["motion"]
    assert hashes(base)["golf_model"] == hashes(changed)["golf_model"]
    assert request_fingerprint(_request(inputs, tmp_path)) != request_fingerprint(
        _request(inputs, tmp_path, motion_path=edited)
    )
    assert base.request_sha256 != changed.request_sha256

    model_copy = tmp_path / "fixture_edited.osim"
    model_copy.write_text(inputs["model"].read_text() + "\n<!-- edited -->\n")
    altered = run_spine_study(
        _request(inputs, tmp_path / "m", model_path=model_copy), code_version="c"
    )
    assert hashes(altered)["segmental_model"] != hashes(base)["segmental_model"]
    assert altered.request_sha256 != base.request_sha256


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_flagged_reserves_make_study_unqualified(inputs, tmp_path) -> None:
    require_opensim()
    receipt = run_spine_study(
        _request(
            inputs,
            tmp_path / "flag",
            reserve_limit_nm=1e-9,
            reserve_max_share=1e-9,
        ),
        code_version="c",
    )
    assert receipt.status == "unqualified"
    assert receipt.reserve["n_flagged"] > 0
    assert any("reserve" in r for r in receipt.status_reasons)


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_stage_failure_is_recorded_as_failed(inputs, tmp_path) -> None:
    require_opensim()
    bad = inputs["root"] / "no_levels.sto"
    bad.write_text(
        "synthetic\nversion=1\nnRows=1\nnColumns=2\ninDegrees=no\n"
        "endheader\ntime\tpelvis_tx\n0.0\t0.0\n",
        encoding="utf-8",
    )
    receipt = run_spine_study(
        _request(inputs, tmp_path / "bad", motion_path=bad), code_version="c"
    )
    assert receipt.status == "failed"
    assert receipt.stages[-1].status == "failed"
    assert receipt.stages[-1].stage == "golf_input"
    assert str(tmp_path) not in json.dumps(receipt.as_dict())


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_missing_input_file_raises(inputs, tmp_path) -> None:
    with pytest.raises(FileNotFoundError, match="motion"):
        run_spine_study(
            _request(inputs, tmp_path, motion_path=tmp_path / "absent.sto"),
            code_version="c",
        )


@pytest.mark.unit
def test_cli_out_supplies_missing_output_dir(tmp_path, monkeypatch) -> None:
    """``--out`` alone satisfies ``output_dir``; the file need not carry it."""
    from src.engines.physics_engines.opensim.python.spine import study

    request_file = tmp_path / "request.json"
    request_file.write_text(
        json.dumps(
            {
                "study_id": "s",
                "motion_path": "m.sto",
                "golf_model_path": "g.osim",
                "model_path": "x.osim",
                "model_id": "fixture",
            }
        ),
        encoding="utf-8",
    )
    seen: list[SpineStudyRequest] = []

    def fake_run(request, env=None):
        seen.append(request)
        raise _Stop

    class _Stop(Exception):
        pass

    monkeypatch.setattr(study, "run_spine_study", fake_run)
    out = tmp_path / "results"
    with pytest.raises(_Stop):
        study.main(["--request", str(request_file), "--out", str(out)])
    assert seen[0].output_dir == out.resolve()
    with pytest.raises(ValueError, match="output_dir"):
        study.main(["--request", str(request_file)])
