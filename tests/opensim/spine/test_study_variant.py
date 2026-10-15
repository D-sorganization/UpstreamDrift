"""Movement variants in the spine study (SPN-13d, #12767).

Every number is INVENTED (same five-level fixture and five-sample synthetic
swing as the SPN-13a tests; the swing file additionally carries the address,
top and impact event header lines a variant window needs).  The tests check
the request -> fingerprint -> receipt contract of a variant run, not
physiology.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.spine.study import (
    SpineStudyRequest,
    request_fingerprint,
    run_spine_study,
    write_receipt,
)
from src.shared.python.pose_estimation.model_files import sha256_of
from src.shared.python.spine.movement_variants import (
    DEFAULT_VARIANTS_PATH,
    movement_variant,
)
from tests.helpers.opensim_support import require_opensim
from tests.opensim.spine.test_study import (
    _law,
    _request,
    _retune,
    _write_motion,
)
from src.engines.physics_engines.opensim.python.spine.fixture_model import (
    FixtureSpec,
    write_fixture_osim,
)

VARIANT = "x_factor_top_scale"
EVENT_LINES = "event_address=0.0\nevent_top=0.2\nevent_impact=0.3\n"


def _with_events(path: Path) -> Path:
    text = path.read_text()
    path.write_text(text.replace("version=1\n", "version=1\n" + EVENT_LINES, 1))
    return path


@pytest.fixture(scope="module")
def inputs(tmp_path_factory) -> dict[str, Path]:
    root = tmp_path_factory.mktemp("study_variant_inputs")
    model = write_fixture_osim(
        FixtureSpec(), root / "fixture.osim", coupling=_law(), muscles=True
    )
    _retune(model)
    narrow = root / "narrow.osim"
    narrow.write_text(model.read_text().replace("-1.0 1.0", "-0.0001 0.0001"))
    return {
        "motion": _with_events(_write_motion(root / "swing.sto")),
        "model": model,
        "narrow": narrow,
        "root": root,
    }


def _variant_request(inputs, out: Path, **kw) -> SpineStudyRequest:
    kw.setdefault("variant_id", VARIANT)
    kw.setdefault("variant_parameter", 1.2)
    return _request(inputs, out, **kw)


# ------------------------------------------------------------- pure contract


@pytest.mark.unit
def test_variant_request_preconditions(tmp_path) -> None:
    ok = {
        "study_id": "s",
        "motion_path": tmp_path / "m.sto",
        "golf_model_path": tmp_path / "g.osim",
        "model_path": tmp_path / "x.osim",
        "model_id": "fixture",
        "output_dir": tmp_path,
    }
    SpineStudyRequest(**ok, variant_id=VARIANT, variant_parameter=1.2)
    SpineStudyRequest(**ok, variant_id=VARIANT)
    low, high = movement_variant(VARIANT).parameter_bounds
    with pytest.raises(ValueError, match="outside bounds"):
        SpineStudyRequest(**ok, variant_id=VARIANT, variant_parameter=high + 1.0)
    with pytest.raises(ValueError, match="outside bounds"):
        SpineStudyRequest(**ok, variant_id=VARIANT, variant_parameter=low - 1.0)
    with pytest.raises(ValueError, match="unknown movement variant"):
        SpineStudyRequest(**ok, variant_id="no_such_variant")
    with pytest.raises(ValueError, match="variant_id"):
        SpineStudyRequest(**ok, variant_parameter=1.0)
    with pytest.raises(TypeError, match="variant_id"):
        SpineStudyRequest(**ok, variant_id=3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="variant_handedness"):
        SpineStudyRequest(**ok, variant_id=VARIANT, variant_handedness="both")


@pytest.mark.unit
def test_no_variant_canonical_dict_is_unchanged(tmp_path) -> None:
    request = SpineStudyRequest(
        study_id="s",
        motion_path=tmp_path / "m.sto",
        golf_model_path=tmp_path / "g.osim",
        model_path=tmp_path / "x.osim",
        model_id="fixture",
        output_dir=tmp_path,
    )
    assert not any("variant" in key for key in request.canonical_dict())


@pytest.mark.unit
def test_variant_fingerprint_covers_variant_parameter_and_config(tmp_path) -> None:
    motion = _write_motion(tmp_path / "swing.sto")
    model = tmp_path / "x.osim"
    model.write_text("<model/>")
    golf = tmp_path / "g.osim"
    golf.write_text("<golf/>")
    base = {
        "study_id": "fp",
        "motion_path": motion,
        "golf_model_path": golf,
        "model_path": model,
        "model_id": "fixture",
        "output_dir": tmp_path,
    }
    plain = request_fingerprint(SpineStudyRequest(**base))
    one = request_fingerprint(
        SpineStudyRequest(**base, variant_id=VARIANT, variant_parameter=1.1)
    )
    two = request_fingerprint(
        SpineStudyRequest(**base, variant_id=VARIANT, variant_parameter=1.2)
    )
    assert len({plain, one, two}) == 3
    again = request_fingerprint(
        SpineStudyRequest(**base, variant_id=VARIANT, variant_parameter=1.1)
    )
    assert again == one


# ------------------------------------------------------------ native pipeline


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_variant_run_is_deterministic_and_records_provenance(inputs, tmp_path) -> None:
    require_opensim()
    first = run_spine_study(_variant_request(inputs, tmp_path / "a"), code_version="c")
    second = run_spine_study(_variant_request(inputs, tmp_path / "b"), code_version="c")
    assert first.status == "pass", (first.status_reasons, first.stages[-1])
    one = write_receipt(first, tmp_path / "a" / "receipt.json").read_bytes()
    two = write_receipt(second, tmp_path / "b" / "receipt.json").read_bytes()
    assert one == two
    variant = first.variant
    assert variant is not None
    assert variant["id"] == f"{VARIANT}@{movement_variant(VARIANT).version}"
    assert variant["parameter"] == 1.2
    assert VARIANT in variant["distribution_source"]
    assert variant["range_check"]["ok"] is True
    assert variant["range_check"]["n_violations"] == 0
    assert variant["config_sha256"] == sha256_of(DEFAULT_VARIANTS_PATH)
    roles = {f.role: f.sha256 for f in first.inputs}
    assert roles["config:spine_movement_variants.json"] == sha256_of(
        DEFAULT_VARIANTS_PATH
    )
    drive = next(s for s in first.stages if s.stage == "drive")
    assert drive.provenance["input_variant"] == variant["distribution_source"]
    assert str(tmp_path) not in json.dumps(first.as_dict())


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_variant_and_baseline_differ_and_baseline_has_no_variant(
    inputs, tmp_path
) -> None:
    require_opensim()
    base = run_spine_study(_request(inputs, tmp_path / "base"), code_version="c")
    varied = run_spine_study(_variant_request(inputs, tmp_path / "var"), code_version="c")
    assert base.status == "pass"
    assert base.variant is None
    assert "variant" not in base.as_dict()
    assert "variant" not in base.request
    assert base.request_sha256 != varied.request_sha256
    assert not any("variant" in f.role for f in base.inputs)


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_model_range_violation_makes_variant_study_unqualified(
    inputs, tmp_path
) -> None:
    require_opensim()
    receipt = run_spine_study(
        _variant_request(inputs, tmp_path / "rng", model_path=inputs["narrow"]),
        code_version="c",
    )
    assert receipt.status == "unqualified", receipt.status_reasons
    assert receipt.variant is not None
    check = receipt.variant["range_check"]
    assert check["ok"] is False
    assert check["n_violations"] > 0
    assert any("range" in reason for reason in receipt.status_reasons)
    assert all(s.status == "ok" for s in receipt.stages)


@pytest.mark.requires_opensim
@pytest.mark.integration
def test_variant_without_events_fails_the_golf_input_stage(inputs, tmp_path) -> None:
    require_opensim()
    bare = _write_motion(inputs["root"] / "bare.sto")
    receipt = run_spine_study(
        _variant_request(inputs, tmp_path / "bare", motion_path=bare),
        code_version="c",
    )
    assert receipt.status == "failed"
    assert receipt.stages[-1].stage == "golf_input"
    assert receipt.variant is not None
    assert "range_check" not in receipt.variant
