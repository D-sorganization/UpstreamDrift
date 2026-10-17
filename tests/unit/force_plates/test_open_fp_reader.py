"""OF-02 (#13066): the fail-closed reader of open-fp swing files.

Files are produced by ``_open_fp_writer`` from the documented format only.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.force_plates import PlateUncertainty
from src.shared.python.force_plates.open_fp_format import (
    PlateDeclarations,
    list_swings,
    read_swing,
    read_swing_csv,
)
from src.shared.python.force_plates.open_fp_format import format as fmt
from tests.unit.force_plates import _open_fp_writer as w

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]


def declared() -> PlateDeclarations:
    return PlateDeclarations(
        uncertainty=PlateUncertainty(
            force_floor_n=2.0,
            force_fraction=0.01,
            cop_m=0.010,
            sensing_depth_sigma_m=0.002,
            source="declared hypothesis for tests",
        ),
        sensing_depth_m=0.015,
        capacity_n=3923.0,
    )


def read(path: Path):
    return read_swing(path, declared=declared())


def meta_with(**changes: Any) -> dict[str, Any]:
    meta = w.default_meta()
    meta.update(changes)
    return meta


# ------------------------------------------------------------ round trips --
@pytest.mark.parametrize("golfer", ["RH", "LH"])
def test_round_trip(tmp_path: Path, golfer: str) -> None:
    trial = read(w.write_swing(tmp_path, golfer=golfer))
    assert trial.handedness == golfer
    assert trial.body_weight_n == 800.0
    np.testing.assert_array_equal(trial.times_s, w.T_S)
    np.testing.assert_array_equal(trial.load_n["left"], w.FZ_L)
    np.testing.assert_array_equal(trial.load_n["right"], w.FZ_R)
    lead = "left" if golfer == "RH" else "right"
    assert trial.lead_foot == lead
    assert trial.trail_foot != lead
    assert trial.cop_min_load_n == 20.0
    assert trial.capacity_n == 3923.0
    assert trial.sensing_depth_m == 0.015
    assert dict(trial.events_s) == w.EVENTS
    assert trial.corner_forces_n is not None and trial.corner_forces_n.shape == (6, 8)
    assert not trial.provenance.qualified
    assert trial.provenance.source_revision == fmt.SOURCE_REVISION


def test_cop_converted_from_millimetres_and_nan_kept(tmp_path: Path) -> None:
    trial = read(w.write_swing(tmp_path))
    np.testing.assert_allclose(trial.cop_m["left"][:4], w.COP_L_MM[:4] * 1e-3, atol=0)
    np.testing.assert_allclose(trial.cop_m["right"], w.COP_R_MM * 1e-3, atol=0)
    assert np.isnan(trial.cop_m["left"][4:]).all()
    expected_net = w.combined_cop_mm(w.FZ_L, w.FZ_R, w.COP_L_MM, w.COP_R_MM) * 1e-3
    np.testing.assert_allclose(trial.cop_m["net"], expected_net, atol=1e-12)


def test_digests_are_sha256_of_the_bytes(tmp_path: Path) -> None:
    json_path = w.write_swing(tmp_path)
    npz_path = json_path.with_suffix(".npz")
    digests = read(json_path).provenance.file_sha256
    assert digests["json"] == hashlib.sha256(json_path.read_bytes()).hexdigest()
    assert digests["npz"] == hashlib.sha256(npz_path.read_bytes()).hexdigest()


def test_impact_source_null_with_null_impact_is_accepted(tmp_path: Path) -> None:
    events = {**w.EVENTS, "impact": None}
    path = w.write_swing(tmp_path, meta=meta_with(events=events, impact_source=None))
    assert read(path).events_s["impact"] is None


# ------------------------------------------------------------ npz checks --
def test_missing_npz_key_rejected(tmp_path: Path) -> None:
    arrays = w.default_arrays()
    del arrays["cop_C"]
    with pytest.raises(ValueError, match="cop_C"):
        read(w.write_swing(tmp_path, arrays=arrays))


def test_extra_npz_key_rejected(tmp_path: Path) -> None:
    arrays = {**w.default_arrays(), "extra": np.zeros(6)}
    with pytest.raises(ValueError, match="extra"):
        read(w.write_swing(tmp_path, arrays=arrays))


def test_misshapen_array_rejected(tmp_path: Path) -> None:
    arrays = {**w.default_arrays(), "cop_L": np.zeros((6, 3))}
    with pytest.raises(ValueError, match="cop_L"):
        read(w.write_swing(tmp_path, arrays=arrays))


def test_inconsistent_length_rejected(tmp_path: Path) -> None:
    arrays = {**w.default_arrays(), "fz_L": w.FZ_L[:5]}
    with pytest.raises(ValueError, match="fz_L"):
        read(w.write_swing(tmp_path, arrays=arrays))


def test_object_array_refused(tmp_path: Path) -> None:
    arrays = {**w.default_arrays(), "t": np.array(list(w.T_S), dtype=object)}
    with pytest.raises(ValueError, match="npz"):
        read(w.write_swing(tmp_path, arrays=arrays))


def test_integer_array_refused(tmp_path: Path) -> None:
    arrays = {**w.default_arrays(), "fz_L": np.arange(6)}
    with pytest.raises(ValueError, match="fz_L"):
        read(w.write_swing(tmp_path, arrays=arrays))


def test_non_increasing_time_rejected(tmp_path: Path) -> None:
    arrays = w.default_arrays()
    arrays["t"][3] = arrays["t"][2]
    with pytest.raises(ValueError, match="strictly increasing"):
        read(w.write_swing(tmp_path, arrays=arrays))


def test_negative_or_nan_load_rejected(tmp_path: Path) -> None:
    for bad in (-1.0, np.nan):
        arrays = w.default_arrays()
        arrays["fz_R"][0] = bad
        with pytest.raises(ValueError, match="fz_R"):
            read(w.write_swing(tmp_path, arrays=arrays))


def test_nonfinite_corner_force_rejected(tmp_path: Path) -> None:
    arrays = w.default_arrays()
    arrays["forces"][1, 2] = np.inf
    with pytest.raises(ValueError, match="forces"):
        read(w.write_swing(tmp_path, arrays=arrays))


def test_cop_threshold_rule_propagates_from_contract(tmp_path: Path) -> None:
    arrays = w.default_arrays()
    arrays["cop_L"][4] = [0.0, 0.0]  # finite CoP at a 10 N load
    with pytest.raises(ValueError, match="threshold"):
        read(w.write_swing(tmp_path, arrays=arrays))


# ------------------------------------------------------------ json checks --
def test_oversized_json_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = w.write_swing(tmp_path)
    monkeypatch.setattr(fmt, "MAX_JSON_BYTES", 10)
    with pytest.raises(ValueError, match="too large"):
        read(path)


def test_oversized_npz_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = w.write_swing(tmp_path)
    monkeypatch.setattr(fmt, "MAX_NPZ_BYTES", 10)
    with pytest.raises(ValueError, match="too large"):
        read(path)


def test_nan_literal_in_json_refused(tmp_path: Path) -> None:
    text = json.dumps(w.default_meta()).replace('"bw_n": 800.0', '"bw_n": NaN')
    with pytest.raises(ValueError, match="NaN"):
        read(w.write_swing(tmp_path, raw_json=text))


def test_malformed_json_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="JSON"):
        read(w.write_swing(tmp_path, raw_json="{not json"))


def test_unknown_top_level_key_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="surprise"):
        read(w.write_swing(tmp_path, meta=meta_with(surprise=1)))


def test_missing_top_level_key_refused(tmp_path: Path) -> None:
    meta = w.default_meta()
    del meta["bw_n"]
    with pytest.raises(ValueError, match="bw_n"):
        read(w.write_swing(tmp_path, meta=meta))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("golfer", "XX"),
        ("bw_n", 0.0),
        ("bw_n", "800"),
        ("bw_n", True),
        ("t0_host", "now"),
        ("created", 5),
        ("impact_source", "guess"),
    ],
)
def test_bad_field_values_refused(tmp_path: Path, field: str, value: Any) -> None:
    with pytest.raises(ValueError, match=field):
        read(w.write_swing(tmp_path, meta=meta_with(**{field: value})))


def test_json_root_must_be_object(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="object"):
        read(w.write_swing(tmp_path, raw_json="[1, 2]"))


# --------------------------------------------------------------- path checks --
def test_id_mismatch_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="id"):
        read(w.write_swing(tmp_path, meta=meta_with(id="20260101-120000-124")))


def test_bad_stem_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="stem"):
        read(w.write_swing(tmp_path, swing_id="not-an-id", meta=meta_with(id="not-an-id")))


def test_path_traversal_refused(tmp_path: Path) -> None:
    w.write_swing(tmp_path)
    with pytest.raises(ValueError, match="stem"):
        read(tmp_path / ".." / "x.json")


def test_wrong_suffix_refused(tmp_path: Path) -> None:
    path = w.write_swing(tmp_path)
    with pytest.raises(ValueError, match=r"\.json"):
        read(path.with_suffix(".npz"))


def test_missing_file_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="regular file"):
        read(tmp_path / f"{w.SWING_ID}.json")


def test_missing_sibling_npz_refused(tmp_path: Path) -> None:
    path = w.write_swing(tmp_path)
    path.with_suffix(".npz").unlink()
    with pytest.raises(ValueError, match="npz"):
        read(path)


def test_symlinked_json_refused(tmp_path: Path) -> None:
    real, linked = tmp_path / "real", tmp_path / "linked"
    real.mkdir()
    linked.mkdir()
    target = w.write_swing(real)
    os.symlink(target, linked / target.name)
    os.symlink(target.with_suffix(".npz"), linked / f"{w.SWING_ID}.npz")
    with pytest.raises(ValueError, match="symlink"):
        read(linked / target.name)


def test_symlinked_npz_refused(tmp_path: Path) -> None:
    real, linked = tmp_path / "real", tmp_path / "linked"
    real.mkdir()
    linked.mkdir()
    target = w.write_swing(real)
    (linked / target.name).write_bytes(target.read_bytes())
    os.symlink(target.with_suffix(".npz"), linked / f"{w.SWING_ID}.npz")
    with pytest.raises(ValueError, match="symlink"):
        read(linked / target.name)


# -------------------------------------------------------------------- events --
def test_unknown_event_name_refused(tmp_path: Path) -> None:
    events = {**w.EVENTS, "bogus": 0.0}
    with pytest.raises(ValueError, match="bogus"):
        read(w.write_swing(tmp_path, meta=meta_with(events=events)))


def test_missing_event_name_refused(tmp_path: Path) -> None:
    events = {k: v for k, v in w.EVENTS.items() if k != "finish"}
    with pytest.raises(ValueError, match="finish"):
        read(w.write_swing(tmp_path, meta=meta_with(events=events)))


@pytest.mark.parametrize("bad", [-0.001, 0.0251, "x"])
def test_event_outside_trial_refused(tmp_path: Path, bad: Any) -> None:
    events = {**w.EVENTS, "finish": bad}
    with pytest.raises(ValueError, match="finish"):
        read(w.write_swing(tmp_path, meta=meta_with(events=events)))


def test_impact_without_impact_source_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="impact_source"):
        read(w.write_swing(tmp_path, meta=meta_with(impact_source=None)))


def test_events_follow_the_time_axis_when_it_does_not_start_at_zero(
    tmp_path: Path,
) -> None:
    arrays = w.default_arrays()
    arrays["t"] = arrays["t"] + 2.0
    trial = read(w.write_swing(tmp_path, arrays=arrays))
    assert trial.events_s["impact"] == pytest.approx(2.015)


# ----------------------------------------------------------------- lead_pct --
def test_lead_pct_inconsistent_with_handedness_refused(tmp_path: Path) -> None:
    path = w.write_swing(tmp_path, golfer="RH", meta=meta_with(golfer="LH"))
    with pytest.raises(ValueError, match="handedness"):
        read(path)


def test_lead_pct_nan_pattern_must_match_total(tmp_path: Path) -> None:
    arrays = w.default_arrays()
    arrays["lead_pct"][0] = np.nan
    with pytest.raises(ValueError, match="lead_pct"):
        read(w.write_swing(tmp_path, arrays=arrays))


# ----------------------------------------------------------------------- csv --
def csv_trial(tmp_path: Path, **overrides: Any):
    kwargs: dict[str, Any] = {
        "handedness": "RH",
        "body_weight_n": 800.0,
        "declared": declared(),
    }
    kwargs.update(overrides)
    return read_swing_csv(w.write_csv(tmp_path), **kwargs)


def test_csv_round_trip(tmp_path: Path) -> None:
    csv_path = w.write_csv(tmp_path)
    trial = read_swing_csv(
        csv_path, handedness="RH", body_weight_n=800.0, declared=declared()
    )
    np.testing.assert_allclose(trial.times_s, w.T_S, atol=1e-4)
    np.testing.assert_allclose(trial.load_n["left"], w.FZ_L, atol=0.005)
    np.testing.assert_allclose(trial.cop_m["right"], w.COP_R_MM * 1e-3, atol=1e-5)
    assert np.isnan(trial.cop_m["left"][4:]).all()
    assert dict(trial.events_s) == {}
    assert trial.handedness == "RH"
    assert trial.provenance.file_sha256["csv"] == hashlib.sha256(
        csv_path.read_bytes()
    ).hexdigest()


def test_csv_other_handedness_is_checked_against_lead_pct(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="handedness"):
        csv_trial(tmp_path, handedness="LH")


def test_csv_wrong_header_refused(tmp_path: Path) -> None:
    path = w.write_csv(tmp_path, header=w.CSV_HEADER.replace("t_s", "time"))
    with pytest.raises(ValueError, match="header"):
        read_swing_csv(path, handedness="RH", body_weight_n=800.0, declared=declared())


def test_csv_requires_handedness_and_body_weight(tmp_path: Path) -> None:
    path = w.write_csv(tmp_path)
    with pytest.raises(TypeError):
        read_swing_csv(path, declared=declared())  # type: ignore[call-arg]
    with pytest.raises(ValueError, match="handedness"):
        read_swing_csv(path, handedness="X", body_weight_n=800.0, declared=declared())
    with pytest.raises(ValueError, match="body_weight_n"):
        read_swing_csv(path, handedness="RH", body_weight_n=0.0, declared=declared())


def test_csv_ragged_row_refused(tmp_path: Path) -> None:
    path = w.write_csv(tmp_path)
    path.write_text(path.read_text(encoding="utf-8") + "0.0300,1.00\n", encoding="utf-8")
    with pytest.raises(ValueError, match="row"):
        read_swing_csv(path, handedness="RH", body_weight_n=800.0, declared=declared())


def test_csv_non_numeric_cell_refused(tmp_path: Path) -> None:
    path = w.write_csv(tmp_path)
    text = path.read_text(encoding="utf-8").replace("400.00", "abc", 1)
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="abc"):
        read_swing_csv(path, handedness="RH", body_weight_n=800.0, declared=declared())


def test_csv_oversized_refused(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = w.write_csv(tmp_path)
    monkeypatch.setattr(fmt, "MAX_CSV_BYTES", 10)
    with pytest.raises(ValueError, match="too large"):
        read_swing_csv(path, handedness="RH", body_weight_n=800.0, declared=declared())


def test_csv_symlink_refused(tmp_path: Path) -> None:
    real = w.write_csv(tmp_path)
    link = tmp_path / "link.csv"
    os.symlink(real, link)
    with pytest.raises(ValueError, match="symlink"):
        read_swing_csv(link, handedness="RH", body_weight_n=800.0, declared=declared())


# ---------------------------------------------------------------- list_swings --
def test_list_swings_ignores_junk_and_orphans(tmp_path: Path) -> None:
    w.write_swing(tmp_path, swing_id="20260101-120000-123")
    w.write_swing(tmp_path, swing_id="20260101-120000-123-2")
    (tmp_path / "20260101-130000-000.json").write_text("{}", encoding="utf-8")  # orphan
    (tmp_path / "notes.json").write_text("{}", encoding="utf-8")
    (tmp_path / "notes.npz").write_bytes(b"")
    (tmp_path / "x.csv").write_text("", encoding="utf-8")
    assert list_swings(tmp_path) == ("20260101-120000-123", "20260101-120000-123-2")


def test_list_swings_does_not_follow_symlinks(tmp_path: Path) -> None:
    real, linked = tmp_path / "real", tmp_path / "linked"
    real.mkdir()
    linked.mkdir()
    target = w.write_swing(real)
    os.symlink(target, linked / target.name)
    os.symlink(target.with_suffix(".npz"), linked / f"{w.SWING_ID}.npz")
    assert list_swings(linked) == ()


def test_list_swings_rejects_non_directory(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="directory"):
        list_swings(tmp_path / "missing")


# --------------------------------------------------------------- declarations --
def test_declarations_validate() -> None:
    base = declared()
    with pytest.raises(TypeError):
        PlateDeclarations(  # type: ignore[call-arg]
            uncertainty=base.uncertainty, sensing_depth_m=0.01
        )
    with pytest.raises(TypeError, match="uncertainty"):
        PlateDeclarations(
            uncertainty="x",  # type: ignore[arg-type]
            sensing_depth_m=0.01,
            capacity_n=1000.0,
        )
    with pytest.raises(ValueError, match="capacity_n"):
        PlateDeclarations(
            uncertainty=base.uncertainty, sensing_depth_m=0.01, capacity_n=0.0
        )
    with pytest.raises(ValueError, match="sensing_depth_m"):
        PlateDeclarations(
            uncertainty=base.uncertainty, sensing_depth_m=-1.0, capacity_n=1000.0
        )
    with pytest.raises(ValueError, match="device"):
        PlateDeclarations(
            uncertainty=base.uncertainty,
            sensing_depth_m=0.01,
            capacity_n=1000.0,
            device=" ",
        )
    assert base.device == "open-fp"


def test_read_swing_rejects_wrong_declared_type(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="declared"):
        read_swing(w.write_swing(tmp_path), declared=object())  # type: ignore[arg-type]
