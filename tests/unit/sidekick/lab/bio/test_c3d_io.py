"""Unit tests for ``_c3d_io`` parsers, unit-scale, sanitization, exports."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from sidekick.lab.bio import _c3d_io as io_mod
from sidekick.lab.bio._c3d_io import (
    build_metadata,
    export_dataframe,
    get_analog_details,
    get_analog_parameters,
    get_events,
    get_point_parameters,
    load_c3d,
    sanitize_for_csv,
    unit_scale,
    validate_export_path,
    write_export,
)
from sidekick.lab.bio._c3d_models import (
    SCHEMA_VERSION,
    C3DEvent,
)
from ._synthetic import _synthetic_c3d_dict

# ----- load_c3d --------------------------------------------------------------


def test_load_c3d_missing_file(tmp_path: Path) -> None:
    bogus = tmp_path / "does_not_exist.c3d"
    with pytest.raises(FileNotFoundError):
        load_c3d(bogus)


def test_load_c3d_no_ezc3d(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(io_mod, "ezc3d", None)
    with pytest.raises(ImportError, match="ezc3d is required"):
        load_c3d(tmp_path / "x.c3d")


# ----- get_point_parameters / get_analog_parameters --------------------------


def test_get_point_parameters() -> None:
    data = _synthetic_c3d_dict()
    pp = get_point_parameters(data, Path("foo.c3d"))
    assert "LABELS" in pp


def test_get_analog_parameters_present() -> None:
    data = _synthetic_c3d_dict(n_analog=2)
    ap = get_analog_parameters(data)
    assert ap is not None
    assert "LABELS" in ap


def test_get_analog_parameters_missing() -> None:
    data = _synthetic_c3d_dict(omit_analog_group=True)
    assert get_analog_parameters(data) is None


# ----- get_analog_details ----------------------------------------------------


def test_analog_details_no_group() -> None:
    data = _synthetic_c3d_dict(omit_analog_group=True)
    labels, rate, units = get_analog_details(data)
    assert labels == []
    assert rate is None
    assert units == []


def test_analog_details_synth_labels_when_empty() -> None:
    # Channel data exists but no labels — generated names
    data = _synthetic_c3d_dict(n_analog=2, analog_labels=[], analog_units=[])
    labels, rate, units = get_analog_details(data)
    assert labels == ["Analog_1", "Analog_2"]
    assert rate == 1000.0
    # units padded
    assert units == ["", ""]


def test_analog_details_unit_truncation() -> None:
    data = _synthetic_c3d_dict(
        n_analog=1, analog_labels=["A1"], analog_units=["V", "extra"]
    )
    labels, _, units = get_analog_details(data)
    assert labels == ["A1"]
    assert units == ["V"]


def test_analog_details_unit_padding() -> None:
    data = _synthetic_c3d_dict(
        n_analog=2, analog_labels=["A1", "A2"], analog_units=["V"]
    )
    _, _, units = get_analog_details(data)
    assert units == ["V", ""]


# ----- get_events ------------------------------------------------------------


def test_get_events_none() -> None:
    data = _synthetic_c3d_dict()
    assert get_events(data) == []


def test_get_events_simple() -> None:
    data = _synthetic_c3d_dict(
        with_events=True,
        event_labels=["A", "B"],
        event_times=[0.1, 0.5],
    )
    events = get_events(data)
    assert events == [C3DEvent("A", 0.1), C3DEvent("B", 0.5)]


def test_get_events_2d_times() -> None:
    data = _synthetic_c3d_dict(
        with_events=True,
        event_labels=["A"],
        event_times=[0.7],
        event_times_2d=True,
    )
    events = get_events(data)
    assert events == [C3DEvent("A", 0.7)]


def test_get_events_missing_times() -> None:
    data = _synthetic_c3d_dict(
        with_events=True,
        event_labels=["A"],
        event_times_missing=True,
    )
    assert get_events(data) == []


def test_get_events_missing_used_infers_from_labels() -> None:
    """Regression for #4753: missing EVENT:USED must not silently drop events.

    Real-world c3d files routinely omit USED while still containing valid
    LABELS/TIMES arrays. We must infer USED from the available metadata
    rather than defaulting to 0.
    """
    data = _synthetic_c3d_dict(
        with_events=True,
        event_labels=["FootStrike", "Toe-off"],
        event_times=[0.5, 1.0],
        event_used_omit=True,  # explicitly omit USED
    )
    events = get_events(data)
    assert events == [C3DEvent("FootStrike", 0.5), C3DEvent("Toe-off", 1.0)]


def test_get_events_explicit_used_parity() -> None:
    """Regression for #4753: explicit USED=N yields same result as inferred."""
    data = _synthetic_c3d_dict(
        with_events=True,
        event_labels=["A", "B"],
        event_times=[0.1, 0.5],
        event_used_omit=False,
        event_used=2,
    )
    events = get_events(data)
    assert events == [C3DEvent("A", 0.1), C3DEvent("B", 0.5)]


def test_get_events_explicit_used_zero_honored() -> None:
    """Regression for #4753: explicit USED=0 still yields no events."""
    data = _synthetic_c3d_dict(
        with_events=True,
        event_labels=["A", "B"],
        event_times=[0.1, 0.5],
        event_used_omit=False,
        event_used=0,
    )
    assert get_events(data) == []


def test_get_events_no_event_group_returns_empty() -> None:
    """Regression for #4753: no EVENT group at all yields [] without error."""
    data = _synthetic_c3d_dict()  # no EVENT group
    assert get_events(data) == []


def test_get_events_skips_non_finite_time() -> None:
    data = _synthetic_c3d_dict(
        with_events=True,
        event_labels=["good", "bad"],
        event_times=[0.5, float("nan")],
    )
    events = get_events(data)
    assert events == [C3DEvent("good", 0.5)]


# ----- build_metadata --------------------------------------------------------


def test_build_metadata_round_trip() -> None:
    data = _synthetic_c3d_dict(
        n_frames=5,
        n_markers=2,
        marker_names=["A", "B"],
        n_analog=1,
        analog_labels=["X"],
        analog_units=["V"],
        with_events=True,
        event_labels=["evt"],
        event_times=[0.01],
    )
    md = build_metadata(data, Path("dummy.c3d"))
    assert md.marker_labels == ["A", "B"]
    assert md.frame_count == 5
    assert md.frame_rate == 100.0
    assert md.units == "m"
    assert md.analog_labels == ["X"]
    assert md.analog_units == ["V"]
    assert md.analog_rate == 1000.0
    assert md.events == [C3DEvent("evt", 0.01)]


# ----- unit_scale ------------------------------------------------------------


def test_unit_scale_none() -> None:
    assert unit_scale("m", None) == 1.0


def test_unit_scale_same() -> None:
    assert unit_scale("m", "m") == 1.0
    assert unit_scale("MM", "mm") == 1.0


@pytest.mark.parametrize(
    "src,dst,expected",
    [
        ("m", "mm", 1000.0),
        ("mm", "m", 0.001),
        ("cm", "mm", 10.0),
        ("in", "mm", 25.4),
        ("ft", "m", 0.3048),
    ],
)
def test_unit_scale_pairs(src: str, dst: str, expected: float) -> None:
    assert unit_scale(src, dst) == pytest.approx(expected)


def test_unit_scale_unsupported_source(caplog) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        assert unit_scale("furlongs", "m") == 1.0
    assert "Unsupported or unknown unit conversion" in caplog.text


def test_unit_scale_unsupported_target(caplog) -> None:
    import logging

    with caplog.at_level(logging.WARNING):
        assert unit_scale("m", "furlongs") == 1.0
    assert "Unsupported or unknown unit conversion" in caplog.text


# ----- sanitize_for_csv ------------------------------------------------------


@pytest.mark.parametrize("prefix", ["=", "+", "-", "@"])
def test_sanitize_for_csv_prefixes(prefix: str) -> None:
    assert sanitize_for_csv(f"{prefix}cmd") == f"'{prefix}cmd"


def test_sanitize_for_csv_safe_string() -> None:
    assert sanitize_for_csv("plain") == "plain"


def test_sanitize_for_csv_non_string() -> None:
    assert sanitize_for_csv(42) == 42
    assert sanitize_for_csv(None) is None


# ----- validate_export_path --------------------------------------------------


def test_validate_export_path_inside_test_env(tmp_path: Path) -> None:
    # tmp_path strings include "pytest" so this is treated as test env
    target = tmp_path / "out.csv"
    validate_export_path(target)


def test_validate_export_path_outside_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Patch Path.cwd to a clean string that doesn't contain "pytest"/"test",
    # and target a clean path likewise. Otherwise the test-env heuristic
    # short-circuits the security check.
    fake_cwd = Path("C:/clean_root_xyz").resolve()
    monkeypatch.setattr(Path, "cwd", classmethod(lambda cls: fake_cwd))
    other = Path("C:/").resolve() / "outside_clean_root" / "out.csv"
    with pytest.raises(ValueError, match="Refusing to output"):
        validate_export_path(other)


# ----- write_export per format ----------------------------------------------


def _sample_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "frame": [0, 1],
            "marker": ["=danger", "safe"],
            "x": [1.0, 2.0],
        }
    )


def test_write_export_csv_sanitizes(tmp_path: Path) -> None:
    df = _sample_df()
    out = tmp_path / "p.csv"
    write_export(out, "csv", df, {"k": "v"}, do_sanitize=True)
    text = out.read_text()
    assert "'=danger" in text
    meta = json.loads((tmp_path / "p_meta.json").read_text())
    assert meta == {"k": "v"}


def test_write_export_csv_no_sanitize(tmp_path: Path) -> None:
    df = _sample_df()
    out = tmp_path / "p.csv"
    write_export(out, "csv", df, {"k": "v"}, do_sanitize=False)
    text = out.read_text()
    assert "=danger" in text
    assert "'=danger" not in text


def test_write_export_json(tmp_path: Path) -> None:
    df = _sample_df()
    out = tmp_path / "p.json"
    write_export(out, "json", df, {"src": "x"}, do_sanitize=False)
    payload = json.loads(out.read_text())
    assert payload["metadata"] == {"src": "x"}
    assert len(payload["data"]) == 2


def test_write_export_npz(tmp_path: Path) -> None:
    df = _sample_df()
    out = tmp_path / "p.npz"
    write_export(out, "npz", df, {"src": "x"}, do_sanitize=False)
    arr = np.load(out, allow_pickle=False)
    assert "frame" in arr.files
    assert "_metadata" in arr.files
    assert json.loads(str(arr["_metadata"])) == {"src": "x"}


def test_write_export_invalid_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="Unsupported export format"):
        write_export(tmp_path / "p.xyz", "xyz", _sample_df(), {}, False)


# ----- export_dataframe -----------------------------------------------------


def test_export_dataframe_infers_format_from_suffix(tmp_path: Path) -> None:
    df = _sample_df()
    out = tmp_path / "auto.csv"
    result = export_dataframe(df, out, None, "src.c3d", "m")
    assert result.exists()
    meta = json.loads((tmp_path / "auto_meta.json").read_text())
    assert meta["schema_version"] == SCHEMA_VERSION
    assert meta["source_file"] == "src.c3d"
    assert meta["units"] == "m"
    assert meta["row_count"] == 2


def test_export_dataframe_no_suffix_no_format(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="File format could not be inferred"):
        export_dataframe(_sample_df(), tmp_path / "noext", None, "x.c3d", "m")


def test_export_dataframe_explicit_format(tmp_path: Path) -> None:
    out = tmp_path / "explicit.bin"
    result = export_dataframe(_sample_df(), out, "json", "src.c3d", "m")
    assert result.exists()
    payload = json.loads(out.read_text())
    assert "metadata" in payload


def test_export_dataframe_creates_parent(tmp_path: Path) -> None:
    target = tmp_path / "nested" / "deep" / "out.csv"
    export_dataframe(_sample_df(), target, None, "x.c3d", "m")
    assert target.exists()


def test_export_dataframe_npz(tmp_path: Path) -> None:
    out = tmp_path / "out.npz"
    result = export_dataframe(_sample_df(), out, None, "x.c3d", "mm")
    assert result.exists()
    arr = np.load(result, allow_pickle=False)
    meta = json.loads(str(arr["_metadata"]))
    assert meta["units"] == "mm"


# ----- patch-based load_c3d sanity ------------------------------------------


def test_load_c3d_via_patch(tmp_path: Path) -> None:
    real_path = tmp_path / "fake.c3d"
    real_path.write_bytes(b"\x00")  # exists check
    fake = _synthetic_c3d_dict()
    with patch.object(io_mod.ezc3d, "c3d", return_value=fake):
        loaded = load_c3d(real_path)
    assert loaded is fake
