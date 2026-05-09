"""Round-trip tests for the Simscape ``.mat`` engine adapter."""

from __future__ import annotations

from pathlib import Path

import pytest

from anthropometrics import ADAPTER_REGISTRY, EngineAdapter, SubjectAnthropometrics
from anthropometrics.engine_adapters import SimscapeAdapter


from ._assertions import assert_subjects_equal


def test_simscape_adapter_satisfies_protocol() -> None:
    assert isinstance(SimscapeAdapter(), EngineAdapter)


def test_simscape_in_registry() -> None:
    assert "simscape" in ADAPTER_REGISTRY
    assert ADAPTER_REGISTRY["simscape"].engine_name == "simscape"


def test_simscape_round_trip_preserves_subject(
    sixteen_segment_subject: SubjectAnthropometrics, tmp_path: Path
) -> None:
    adapter = SimscapeAdapter()
    out = tmp_path / "subject.mat"
    adapter.export(sixteen_segment_subject, out)
    assert out.exists() and out.stat().st_size > 0
    restored = adapter.import_back(out)
    assert_subjects_equal(restored, sixteen_segment_subject)


def test_simscape_export_rejects_non_subject(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        SimscapeAdapter().export("nope", tmp_path / "x.mat")  # type: ignore[arg-type]


def test_simscape_missing_key_raises(tmp_path: Path) -> None:
    """Missing required keys raise ValueError."""
    from scipy.io import savemat

    bad = tmp_path / "bad.mat"
    savemat(str(bad), {"only_this": 1.0})
    with pytest.raises(ValueError, match="missing required key"):
        SimscapeAdapter().import_back(bad)


def test_simscape_round_trip_handles_no_age(
    sixteen_segment_subject: SubjectAnthropometrics, tmp_path: Path
) -> None:
    """``age_years=None`` round-trips via the ``nan`` sentinel."""
    from dataclasses import replace

    no_age = replace(sixteen_segment_subject, age_years=None)
    adapter = SimscapeAdapter()
    out = tmp_path / "subject.mat"
    adapter.export(no_age, out)
    restored = adapter.import_back(out)
    assert restored.age_years is None
    assert_subjects_equal(restored, no_age)
