"""A failed JSON recording export must not destroy the previous artifact."""

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.data_io import export as export_module
from src.shared.python.data_io.export import export_recording_all_formats


@pytest.mark.unit
@pytest.mark.parametrize("existing", [False, True])
@pytest.mark.parametrize("failure", ["unsupported", "circular"])
def test_failed_json_serialization_preserves_recording(
    tmp_path: Path, existing: bool, failure: str
) -> None:
    output = tmp_path / "recording.json"
    original = b'{"previous": "valuable recording"}\n'
    if existing:
        output.write_bytes(original)
    invalid: Any = object()
    error: type[Exception] = TypeError
    if failure == "circular":
        invalid = []
        invalid.append(invalid)
        error = ValueError

    with pytest.raises(error):
        export_recording_all_formats(
            str(output.with_suffix("")),
            {"times": np.arange(3), "invalid": invalid},
            formats=["json"],
        )

    assert output.exists() is existing
    if existing:
        assert output.read_bytes() == original
    assert list(tmp_path.glob(".*.tmp")) == []


@pytest.mark.unit
@pytest.mark.parametrize("existing", [False, True])
def test_failed_json_publication_preserves_recording(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, existing: bool
) -> None:
    output = tmp_path / "recording.json"
    original = b'{"previous": "valuable recording"}\n'
    if existing:
        output.write_bytes(original)

    def refuse_replace(source: Any, destination: Any) -> None:
        assert Path(destination) == output
        assert json.loads(Path(source).read_text())["times"] == [0, 1, 2]
        raise OSError("replacement refused")

    monkeypatch.setattr(export_module.os, "replace", refuse_replace)
    with pytest.raises(OSError, match="replacement refused"):
        export_recording_all_formats(
            str(output.with_suffix("")), {"times": np.arange(3)}, formats=["json"]
        )

    assert output.exists() is existing
    if existing:
        assert output.read_bytes() == original
    assert list(tmp_path.glob(".*.tmp")) == []
