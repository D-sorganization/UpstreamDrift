"""Hypothesis source binding cannot cross a declared camera within one archive."""

import pytest

from hypothesis_fixture import imported_capture

pytestmark = pytest.mark.unit


def test_mixed_source_cameras_reject_even_in_an_authentic_import(
    fit_case, tmp_path
) -> None:
    from src.shared.python.workspace.necromatcher_capture_identity import (
        capture_identity,
    )

    library, _, _ = fit_case
    captured = imported_capture(
        library, tmp_path / "mixed-capture", camera_ids=("first", "second", "first")
    )
    with pytest.raises(ValueError, match="camera|shot|identity"):
        capture_identity(library, captured.dataset_id)
