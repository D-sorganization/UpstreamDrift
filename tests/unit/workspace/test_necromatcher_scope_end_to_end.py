"""Native-bound fit admission uses registered original capture receipts."""

from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from src.shared.python.workspace import import_fit_source_scope_review
from src.shared.python.workspace.necromatcher_capture_identity import capture_identity
from src.shared.python.workspace.necromatcher_fit_jobs import (
    NativeRefitOptions,
    start_native_refit,
)
from tests.unit.workspace.hypothesis_fixture import saved_parent
from tests.unit.workspace.test_scope_fixtures import fixture_review_artifact

pytestmark = pytest.mark.unit


def test_original_capture_review_reaches_admitted_native_request(
    native_fit_case: tuple,
    tmp_path: Path,
) -> None:
    library, parent = saved_parent(native_fit_case, tmp_path)
    saved_before = library.load_fit("parent")
    parent_path = Path(library.load_asset("parent").path)
    parent_bytes = parent_path.read_bytes()
    identity = capture_identity(library, parent["capture_id"])
    reference = fixture_review_artifact(tmp_path, identity, 0, 3)
    scope = import_fit_source_scope_review(
        library, "parent", Path(reference.path).read_bytes()
    )
    admitted = []
    service = SimpleNamespace(start=lambda spec, **kwargs: admitted.append(spec))
    _, root = start_native_refit(
        library,
        "parent",
        "scoped-candidate",
        NativeRefitOptions((0, 2), 2, (1.0,) * len(parent["coordinate_order"])),
        service,
        source_scope=scope,
    )
    request = json.loads((root / "request.json").read_bytes())
    assert len(admitted) == 1
    assert request["source_scope"] == scope.to_record()
    assert request["source_scope_binding"]["frame_indices"] == [0, 2]
    assert (
        request["source_scope_binding"]["source_clock_sha256"]
        == identity.source_clock_sha256
    )
    assert library.load_fit("parent") == saved_before
    assert parent_path.read_bytes() == parent_bytes
    with pytest.raises(KeyError):
        library.load_fit("scoped-candidate")
