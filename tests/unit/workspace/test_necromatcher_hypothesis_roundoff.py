"""Authored rebinding retains saved references and bounds canonical roundoff."""

from copy import deepcopy
import json

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _rounded_parent(case, tmp_path, delta):
    from src.shared.python.workspace import NativeHypothesisRequest

    library, request, parent = case
    parent = deepcopy(parent)
    parent["q"][0][3] += delta
    parent["q"][1][3] += delta
    path = tmp_path / "rounded.json"
    path.write_text(json.dumps(parent), encoding="utf-8")
    asset = library.add_fit("rounded-parent", "practice", path)
    record = request.to_record()
    record["parents"].update(
        source_fit_id=asset.dataset_id, source_fit_hash=asset.metadata["hash"]
    )
    record["mapping"]["reference_pose"] = parent["q"][0]
    return library, NativeHypothesisRequest.from_record(record), parent


@pytest.mark.parametrize("delta", [float(np.nextafter(0.1, np.inf) - 0.1), 5e-13])
def test_roundoff_saved_reference_can_bind_and_publish(
    hypothesis_case, tmp_path, delta
):
    from src.shared.python.workspace import author_native_hypothesis
    from src.shared.python.workspace.necromatcher_hypothesis import (
        bind_native_hypothesis,
    )

    library, request, parent = _rounded_parent(hypothesis_case, tmp_path, delta)
    bound = bind_native_hypothesis(library, "rounded-parent", request)
    assert bound.request.mapping.reference_pose == tuple(parent["q"][0])
    asset = author_native_hypothesis(
        library, "rounded-parent", "roundoff-seed", request
    )
    saved = library.load_fit(asset.dataset_id)
    np.testing.assert_allclose(saved["q"], parent["q"], rtol=0, atol=1e-12)
    assert saved["evidence"]["original_fit"]["optimizer_ran"] is False


def test_hypothesis_rejects_legacy_valid_but_excessive_roundoff(
    hypothesis_case, tmp_path
):
    from src.shared.python.workspace.necromatcher_hypothesis import (
        bind_native_hypothesis,
    )

    library, request, _ = _rounded_parent(hypothesis_case, tmp_path, 2e-12)
    with pytest.raises(ValueError, match="roundoff|canonical"):
        bind_native_hypothesis(library, "rounded-parent", request)
