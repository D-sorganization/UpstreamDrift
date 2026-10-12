"""Saved-spline recall validates identities without loading an engine SDK."""

from copy import deepcopy
import hashlib
import json

import pytest

pytestmark = pytest.mark.unit


def _fit():
    definition = {"coordinate_order": ["hip"]}
    return {
        "coordinate_order": ["hip"],
        "provenance": {"native_definition": definition},
        "evidence": {
            "original_fit": {
                "coordinate_order": ["hip"],
                "free_coordinates": ["hip"],
                "knot_times": [0.0, 0.2],
                "spline_coefficients": [0.1, 0.2, 0.0, 0.0],
            }
        },
    }


def test_legacy_spline_recall_binds_exact_native_definition():
    from src.shared.python.workspace.necromatcher_spline import preserved_fit_spline

    fit = _fit()
    start = preserved_fit_spline(fit)
    assert start is not None
    assert (
        start.model_sha
        == hashlib.sha256(
            json.dumps(fit["provenance"]["native_definition"], allow_nan=False).encode()
        ).hexdigest()
    )
    assert start.knot_times == (0.0, 0.2)


def test_absent_legacy_spline_is_unavailable_but_partial_record_rejects():
    from src.shared.python.workspace.necromatcher_spline import preserved_fit_spline

    fit = _fit()
    fit["evidence"]["original_fit"] = {}
    assert preserved_fit_spline(fit) is None
    fit["evidence"]["original_fit"]["knot_times"] = [0, 1]
    with pytest.raises(ValueError, match="Incomplete"):
        preserved_fit_spline(fit)


@pytest.mark.parametrize(
    "field", ["model_sha", "coordinate_order", "coefficient_sha256"]
)
def test_declared_spline_identity_corruption_is_never_hidden(field):
    from src.shared.python.workspace.necromatcher_spline import preserved_fit_spline

    fit = _fit()
    record = preserved_fit_spline(fit).to_record()
    fit["evidence"]["original_fit"]["spline_start"] = deepcopy(record)
    fit["evidence"]["original_fit"]["spline_start"][field] = (
        ["other"] if field == "coordinate_order" else "wrong"
    )
    with pytest.raises(ValueError):
        preserved_fit_spline(fit)


def test_refit_plan_exposes_authoritative_recipe_and_exact_spline_clock():
    from types import SimpleNamespace
    from src.shared.python.workspace import refit_plan

    fit = _fit()
    fit.update(frame_indices=[0, 2], coordinate_units=["rad"], qualification="research")
    fit["provenance"]["request_options"] = {"config": {"prior_weight": 9.0}}
    fit["evidence"]["original_fit"]["config"] = {"prior_weight": 0.25}
    plan = refit_plan(SimpleNamespace(load_fit=lambda _: fit), "source")
    assert plan["baseline_config"]["prior_weight"] == 0.25
    assert plan["preserved_spline"] == {
        "available": True,
        "knot_count": 2,
        "source_interval": [0.0, 0.2],
        "reason": "Exact saved coefficients and source interval are available",
    }
