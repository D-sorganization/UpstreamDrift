"""Schema-only tests for the golf metric-definitions registry (#14422)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

pytestmark = pytest.mark.unit


def _valid_kwargs() -> dict:
    from src.shared.python.biomechanics.metric_definitions.schema import (
        FilterDifferentiationDescriptor,
    )

    return {
        "metric_id": "example.x_factor_line",
        "segments": ("thorax", "pelvis"),
        "landmark_proxy": "joint_centers",
        "dimension": "3d",
        "reference_frame": "lab_target_line",
        "cardan_order": "ZXY",
        "sign_convention": "right_handed_positive_on_trail_side",
        "units": "deg",
        "event_anchors": ("address", "top", "impact"),
        "normalisation": "percent_body_weight",
        "source": "measured",
        "definition_version": "1.0.0",
        "cites": ("Brown2013",),
        "filter_descriptor": FilterDifferentiationDescriptor(
            lowpass_cutoff_hz=15.0,
            method="butterworth_4th_order",
            min_sampling_hz=120.0,
        ),
    }


def test_valid_metric_definition_accepts_required_fields() -> None:
    from src.shared.python.biomechanics.metric_definitions.schema import (
        GolfMetricDefinitionEntry,
    )

    entry = GolfMetricDefinitionEntry(**_valid_kwargs())
    assert entry.metric_id == "example.x_factor_line"
    assert entry.dimension == "3d"
    assert entry.source == "measured"


@pytest.mark.parametrize("drop", list(_valid_kwargs().keys()))
def test_rejects_missing_required_field(drop: str) -> None:
    from src.shared.python.biomechanics.metric_definitions.schema import (
        GolfMetricDefinitionEntry,
    )

    kw = _valid_kwargs()
    del kw[drop]
    with pytest.raises(ValidationError):
        GolfMetricDefinitionEntry(**kw)


def test_rejects_empty_cites() -> None:
    from src.shared.python.biomechanics.metric_definitions.schema import (
        GolfMetricDefinitionEntry,
    )

    kw = _valid_kwargs()
    kw["cites"] = ()
    with pytest.raises(ValidationError):
        GolfMetricDefinitionEntry(**kw)


def test_filter_descriptor_requires_positive_min_sampling_hz() -> None:
    from src.shared.python.biomechanics.metric_definitions.schema import (
        FilterDifferentiationDescriptor,
    )

    with pytest.raises(ValidationError):
        FilterDifferentiationDescriptor(
            lowpass_cutoff_hz=None,
            method="central_difference",
            min_sampling_hz=0.0,
        )
