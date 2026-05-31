"""Tests for CC-33 viewport provider evaluation."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.simulation_backends.protocol import Trace
from src.shared.python.visualization.viewport import (
    ProviderAvailability,
    ViewportOverlayPayload,
    ViewportProvider,
    evaluate_viewport_providers,
    select_viewport_provider,
    selected_viewport_decision,
)


def _trace(**kwargs: object) -> Trace:
    t = np.array([0.0, 0.01, 0.02])
    q = np.array(
        [
            [0.0, 0.0, 1.0, 1.0],
            [0.1, 0.0, 1.1, 1.0],
            [0.2, 0.0, 1.2, 1.0],
        ]
    )
    v = np.zeros_like(q)
    return Trace(
        t=t,
        q=q,
        v=v,
        dt=0.01,
        backend="unit",
        meta={"convention": "canonical-v2", "frame": "world_Zup", "units": "SI"},
        **kwargs,
    )


def _checker(available: set[str]):
    def check(module: str) -> bool:
        return module in available

    return check


def test_provider_metadata_records_meshcat_decision() -> None:
    decision = selected_viewport_decision()
    assert decision.provider == ViewportProvider.MESHCAT
    assert decision.selected_default
    assert decision.supports_embedding
    assert decision.supports_markers
    assert decision.supports_wrenches


def test_evaluate_providers_does_not_require_optional_deps() -> None:
    statuses = evaluate_viewport_providers(import_checker=_checker(set()))
    assert {status.metadata.provider for status in statuses} == set(ViewportProvider)
    assert all(
        status.availability == ProviderAvailability.UNAVAILABLE for status in statuses
    )
    assert all(status.degradation_reason for status in statuses)


def test_selects_meshcat_when_available() -> None:
    selection = select_viewport_provider(import_checker=_checker({"meshcat"}))
    assert not selection.degraded
    assert selection.selected is not None
    assert selection.selected.metadata.provider == ViewportProvider.MESHCAT


def test_preferred_unavailable_provider_degrades_without_silent_fallback() -> None:
    selection = select_viewport_provider(
        ViewportProvider.RERUN,
        import_checker=_checker({"meshcat"}),
    )
    assert selection.degraded
    assert selection.reason is not None
    assert "Rerun" in selection.reason
    assert "rerun" in selection.reason


def test_unknown_provider_raises_clear_error() -> None:
    with pytest.raises(ValueError, match="unknown viewport provider"):
        select_viewport_provider("unknown", import_checker=_checker({"meshcat"}))


def test_payload_from_trace_carries_markers_and_wrench() -> None:
    markers = np.arange(18, dtype=float).reshape(3, 2, 3)
    contacts = np.zeros((3, 1, 3), dtype=float)
    wrench = np.arange(18, dtype=float).reshape(3, 6)
    payload = ViewportOverlayPayload.from_trace(
        _trace(markers=markers, contacts=contacts, wrench=wrench),
        marker_names=("lead_wrist", "trail_wrist"),
    )
    np.testing.assert_allclose(payload.trajectory_xyz[:, 0], [0.0, 0.1, 0.2])
    assert payload.marker_names == ("lead_wrist", "trail_wrist")
    assert payload.has_marker_overlay
    assert payload.has_wrench_overlay
    np.testing.assert_allclose(payload.wrench, wrench)


def test_payload_rejects_non_canonical_frame() -> None:
    trace = _trace()
    trace.meta["frame"] = "Y_up"
    with pytest.raises(ValueError, match="world_Zup"):
        ViewportOverlayPayload.from_trace(trace)


def test_payload_rejects_marker_name_mismatch() -> None:
    markers = np.zeros((3, 2, 3), dtype=float)
    with pytest.raises(ValueError, match="marker_names"):
        ViewportOverlayPayload.from_trace(
            _trace(markers=markers),
            marker_names=("only_one",),
        )


def test_fixed_base_trace_uses_origin_metadata() -> None:
    trace = Trace(
        t=np.array([0.0, 0.01]),
        q=np.zeros((2, 2)),
        v=np.zeros((2, 2)),
        meta={"viewport_origin_xyz": [1.0, 2.0, 3.0]},
    )
    payload = ViewportOverlayPayload.from_trace(trace)
    np.testing.assert_allclose(
        payload.trajectory_xyz,
        [[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]],
    )
