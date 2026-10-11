"""Strict source/chart contracts for the sampled native path-work diagnostic."""

from __future__ import annotations

import hashlib
import math

import pytest

from src.engines.physics_engines.opensim.python.tour_matching.native_path_work_audit import (
    NativePathWorkIdentity,
    audit_native_path_work,
)

pytestmark = pytest.mark.unit


def _identity(source: bytes = b"owned native model") -> NativePathWorkIdentity:
    return NativePathWorkIdentity(
        source_sha256=hashlib.sha256(source).hexdigest(),
        path_component="/forceset/cable/geometrypath",
        coordinate_path="/jointset/base/angle",
        point_paths=(
            "/forceset/cable/pathpointset/origin",
            "/forceset/cable/pathpointset/guide",
        ),
        source_component_paths=(
            "/forceset/cable/geometrypath",
            "/jointset/base/angle",
            "/forceset/cable/pathpointset/origin",
            "/forceset/cable/pathpointset/guide",
        ),
        chart_bounds=(-1.0, 1.0),
        active_bounds=(-0.5, 0.7),
        function_bounds=(-0.8, 0.6),
    )


def test_sampled_path_work_is_bound_to_source_and_intersected_chart() -> None:
    source = b"owned native model"
    identity = _identity(source)
    audit = audit_native_path_work(
        identity,
        source,
        q_values=(-0.2, 0.2),
        step_sizes=(1e-4, 1e-5),
        sample_path=lambda q: (q, 1.0 - 0.1 * q),
        native_moment_arm=lambda _q: 0.1,
        force_moment_arm=lambda _q: 0.1,
    )
    assert identity.surveyed_bounds == (-0.5, 0.6)
    assert audit.status == "workless_identity_satisfied_at_tested_samples"
    assert audit.evidence_scope == "sampled-domain-only"
    assert len(audit.samples) == 4
    assert all(
        sample.achieved_span_rad == pytest.approx(2 * sample.step_rad)
        for sample in audit.samples
    )
    assert max(sample.workless_residual_m for sample in audit.samples) < 1e-7


def test_moving_guide_residual_is_preserved_while_native_force_agrees() -> None:
    source = b"owned moving guide source"
    identity = _identity(source)
    audit = audit_native_path_work(
        identity,
        source,
        q_values=(0.0,),
        step_sizes=(1e-4, 1e-5),
        sample_path=lambda q: (q, 1.0 - 0.13 * q),
        native_moment_arm=lambda _q: 0.1,
        force_moment_arm=lambda _q: 0.1,
    )
    assert audit.status == "workless_identity_not_satisfied"
    assert all(sample.force_residual_m < 1e-12 for sample in audit.samples)
    assert all(
        sample.workless_residual_m == pytest.approx(0.03, abs=1e-10)
        for sample in audit.samples
    )


def test_source_component_domain_and_achieved_state_mutations_fail_closed() -> None:
    source = b"owned native model"
    identity = _identity(source)
    arguments = {
        "q_values": (0.0,),
        "step_sizes": (1e-4,),
        "sample_path": lambda q: (q, 1.0),
        "native_moment_arm": lambda _q: 0.0,
        "force_moment_arm": lambda _q: 0.0,
    }
    with pytest.raises(ValueError, match="source bytes"):
        audit_native_path_work(identity, b"mutated model", **arguments)
    with pytest.raises(ValueError, match="component inventory"):
        NativePathWorkIdentity(
            source_sha256=identity.source_sha256,
            path_component="/forceset/other/geometrypath",
            coordinate_path=identity.coordinate_path,
            point_paths=identity.point_paths,
            source_component_paths=identity.source_component_paths,
            chart_bounds=identity.chart_bounds,
            active_bounds=identity.active_bounds,
            function_bounds=identity.function_bounds,
        )
    with pytest.raises(ValueError, match="intersected chart"):
        audit_native_path_work(
            identity,
            source,
            q_values=(0.6,),
            step_sizes=(1e-4,),
            sample_path=lambda q: (q, 1.0),
            native_moment_arm=lambda _q: 0.0,
            force_moment_arm=lambda _q: 0.0,
        )
    with pytest.raises(ValueError, match="achieve"):
        audit_native_path_work(
            identity,
            source,
            q_values=(0.0,),
            step_sizes=(1e-4,),
            sample_path=lambda q: (q + 1e-4, 1.0),
            native_moment_arm=lambda _q: 0.0,
            force_moment_arm=lambda _q: 0.0,
        )


@pytest.mark.parametrize("bad", [math.nan, math.inf, -1.0, 0.0])
def test_invalid_tolerance_is_rejected(bad: float) -> None:
    source = b"owned native model"
    with pytest.raises(ValueError, match="tolerance"):
        audit_native_path_work(
            _identity(source),
            source,
            q_values=(0.0,),
            step_sizes=(1e-4,),
            sample_path=lambda q: (q, 1.0),
            native_moment_arm=lambda _q: 0.0,
            force_moment_arm=lambda _q: 0.0,
            tolerance_m=bad,
        )
