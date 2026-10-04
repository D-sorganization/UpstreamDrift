"""Explicit restriction options preserve the existing eight-field public API."""

from dataclasses import asdict

import pytest

from src.shared.python.motion_matching.historical_fit import ImageFitConfig
from src.shared.python.workspace.necromatcher_fit_jobs import NativeRefitOptions

pytestmark = pytest.mark.unit


def test_restriction_mode_requires_explicit_operation_and_keeps_eight_fields() -> None:
    options = NativeRefitOptions(
        (0, 1),
        2,
        (1.0,),
        operation="restrict_initialization",
        initialization_source="restricted_spline",
    )
    assert len(asdict(options)) == 8
    assert options.config.initialization_policy == "strict"


@pytest.mark.parametrize(
    "operation,source",
    [
        ("fit", "restricted_spline"),
        ("author_initialization", "restricted_spline"),
        ("restrict_initialization", "sampled_parent"),
        ("restrict_initialization", "preserved_spline"),
    ],
)
def test_crossed_restriction_options_reject(operation: str, source: str) -> None:
    with pytest.raises(ValueError):
        NativeRefitOptions(
            (0, 1), 2, (1.0,), operation=operation, initialization_source=source
        )


def test_restriction_rejects_authored_projection_policy() -> None:
    config = ImageFitConfig(
        coordinate_bounds=(("hip", -1.0, 1.0),),
        initialization_policy="authored_range_project_zero_slopes",
    )
    with pytest.raises(ValueError):
        NativeRefitOptions(
            (0, 1),
            2,
            (1.0,),
            config,
            operation="restrict_initialization",
            initialization_source="restricted_spline",
        )


def test_default_and_preserved_modes_keep_their_declared_contract() -> None:
    default = NativeRefitOptions((0, 1), 2, (1.0,))
    preserved = NativeRefitOptions(
        (0, 1), 2, (1.0,), initialization_source="preserved_spline"
    )
    assert default.operation == preserved.operation == "fit"
    assert default.initialization_source == "sampled_parent"
    assert preserved.initialization_source == "preserved_spline"
    assert len(asdict(default)) == len(asdict(preserved)) == 8
