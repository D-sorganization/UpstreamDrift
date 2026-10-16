"""Pure-data tests for the Spine Viewer core (SPN-15b, #12772). No Qt."""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.opensim.python.spine.study_aggregation import (
    AggregationIntegrityError,
)
from src.shared.python.spine.levels import SpinalLevel
from src.shared.python.spine.segmental import LevelTransform, SegmentalPose
from src.tools.spine_viewer import core

from .conftest import LEVELS, SOURCE

pytestmark = pytest.mark.unit


def test_open_study_carries_receipt_metadata(study_factory) -> None:
    view = core.open_study(study_factory())
    assert view.study_id == "s1"
    assert view.distribution_source == SOURCE
    assert view.levels == LEVELS
    assert view.scientific_status == "unqualified"
    assert view.series.compression_n().shape == (4, 2)


def test_unqualified_is_the_default_and_never_promoted_by_pass(study_factory) -> None:
    view = core.open_study(study_factory(status="pass"))
    assert view.scientific_status == "unqualified"
    assert "UNQUALIFIED" in core.status_banner(view)


def test_qualified_requires_pass_and_declared_status(study_factory) -> None:
    ok = core.open_study(study_factory(status="pass", scientific_status="qualified"))
    assert ok.scientific_status == "qualified"
    assert core.status_banner(ok).startswith("QUALIFIED")
    bad = core.open_study(
        study_factory(status="unqualified", scientific_status="qualified")
    )
    assert bad.scientific_status == "unqualified"
    assert "UNQUALIFIED" in core.status_banner(bad)


def test_banner_states_distribution_source_and_reasons(study_factory) -> None:
    view = core.open_study(study_factory(status="unqualified"))
    banner = core.status_banner(view)
    assert SOURCE in banner
    assert "invented reason" in banner


def test_heatmap_values_signs_and_status(study_factory) -> None:
    view = core.open_study(study_factory())
    comp = core.view_heatmap(view, "compression")
    assert comp.values[2, 0] == 300.0
    tension = core.view_heatmap(view, "tension")
    assert tension.values[2, 0] == -300.0
    assert comp.scientific_status == "unqualified"
    assert comp.distribution_source == SOURCE


def test_heatmap_figure_title_says_unqualified(study_factory) -> None:
    view = core.open_study(study_factory())
    figure = core.heatmap_figure(view, "compression")
    texts = " ".join(
        t.get_text() for t in figure.findobj(lambda a: hasattr(a, "get_text"))
    )
    assert "UNQUALIFIED" in texts


def test_tampered_artifact_fails_closed(study_factory) -> None:
    receipt = study_factory()
    (receipt.parent / "s1_loads.npz").write_bytes(b"corrupt")
    with pytest.raises(AggregationIntegrityError):
        core.open_study(receipt)


def test_missing_receipt_raises_oserror(tmp_path) -> None:
    with pytest.raises(OSError):
        core.open_study(tmp_path / "nope.json")


def test_unknown_quantity_rejected(study_factory) -> None:
    view = core.open_study(study_factory())
    with pytest.raises(ValueError, match="quantity"):
        core.view_heatmap(view, "bogus")


def test_type_checks(study_factory) -> None:
    with pytest.raises(TypeError):
        core.view_heatmap("not a view", "compression")  # type: ignore[arg-type]
    with pytest.raises(TypeError):
        core.status_banner(None)  # type: ignore[arg-type]


def test_angle_stack_inherits_study_provenance(study_factory) -> None:
    view = core.open_study(study_factory())
    poses = [
        SegmentalPose(
            tuple(
                LevelTransform.from_isb_angles(lv, np.deg2rad([0.0, 0.0, -a]))
                for lv, a in zip(LEVELS, (10.0, 20.0), strict=True)
            )
        )
        for _ in range(2)
    ]
    stack = core.view_angle_stack(
        view, [0.0, 1.0], poses, "x_factor_jcs_axial", handedness="right"
    )
    assert stack.distribution_source == SOURCE
    assert stack.scientific_status == "unqualified"
    assert stack.levels == tuple(SpinalLevel[lv.name] for lv in LEVELS)
