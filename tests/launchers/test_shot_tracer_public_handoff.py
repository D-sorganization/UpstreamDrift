"""Public display accepts validated samples without I/O or simulation."""

from dataclasses import replace

import numpy as np
import pytest

from src.launchers._shot_tracer_trajectory_import import ImportedTrajectoryCurve

pytestmark = pytest.mark.unit


def curve() -> ImportedTrajectoryCurve:
    return ImportedTrajectoryCurve(
        "research / recorded",
        np.array([[0.0, 0.0, 0.0], [3.25, -1.5, 2.0]]),
        "owned-run",
        "research",
        "recorded",
        "flight_xfwd_yleft_zup",
    )


def test_public_display_detaches_exact_samples_and_replaces_label(tracer_widget):
    original = curve()
    expected = original.positions.copy()
    tracer_widget.display_imported_trajectory(original)
    stored = tracer_widget.imported_trajectories[original.label]
    assert stored is not original
    assert not np.shares_memory(stored.positions, original.positions)
    original.positions[:] = 999
    np.testing.assert_array_equal(stored.positions, expected)
    assert not stored.positions.flags.writeable
    replacement = replace(curve(), positions=expected + 1)
    tracer_widget.display_imported_trajectory(replacement)
    assert tracer_widget.imported_list.count() == 1
    np.testing.assert_array_equal(
        tracer_widget.imported_trajectories[original.label].positions, expected + 1
    )


@pytest.mark.parametrize(
    "changes",
    [
        {"positions": np.zeros((0, 3))},
        {"positions": np.zeros((2, 2))},
        {"positions": np.array([[np.nan, 0, 0]])},
        {"positions": np.array([[np.inf, 0, 0]])},
        {"positions": np.ones((1, 3), dtype=bool)},
        {"positions": np.ones((1, 3), dtype=complex)},
        {"positions": [[0, 0, 0]]},
        {"frame_id": "app_xtarget_yup_zright"},
        {"label": " "},
        {"source_id": ""},
        {"model_family": ""},
        {"model_name": ""},
    ],
)
def test_invalid_curve_rejected_before_publication(tracer_widget, changes):
    tracer_widget.display_imported_trajectory(curve())
    previous = dict(tracer_widget.imported_trajectories)
    with pytest.raises((TypeError, ValueError)):
        tracer_widget.display_imported_trajectory(replace(curve(), **changes))
    assert tracer_widget.imported_trajectories == previous
    assert tracer_widget.imported_list.count() == 1


def test_non_dto_rejected_before_publication(tracer_widget):
    with pytest.raises(TypeError):
        tracer_widget.display_imported_trajectory({"positions": [[0, 0, 0]]})
    assert tracer_widget.imported_list.count() == 0
