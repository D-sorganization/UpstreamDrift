"""Simulation traces enter the existing explicit motion mapping workflow."""

import json
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("h5py")

from src.motion_capture.reference.importers import load_motion_draft
from src.shared.python.simulation_backends.protocol import Trace
from src.shared.python.simulation_backends.trace_io import write_trace

pytestmark = pytest.mark.unit


def trace_file(path: Path, *, markers: bool = True, names: str | None = None) -> None:
    meta = {"frame": "world_Zup", "model_identity": "qualified-model"}
    if names is not None:
        meta["marker_names_json"] = names
    trace = Trace(
        t=np.array([0, 0.1]),
        q=np.zeros((2, 1)),
        v=np.zeros((2, 1)),
        backend="fixture-backend",
        meta=meta,
        markers=np.array([[[1, 2, 3]], [[np.nan, np.nan, np.nan]]])
        if markers
        else None,
    )
    write_trace(trace, path)


def test_trace_import_preserves_clock_names_gaps_and_provenance(tmp_path: Path):
    path = tmp_path / "motion.h5"
    trace_file(path, names=json.dumps(["clubhead"]))
    draft = load_motion_draft(path)
    assert draft.names == ("clubhead",)
    assert draft.time_s == (0, 0.1)
    np.testing.assert_array_equal(draft.points[0, 0], [1, 2, 3])
    assert np.isnan(draft.points[1, 0]).all()
    assert draft.canonical and draft.units_declared
    assert draft.model_identity == "fixture-backend / qualified-model"
    assert draft.source.format == "simulation-trace/2"


def test_unnamed_trace_exposes_indices_for_explicit_mapping(tmp_path: Path):
    path = tmp_path / "unnamed.hdf5"
    trace_file(path)
    assert load_motion_draft(path).names == ("marker_0",)


@pytest.mark.parametrize("names", ['["a", "b"]', "[3]", '"wrist"'])
def test_invalid_trace_names_are_rejected(tmp_path: Path, names: str):
    path = tmp_path / "bad.h5"
    trace_file(path, names=names)
    with pytest.raises(ValueError, match="marker names"):
        load_motion_draft(path)


def test_trace_without_markers_is_an_explicit_capability_gap(tmp_path: Path):
    path = tmp_path / "state-only.h5"
    trace_file(path, markers=False)
    with pytest.raises(ValueError, match="marker trajectories"):
        load_motion_draft(path)


def test_compressed_trace_budget_is_checked_before_decoding(tmp_path: Path):
    import h5py

    path = tmp_path / "large.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "2.0.0"
        handle.create_dataset(
            "markers", shape=(1_000_000, 3, 3), dtype="f8", chunks=True
        )
    with pytest.raises(ValueError, match="Decoded trace exceeds"):
        load_motion_draft(path)


def test_trace_external_links_are_not_followed(tmp_path: Path):
    import h5py

    path = tmp_path / "linked.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = "2.0.0"
        handle["markers"] = h5py.ExternalLink("outside.h5", "markers")
    with pytest.raises(ValueError, match="links"):
        load_motion_draft(path)


def club_trace(path: Path) -> None:
    trace = Trace(
        t=np.array([0, 0.1]),
        q=np.zeros((2, 1)),
        v=np.zeros((2, 1)),
        markers=np.ones((2, 3, 3)),
        backend="test",
        meta={
            "frame": "world_Zup",
            "marker_names_json": '["wrist", "grip", "head"]',
            "edges_json": "[[0,1],[1,2]]",
            "club_edges_json": "[[1,2]]",
        },
    )
    write_trace(trace, path)


def test_declared_body_and_club_connections_survive_mapping(tmp_path: Path):
    from src.motion_capture.reference.importers import finish_motion_import

    path = tmp_path / "club.h5"
    club_trace(path)
    draft = load_motion_draft(path)
    assert draft.edges == ((0, 1), (1, 2))
    assert draft.club_edges == ((1, 2),)
    asset = finish_motion_import(
        draft,
        title="Club",
        units="m",
        axes=("+X", "+Y", "+Z"),
        joint_names=draft.names,
        edges=draft.edges,
    )
    assert asset.club_edges == ((1, 2),)
    assert asset.edges == ((0, 1), (1, 2))


@pytest.mark.parametrize(
    "edges,clubs",
    [
        ("[[-1,1]]", "[]"),
        ("[[0,3]]", "[]"),
        ("[[0,0]]", "[]"),
        ("[[true,1]]", "[]"),
        ("[[0,1],[0,1]]", "[]"),
        ("[[0,1]]", "[[1,2]]"),
    ],
)
def test_invalid_topology_is_rejected(tmp_path: Path, edges: str, clubs: str):
    import h5py

    path = tmp_path / "bad-topology.h5"
    club_trace(path)
    with h5py.File(path, "a") as handle:
        handle.attrs["meta_edges_json"] = edges
        handle.attrs["meta_club_edges_json"] = clubs
    with pytest.raises(ValueError, match="connections"):
        load_motion_draft(path)
