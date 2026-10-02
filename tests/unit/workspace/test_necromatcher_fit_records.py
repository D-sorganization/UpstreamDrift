"""Canonical fit records distinguish coordinate expansion from optimization."""

from dataclasses import asdict, replace
import hashlib
import json
import numpy as np
import pytest
from src.shared.python.motion_matching.historical_fit import (
    ImageFitConfig,
    ImageFitResult,
    ImageSplineStart,
    expand_image_spline_coordinates,
)
from src.shared.python.workspace.necromatcher_fit_records import (
    build_native_fit_payload,
)

pytestmark = pytest.mark.unit


def specimen(source=None):
    if source is None:
        definition = {"coordinate_order": ["arm", "wrist"]}
        source = {
            "coordinate_order": ["arm", "wrist"],
            "provenance": {"native_definition": definition},
            "evidence": {"original_fit": {"camera": {}, "attachments": {}}},
        }
    order = tuple(source["coordinate_order"])
    definition = source["provenance"]["native_definition"]
    sha = hashlib.sha256(json.dumps(definition, allow_nan=False).encode()).hexdigest()
    oldfree = (order[0],)
    desired = (order[0], order[-1])
    times = np.array([0.0, 0.2])
    old = ImageSplineStart.from_coefficients(times, np.zeros(4), order, oldfree, sha)
    original = source["evidence"]["original_fit"]
    original.update(
        knot_times=list(old.knot_times),
        spline_coefficients=list(old.spline_coefficients),
        coordinate_order=list(order),
        free_coordinates=list(oldfree),
        spline_start=old.to_record(),
    )
    reference = np.zeros(len(order))
    expansion = expand_image_spline_coordinates(old, desired, reference)
    source["q"] = np.zeros((2, len(order))).tolist()
    source.setdefault("frame_indices", [0, 2])
    source.setdefault("frames", [{}, {}])
    original.update(source_times=times.tolist(), q=source["q"], frame_indices=[0, 2])
    result = ImageFitResult(
        times,
        np.zeros((2, len(order))),
        1.0,
        1.0,
        np.zeros((2, 1)),
        2,
        sha,
        order,
        False,
        "Unoptimized expanded seed",
        times,
        np.asarray(expansion.expanded_start.spline_coefficients),
        desired,
        optimizer_ran=False,
        initial_spline=expansion.expanded_start,
    )
    request = {
        "source_fit_id": "parent",
        "source_fit_hash": "sha256:" + "a" * 64,
        "execution_stamp": {},
        "options": {
            "frame_indices": [0, 2],
            "config": asdict(ImageFitConfig()),
            "operation": "coordinate_expansion",
            "initialization_source": "preserved_spline",
        },
    }
    stamp = dict.fromkeys(("started_at_utc", "source_sha256", "runtime_sha256"), "test")
    dense = ((0, 2), source.get("frames", [{}, {}]), result.q)
    return request, source, result, dense, stamp, expansion


def test_public_seed_record_is_fresh_and_not_optimizer_receipt():
    request, source, result, dense, stamp, expansion = specimen()
    record = build_native_fit_payload(
        request, source, result, dense, stamp, 0.0, coordinate_expansion=expansion
    )
    assert record["provenance"]["coordinate_expansion"] == asdict(expansion)
    assert record["provenance"]["operation"] == "coordinate_expansion"
    assert record["provenance"]["warm_start_fit_hash"] == request["source_fit_hash"]
    assert record["evidence"]["original_fit"]["optimizer_ran"] is False
    assert record["evidence"]["original_fit"]["initialization"] is None
    assert (
        record["evidence"]["original_fit"]["spline_start"]
        == expansion.expanded_start.to_record()
    )
    assert "coordinate_expansion_only" in record["evidence"]["rejection_reasons"]
    assert "authored_initialization_only" not in record["evidence"]["rejection_reasons"]
    assert "historical_anatomy_unqualified" in record["evidence"]["rejection_reasons"]


@pytest.mark.parametrize(
    "change",
    [
        "optimized",
        "converged",
        "model",
        "free",
        "coefficients",
        "initial",
        "oldhash",
        "operation",
        "missing",
    ],
)
def test_coordinate_seed_rejects_transplanted_or_inconsistent_result(change):
    request, source, result, dense, stamp, expansion = specimen()
    if change == "optimized":
        result = replace(result, optimizer_ran=True)
    if change == "converged":
        result = replace(result, converged=True, optimizer_ran=True)
    if change == "model":
        result = replace(result, model_sha="wrong")
    if change == "free":
        result = replace(result, free_coordinates=("wrist", "arm"))
    if change == "coefficients":
        result = replace(result, spline_coefficients=np.ones(8))
    if change == "initial":
        result = replace(result, initial_spline=None)
    if change == "oldhash":
        expansion = replace(expansion, original_coefficient_sha256="sha256:" + "b" * 64)
    if change == "operation":
        request["options"]["operation"] = "fit"
    if change == "missing":
        expansion = None
    with pytest.raises(ValueError, match="expansion|seed"):
        build_native_fit_payload(
            request, source, result, dense, stamp, 0.0, coordinate_expansion=expansion
        )


def test_public_seed_schema_add_fit_recall(native_fit_case):
    library, path, source = native_fit_case
    request, source, result, dense, stamp, expansion = specimen(source)
    record = build_native_fit_payload(
        request, source, result, dense, stamp, 0.0, coordinate_expansion=expansion
    )
    path.write_text(json.dumps(record))
    saved = library.add_fit("expanded-seed", "practice", path)
    recalled = library.load_fit(saved.dataset_id)
    assert recalled["model_hash"] == source["model_hash"]
    assert recalled["evidence"]["original_fit"]["free_coordinates"] == list(
        expansion.expanded_start.free_coordinates
    )
    assert recalled["physical_time_qualified"] is False
    assert recalled["dynamics_replayed"] is False


def test_public_builder_facade_import_does_not_import_native_sdk():
    import subprocess
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parents[3]
    script = """import sys
from pathlib import Path
for path in (Path.cwd()/'src',Path.cwd()/'src/shared/python',Path.cwd()/'vendor/ud-tools/src'):
    sys.path.insert(0,str(path))
import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'mujoco','pydrake','pinocchio','opensim'}:
        raise RuntimeError('native SDK import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
from src.shared.python.workspace import build_native_fit_payload
assert callable(build_native_fit_payload)
"""
    process = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr


@pytest.mark.parametrize(
    "change",
    ["locked_reference", "dense_q", "indices", "frames", "training_q", "training_time"],
)
def test_coordinate_seed_rejects_changed_parent_motion_or_dense_evidence(change):
    request, source, result, dense, stamp, expansion = specimen()
    if change == "locked_reference":
        source["q"][1][-1] = 0.1
    if change == "dense_q":
        dense = (dense[0], dense[1], dense[2] + 0.1)
    if change == "indices":
        dense = ((1, 2), dense[1], dense[2])
    if change == "frames":
        dense = (dense[0], [{"different": True}, dense[1][1]], dense[2])
    if change == "training_q":
        source["evidence"]["original_fit"]["q"] = [[0.1, 0.1], [0.1, 0.1]]
    if change == "training_time":
        source["evidence"]["original_fit"]["source_times"] = [0.0, 0.3]
    with pytest.raises(ValueError, match="expansion|seed"):
        build_native_fit_payload(
            request, source, result, dense, stamp, 0.0, coordinate_expansion=expansion
        )


def test_coordinate_seed_rejects_changed_requested_training_frame_identity():
    request, source, result, dense, stamp, expansion = specimen()
    request["options"]["frame_indices"] = [1, 2]
    with pytest.raises(ValueError, match="expansion|seed"):
        build_native_fit_payload(
            request, source, result, dense, stamp, 0.0, coordinate_expansion=expansion
        )


@pytest.mark.parametrize("native_first", [False, True])
def test_historical_fit_first_clean_import_order(native_first):
    import importlib.util
    import subprocess
    import sys
    from pathlib import Path

    if native_first and importlib.util.find_spec("mujoco") is None:
        pytest.skip("Native SDK unavailable for actual driver import order")
    root = Path(__file__).resolve().parents[3]
    script = """import sys
from pathlib import Path
for path in (Path.cwd()/'src',Path.cwd()/'src/shared/python',Path.cwd()/'vendor/ud-tools/src'):
    sys.path.insert(0,str(path))
"""
    if native_first:
        script += "import mujoco\n"
    else:
        script += """import builtins
original=builtins.__import__
def guarded(name,*args,**kwargs):
    if name.split('.')[0] in {'mujoco','pydrake','pinocchio','opensim'}:
        raise RuntimeError('native SDK import: '+name)
    return original(name,*args,**kwargs)
builtins.__import__=guarded
"""
    script += """from src.shared.python.motion_matching.historical_fit import (
    ImageFitConfig, SplineCoordinateExpansion, initialize_image_trajectory)
from src.shared.python.workspace import build_native_fit_payload
assert callable(build_native_fit_payload)
assert callable(initialize_image_trajectory)
"""
    process = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        capture_output=True,
        text=True,
        check=False,
    )
    assert process.returncode == 0, process.stderr
