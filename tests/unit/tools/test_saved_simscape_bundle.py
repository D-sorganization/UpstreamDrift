"""Real saved MATLAB import contract; no MATLAB runtime or simulation needed."""

from __future__ import annotations
import hashlib
import json
import shutil
from pathlib import Path
import numpy as np
import pytest
from scipy.io import loadmat, savemat
from src.shared.python.simulation_store.replay_bundle import load_simscape_bundle

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "docs/development/simscape_tour_matching/native_evidence"


@pytest.fixture
def bundle(tmp_path: Path) -> Path:
    run = EVIDENCE / "two_window_fit_9967_102"
    sources = {
        "model": EVIDENCE / "native_geometry_spec_9967.json",
        "candidate": run / "returned-candidate.json",
        "trajectory": run / "qualified_candidate_replay.mat",
        "target": run / "returned-replay.npz",
        "report": run / "qualified_candidate_replay.json",
    }
    artifacts = {}
    for key, source in sources.items():
        dest = tmp_path / source.name
        shutil.copyfile(source, dest)
        artifacts[key] = {
            "path": dest.name,
            "sha256": hashlib.sha256(dest.read_bytes()).hexdigest(),
        }
    manifest = {
        "schema_version": "upstreamdrift/saved-simscape-replay/1",
        "run_id": "simscape-returned102",
        "engine": "simscape",
        "status": "rejected",
        "duration_s": 0.85,
        "artifacts": artifacts,
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(manifest))
    return path


def change(path: Path, key: str, value: object) -> None:
    doc = json.loads(path.read_text())
    doc[key] = value
    path.write_text(json.dumps(doc))


def test_import_actual_matlab_state_and_read_only_channels(bundle: Path) -> None:
    result = load_simscape_bundle(bundle)
    raw = loadmat(bundle.parent / "qualified_candidate_replay.mat")
    np.testing.assert_array_equal(result.arrays["q"], raw["q"])
    np.testing.assert_array_equal(result.arrays["markers_m"], raw["prediction"])
    assert result.arrays["q"].shape == (307, 27)
    assert result.status == "rejected"
    assert not result.arrays["tau_valid"].any()
    assert np.isnan(result.arrays["tau"]).all()
    assert not result.arrays["tau_valid"].flags.writeable
    assert not result.arrays["q"].flags.writeable
    assert result.coordinate_names[0] == result.candidate["coordinate_names"][0]
    with pytest.raises(TypeError):
        result.arrays["q"] = np.zeros((1, 1))


@pytest.mark.parametrize(
    "key,value",
    [
        ("schema_version", "unknown"),
        ("engine", "mujoco"),
        ("status", "accepted"),
        ("duration_s", 1.8),
    ],
)
def test_reject_mislabelled_bundle(bundle: Path, key: str, value: object) -> None:
    change(bundle, key, value)
    with pytest.raises(ValueError):
        load_simscape_bundle(bundle)


def test_reject_changed_artifact_before_parsing(bundle: Path) -> None:
    with (bundle.parent / "qualified_candidate_replay.mat").open("ab") as stream:
        stream.write(b"changed")
    with pytest.raises(ValueError, match="hash"):
        load_simscape_bundle(bundle)


def test_reject_path_escape(bundle: Path) -> None:
    doc = json.loads(bundle.read_text())
    doc["artifacts"]["model"]["path"] = "../outside.json"
    bundle.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="relative|escape"):
        load_simscape_bundle(bundle)


def test_reject_nonfinite_states_even_with_matching_hash(bundle: Path) -> None:
    path = bundle.parent / "qualified_candidate_replay.mat"
    raw = loadmat(path)
    raw["q"][3, 2] = np.nan
    savemat(
        path,
        {k: v for k, v in raw.items() if not k.startswith("__")},
        long_field_names=True,
    )
    doc = json.loads(bundle.read_text())
    doc["artifacts"]["trajectory"]["sha256"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    bundle.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="finite"):
        load_simscape_bundle(bundle)
