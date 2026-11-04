"""The MuJoCo run path writes a run manifest next to its outputs (Issue #14432 / Epic #14436)."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

import pytest

mujoco = pytest.importorskip("mujoco")

from src.sim_run.manifest import RunManifestError, validate_run_manifest
from src.sim_run.manifest_emitter import MANIFEST_FILENAME, emit_run_manifest
from src.sim_run.mujoco_run import TRAJECTORY_FILENAME, run_mujoco

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

_TINY_XML = """<mujoco model="tiny_pendulum">
  <option timestep="0.002"/>
  <worldbody>
    <body name="link" pos="0 0 1">
      <joint name="hinge" type="hinge" axis="0 1 0"/>
      <geom type="capsule" fromto="0 0 0 0 0 -0.5" size="0.03" mass="1"/>
    </body>
  </worldbody>
  <actuator><motor joint="hinge" ctrlrange="-1 1"/></actuator>
</mujoco>"""
_SCHEMA = Path("schemas") / "run-manifest-v1.json"
_COMMIT = "a" * 40
_CREATED = datetime(2026, 11, 4, 3, 0, tzinfo=timezone.utc)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _run(out: Path, **overrides: object):
    kwargs: dict = {
        "model_id": "test/tiny_pendulum",
        "output_dir": out,
        "duration": 0.05,
        "seed": 7,
        "qvel_noise": 0.1,
        "ud_commit": _COMMIT,
        "created_at": _CREATED,
    }
    kwargs.update(overrides)
    return run_mujoco(_TINY_XML, **kwargs)


def test_run_writes_a_valid_manifest_next_to_its_outputs(tmp_path: Path) -> None:
    manifest = _run(tmp_path)
    written = json.loads((tmp_path / MANIFEST_FILENAME).read_text(encoding="utf-8"))
    assert validate_run_manifest(written) == manifest
    assert manifest.engine.name == "mujoco"
    assert manifest.engine.version == mujoco.__version__
    expected = hashlib.sha256(_TINY_XML.encode("utf-8")).hexdigest()
    assert manifest.model.revision == f"sha256:{expected}"
    assert manifest.parameters["timestep_s"] == pytest.approx(0.002)
    assert manifest.seed == 7


def test_manifest_records_the_output_artifact_hashes(tmp_path: Path) -> None:
    manifest = _run(tmp_path)
    assert set(manifest.outputs) == {TRAJECTORY_FILENAME}
    assert manifest.outputs[TRAJECTORY_FILENAME] == _sha256(
        tmp_path / TRAJECTORY_FILENAME
    )


def test_same_seed_reproduces_and_another_seed_differs(tmp_path: Path) -> None:
    first = _run(tmp_path / "a").outputs[TRAJECTORY_FILENAME]
    again = _run(tmp_path / "b").outputs[TRAJECTORY_FILENAME]
    other = _run(tmp_path / "c", seed=8).outputs[TRAJECTORY_FILENAME]
    assert first == again
    assert first != other


def test_input_digest_identifies_the_initial_condition(tmp_path: Path) -> None:
    seeded = _run(tmp_path / "a").input_data.sha256
    assert _run(tmp_path / "b").input_data.sha256 == seeded
    assert _run(tmp_path / "c", seed=8).input_data.sha256 != seeded


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"duration": 0.0}, "duration"),
        ({"qvel_noise": -1.0}, "qvel_noise"),
        ({"seed": -1}, "seed"),
    ],
)
def test_run_rejects_bad_preconditions(
    tmp_path: Path, overrides: dict, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        _run(tmp_path, **overrides)


def test_emitter_rejects_outputs_outside_the_run_dir(tmp_path: Path) -> None:
    (tmp_path / "run").mkdir()
    outside = tmp_path / "elsewhere.npz"
    outside.write_bytes(b"x")
    with pytest.raises(ValueError, match="inside"):
        emit_run_manifest(
            tmp_path / "run",
            outputs=[outside],
            model_id="test/m",
            revision="sha256:" + "0" * 64,
            parameters={},
            input_sha256="0" * 64,
            seed=0,
            engine_name="mujoco",
            engine_version="0",
            ud_commit=_COMMIT,
            created_at=_CREATED,
        )


def test_emitter_rejects_a_missing_output(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        emit_run_manifest(
            tmp_path,
            outputs=[tmp_path / "missing.npz"],
            model_id="test/m",
            revision="sha256:" + "0" * 64,
            parameters={},
            input_sha256="0" * 64,
            seed=0,
            engine_name="mujoco",
            engine_version="0",
            ud_commit=_COMMIT,
            created_at=_CREATED,
        )


@pytest.mark.parametrize(
    "outputs",
    [
        {"/abs/trajectory.npz": "0" * 64},
        {"../trajectory.npz": "0" * 64},
        {"trajectory.npz": "not-a-digest"},
    ],
)
def test_manifest_outputs_must_be_relative_names_with_digests(
    outputs: dict[str, str],
) -> None:
    fixture = Path(__file__).resolve().parents[2] / "fixtures" / "sim_run"
    data = json.loads((fixture / "run_manifest_valid_v1.json").read_text("utf-8"))
    data["outputs"] = outputs
    with pytest.raises(RunManifestError, match="outputs"):
        validate_run_manifest(data)
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads((fixture.parents[2] / _SCHEMA).read_text("utf-8"))
    with pytest.raises(jsonschema.ValidationError):
        jsonschema.validate(data, schema)
