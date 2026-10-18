"""CLI appearance flags travel to the worker job and the MuJoCo scene (HRG-B2, #13167)."""

from __future__ import annotations

import json
from pathlib import Path
import re

import numpy as np
import pytest

from src.shared.python.model_appearance import library
from src.shared.python.model_appearance.appearance_selection import hex_to_rgb
from src.shared.python.model_appearance.defaults import default_document_dict
from src.shared.python.model_appearance.schema import document_from_dict
from src.tools.native_viewer_export import cli
from src.tools.native_viewer_export.backends._subprocess import stage_job
from src.tools.native_viewer_export.backends._worker_job import WorkerJob
from src.tools.native_viewer_export.backends.mujoco_worker import build_scene_xml
from src.tools.native_viewer_export.core import ExportSettings, SwingInput
from src.shared.python.motion_matching.same_input import InputBundle

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
BASE = ["--bundle", "b.npz", "--out", "o", "--swing", "s"]


def _settings(*extra: str) -> ExportSettings:
    return cli.build_settings(cli.build_parser().parse_args([*BASE, *extra]))


def test_no_flags_leaves_the_contract_default() -> None:
    assert _settings().appearance is None


def test_flags_resolve_into_settings() -> None:
    s = _settings(
        "--skin-tone",
        "skin_6",
        "--hair",
        "#336699",
        "--clothing",
        "golf_polo_shorts",
        "--top-color",
        "#aa0000",
        "--bottom-color",
        "#00aa00",
    )
    doc = document_from_dict(s.appearance)
    assert doc.skin_tone == "skin_6"
    assert doc.clothing == "golf_polo_shorts"
    mats = library.library_materials(doc)
    assert mats[doc.head.headwear_material].rgba[:3] == pytest.approx(
        hex_to_rgb("#336699")
    )
    assert mats["polo_navy"].rgba[:3] == pytest.approx(hex_to_rgb("#aa0000"))


def test_appearance_file_is_the_base_and_flags_win(tmp_path: Path) -> None:
    path = tmp_path / "x.appearance.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": "appearance-v1",
                "skin_tone": "skin_5",
                "clothing": "golf_polo_shorts",
            }
        ),
        encoding="utf-8",
    )
    s = _settings("--appearance", str(path), "--skin-tone", "skin_7")
    assert s.appearance["skin_tone"] == "skin_7"
    assert s.appearance["clothing"] == "golf_polo_shorts"


def test_bad_inputs_are_value_errors(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        _settings("--hair", "hair_teal")
    with pytest.raises(ValueError, match="hex"):
        _settings("--top-color", "red")
    with pytest.raises(ValueError):
        _settings("--appearance", str(tmp_path / "missing.json"))


def _swing() -> SwingInput:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    order = tuple(spec["coordinate_order"])
    q = np.zeros((3, len(order)))
    bundle = InputBundle(
        spec_bytes=SPEC.read_bytes(),
        coordinate_order=order,
        dt_s=0.001,
        q0=q[0],
        v0=q[0],
        efforts=np.zeros((2, len(order))),
        reference_q=q,
        reference_v=q.copy(),
        reference_engine="mujoco",
        provenance={},
    )
    return SwingInput(bundle, q, "s", "Driver", "mujoco")


def test_worker_job_carries_the_appearance(tmp_path: Path) -> None:
    settings = _settings("--hair", "#336699")
    job = stage_job(_swing(), settings, [0], None, tmp_path)
    assert job.appearance == settings.appearance
    job.dump(tmp_path / "job.json")
    assert WorkerJob.load(tmp_path / "job.json").appearance == settings.appearance
    (tmp_path / "p").mkdir()
    assert (
        stage_job(_swing(), _settings(), [0], None, tmp_path / "p").appearance is None
    )


def _material_rgba(xml: str, name: str) -> tuple[float, ...]:
    match = re.search(rf'<material name="{name}" rgba="([^"]+)"', xml)
    assert match is not None, name
    return tuple(float(v) for v in match.group(1).split())


def test_mujoco_worker_scene_uses_the_job_colours() -> None:
    spec = SPEC.read_bytes()
    custom = _settings("--hair", "#ff0000").appearance
    default_xml, _ = build_scene_xml(spec, None)
    custom_xml, _ = build_scene_xml(spec, custom)
    assert _material_rgba(custom_xml, "mat_custom_hair") == (1.0, 0.0, 0.0, 1.0)
    assert "mat_custom_hair" not in default_xml
    # passing the contract document explicitly is the same scene as passing nothing
    explicit, _ = build_scene_xml(spec, default_document_dict())
    assert explicit == default_xml
