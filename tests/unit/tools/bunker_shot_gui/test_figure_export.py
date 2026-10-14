"""Stamped figure exports and the reader that refuses corrupt ones (#9545).

Acceptance item: "Synchronized animation, field slices and charts retain
tier/verdict/source stamps in screenshots and exports; corruption is
rejected." The in-frame stamp already survives a screenshot; these tests pin
the export half -- a PNG that carries the tier, verdict and sources as data,
bound by a digest to its own pixels, and a reader that refuses a tampered,
unstamped or foreign file instead of trusting it.

Headless: matplotlib's Agg canvas and Pillow only, real F0 evaluations from
the shared ``conftest`` fixtures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matplotlib.figure import Figure
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from bunkershot3d.solvers import EnvelopeStatus, FidelityTier
from src.tools.bunker_shot_gui.export import (
    ANIMATION_SCHEMA_VERSION,
    EXPORT_METADATA_KEY,
    EXPORT_SCHEMA_VERSION,
    ExportIntegrityError,
    ExportStamp,
    evaluation_stamp,
    export_figure,
    export_vtk_stamped_animation,
    export_vtk_stamped_frame,
    header_digest,
    load_workbench_evaluation,
    read_animation_export,
    read_export,
    save_workbench_evaluation,
)
from src.tools.bunker_shot_gui.render import sole_load_still, stamp_axes
from src.tools.bunker_shot_gui.report import (
    MODEL_SOURCE,
    evaluation_sources,
    validity_stamp,
)

pytestmark = pytest.mark.unit

_DIGEST = "a" * 64


def _stamped_figure(status: EnvelopeStatus, tier: FidelityTier) -> Figure:
    """A small figure carrying the in-frame stamp for ``status``/``tier``."""
    figure = Figure(figsize=(2.0, 1.5), dpi=50)
    axes = figure.add_subplot()
    axes.plot([0.0, 1.0], [0.0, 1.0])
    stamp_axes(axes, status, tier)
    return figure


def _stamp(**overrides: Any) -> ExportStamp:
    values: dict[str, Any] = {
        "view": "chart",
        "status": EnvelopeStatus.BEYOND_VALIDATION,
        "tier": FidelityTier.F0,
        "sources": {"model": MODEL_SOURCE},
    }
    values.update(overrides)
    return ExportStamp(**values)


def _rewrite(path: Path, pixels: np.ndarray, text: str | None) -> None:
    """Write ``pixels`` back to ``path`` with ``text`` as the export chunk."""
    info = PngInfo()
    if text is not None:
        info.add_itxt(EXPORT_METADATA_KEY, text)
    Image.fromarray(pixels, mode="RGBA").save(path, format="PNG", pnginfo=info)


def _load(path: Path) -> tuple[np.ndarray, dict[str, Any]]:
    with Image.open(path) as image:
        image.load()
        header = json.loads(image.text[EXPORT_METADATA_KEY])
        return np.array(image), header


@pytest.fixture
def chart_export(tmp_path: Path) -> Path:
    stamp = _stamp()
    path = tmp_path / "chart.png"
    export_figure(_stamped_figure(stamp.status, stamp.tier), path, stamp)
    return path


# ------------------------------------------------------------------ round trip


def test_a_stamped_evaluation_still_round_trips(
    nominal_evaluation, tmp_path: Path
) -> None:  # type: ignore[no-untyped-def]
    stamp = evaluation_stamp(
        nominal_evaluation, view="sole_load", record_digest=_DIGEST
    )
    figure = sole_load_still(nominal_evaluation.shot.sole_field)
    written = export_figure(figure, tmp_path / "sole.png", stamp)

    reloaded = read_export(tmp_path / "sole.png")

    assert reloaded.stamp == stamp
    assert reloaded.digest == written.digest
    assert reloaded.stamp.status is nominal_evaluation.shot.status
    assert reloaded.stamp.tier is nominal_evaluation.shot.fidelity_tier
    assert reloaded.stamp.record_digest == _DIGEST
    assert reloaded.shape == written.shape
    assert reloaded.shape[2] == 4


def test_the_export_carries_the_in_frame_words_as_data(chart_export: Path) -> None:
    _, header = _load(chart_export)

    assert header["schema_version"] == EXPORT_SCHEMA_VERSION
    stamp = header["stamp"]
    assert stamp["text"] == validity_stamp(
        EnvelopeStatus.BEYOND_VALIDATION, FidelityTier.F0
    )
    assert stamp["status"] == "beyond_validation"
    assert stamp["tier"] == "F0"
    assert stamp["sources"] == {"model": MODEL_SOURCE}
    assert header["predictive"] is False


def test_evaluation_sources_are_the_api_record_sources(
    nominal_evaluation, tour_swing
) -> None:  # type: ignore[no-untyped-def]
    from src.api.routes.bunker_workbench import evaluation_record

    record = evaluation_record(
        nominal_evaluation, tour_swing, handedness="right", objective=None
    )
    assert record["sources"] == evaluation_sources(nominal_evaluation)
    assert evaluation_stamp(nominal_evaluation, view="x").sources == record["sources"]


def test_the_write_leaves_no_staging_file(chart_export: Path) -> None:
    assert sorted(p.name for p in chart_export.parent.iterdir()) == ["chart.png"]


# ---------------------------------------------------------------- preconditions


def test_a_figure_without_its_stamp_is_not_exported(tmp_path: Path) -> None:
    figure = Figure(figsize=(2.0, 1.5), dpi=50)
    figure.add_subplot().plot([0.0, 1.0], [1.0, 0.0])

    with pytest.raises(ValueError, match="in-frame stamp"):
        export_figure(figure, tmp_path / "bare.png", _stamp())
    assert not (tmp_path / "bare.png").exists()


def test_a_figure_stamped_with_another_verdict_is_not_exported(
    tmp_path: Path,
) -> None:
    figure = _stamped_figure(EnvelopeStatus.EXTRAPOLATED, FidelityTier.F0)

    with pytest.raises(ValueError, match="in-frame stamp"):
        export_figure(figure, tmp_path / "wrong.png", _stamp())


def test_only_png_is_written(tmp_path: Path) -> None:
    stamp = _stamp()
    with pytest.raises(ValueError, match=r"\.png"):
        export_figure(
            _stamped_figure(stamp.status, stamp.tier), tmp_path / "a.jpg", stamp
        )


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"view": ""}, "view"),
        ({"sources": {}}, "sources"),
        ({"sources": {"sand": "x"}}, "model"),
        ({"record_digest": "ABC"}, "record_digest"),
        ({"status": "beyond_validation"}, "EnvelopeStatus"),
        ({"tier": "F0"}, "FidelityTier"),
    ],
)
def test_a_malformed_stamp_is_refused(overrides: dict[str, Any], message: str) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        _stamp(**overrides)


# ------------------------------------------------------------------- rejection


def test_an_unstamped_png_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "plain.png"
    _stamped_figure(EnvelopeStatus.WITHIN, FidelityTier.F0).savefig(path)

    with pytest.raises(ExportIntegrityError, match="unstamped"):
        read_export(path)


def test_edited_pixels_are_refused(chart_export: Path) -> None:
    pixels, header = _load(chart_export)
    pixels[0, 0] = 255 - pixels[0, 0]
    _rewrite(chart_export, pixels, json.dumps(header))

    with pytest.raises(ExportIntegrityError, match="pixels"):
        read_export(chart_export)


def test_a_promoted_verdict_is_refused(chart_export: Path) -> None:
    pixels, header = _load(chart_export)
    header["stamp"]["status"] = "within"
    _rewrite(chart_export, pixels, json.dumps(header))

    with pytest.raises(ExportIntegrityError, match="digest"):
        read_export(chart_export)


def test_a_re_signed_stamp_that_contradicts_its_words_is_refused(
    chart_export: Path,
) -> None:
    from src.tools.bunker_shot_gui.export import header_digest

    pixels, header = _load(chart_export)
    header["stamp"]["status"] = "within"
    header["digest"] = header_digest(header)
    _rewrite(chart_export, pixels, json.dumps(header))

    with pytest.raises(ExportIntegrityError, match="stamp text"):
        read_export(chart_export)


def test_a_predictive_claim_is_refused_even_when_re_signed(
    chart_export: Path,
) -> None:
    from src.tools.bunker_shot_gui.export import header_digest

    pixels, header = _load(chart_export)
    header["predictive"] = True
    header["digest"] = header_digest(header)
    _rewrite(chart_export, pixels, json.dumps(header))

    with pytest.raises(ExportIntegrityError, match="predictive"):
        read_export(chart_export)


def test_an_unknown_schema_is_refused(chart_export: Path) -> None:
    pixels, header = _load(chart_export)
    header["schema_version"] = "bunker-workbench-figure/99"
    _rewrite(chart_export, pixels, json.dumps(header))

    with pytest.raises(ExportIntegrityError, match="schema"):
        read_export(chart_export)


@pytest.mark.parametrize("text", ["not json", "[1, 2]", '{"schema_version": 1}'])
def test_a_garbled_stamp_chunk_is_refused(chart_export: Path, text: str) -> None:
    pixels, _ = _load(chart_export)
    _rewrite(chart_export, pixels, text)

    with pytest.raises(ExportIntegrityError):
        read_export(chart_export)


def test_a_file_that_is_not_a_png_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "junk.png"
    path.write_bytes(b"\x89PNG not really")

    with pytest.raises(ExportIntegrityError, match="readable PNG"):
        read_export(path)


def test_a_missing_file_is_a_plain_file_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_export(tmp_path / "absent.png")


# ---------------------------------------------------------------- save / load


def test_save_and_load_workbench_evaluation_round_trips(
    nominal_evaluation, tour_swing, tmp_path: Path
) -> None:
    path = tmp_path / "nominal_eval.json"
    saved_path = save_workbench_evaluation(
        nominal_evaluation, path, swing=tour_swing, handedness="right"
    )
    assert saved_path == path
    assert saved_path.is_file()

    reloaded = load_workbench_evaluation(saved_path)
    assert reloaded["schema_version"] == "bunker-workbench-evaluation/1"
    assert reloaded["predictive"] is False
    assert reloaded["inputs"]["design"]["name"] == "nominal"
    assert reloaded["inputs"]["swing"]["clubhead_speed_mps"] == 25.0
    assert reloaded["shot"]["peak_force_n"] == pytest.approx(
        nominal_evaluation.shot.peak_force_n
    )


def test_save_workbench_evaluation_requires_json_suffix(
    nominal_evaluation, tmp_path: Path
) -> None:
    with pytest.raises(ValueError, match=r"\.json"):
        save_workbench_evaluation(nominal_evaluation, tmp_path / "bad.txt")


def test_load_workbench_evaluation_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_workbench_evaluation(tmp_path / "missing.json")


def test_load_workbench_evaluation_rejects_corrupted_json(tmp_path: Path) -> None:
    bad = tmp_path / "corrupt.json"
    bad.write_text("not json", encoding="utf-8")
    with pytest.raises(ExportIntegrityError, match="readable JSON"):
        load_workbench_evaluation(bad)


def test_load_workbench_evaluation_rejects_non_dict_json(tmp_path: Path) -> None:
    bad = tmp_path / "list.json"
    bad.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ExportIntegrityError, match="JSON object"):
        load_workbench_evaluation(bad)


def test_load_workbench_evaluation_rejects_unknown_schema(
    nominal_evaluation, tour_swing, tmp_path: Path
) -> None:
    path = tmp_path / "schema.json"
    save_workbench_evaluation(
        nominal_evaluation, path, swing=tour_swing, handedness="right"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["schema_version"] = "bunker-workbench-evaluation/999"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ExportIntegrityError, match="schema"):
        load_workbench_evaluation(path)


def test_load_workbench_evaluation_rejects_tampered_payload(
    nominal_evaluation, tour_swing, tmp_path: Path
) -> None:
    path = tmp_path / "tampered.json"
    save_workbench_evaluation(
        nominal_evaluation, path, swing=tour_swing, handedness="right"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["shot"]["peak_force_n"] = 99999.0
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ExportIntegrityError, match="digest"):
        load_workbench_evaluation(path)


def test_load_workbench_evaluation_rejects_predictive_claim(
    nominal_evaluation, tour_swing, tmp_path: Path
) -> None:
    import hashlib

    from src.tools.bunker_shot_gui.export import canonical_json

    path = tmp_path / "predictive.json"
    save_workbench_evaluation(
        nominal_evaluation, path, swing=tour_swing, handedness="right"
    )
    data = json.loads(path.read_text(encoding="utf-8"))
    data["predictive"] = True
    body = {k: v for k, v in data.items() if k != "digest"}
    data["digest"] = hashlib.sha256(canonical_json(body).encode("utf-8")).hexdigest()
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ExportIntegrityError, match="predictive"):
        load_workbench_evaluation(path)


# ------------------------------------------------------------- VTK export


def test_export_vtk_stamped_frame_round_trips(tmp_path: Path) -> None:
    stamp = _stamp(view="vtk_frame")
    pixels = np.zeros((64, 64, 4), dtype=np.uint8)
    pixels[:, :, 0] = 120
    pixels[:, :, 3] = 255
    path = tmp_path / "vtk_frame.png"
    exported = export_vtk_stamped_frame(pixels, path, stamp)
    assert exported.path == path
    assert exported.shape == (64, 64, 4)

    reloaded = read_export(path)
    assert reloaded.stamp == stamp
    assert reloaded.digest == exported.digest


def test_export_vtk_stamped_frame_requires_png(tmp_path: Path) -> None:
    stamp = _stamp()
    pixels = np.zeros((10, 10, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match=r"\.png"):
        export_vtk_stamped_frame(pixels, tmp_path / "frame.jpg", stamp)


def test_export_vtk_stamped_frame_rejects_unsupported_type(tmp_path: Path) -> None:
    stamp = _stamp()
    with pytest.raises(TypeError, match="artists must be"):
        export_vtk_stamped_frame("invalid", tmp_path / "f.png", stamp)


def test_export_vtk_stamped_animation_round_trips(tmp_path: Path) -> None:
    stamp = _stamp(view="vtk_animation")
    frames = [
        np.full((32, 32, 4), fill_value=val, dtype=np.uint8) for val in (50, 100, 150)
    ]
    path = tmp_path / "anim.gif"
    exported = export_vtk_stamped_animation(frames, path, stamp, fps=10)
    assert exported.path == path
    assert exported.frame_count == 3
    assert exported.duration_s > 0

    reloaded = read_animation_export(path)
    assert reloaded.stamp == stamp
    assert reloaded.digest == exported.digest
    assert reloaded.frame_count == 3
    assert reloaded.duration_s == pytest.approx(exported.duration_s)


def test_export_vtk_stamped_animation_requires_gif(tmp_path: Path) -> None:
    stamp = _stamp()
    frames = [np.zeros((10, 10, 4), dtype=np.uint8)]
    with pytest.raises(ValueError, match=r"\.gif"):
        export_vtk_stamped_animation(frames, tmp_path / "anim.mp4", stamp)


def test_export_vtk_stamped_animation_requires_frames(tmp_path: Path) -> None:
    stamp = _stamp()
    with pytest.raises(ValueError, match="no frames"):
        export_vtk_stamped_animation([], tmp_path / "anim.gif", stamp)


def test_read_animation_export_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_animation_export(tmp_path / "missing.gif")


def test_read_animation_export_rejects_unstamped_gif(tmp_path: Path) -> None:
    path = tmp_path / "unstamped.gif"
    img = Image.new("RGBA", (16, 16), color=(255, 0, 0, 255))
    img.save(path, format="GIF")
    with pytest.raises(ExportIntegrityError, match="valid stamped GIF"):
        read_animation_export(path)


def test_read_animation_export_rejects_tampered_digest(tmp_path: Path) -> None:
    stamp = _stamp(view="vtk_animation")
    frames = [np.zeros((16, 16, 4), dtype=np.uint8)]
    path = tmp_path / "tampered.gif"
    export_vtk_stamped_animation(frames, path, stamp)

    # Read GIF comment, tamper with fps, re-save
    with Image.open(path) as img:
        comment = img.info["comment"]
        data = json.loads(comment)
        data["fps"] = 999  # breaks digest
        img.save(path, format="GIF", comment=json.dumps(data).encode("utf-8"))

    with pytest.raises(ExportIntegrityError, match="changed after writing"):
        read_animation_export(path)


def test_read_animation_export_rejects_predictive_claim(tmp_path: Path) -> None:
    stamp = _stamp(view="vtk_animation")
    frames = [np.zeros((16, 16, 4), dtype=np.uint8)]
    path = tmp_path / "predictive.gif"
    export_vtk_stamped_animation(frames, path, stamp)

    with Image.open(path) as img:
        comment = img.info["comment"]
        data = json.loads(comment)
        data["predictive"] = True
        data["digest"] = header_digest(data)
        img.save(path, format="GIF", comment=json.dumps(data).encode("utf-8"))

    with pytest.raises(ExportIntegrityError, match="predictive"):
        read_animation_export(path)
