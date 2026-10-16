"""Offscreen contract tests for the PyQt6 Model Explorer, one run per engine (#12878).

For every engine adapter the explorer must be able to

1. **open** the canonical model (the anthro driver specification),
2. **explore** it: expose the body tree, the joints and select an object,
3. **load results** (a committed matched-swing run for that engine),
4. **scrub** the playback, and
5. **export an MP4** through the app's ``Export video...`` action.

Engines whose native package is not installed skip the native-load test with
an explicit reason; the explorer-level stages are engine-package-free (they
consume the engine's exported artifact: MJCF, URDF or OSIM) and always run.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable, Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

pytest.importorskip("PyQt6", reason="Model Explorer contract needs PyQt6")

from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox  # noqa: E402

from tests.support.real_engines import real_module  # noqa: E402
from src.tools.matched_swing_browser.model import (  # noqa: E402
    MatchedSwingBrowserModel,
)
from src.tools.model_explorer.gui import MainWidget  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = (
    REPO_ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
ENGINES = ("mujoco", "drake", "pinocchio", "opensim", "myosuite", "simscape")
# The canonical full-body model has 41 scalar joints plus the 3 club DOF.
CANONICAL_COORDINATES = 44
MIN_BODIES = 24


@dataclass(frozen=True)
class EngineCase:
    """How the explorer reaches one engine's canonical model."""

    name: str
    suffix: str | None
    export: Callable[[bytes], str] | None
    native_package: str | None
    skip_reason: str | None = None


def _export_mjcf(spec_bytes: bytes) -> str:
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

    return export_full_body_mjcf(spec_bytes, visual=True)[0]


def _export_urdf(spec_bytes: bytes) -> str:
    from src.engines.physics_engines.drake.python.full_body_urdf import (
        export_full_body_urdf,
    )

    return export_full_body_urdf(spec_bytes)[0]


def _export_osim(spec_bytes: bytes) -> str:
    from src.engines.physics_engines.opensim.python.full_body_osim import (
        export_full_body_osim,
    )

    return export_full_body_osim(spec_bytes, club_geometry_ref=None)[0]


CASES: dict[str, EngineCase] = {
    "mujoco": EngineCase("mujoco", ".xml", _export_mjcf, "mujoco"),
    "drake": EngineCase("drake", ".urdf", _export_urdf, "pydrake"),
    "pinocchio": EngineCase("pinocchio", ".urdf", _export_urdf, "pinocchio"),
    "opensim": EngineCase("opensim", ".osim", _export_osim, "opensim"),
    "myosuite": EngineCase(
        "myosuite",
        None,
        None,
        "myosuite",
        skip_reason="MyoSuite has no anthro-driver model exporter; it replays "
        "results only (kinematic-only engine)",
    ),
    "simscape": EngineCase(
        "simscape",
        None,
        None,
        None,
        skip_reason="Simscape models are MATLAB-licensed and have no explorer "
        "artifact; the explorer loads its replay results only",
    ),
}


@pytest.fixture(scope="session")
def qapp() -> QApplication:
    inst = QApplication.instance()
    return inst if isinstance(inst, QApplication) else QApplication(["explorer"])


@pytest.fixture
def explorer(qapp: QApplication) -> Generator[MainWidget, None, None]:
    widget = MainWidget()
    yield widget
    widget.cleanup()
    widget.deleteLater()


@pytest.fixture(scope="module")
def spec_bytes() -> bytes:
    return SPEC_PATH.read_bytes()


@pytest.fixture
def warnings_box() -> Generator[list[str], None, None]:
    """Capture message boxes so a modal dialog can never block the suite."""
    shown: list[str] = []

    def _record(*args: Any, **_kwargs: Any) -> int:
        shown.append(" ".join(str(a) for a in args[1:]))
        return int(QMessageBox.StandardButton.Ok)

    with (
        patch.object(QMessageBox, "warning", _record),
        patch.object(QMessageBox, "critical", _record),
        patch.object(QMessageBox, "information", _record),
    ):
        yield shown


def _model_artifact(case: EngineCase, spec_bytes: bytes, tmp_path: Path) -> Path:
    if case.export is None or case.suffix is None:
        pytest.skip(case.skip_reason or f"{case.name}: no explorer model artifact")
    path = tmp_path / f"anthro_driver_{case.name}{case.suffix}"
    path.write_text(case.export(spec_bytes), encoding="utf-8")
    return path


def _has_marker_playback(npz_path: Path) -> bool:
    """Schema check independent of the implementation under test."""
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            keys = set(data.files)
            frames = len(data["time_s"]) if "time_s" in keys else 0
    except (OSError, ValueError):
        return False
    packaged = {"model_markers_m", "target_markers_m"} <= keys
    replay = {"markers_m", "target_m"} <= keys
    return frames >= 2 and (packaged or replay)


def _results_for(engine: str) -> Path:
    model = MatchedSwingBrowserModel()
    for row in model.load_ledger():
        if row.engine != engine:
            continue
        npz = model.resolve_artifact_path(row, "npz")
        if npz is not None and _has_marker_playback(npz):
            return npz
    pytest.fail(f"no committed matched-swing run with marker playback for {engine}")


# --- 1 + 2: open and explore -------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("engine", ENGINES)
def test_open_and_explore_canonical_model(
    engine: str,
    explorer: MainWidget,
    spec_bytes: bytes,
    tmp_path: Path,
    warnings_box: list[str],
) -> None:
    artifact = _model_artifact(CASES[engine], spec_bytes, tmp_path)

    assert explorer.load_model(artifact) is True, warnings_box
    assert warnings_box == []

    view = explorer.visualization_widget
    links, joints = view.get_link_names(), view.get_joint_names()
    assert len(links) >= MIN_BODIES, f"{engine}: blank body tree ({len(links)})"
    assert len(joints) >= CANONICAL_COORDINATES - 3, (
        f"{engine}: explorer shows {len(joints)} joints, "
        f"canonical model has {CANONICAL_COORDINATES} coordinates"
    )

    view.select_object(links[1])  # exploring must not raise
    assert explorer.current_file_path == artifact


# --- engine-native load (skips when the engine is not installed) ------------


@pytest.mark.integration
@pytest.mark.parametrize("engine", ENGINES)
def test_native_engine_loads_canonical_model(
    engine: str, spec_bytes: bytes, tmp_path: Path
) -> None:
    case = CASES[engine]
    if case.native_package is None or case.export is None:
        pytest.skip(case.skip_reason or f"{engine}: no native model contract")
    real_module(case.native_package)
    spec: dict[str, Any] = json.loads(spec_bytes)

    if engine == "mujoco":
        import mujoco

        assert mujoco.MjModel.from_xml_string(case.export(spec_bytes)).nq == (
            CANONICAL_COORDINATES
        )
    elif engine == "drake":
        from src.engines.physics_engines.drake.python.full_body_model import (
            FullBodyDrakeModel,
        )

        assert len(FullBodyDrakeModel(spec).names) == CANONICAL_COORDINATES
    elif engine == "pinocchio":
        from src.engines.physics_engines.pinocchio.python.native_model import (
            build_full_body_pinocchio_model,
        )

        assert build_full_body_pinocchio_model(spec) is not None
    elif engine == "opensim":
        import opensim

        path = _model_artifact(case, spec_bytes, tmp_path)
        model = opensim.Model(str(path))
        model.initSystem()
        assert model.getNumBodies() >= MIN_BODIES
    else:  # pragma: no cover - guarded by the skip above
        pytest.skip(f"{engine}: no native check")


# --- 3 + 4 + 5: results, scrub, export ---------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("engine", ENGINES)
def test_load_results_and_scrub(
    engine: str, explorer: MainWidget, warnings_box: list[str]
) -> None:
    npz = _results_for(engine)

    assert explorer.load_results(npz, engine) is True, warnings_box
    frames = explorer.results_frame_count
    assert frames >= 2

    for index in (0, frames // 2, frames - 1):
        explorer.scrub_results(index)
        assert explorer.current_result_frame == index

    with pytest.raises(ValueError, match="frame"):
        explorer.scrub_results(frames)


@pytest.mark.unit
@pytest.mark.parametrize("engine", ENGINES)
def test_export_mp4_through_the_app_action(
    engine: str, explorer: MainWidget, tmp_path: Path, warnings_box: list[str]
) -> None:
    import cv2

    assert explorer.load_results(_results_for(engine), engine) is True
    out = tmp_path / f"{engine}.mp4"

    with patch.object(QFileDialog, "getSaveFileName", return_value=(str(out), "MP4")):
        explorer.export_video_action.trigger()

    assert out.is_file() and out.stat().st_size > 0, warnings_box
    capture = cv2.VideoCapture(str(out))
    try:
        assert capture.isOpened()
        assert int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) >= 2
    finally:
        capture.release()


# --- design by contract: fail closed with an actionable message -------------


@pytest.mark.unit
def test_export_action_disabled_until_results_are_loaded(
    explorer: MainWidget,
) -> None:
    assert explorer.export_video_action.isEnabled() is False


@pytest.mark.unit
def test_loading_missing_results_is_actionable(
    explorer: MainWidget, tmp_path: Path, warnings_box: list[str]
) -> None:
    missing = tmp_path / "nope.npz"

    assert explorer.load_results(missing) is False
    assert any("nope.npz" in text for text in warnings_box)
    assert explorer.results_frame_count == 0
    assert explorer.export_video_action.isEnabled() is False


@pytest.mark.unit
def test_loading_results_without_markers_names_the_missing_arrays(
    explorer: MainWidget, tmp_path: Path, warnings_box: list[str]
) -> None:
    bad = tmp_path / "joints_only.npz"
    np.savez(bad, time_s=np.linspace(0, 1, 5), q=np.zeros((5, 3)))

    assert explorer.load_results(bad) is False
    assert any("marker" in text.lower() for text in warnings_box)


@pytest.mark.unit
def test_opening_empty_mjcf_is_rejected_not_blank(
    explorer: MainWidget, tmp_path: Path, warnings_box: list[str]
) -> None:
    empty = tmp_path / "empty.xml"
    empty.write_text("<mujoco model='x'><worldbody/></mujoco>", encoding="utf-8")

    assert explorer.load_model(empty) is False
    assert any("empty.xml" in text for text in warnings_box)


@pytest.mark.unit
def test_matched_browser_export_button_follows_playability(
    qapp: QApplication,
) -> None:
    """The results browser must not offer Export for runs it cannot render."""
    from src.tools.matched_swing_browser.gui import MatchedSwingBrowserWidget

    widget = MatchedSwingBrowserWidget()
    try:
        for engine in ("mujoco", "pinocchio"):
            widget.engine_combo.setCurrentIndex(widget.engine_combo.findText(engine))
            for row_index in range(widget.table.rowCount()):
                widget.table.selectRow(row_index)
                npz = widget._model.resolve_artifact_path(widget._selected_row, "npz")
                playable = npz is not None and _has_marker_playback(npz)
                assert widget.export_video_button.isEnabled() is playable, (
                    f"{engine} row {row_index}: export enabled="
                    f"{widget.export_video_button.isEnabled()} but playable={playable}"
                )
    finally:
        widget.cleanup()
