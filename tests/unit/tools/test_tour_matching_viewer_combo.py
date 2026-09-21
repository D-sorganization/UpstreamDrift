"""Unit tests for Tour Matching Viewer multi-candidate combo and GIF export (MV-03 #10479 / #10354).

Validates:
1. Multi-candidate combo replay holding up to four candidates on a shared timeline.
2. Per-engine marker RMS side-by-side computation.
3. Engine colors matching ENGINE_COLORS specification.
4. Exporting animation GIF via the shared renderer.
5. Embed adapter registration in launcher discovery.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.engine_core.engine_availability import skip_if_unavailable
from src.tools.tour_matching_viewer.core import (
    ENGINE_COLORS,
    MultiCandidateReplay,
    ReplayData,
    export_animation_gif,
)

pytestmark = [
    skip_if_unavailable("pyqt6"),
    pytest.mark.unit,
]


ROOT = Path(__file__).resolve().parents[3]
SPEC_PATH = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"


def _get_spec() -> dict[str, Any]:
    return json.loads(SPEC_PATH.read_text(encoding="utf-8"))


def _build_synthetic_replay(engine: str, offset: float) -> ReplayData:
    spec = _get_spec()
    coords_order = tuple(spec["coordinate_order"])
    n_coords = len(coords_order)
    time_s = np.linspace(0.0, 0.3, 4)
    q = np.ones((4, n_coords), dtype=float) * offset
    model_markers = np.ones((4, 5, 3), dtype=float) * (offset + 0.01)
    target_markers = np.ones((4, 5, 3), dtype=float) * offset
    valid_mask = np.ones((4, 5), dtype=bool)
    return ReplayData(
        time_s=time_s,
        coordinates=q,
        model_markers_m=model_markers,
        target_markers_m=target_markers,
        valid_mask=valid_mask,
        coordinate_names=coords_order,
    )


def test_multi_candidate_replay_construction_and_rms() -> None:
    r_mujoco = _build_synthetic_replay("mujoco", 0.1)
    r_pinocchio = _build_synthetic_replay("pinocchio", 0.2)

    combo = MultiCandidateReplay(
        candidates=(
            ("mujoco", r_mujoco),
            ("pinocchio", r_pinocchio),
        )
    )

    assert len(combo.candidates) == 2
    assert combo.frame_count == 4
    rms_dict = combo.get_per_engine_rms(0)
    assert "mujoco" in rms_dict
    assert "pinocchio" in rms_dict
    # Marker difference was 0.01 m = 10 mm
    assert np.isclose(rms_dict["mujoco"], 0.01, atol=1e-4)
    assert np.isclose(rms_dict["pinocchio"], 0.01, atol=1e-4)


def test_multi_candidate_replay_max_four_candidates() -> None:
    replays = [
        ("mujoco", _build_synthetic_replay("mujoco", 0.1)),
        ("pinocchio", _build_synthetic_replay("pinocchio", 0.2)),
        ("drake", _build_synthetic_replay("drake", 0.3)),
        ("opensim", _build_synthetic_replay("opensim", 0.4)),
    ]
    combo = MultiCandidateReplay(candidates=tuple(replays))
    assert len(combo.candidates) == 4

    # 5th candidate exceeds max of four
    excess = replays + [("simscape", _build_synthetic_replay("simscape", 0.5))]
    with pytest.raises(ValueError, match="at most 4"):
        MultiCandidateReplay(candidates=tuple(excess))


def test_engine_colors_mapping() -> None:
    for eng in ("mujoco", "pinocchio", "drake", "opensim", "default"):
        assert eng in ENGINE_COLORS
        assert ENGINE_COLORS[eng].startswith("#")


def test_export_animation_gif(tmp_path: Path) -> None:
    spec = _get_spec()
    replay = _build_synthetic_replay("mujoco", 0.05)
    out_gif = tmp_path / "test_swing.gif"

    res = export_animation_gif(
        replay,
        spec=spec,
        output_path=out_gif,
        fps=10,
        dpi=50,
        max_frames=3,
    )

    assert res.exists()
    assert res.stat().st_size > 0


def test_launcher_embed_registration() -> None:
    from src.shared.python.launcher_embed import (
        EMBEDDABLE_TOOL_REGISTRY,
        get_embeddable_tool,
    )
    import src.tools.tour_matching_viewer  # noqa: F401

    assert "tour_matching_viewer" in EMBEDDABLE_TOOL_REGISTRY
    tool = get_embeddable_tool("tour_matching_viewer")
    assert tool is not None
    caps = tool.embed_capabilities()
    assert caps.supports_embedded is True


@pytest.fixture(scope="module")
def qapp():  # noqa: ANN201
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


def test_tour_matching_viewer_widget_rejection_and_capabilities(qapp) -> None:  # noqa: ANN001
    from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

    widget = TourMatchingViewerWidget(spec_path=SPEC_PATH)
    assert widget._rejection_banner.isHidden() is True
    assert "Forces: Unsupported" in widget._capabilities_label.text()

    # Rejection status shows conspicuous banner
    widget.set_acceptance(False, "Marker error > 15 mm")
    assert widget._rejection_banner.isHidden() is False
    assert "Marker error > 15 mm" in widget._rejection_banner.text()
    assert "#d9534f" in widget._rejection_banner.styleSheet()

    # Acceptance hides banner
    widget.set_acceptance(True)
    assert widget._rejection_banner.isHidden() is True
    widget.cleanup()


def test_tour_matching_viewer_widget_multi_candidate_loading(qapp) -> None:  # noqa: ANN001
    from src.tools.tour_matching_viewer.gui import TourMatchingViewerWidget

    widget = TourMatchingViewerWidget(spec_path=SPEC_PATH)
    r1 = _build_synthetic_replay("mujoco", 0.01)
    r2 = _build_synthetic_replay("pinocchio", 0.02)

    widget.load_multi_candidates([("mujoco", r1), ("pinocchio", r2)])
    assert widget._multi_replay is not None
    assert len(widget._multi_replay.candidates) == 2
    assert "Multi-Candidate Replay" in widget._title_label.text()
    assert "Muj" in widget._rms_label.text()
    assert "Pin" in widget._rms_label.text()
    widget.cleanup()
