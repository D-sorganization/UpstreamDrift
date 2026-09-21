"""MyoSuite golfer scene path and marker site metadata (MS-51/52)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from src.shared.python.contracts import precondition

_ENGINE_ROOT = Path(__file__).resolve().parents[1]
_PLACEHOLDER_MYOBODY = (
    _ENGINE_ROOT.parent / "mujoco" / "myo_sim" / "body" / "myobody.xml"
)


@dataclass(frozen=True)
class GolferScene:
    """Resolved scene assets for kinematic replay."""

    xml_path: Path
    marker_sites: dict[str, dict[str, Any]]
    topology_note: str

    @property
    def is_placeholder(self) -> bool:
        return self.xml_path.name in {"myobody.xml", "myoupperbody.xml"} or (
            "placeholder" in self.xml_path.name
        )


@precondition(lambda: True, "scene resolution")
def resolve_golfer_scene(
    marker_sites: dict[str, dict[str, Any]] | None = None,
) -> GolferScene:
    """Return the best available MyoSuite golfer MJCF and marker site table."""
    if not _PLACEHOLDER_MYOBODY.is_file():
        raise FileNotFoundError(
            f"MyoSuite body MJCF missing: {_PLACEHOLDER_MYOBODY}. "
            "Run MS-51 setup_myosuite_models or vendor myo_sim."
        )
    default_sites = {
        "WaistLeft": {"body": "pelvis", "pos": [-0.12, 0.0, 0.0]},
        "WaistRight": {"body": "pelvis", "pos": [0.12, 0.0, 0.0]},
        "BackTop": {"body": "torso", "pos": [0.0, 0.0, 0.35]},
    }
    sites = marker_sites or default_sites
    return GolferScene(
        xml_path=_PLACEHOLDER_MYOBODY,
        marker_sites=dict(sites),
        topology_note=(
            "Vendored MyoBody placeholder MJCF until MS-51 pins myo_sim and "
            "generates golfer_myobody.xml with club weld and foot contacts."
        ),
    )
