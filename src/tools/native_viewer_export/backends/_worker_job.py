"""Job description exchanged with subprocess workers (OpenSim, MyoSuite)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.shared.python.body_part_viz.axial_loads import AxialLoadFrame
    from src.shared.python.body_part_viz.force_colors import ForceColorScale
    from src.shared.python.force_overlay.glyphs import GlyphSet


@dataclass(frozen=True)
class WorkerJob:
    """Everything a render worker needs; frames are written as ``.npy`` files."""

    bundle_path: str
    q_path: str
    indices: list[int]
    views: list[str]
    width: int
    height: int
    lookat_m: list[float]
    distance_m: float | None
    out_dir: str
    glyphs_path: str | None = None  # JSON list of GlyphSet dicts aligned to indices
    #: look-at point per view and position (tracking views follow the grip midpoint)
    lookats: dict[str, list[list[float]]] | None = None
    #: decorative golf ball (``BallPlacement.to_dict()``) and its visibility per position
    ball: dict[str, Any] | None = None
    ball_visible: list[bool] | None = None
    #: axial-load frames (JSON list aligned to indices) and the ``ForceColorScale``
    #: dict that shades the segments (MuJoCo worker only)
    axial_path: str | None = None
    shading: dict[str, Any] | None = None

    def lookat_for(self, view: str, position: int) -> list[float]:
        """Look-at point of ``view`` at ``position`` (static ``lookat_m`` if unset)."""
        if self.lookats is None or view not in self.lookats:
            return self.lookat_m
        return self.lookats[view][position]

    def dump(self, path: Path) -> None:
        path.write_text(json.dumps(asdict(self)), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> WorkerJob:
        data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return cls(**data)

    def frame_path(self, view: str, position: int) -> Path:
        return Path(self.out_dir) / f"{view}_{position:05d}.npy"

    def load_axial_frames(self) -> list[AxialLoadFrame | None] | None:
        """Per-position axial frames staged for this job, or ``None`` if unshaded."""
        if not self.axial_path:
            return None
        from src.tools.native_viewer_export.shading import frames_from_json

        return frames_from_json(Path(self.axial_path))

    def load_shading(self) -> ForceColorScale | None:
        """The staged segment-shading scale, or ``None`` when shading is off."""
        if self.shading is None:
            return None
        from src.shared.python.body_part_viz.force_colors import ForceColorScale

        return ForceColorScale.from_dict(self.shading)

    def load_glyph_sets(self) -> list[GlyphSet] | None:
        """Per-frame glyph sets staged for this job, or ``None`` without overlays."""
        if not self.glyphs_path:
            return None
        from src.shared.python.force_overlay.glyphs import GlyphSet

        docs = json.loads(Path(self.glyphs_path).read_text(encoding="utf-8"))
        return [GlyphSet.from_dict(d) for d in docs]
